/***
 * Multi-Phase star formation
 *
 * The algorithm here is based on Springel Hernequist 2003, and Okamoto 2010.
 *
 * The source code originally came from sfr_eff.c in Gadget-3. This version has
 * been heavily rewritten to add support for new wind models, new star formation
 * criterions, and more importantly, use the new tree walker routines.
 *
 * I (Yu Feng) feel it is appropriate to release this file with a free license,
 * because the implementation here has diverged from the original code by too far.
 *
 * Functions for self-gravity starformation condition and H2 are derived from Gadget-P
 * and used with permission of Phil Hopkins. Please cite the requisite papers if you use them.
 *
 * */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_machine.h>
#include "libgadget/timebinmgr.h"
#include "physconst.h"
#include "sfr_eff.h"
#include "cooling.h"
#include "slotsmanager.h"
#include "walltime.h"
#include "winds.h"
#include "starcluster.h"
/*Only for the star slot reservation*/
#include "forcetree.h"
#include "domain.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/* StarClusterReffRelation: the median size-mass relation R_eff = R4 pc (M_cl/1e4 Msun)^beta
 * that starcluster_sample_reff_pc() draws about, and the lognormal scatter (dex) that goes
 * with it (the relation's own scatter is used unless StarClusterReffScatter >= 0 replaces
 * it).  Both are PROJECTED effective radii -- the CW seed-mass model applies its own
 * r_max = 1.4 R_eff, so no 3-D deprojection belongs here. */
#define SC_REFF_BG21 0   /* Brown & Gnedin 2021, LEGUS clusters of 1-10 Myr ("young"): 2.365 pc, 0.180, 0.32 dex (default) */
#define SC_REFF_GBF  1   /* the relation used before this switch existed: 1.4 pc, 0.25, 0.5 dex */
static const double sc_reff_R4_pc[2] = {2.365, 1.4};
static const double sc_reff_beta[2]  = {0.180, 0.25};
static const double sc_reff_sigma[2] = {0.32,  0.5};
static const char * sc_reff_name[2]  = {"BG21", "GBF"};

/*Parameters of the star formation model*/
static struct SFRParams
{
    /* Master switch enabling star formation*/
    int StarformationOn;
    enum StarformationCriterion StarformationCriterion;  /*!< Type of star formation model. */
    int WindOn; /* if Wind is enabled */
    /*Star formation parameters*/
    double CritOverDensity;
    double CritPhysDensity;
    double OverDensThresh;
    double PhysDensThresh;
    double EgySpecSN;
    double FactorSN;
    double EgySpecCold;
    double FactorEVP;
    double TempSupernova;
    double TempClouds;
    double MaxSfrTimescale;
    int BHFeedbackUseTcool;
    int StarClusterOn; /* if star cluster bh seeding formation is enabled */
    int StarClusterSampling; /* if star cluster mass sampling is enabled (requires StarClusterOn) */
    int StarClusterICMFcutoff; /* if 1, ICMF has exp(-m/Mcut) cutoff; if 0, pure power law m^-2 on [1e2,1e8] */
    int SeedSecFOFcomSample; /* combined per-secFOF cluster sampling for BH seeding; skips per-star sampling */
    int SCmasscapSecFOFstarmass; /* if 1, cap the combined-sampled SC mass at the group's unseeded stellar mass */
    /* If > 0, use this fixed effective radius (in pc) for every seeded star cluster
     * instead of the size-mass relation. Default 0 (use the size-mass relation). */
    double StarClusterFixReff;
    /* Which size-mass relation (SC_REFF_*) and scatter starcluster_sample_reff_pc draws
     * from when StarClusterFixReff = 0.  Only used with SecondFOFOn = 1: the secFOF
     * star-cluster seeding is the sampler's only caller. */
    int StarClusterReffRelation;
    /* Lognormal scatter (dex) that starcluster_sample_reff_pc actually draws with: the
     * selected relation's own value (sc_reff_sigma) unless the parameter
     * StarClusterReffScatter is >= 0, in which case that value replaces it. */
    double StarClusterReffSigma;
    /*!< may be used to set a floor for the gas temperature */
    double MinGasTemp;
    /* Precomputed constants for M_cstar calculation (in code units) */
    double t_sn_code;    /* Supernova timescale t_sn = 3 Myr in code time */
    double phi_fb_code;  /* Feedback efficiency phi_fb = 0.16 cm^2/s^3 in code units */
    /* Mass limits for msc_ave calculation (in code mass units) */
    double msc_min_code; /* 1e2 Msun in code mass */
    double msc_max_code; /* 1e8 Msun in code mass */
    double msc_ave_powerlaw_code; /* precomputed <m> of the pure power law m^-2 on [msc_min,msc_max] (StarClusterICMFcutoff=0); cutoff-independent constant */
    double msc_seed_thresh_code; /* 1e4 Msun in code mass: "massive cluster" cutoff for bhseed_msc */
    double msc_multiseed_thresh_code; /* 1e8 Msun in code mass: per-secFOF multi-seed threshold (M_SC>this seeds floor(M_SC/1e8) BHs) */
    /* StarClusterCFEsigma: 0 = tabulated Gamma(P/k_B), i.e. sigma_loc = sqrt(P/rho) (original); 1 / 2 = Kruijssen (2012)
     * eq. 26 with the per-step measured 1-D dispersion of the star-forming neighbours (VDisp_gas_sf) / of all gas neighbours (VDisp_gas_all) */
    int StarClusterCFEsigma;
    double cs_cold_code;  /* cold-gas sound speed of the Kruijssen model, 0.3 km/s, in code velocity */
    double t_inc_code;    /* incomplete-star-formation time of the Kruijssen model, 10 Myr, in code time (with h, like t_sn_code) */

    /* Unit conversion factor for the sfr_due_to_h2 function*/
    double tau_fmol_unit;
    /*Lyman alpha forest specific star formation.*/
    double QuickLymanAlphaProbability;
    double QuickLymanAlphaTempThresh;
    /* Number of stars to create from each gas particle*/
    int Generations;
    /* Average starting mass for a gas particle.*/
    double avg_baryon_mass;
    /* U = temp_to_u / meanweight  * T
     * temp_to_u = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) / (UnitEnergy_in_cgs / UnitMass_in_g)*/
    double temp_to_u;
    /* COnversion factor from internal SFR units to solar masses per year*/
    double UnitSfr_in_solar_per_year;
    /* Conversion factor from code pressure (GAMMA_MINUS1 * rho_phys * u) to P/k_B [K/cm^3] */
    double pressure_to_pkb;
    /* The temperature boost from reionisation. Following 1807.09282,
     * we use a fixed, density independent value of 20000 K. I also tried
     * their eq. 3-4 but found that for this low resolution (of the UV grid) the
     * density gradients were too small and Treion was only 10 K.
     */
    double HIReionTemp;
    /* Input files for the various cooling modules*/
    char TreeCoolFile[100];
    char J21CoeffFile[100];
    char MetalCoolFile[100];
    char UVFluctuationFile[100];
    /* Boost SF for dense gas*/
    int BoostSFDenseGas;
    double BoostSFOverDenseFactor;
    /* File with the helium reionization table*/
    char ReionHistFile[100];
} sfr_params;

int get_generations(void)
{
    return sfr_params.Generations;
}

/* Structure storing the results of an evaluation of the star formation model*/
struct sfr_eeqos_data
{
    /* Relaxation time*/
    double trelax;
    /* Star formation timescale*/
    double tsfr;
    /* Internal energy of the gas in the hot phase. */
    double egyhot;
    /* Internal energy of the gas in the cold phase.*/
    double egycold;
    /* Fraction of the gas in the cold cloud phase. */
    double cloudfrac;
    /* Electron fraction after cooling. */
    double ne;
};

/* Computes properties of the gas on star forming equation of state*/
static struct sfr_eeqos_data get_sfr_eeqos(struct particle_data * part, struct sph_particle_data * sph, double dtime, struct UVBG *local_uvbg, const double redshift, const double a3inv);

/*Cooling only: no star formation*/
static void cooling_direct(int i, const double redshift, const double a3inv, const double hubble, const double GravInternal, const struct UVBG * const GlobalUVBG);

static void cooling_relaxed(int i, double dtime, struct UVBG * local_uvbg, const double redshift, const double a3inv, struct sfr_eeqos_data sfr_data, const struct UVBG * const GlobalUVBG);

/* Update the active particle list when a new star is formed.*/
static int add_new_particle_to_active(const int parent, const int child, ActiveParticles * act);
static int copy_gravaccel_new_particle(const int parent, const int child, MyFloat (* GravAccel)[3], int64_t nstoredgravaccel);

static int make_particle_star(int child, int parent, int placement, double Time, const double GravInternal, const RandTable * const rnd);
static int starformation(int i, double *localsfr, MyFloat * sm_out, MyFloat * sum_sm, MyFloat * sum_dtime, MyFloat * GradRho, const double redshift, const double a3inv, const double hubble, const double GravInternal, const struct UVBG * const GlobalUVBG, const RandTable * const rnd);
static int quicklyastarformation(int i, const double a3inv, const RandTable * const rnd);
static double get_sfr_factor_due_to_selfgravity(int i, const double atime, const double a3inv, const double hubble, const double GravInternal);
static double get_sfr_factor_due_to_h2(int i, MyFloat * GradRho_mag, const double atime);
static double get_starformation_rate_full(int i, MyFloat * GradRho, struct sfr_eeqos_data sfr_data, const double atime, const double a3inv, const double hubble, const double GravInternal);
static double get_egyeff(double redshift, double dens, struct UVBG * uvbg);
static double find_star_mass(int i, const double avg_baryon_mass);
/*Get enough memory for new star slots. This may be excessively slow! Don't do it too often.*/
static int * sfr_reserve_slots(ActiveParticles * act, int * NewStars, int NumNewStar, ForceTree * tt);

/* Convert entropy to internal energy*/
static double entropy_to_u(const double density, const double a3inv)
{
    return exp(GAMMA_MINUS1 * log(density * a3inv))/GAMMA_MINUS1;
//     return pow(density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
}

/*Set the parameters of the SFR module*/
void set_sfr_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /*Star formation parameters*/
        sfr_params.StarformationCriterion = (enum StarformationCriterion) param_get_enum(ps, "StarformationCriterion");
        sfr_params.CritOverDensity = param_get_double(ps, "CritOverDensity");
        sfr_params.CritPhysDensity = param_get_double(ps, "CritPhysDensity");
        sfr_params.WindOn = param_get_int(ps, "WindOn");
        sfr_params.BoostSFDenseGas = param_get_int(ps, "BoostSFDenseGas");
        sfr_params.BoostSFOverDenseFactor = param_get_double(ps, "BoostSFOverDenseFactor");

        sfr_params.FactorSN = param_get_double(ps, "FactorSN");
        sfr_params.FactorEVP = param_get_double(ps, "FactorEVP");
        sfr_params.TempSupernova = param_get_double(ps, "TempSupernova");
        sfr_params.TempClouds = param_get_double(ps, "TempClouds");
        sfr_params.MaxSfrTimescale = param_get_double(ps, "MaxSfrTimescale");
        sfr_params.Generations = param_get_int(ps, "Generations");
        if(sfr_params.Generations > 14)
            endrun(0, "Generations is %d, only space in the bitfield for 14.\n", sfr_params.Generations);
        sfr_params.MinGasTemp = param_get_double(ps, "MinGasTemp");
        sfr_params.BHFeedbackUseTcool = param_get_int(ps, "BHFeedbackUseTcool");
        if(sfr_params.BHFeedbackUseTcool > 3 || sfr_params.BHFeedbackUseTcool < 0)
            endrun(0, "BHFeedbackUseTcool mode %d not supported\n", sfr_params.BHFeedbackUseTcool);
        /*Lyman-alpha forest parameters*/
        sfr_params.StarClusterOn = param_get_int(ps, "StarClusterOn");
        sfr_params.StarClusterSampling = param_get_int(ps, "StarClusterSampling");
        if(sfr_params.StarClusterSampling && !sfr_params.StarClusterOn)
            endrun(0, "StarClusterSampling = 1 requires StarClusterOn = 1\n");
        sfr_params.StarClusterICMFcutoff = param_get_int(ps, "StarClusterICMFcutoff");
        sfr_params.SeedSecFOFcomSample = param_get_int(ps, "SeedSecFOFcomSample");
        if(sfr_params.SeedSecFOFcomSample && !sfr_params.StarClusterOn)
            endrun(0, "SeedSecFOFcomSample = 1 requires StarClusterOn = 1\n");
        sfr_params.SCmasscapSecFOFstarmass = param_get_int(ps, "SCmasscapSecFOFstarmass");
        sfr_params.StarClusterFixReff = param_get_double(ps, "StarClusterFixReff");
        if(sfr_params.StarClusterFixReff < 0)
            endrun(0, "StarClusterFixReff (%g) must be >= 0.\n", sfr_params.StarClusterFixReff);
        /* StarClusterReffRelation: only meaningful with SecondFOFOn = 1 (and StarClusterOn),
         * where the secFOF seeding samples cluster radii.  An unrecognised value is an
         * error there; without the second FOF the parameter is inert and BG21 is kept. */
        {
            const char * rel = param_get_string(ps, "StarClusterReffRelation");
            const int SecondFOFOn = param_get_int(ps, "SecondFOFOn");
            if(strcmp(rel, "BG21") == 0)
                sfr_params.StarClusterReffRelation = SC_REFF_BG21;
            else if(strcmp(rel, "GBF") == 0)
                sfr_params.StarClusterReffRelation = SC_REFF_GBF;
            else if(SecondFOFOn)
                endrun(0, "StarClusterReffRelation = '%s' is not supported: use 'BG21' or 'GBF'.\n", rel);
            else {
                sfr_params.StarClusterReffRelation = SC_REFF_BG21;
                message(0, "StarClusterReffRelation = '%s' ignored (SecondFOFOn = 0); BG21 kept.\n", rel);
            }
            /* StarClusterReffScatter: < 0 (default) keeps the scatter that belongs to the
             * selected relation; >= 0 replaces it (0 puts every cluster on the median
             * relation).  Like the relation itself it only matters where the sampler is
             * called, i.e. with SecondFOFOn = 1 and StarClusterFixReff = 0. */
            const double scat = param_get_double(ps, "StarClusterReffScatter");
            const int r = sfr_params.StarClusterReffRelation;
            sfr_params.StarClusterReffSigma = scat < 0 ? sc_reff_sigma[r] : scat;
            if(SecondFOFOn && sfr_params.StarClusterOn && sfr_params.StarClusterFixReff <= 0) {
                message(0, "StarClusterReffRelation = %s: star-cluster R_eff = %g pc (M_cl/1e4 Msun)^%g with %g dex lognormal scatter (%s).\n",
                        sc_reff_name[r], sc_reff_R4_pc[r], sc_reff_beta[r], sfr_params.StarClusterReffSigma,
                        scat < 0 ? "the relation's own value" : "set by StarClusterReffScatter");
            }
            else if(scat >= 0)
                message(0, "StarClusterReffScatter = %g ignored (%s).\n", scat,
                        sfr_params.StarClusterFixReff > 0 ? "StarClusterFixReff > 0 fixes every cluster radius" : "no star-cluster radii are sampled");
        }
        sfr_params.StarClusterCFEsigma = param_get_int(ps, "StarClusterCFEsigma");
        if(sfr_params.StarClusterCFEsigma < 0 || sfr_params.StarClusterCFEsigma > 2)
            endrun(0, "StarClusterCFEsigma = %d is not supported (0, 1 or 2)\n", sfr_params.StarClusterCFEsigma);
        if(sfr_params.StarClusterCFEsigma > 0 && !sfr_params.StarClusterOn)
            endrun(0, "StarClusterCFEsigma > 0 requires StarClusterOn = 1\n");
        if(sfr_params.StarClusterCFEsigma > 0)
            message(0, "StarClusterCFEsigma = %d: cluster formation efficiency from Kruijssen (2012) eq. 26 with the per-step measured "
                       "1-D dispersion of %s gas neighbours (sigma = 0, i.e. Gamma = sSFR_ff t_sn / (eps_core t_ff), when it is not defined)\n",
                    sfr_params.StarClusterCFEsigma, sfr_params.StarClusterCFEsigma == 1 ? "the star-forming" : "all");
        if(sfr_params.StarClusterOn) {
            int GasTidalField = param_get_int(ps, "GasTidalField");
            int SCgasVDisp = param_get_int(ps, "SCgasVDisp");
            if(!GasTidalField || !SCgasVDisp)
                endrun(0, "StarClusterOn = 1 requires GasTidalField = 1 and SCgasVDisp = 1\n");
        }
        sfr_params.QuickLymanAlphaProbability = param_get_double(ps, "QuickLymanAlphaProbability");
        sfr_params.QuickLymanAlphaTempThresh = param_get_double(ps, "QuickLymanAlphaTempThresh");
        sfr_params.HIReionTemp = param_get_double(ps, "HIReionTemp");

        /* File names*/
        param_get_string2(ps, "TreeCoolFile", sfr_params.TreeCoolFile, sizeof(sfr_params.TreeCoolFile));
        param_get_string2(ps, "J21CoeffFile", sfr_params.J21CoeffFile, sizeof(sfr_params.J21CoeffFile));
        param_get_string2(ps, "UVFluctuationfile", sfr_params.UVFluctuationFile, sizeof(sfr_params.UVFluctuationFile));
        param_get_string2(ps, "MetalCoolFile", sfr_params.MetalCoolFile, sizeof(sfr_params.MetalCoolFile));
        param_get_string2(ps, "ReionHistFile", sfr_params.ReionHistFile, sizeof(sfr_params.ReionHistFile));
    }
    MPI_Bcast(&sfr_params, sizeof(struct SFRParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* cooling and star formation routine.*/
void
cooling_and_starformation(ActiveParticles * act, double Time, double dloga, ForceTree * tree, struct grav_accel_store GravAccel, DomainDecomp * ddecomp, Cosmology *CP, MyFloat * GradRho, RandTable * rnd, FILE * FdSfr, int **NewStars_out, int64_t *NumNewStar_out)
{
    /*This is a queue for the new stars and their parents, so we can reallocate the slots after the main cooling loop.*/
    gadget_thread_arrays NewStarThread = {0}, NewParentThread = {0}, MaybeWindThread = {0};

    /*Need to capture this so that when NumActiveParticle increases during the loop
     * we don't add extra loop iterations on particles with invalid slots.*/
    const int nactive = act->NumActiveParticle;
    const double a3inv = 1./(Time * Time * Time);
    const double hubble = hubble_function(CP, Time);

    if(sfr_params.StarformationOn) {
        /* Maximally we need the active gas particles*/
        NewStarThread = gadget_setup_thread_arrays("NewStars", 0, act->NumActiveHydro);
        NewParentThread = gadget_setup_thread_arrays("NewParents", 1, act->NumActiveHydro);
    }

    MyFloat * StellarMass = NULL;
    if(sfr_params.WindOn && winds_are_subgrid()) {
        StellarMass = (MyFloat *) mymalloc("StellarMass", SlotsManager->info[0].size * sizeof(MyFloat));
        MaybeWindThread = gadget_setup_thread_arrays("MaybeWind", 0, act->NumActiveHydro);
    }

    /* Get the global UVBG for this redshift. */
    const double redshift = 1./Time - 1;
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);
    double sum_sm = 0, localsfr = 0, sum_dtime = 0;
    int64_t sum_sf_part = 0;

    /* First decide which stars are cooling and which starforming. If star forming we add them to a list.
     * Note the dynamic scheduling: individual particles may have very different loop iteration lengths.
     * Cooling is much slower than sfr. I tried splitting it into a separate loop instead, but this was faster.*/
    #pragma omp parallel reduction(+:localsfr) reduction(+: sum_sm) reduction(+:sum_dtime) reduction(+:sum_sf_part)
    {
        int i;
        const int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(i=0; i < nactive; i++)
        {
            /*Use raw particle number if active_set is null, otherwise use active_set*/
            const int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
            /* Skip non-gas or garbage particles */
            if(P[p_i].Type != 0 || P[p_i].IsGarbage || P[p_i].Mass <= 0)
                continue;

            int shall_we_star_form = 0;
            if(sfr_params.StarformationOn) {
                /*Reduce delaytime for wind particles.*/
                winds_evolve(p_i, a3inv, hubble);
                /* check whether we are star forming gas.*/
                if(sfr_params.QuickLymanAlphaProbability > 0)
                    shall_we_star_form = quicklyastarformation(p_i, a3inv, rnd);
                else
                    shall_we_star_form = sfreff_on_eeqos(&SPHP(p_i), a3inv);
            }

            if(shall_we_star_form) {
                int newstar = -1;
                MyFloat sm = 0;
                sum_sf_part++;
                if(sfr_params.QuickLymanAlphaProbability > 0) {
                    /*New star is always the same particle as the parent for quicklya*/
                    newstar = p_i;
                    sum_sm += P[p_i].Mass;
                    sm = P[p_i].Mass;
                } else {
                    newstar = starformation(p_i, &localsfr, &sm, &sum_sm, &sum_dtime, GradRho, redshift, a3inv, hubble, CP->GravInternal, &GlobalUVBG, rnd);
                }
                /*Add this particle to the stellar conversion queue if necessary.*/
                if(newstar >= 0) {
                    NewStarThread.srcs[tid][NewStarThread.sizes[tid]] = newstar;
                    NewStarThread.sizes[tid]++;
                    NewParentThread.srcs[tid][NewParentThread.sizes[tid]] = p_i;
                    NewParentThread.sizes[tid]++;
                }
                /* Add this particle to the queue for consideration to spawn a wind.
                 * Only for subgrid winds. */
                if(MaybeWindThread.sizes && newstar < 0) {
                    MaybeWindThread.srcs[tid][MaybeWindThread.sizes[tid]] = p_i;
                    StellarMass[P[p_i].PI] = sm;
                    MaybeWindThread.sizes[tid]++;
                }
            }
            else
                cooling_direct(p_i, redshift, a3inv, hubble, CP->GravInternal, &GlobalUVBG);
        }
    }

    report_memory_usage("SFR");

    walltime_measure("/Cooling/Cooling");

    /* Do subgrid winds*/
    if(sfr_params.WindOn && winds_are_subgrid()) {
        int * MaybeWind;
        int64_t NumMaybeWind = gadget_compact_thread_arrays(&MaybeWind, &MaybeWindThread);
        winds_subgrid(MaybeWind, NumMaybeWind, Time, StellarMass, rnd);
        myfree(MaybeWind);
        myfree(StellarMass);
    }

    int * NewStars = NewStarThread.dest;
    int * NewParents = NewParentThread.dest;
    int64_t NumNewStar = 0;

    /*Merge step for the queue.*/
    if(NewStars) {
        int64_t NumNewParent = gadget_compact_thread_arrays(&NewParents, &NewParentThread);
        NumNewStar = gadget_compact_thread_arrays(&NewStars, &NewStarThread);
        if(NumNewStar != NumNewParent)
            endrun(3,"%lu new stars, but %lu new parents!\n",NumNewStar, NumNewParent);
        /*Shrink star memory as we keep it for the wind model*/
        NewStars = (int *) myrealloc(NewStars, sizeof(int) * NumNewStar);
    }

    if(!sfr_params.StarformationOn)
        return;

    /*Get some empty slots for the stars*/
    int firststarslot = SlotsManager->info[4].size;
    /* We ran out of slots! We must be forming a lot of stars.
     * There are things in the way of extending the slot list, so we have to move them.
     * The code in sfr_reserve_slots is not elegant, but I cannot think of a better way.*/
    if(sfr_params.StarformationOn && (SlotsManager->info[4].size + NumNewStar >= SlotsManager->info[4].maxsize)) {
        if(NewParents)
            NewParents = (int *) myrealloc(NewParents, sizeof(int) * NumNewStar);
        NewStars = sfr_reserve_slots(act, NewStars, NumNewStar, tree);
    }
    SlotsManager->info[4].size += NumNewStar;

    int64_t stars_converted = 0, stars_spawned = 0, stars_spawned_gravity = 0;
    int i;
    double sum_mass_stars = 0;

    /*Now we turn the particles into stars*/
    #pragma omp parallel for schedule(static) reduction(+:stars_converted) reduction(+:stars_spawned) reduction(+:sum_mass_stars) reduction(+:stars_spawned_gravity)
    for(i=0; i < NumNewStar; i++)
    {
        int child = NewStars[i];
        int parent = NewParents[i];
        make_particle_star(child, parent, firststarslot+i, Time, CP->GravInternal, rnd);
        sum_mass_stars += P[child].Mass;
        if(child == parent)
            stars_converted++;
        else {
            /* Accumulate the spawned star mass on the parent gas particle */
            SPHP(parent).SumSpawnedMass += P[child].Mass;
            /* Update the active particle list when a new star is formed.*/
            stars_spawned_gravity += add_new_particle_to_active(parent, child, act);
            copy_gravaccel_new_particle(parent, child, GravAccel.GravAccel, GravAccel.nstore);
            stars_spawned++;
        }
    }
    act->NumActiveGravity += stars_spawned_gravity;

    /*Done with the parents*/
    myfree(NewParents);

    double total_sum_mass_stars = 0, total_sm = 0, totsfrrate = 0, total_sum_dtime = 0;
    int64_t total_sum_part = 0;

    MPI_Reduce(&localsfr, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_dtime, &total_sum_dtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_sf_part, &total_sum_part, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

    int64_t tot_spawned=0, tot_converted=0;
    MPI_Reduce(&stars_spawned, &tot_spawned, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stars_converted, &tot_converted, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

    if(FdSfr && total_sm > 0)
    {
        double rate = 0;

        if(total_sum_dtime > 0)
            rate = total_sm * total_sum_part / total_sum_dtime;

        /* convert to solar masses per yr */
        double rate_in_msunperyear = rate * sfr_params.UnitSfr_in_solar_per_year;

        /* Format:
         * Time = current scale factor,
         * total_sm = expected change in stellar mass this timestep.
         * This is: sigma_i dM_* = p_* M_* = M_i (1 - exp(-sm_i / M_i))
         * totsfrrate = current star formation rate in active particles in Msun/year.
         * This is: sigma_i dM_* / dt_i
         * rate_in_msunperyear = expected stellar mass formation rate in Msun/year from total_sm,
         * This is sigma_i dM_* / mean(dt_i), where dt_i is over the starforming particles, and may be
         * moderately different from columns 1 and 2.
         * total_sum_mass_stars = actual mass of stars formed this timestep (discretized total_sm).
         * This should be a noisier version of total_sm.
         * total_sum_dtime / total_sum_part : this is the average timsetep (dt) for the currently active star particles
         * total_sum_part: the number of actively star-forming particles
         * tot_new stars: number of new star particles spawned or converted this timestep
         * */
        fprintf(FdSfr, "%.12g %g %g %g %g %g %ld %ld\n", Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars, total_sum_dtime / total_sum_part, total_sum_part, tot_spawned + tot_converted);
        fflush(FdSfr);
    }

    if(tot_spawned || tot_converted)
        message(0, "SFR: spawned %ld stars, converted %ld gas particles into stars\n", tot_spawned, tot_converted);

    walltime_measure("/Cooling/StarFormation");

    /* Now apply the wind model using the list of new stars.*/
    if(sfr_params.WindOn && !winds_are_subgrid())
        winds_and_feedback(NewStars, NumNewStar, Time, rnd, tree, ddecomp);

    /* Return the NewStars list to the caller if requested (e.g. for BH
     * seeding from newly formed SC particles), otherwise free it here. */
    if(NewStars_out && NumNewStar_out) {
        *NewStars_out = NewStars;
        *NumNewStar_out = NumNewStar;
    } else {
        myfree(NewStars);
    }
}

/* Get enough memory for new star slots. This may be excessively slow! Don't do it too often.
 * It is also not elegant, but I couldn't think of a better way. May be fragile and need updating
 * if memory allocation patterns change. */
static int *
sfr_reserve_slots(ActiveParticles * act, int * NewStars, int NumNewStar, ForceTree * tree)
{
        /* SlotsManager is below Nodes and ActiveParticleList,
         * so we need to move them out of the way before we extend Nodes.
         * This is quite slow, but need not be collective and is faster than a tree rebuild.
         * Try not to do this too often.*/
        message(1, "Need %ld star slots, more than %ld available. Try increasing SlotsIncreaseFactor on restart.\n", SlotsManager->info[4].size, SlotsManager->info[4].maxsize);
        /*Move the NewStar array to upper memory*/
        int * new_star_tmp = NULL;
        if(NewStars) {
            new_star_tmp = (int *) mymalloc2("newstartmp", NumNewStar*sizeof(int));
            memmove(new_star_tmp, NewStars, NumNewStar * sizeof(int));
            myfree(NewStars);
        }
        /*Move the tree to upper memory*/
        struct NODE * nodes_base_tmp=NULL;
        int *Father_tmp=NULL;
        int *ActiveParticle_tmp=NULL;
        if(force_tree_allocated(tree)) {
            nodes_base_tmp = (struct NODE *) mymalloc2("nodesbasetmp", tree->numnodes * sizeof(struct NODE));
            memmove(nodes_base_tmp, tree->Nodes_base, tree->numnodes * sizeof(struct NODE));
            myfree(tree->Nodes_base);
            Father_tmp = (int *) mymalloc2("Father_tmp", PartManager->MaxPart * sizeof(int));
            memmove(Father_tmp, tree->Father, PartManager->MaxPart * sizeof(int));
            myfree(tree->Father);
        }
        if(act->ActiveParticle) {
            ActiveParticle_tmp = (int *) mymalloc2("ActiveParticle_tmp", act->NumActiveParticle * sizeof(int));
            memmove(ActiveParticle_tmp, act->ActiveParticle, act->NumActiveParticle * sizeof(int));
            myfree(act->ActiveParticle);
        }
        /*Now we can extend the slots! */
        int64_t atleast[6];
        int64_t i;
        for(i = 0; i < 6; i++)
            atleast[i] = SlotsManager->info[i].maxsize;
        atleast[4] += NumNewStar;
        slots_reserve(1, atleast, SlotsManager);

        /*And now we need our memory back in the right place*/
        if(ActiveParticle_tmp) {
            act->ActiveParticle = (int *) mymalloc("ActiveParticle", sizeof(int)*(act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
            memmove(act->ActiveParticle, ActiveParticle_tmp, act->NumActiveParticle * sizeof(int));
            myfree(ActiveParticle_tmp);
        }
        if(force_tree_allocated(tree)) {
            tree->Father = (int *) mymalloc("Father", PartManager->MaxPart * sizeof(int));
            memmove(tree->Father, Father_tmp, PartManager->MaxPart * sizeof(int));
            myfree(Father_tmp);
            tree->Nodes_base = (struct NODE *) mymalloc("Nodes_base", tree->numnodes * sizeof(struct NODE));
            memmove(tree->Nodes_base, nodes_base_tmp, tree->numnodes * sizeof(struct NODE));
            myfree(nodes_base_tmp);
            /*Don't forget to update the Node pointer as well as Node_base!*/
            tree->Nodes = tree->Nodes_base - tree->firstnode;
        }
        if(new_star_tmp) {
            NewStars = (int *) mymalloc("NewStars", NumNewStar*sizeof(int));
            memmove(NewStars, new_star_tmp, NumNewStar * sizeof(int));
            myfree(new_star_tmp);
        }
        return NewStars;
}

/* Cluster formation efficiency of a gas particle (slot data sp) with physical density rho_phys [code units].
 * StarClusterCFEsigma = 0: the tabulated Kruijssen (2012) Gamma(P/k_B) at the thermodynamic pressure, i.e. sigma_loc = sqrt(P/rho)
 *   (the original prescription; Pressure_over_kB is formed by the caller exactly as before, so mode 0 is unchanged).
 * 1 / 2: Kruijssen (2012) eq. 26 evaluated directly from rho_phys and the 1-D gas velocity dispersion measured in the density
 *   treewalk at the particle's last active step (SCgasVDisp): VDisp_gas_sf, the star-forming neighbours only (0 when there are
 *   fewer than VDISP_SF_MINNGB of them, which is then the sigma = 0 limit of the model), or VDisp_gas_all, all gas neighbours.
 *   The stored dispersions are internal velocities (a x peculiar), hence the division by atime. */
static double
gas_cluster_formation_efficiency(const struct sph_particle_data * sp, const double rho_phys, const double Pressure_over_kB, const double atime, const double GravInternal)
{
    if(sfr_params.StarClusterCFEsigma == 0)
        return get_cluster_formation_efficiency(Pressure_over_kB);
    const double sigma = (sfr_params.StarClusterCFEsigma == 1 ? sp->VDisp_gas_sf : sp->VDisp_gas_all) / atime;
    return get_cluster_formation_efficiency_k12(rho_phys, sigma, sfr_params.cs_cold_code, GravInternal,
                                                sfr_params.t_sn_code, sfr_params.phi_fb_code, sfr_params.t_inc_code);
}

static void
cooling_direct(int i, const double redshift, const double a3inv, const double hubble, const double GravInternal, const struct UVBG * const GlobalUVBG)
{
    /*  the actual time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift);
    double dtime = dloga / hubble;

    double ne = SPHP(i).Ne;	/* electron abundance (gives ionization state and mean molecular weight) */

    const double enttou = entropy_to_u(SPHP(i).Density, a3inv);

    /* Current internal energy including adiabatic change*/
    double uold = SPHP(i).Entropy * enttou;
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  SPHP(i).local_J21;
    zreion = SPHP(i).zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, GlobalUVBG, P[i].Pos, PartManager->CurrentParticleOffset, localJ21, zreion);
    double lasttime = exp(loga_from_ti(P[i].Ti_drift - dti_from_timebin(P[i].TimeBinHydro)));
    double lastred = 1/lasttime - 1;
    double unew;
    /* The particle reionized this timestep, bump the temperature to the HI reionization temperature.
     * We only do this for non-star-forming gas.*/
    if(sfr_params.HIReionTemp > 0 && uvbg.zreion >= redshift && uvbg.zreion < lastred) {
        /* We assume singly ionised helium at the time of reionisation */
        /* The 100% correct thing to do is to solve for the equilibrium ne based on the local UVBG
         * then calculate the mean weight based on this. The current approach will cause
         * a boost in reionisation temperatures proportional to the residual neutral fraction,
         * which should be relatively small most of the time. The 6 is because helium is singly
         * ionized, not doubly so.*/
        /* TODO: Make sure that not setting SPHP.Ne(i) here doesn't mess up anything between
         * now and the next cooling call when it gets set properly */
        const double meanweight = 4 / (8 - 6 * (1 - HYDROGEN_MASSFRAC));
        unew = sfr_params.temp_to_u / meanweight * sfr_params.HIReionTemp;
        //We don't want gas to cool by ionising
        if(uold > unew) unew = uold;
    }
    else {
        /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
        const double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);
        const double MinEgySpec = sfr_params.temp_to_u/meanweight * sfr_params.MinGasTemp;
        unew = DoCooling(redshift, uold, SPHP(i).Density * a3inv, dtime, &uvbg, &ne, SPHP(i).Metallicity, MinEgySpec, P[i].HeIIIionized);
    }

    SPHP(i).Ne = ne;
    /* Update the entropy. This is done after synchronizing kicks and drifts, as per run.c.*/
    SPHP(i).Entropy = unew / enttou;
    /* Cooling gas is not forming stars*/
    SPHP(i).Sfr = 0;

    /* Update the gas ClusterFormationEfficiency based on current density and entropy (and the measured dispersion, StarClusterCFEsigma > 0) */
    if (sfr_params.StarClusterOn) {
        double Pressure_over_kB = GAMMA_MINUS1 * SPHP(i).Density * a3inv
                                  * unew * sfr_params.pressure_to_pkb;
        SPHP(i).ClusterFormationEfficiency = gas_cluster_formation_efficiency(&SPHP(i), SPHP(i).Density * a3inv, Pressure_over_kB, 1. / (1 + redshift), GravInternal);
    }
}

/* Returns the density threshold for star formation in comoving units*/
double
sfr_density_threshold(const double atime)
{
    double thresh;
    if(sfr_params.QuickLymanAlphaProbability > 0)
        thresh = sfr_params.OverDensThresh;
    else {
        thresh = sfr_params.PhysDensThresh * (atime * atime * atime);
        if(thresh < sfr_params.OverDensThresh)
            thresh = sfr_params.OverDensThresh;
    }
    return thresh;
}

/* returns 1 if the particle is on the effective equation of state,
 * cooling via the relaxation equation and maybe forming stars.
 * 0 if the particle does not form stars, instead cooling normally.*/
int
sfreff_on_eeqos(const struct sph_particle_data * sph, const double a3inv)
{
    int flag = 0;
    /* no sfr: normal cooling*/
    if(!sfr_params.StarformationOn) {
        return 0;
    }

    if(sph->Density * a3inv >= sfr_params.PhysDensThresh)
        flag = 1;

    if(sph->Density < sfr_params.OverDensThresh)
        flag = 0;

    if(sph->DelayTime > 0)
        flag = 0;   /* only normal cooling for particles in the wind */

    /* The model from 0904.2572 makes gas not star forming if more than 0.5 dex above
     * the effective equation of state (at z=0). This in practice means black hole heated.*/
    if(flag == 1 && sfr_params.BHFeedbackUseTcool == 2) {
        //Redshift is the argument
        double redshift = cbrt(a3inv)-1;
        struct UVBG uvbg = get_global_UVBG(redshift);
        double egyeff = get_egyeff(redshift, sph->Density, &uvbg);
        const double enttou = entropy_to_u(sph->Density, a3inv);
        double unew = sph->Entropy * enttou;
        /* 0.5 dex = 10^0.5 = 3.2 */
        if(unew >= egyeff * 3.2)
            flag = 0;
    }
    return flag;
}

/*Get the neutral fraction of a particle correctly, accounting for being on the star-forming equation of state*/
double get_neutral_fraction_sfreff(double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata)
{
    double nh0;
    const double a3inv = (1+redshift)*(1+redshift)*(1+redshift);
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  sphdata->local_J21;
    zreion = sphdata->zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, &GlobalUVBG, partdata->Pos, PartManager->CurrentParticleOffset, localJ21, zreion);
    double physdens = sphdata->Density * a3inv;

    if(sfr_params.QuickLymanAlphaProbability > 0 || !sfreff_on_eeqos(sphdata, a3inv)) {
        /*This gets the neutral fraction for standard gas*/
        double InternalEnergy = sphdata->Entropy * entropy_to_u(sphdata->Density, a3inv);
        nh0 = GetNeutralFraction(InternalEnergy, physdens, &uvbg, sphdata->Ne);
    }
    else {
        /* This gets the neutral fraction for gas on the star-forming equation of state.
         * This needs special handling because the cold clouds have a different neutral
         * fraction than the hot gas*/
        double dloga = get_dloga_for_bin(partdata->TimeBinHydro, partdata->Ti_drift);
        double dtime = dloga / hubble;
        struct sfr_eeqos_data sfr_data = get_sfr_eeqos(partdata, sphdata, dtime, &uvbg, redshift, a3inv);
        double nh0cold = GetNeutralFraction(sfr_params.EgySpecCold, physdens, &uvbg, sfr_data.ne);
        double nh0hot = GetNeutralFraction(sfr_data.egyhot, physdens, &uvbg, sfr_data.ne);
        nh0 =  nh0cold * sfr_data.cloudfrac + (1-sfr_data.cloudfrac) * nh0hot;
    }
    return nh0;
}

double get_helium_neutral_fraction_sfreff(int ion, double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata)
{
    const double a3inv = (1+redshift)*(1+redshift)*(1+redshift);
    double helium;
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  sphdata->local_J21;
    zreion = sphdata->zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, &GlobalUVBG, partdata->Pos, PartManager->CurrentParticleOffset, localJ21, zreion);
    double physdens = sphdata->Density * a3inv;

    if(sfr_params.QuickLymanAlphaProbability > 0 || !sfreff_on_eeqos(sphdata, a3inv)) {
        /*This gets the neutral fraction for standard gas*/
        double InternalEnergy = sphdata->Entropy * entropy_to_u(sphdata->Density, a3inv);
        helium = GetHeliumIonFraction(ion, InternalEnergy, physdens, &uvbg, sphdata->Ne);
    }
    else {
        /* This gets the neutral fraction for gas on the star-forming equation of state.
         * This needs special handling because the cold clouds have a different neutral
         * fraction than the hot gas*/
        double dloga = get_dloga_for_bin(partdata->TimeBinHydro, partdata->Ti_drift);
        double dtime = dloga / hubble;
        struct sfr_eeqos_data sfr_data = get_sfr_eeqos(partdata, sphdata, dtime, &uvbg, redshift, a3inv);
        double nh0cold = GetHeliumIonFraction(ion, sfr_params.EgySpecCold, physdens, &uvbg, sfr_data.ne);
        double nh0hot = GetHeliumIonFraction(ion, sfr_data.egyhot, physdens, &uvbg, sfr_data.ne);
        helium =  nh0cold * sfr_data.cloudfrac + (1-sfr_data.cloudfrac) * nh0hot;
    }
    return helium;
}
/* This function turns a particle into a star. It returns 1 if a particle was
 * converted and 2 if a new particle was spawned. This is used
 * above to set stars_{spawned|converted}*/
/* Wrapper for gsl_sf_expint_E1 that treats underflow as zero instead of
 * triggering GSL's fatal error handler.
 *
 * E1(x) underflows to ~0 once exp(-x)/x is too small to represent (large x,
 * i.e. small cutoff masses). On underflow gsl_sf_expint_E1_e flags GSL_EUNDRFLW
 * *by invoking the global GSL error handler* before it returns. gadget/main.c
 * installs a handler that calls endrun(), and that handler is a single global
 * shared by every OpenMP thread.
 *
 * The previous implementation toggled the handler off/on around the call. That
 * is correct only in serial: make_particle_star() calls this from inside an
 * "omp parallel for" (see cooling_and_starformation), so concurrent threads
 * raced on the global handler — one thread would restore it while another was
 * still inside gsl_sf_expint_E1_e hitting an underflow, aborting the run with a
 * spurious "GSL_ERROR ... errno:15 ... underflow".
 *
 * Fix: never let GSL reach the underflow branch. Reproduce GSL's own threshold
 * (GSL 2.6 src/specfunc/expint.c returns UNDERFLOW for x > xmax) and return 0
 * directly, so the global handler is never invoked. No handler toggling => no
 * global side effects => thread-safe. */
static double safe_expint_E1(double x)
{
    const double xmaxt = -GSL_LOG_DBL_MIN;
    const double xmax  = xmaxt - log(xmaxt);
    if(x >= xmax)
        return 0.0;
    gsl_sf_result result;
    int status = gsl_sf_expint_E1_e(x, &result);
    if(status == GSL_EUNDRFLW)
        return 0.0;
    if(status)
        endrun(2001, "GSL_ERROR in safe_expint_E1: x=%g, errno:%d, error: %s\n",
               x, status, gsl_strerror(status));
    return result.val;
}

/* Average cluster mass <m> of the pure power law n(m) ~ m^-2 on [msc_min, msc_max]
 * (StarClusterICMFcutoff = 0), in code mass units. Independent of any cutoff:
 *   <m> = ln(m_max/m_min) / (1/m_min - 1/m_max). */
static double msc_ave_powerlaw(void)
{
    double m_min = sfr_params.msc_min_code;
    double m_max = sfr_params.msc_max_code;
    double denom = 1.0 / m_min - 1.0 / m_max;
    if(denom > 0)
        return log(m_max / m_min) / denom;
    return 0;
}

/* Inverse-CDF draw of one cluster mass from the pure power law n(m) ~ m^-2 on
 * [msc_min, msc_max] (StarClusterICMFcutoff = 0), given a uniform deviate u in
 * [0,1). Closed form, no cutoff and no bisection:
 *   CDF(m) = (1/m_min - 1/m) / (1/m_min - 1/m_max)
 *   => m = 1 / (1/m_min - u*(1/m_min - 1/m_max)). */
static double msc_sample_powerlaw(double u)
{
    double inv_min = 1.0 / sfr_params.msc_min_code;
    double inv_max = 1.0 / sfr_params.msc_max_code;
    return 1.0 / (inv_min - u * (inv_min - inv_max));
}

/* Average cluster mass <m> of n(m) ~ m^-2 exp(-m/Mcut) over [msc_min, msc_max],
 * in code mass units. Returns 0 if Mcut <= 0 or the normalisation is non-positive.
 *   <m> = Mcut * [E1(x_min) - E1(x_max)] /
 *         [exp(-x_min)/x_min - exp(-x_max)/x_max + E1(x_max) - E1(x_min)],   x = m/Mcut.
 * When StarClusterICMFcutoff = 0 the exponential cutoff is dropped and the pure
 * power-law average (a cutoff-independent constant precomputed at init) is
 * returned instead. */
static double msc_ave_from_cutoff(double Mcut)
{
    if(Mcut <= 0)
        return 0;
    if(!sfr_params.StarClusterICMFcutoff)
        return sfr_params.msc_ave_powerlaw_code;
    double x_min = sfr_params.msc_min_code / Mcut;
    double x_max = sfr_params.msc_max_code / Mcut;
    double E1_min = safe_expint_E1(x_min);
    double E1_max = safe_expint_E1(x_max);
    double numer = E1_min - E1_max;
    double denom = exp(-x_min) / x_min - exp(-x_max) / x_max + E1_max - E1_min;
    if(denom > 0)
        return Mcut * numer / denom;
    return 0;
}

/* ---------------- shared ICMF draw (used by every star-cluster sampler) ----------------
 * The three samplers below (summed seed mass, per-cluster mass list, and the
 * StarClusterDetails full-population pass) MUST draw the identical cluster
 * population from the same rand_id, so the Poisson draw and the per-cluster
 * inverse-CDF live here in one place instead of being copied per sampler. */

/* Poisson draw of the cluster count: normal approximation for large lambda,
 * Knuth's product method otherwise. RNG offsets +10,+11 (and +10.. for Knuth).
 * errcode identifies the caller in the (never-expected) runaway-iteration abort. */
static int msc_poisson_draw(double lambda, uint64_t rand_id, const RandTable * const rnd, int errcode)
{
    if(lambda > 30) {
        double u1 = get_random_number(rand_id + 10, rnd);
        double u2 = get_random_number(rand_id + 11, rnd);
        if(u1 < 1e-20)
            u1 = 1e-20;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        int sample = (int)(lambda + sqrt(lambda) * z + 0.5);
        return sample > 0 ? sample : 0;
    }
    if(!(lambda > 0))       /* negated so a NaN lambda cannot enter the Knuth loop */
        return 0;
    const int max_iter = 200;
    double L = exp(-lambda);
    double p = 1.0;
    int k = 0;
    uint64_t seed = rand_id + 10;
    do {
        k++;
        p *= get_random_number(seed, rnd);
        seed++;
        if(k > max_iter)
            endrun(errcode, "Star-cluster Poisson sampling exceeded %d iterations for lambda=%g, rand_id=%ld\n",
                   max_iter, lambda, (long)rand_id);
    } while(p > L);
    return k - 1;
}

/* Cutoff-dependent constants of the ICMF inverse-CDF, computed once per draw. */
struct msc_icmf {
    int    use_cutoff;          /* StarClusterICMFcutoff */
    double Mcut;
    double x_min, x_max;        /* msc_min/max in units of Mcut */
    double E1_xmin;
    double emxmin_over_xmin;
    double g_norm;
};

/* StarClusterICMFcutoff = 1: n(m) ~ m^-2 exp(-m/Mcut), inverse-CDF via bisection,
 *   CDF(x) prop. e^{-x_min}/x_min - e^{-x}/x + E1(x) - E1(x_min),  x = m/Mcut.
 * StarClusterICMFcutoff = 0: pure power law n(m) ~ m^-2 on [1e2,1e8] Msun
 *   (closed-form inverse-CDF, cutoff-independent), so nothing to precompute. */
static void msc_icmf_init(struct msc_icmf * p, double Mcut)
{
    memset(p, 0, sizeof(*p));
    p->use_cutoff = sfr_params.StarClusterICMFcutoff;
    p->Mcut = Mcut;
    if(!p->use_cutoff)
        return;
    p->x_min = sfr_params.msc_min_code / Mcut;
    p->x_max = sfr_params.msc_max_code / Mcut;
    p->E1_xmin = safe_expint_E1(p->x_min);
    p->emxmin_over_xmin = exp(-p->x_min) / p->x_min;
    p->g_norm = p->emxmin_over_xmin - exp(-p->x_max) / p->x_max
              + safe_expint_E1(p->x_max) - p->E1_xmin;
}

/* One cluster mass (code units) from the uniform deviate u in [0,1). */
static double msc_icmf_sample(const struct msc_icmf * p, double u)
{
    if(!p->use_cutoff)
        return msc_sample_powerlaw(u);
    double target = u * p->g_norm;
    double lo = p->x_min, hi = p->x_max;
    for(int iter = 0; iter < 50; iter++) {
        double mid = 0.5 * (lo + hi);
        double g_mid = p->emxmin_over_xmin - exp(-mid) / mid
                     + safe_expint_E1(mid) - p->E1_xmin;
        if(g_mid < target)
            lo = mid;
        else
            hi = mid;
    }
    return p->Mcut * 0.5 * (lo + hi);
}

/* ---- SCmasscapSecFOFstarmass: draw-order truncation at Mcut ----
 * A group cannot turn more stellar mass into clusters than it has unseeded stars, so
 * the drawn population is truncated in DRAW ORDER at Mcut (= the group's unseeded
 * stellar mass): clusters are accepted until the running total reaches Mcut, the
 * cluster that crosses it is shortened to the remaining budget, and every later
 * cluster is dropped.  The total therefore lands exactly on Mcut -- unless the
 * remainder is below the ICMF floor msc_min_code (100 Msun), in which case the
 * crossing cluster is dropped whole rather than shortened: a shortened cluster is no
 * longer an ICMF draw, and one lighter than the mass function's own lower limit is not
 * a cluster the model can represent (its size, VMS mass and seed mass would all be
 * extrapolated off the bottom of their fits).  The draw then lands just under Mcut,
 * short by less than 100 Msun.
 *
 * Draw order matters.  With StarClusterICMFcutoff = 0 the ICMF is a pure m^-2 with no
 * knowledge of the host, so a small group can draw a cluster heavier than all its
 * stars; truncating a prefix of the draw removes clusters without regard to their
 * mass, which leaves the surviving subset an unbiased sample of the ICMF.  Truncating
 * after the descending sort in starcluster_combined_seed_masslist would instead only
 * ever delete the lightest clusters -- a cap in name only, since the offending heavy
 * cluster is exactly the one that would survive.
 *
 * All three samplers below share this so they keep drawing the identical population
 * (see the invariant note above msc_poisson_draw): the budget is consumed by EVERY
 * drawn cluster, including ones a caller then discards for being outside its mass
 * window, so the truncation point does not depend on which sampler is asking. */
struct msc_budget {
    int    on;      /* cap active for this draw */
    double left;    /* stellar mass still available (code units) */
};

static void msc_budget_init(struct msc_budget * b, double Mcut, int allow)
{
    b->on = allow && sfr_params.SCmasscapSecFOFstarmass;
    b->left = Mcut;
}

/* Charge one drawn cluster against the remaining budget and return the mass that
 * actually forms: the cluster itself while the budget covers it, the remainder for the
 * cluster that crosses Mcut, then 0 -- which every caller treats as end-of-draw.
 * A remainder below msc_min_code (the 100 Msun ICMF floor) cannot stand as a cluster,
 * so the crossing cluster is dropped instead of shortened and the budget is closed. */
static double msc_budget_take(struct msc_budget * b, double mass)
{
    if(!b->on)
        return mass;
    if(b->left <= 0)
        return 0;
    if(mass > b->left) {
        if(b->left < sfr_params.msc_min_code) {
            b->left = 0;    /* leftover too small to be a cluster: it forms nothing */
            return 0;
        }
        mass = b->left;
    }
    b->left -= mass;
    return mass;
}

/* Combined per-secFOF star-cluster sampling for BH seeding (SeedSecFOFcomSample).
 * Draws ONE cluster population for a whole secFOF group:
 *   - mass function n(m) ~ m^-2 exp(-m/Mcut) on [1e2, 1e8] Msun, cutoff Mcut = group
 *     total (unseeded) stellar mass;
 *   - n = sum_mGamma / <m>; N ~ Poisson(n); then N cluster masses;
 *   - returns the summed mass of sampled clusters above 1e4 Msun (bhseed_msc);
 *   - if total_sampled_out != NULL, also returns there the summed mass of ALL
 *     sampled clusters (the full draw, no 1e4 Msun threshold).
 * All masses are in code units. rand_id seeds the (reproducible) RNG draws.
 * NOT OpenMP-safe: safe_expint_E1 toggles the global GSL error handler, so this
 * must be called from a serial context. */
double starcluster_combined_bhseed_msc(double Mcut, double sum_mGamma,
                                       uint64_t rand_id, const RandTable * const rnd,
                                       double * total_sampled_out, int allow_cap)
{
    if(total_sampled_out)
        *total_sampled_out = 0;
    if(Mcut <= 0 || sum_mGamma <= 0)
        return 0;
    double m_ave = msc_ave_from_cutoff(Mcut);
    if(m_ave <= 0)
        return 0;
    double lambda = sum_mGamma / m_ave;

    int N = msc_poisson_draw(lambda, rand_id, rnd, 7774);
    if(N <= 0)
        return 0;

    /* Sample N cluster masses from the ICMF, summing those above the massive-cluster
     * threshold (1e4 Msun). RNG offsets +300+s, matching the per-star sampler.
     * SCmasscapSecFOFstarmass truncates the draw in order at Mcut, so the loop ends
     * as soon as the group's unseeded stellar mass is spent (bhseed_msc <=
     * total_sampled <= Mcut, the gap below Mcut being at most one sub-100-Msun
     * remainder).  allow_cap == 0 disables it for the per-particle sampler
     * (SeedSecFOFcomSampleParticle), which passes Mcut = min(M_cstar, group stellar
     * mass) per star: there the cap belongs on the group-summed total, applied by the
     * caller, not on each per-star draw. */
    struct msc_icmf icmf;
    msc_icmf_init(&icmf, Mcut);
    struct msc_budget budget;
    msc_budget_init(&budget, Mcut, allow_cap);

    double bhseed_msc = 0;
    double total_sampled = 0;
    for(int s = 0; s < N; s++) {
        double u_s = get_random_number(rand_id + 300 + (uint64_t)s, rnd);
        double mass = msc_budget_take(&budget, msc_icmf_sample(&icmf, u_s));
        if(mass <= 0)
            break;              /* stellar mass budget spent: the rest never forms */
        total_sampled += mass;
        if(mass > sfr_params.msc_seed_thresh_code)
            bhseed_msc += mass;
    }
    if(total_sampled_out)
        *total_sampled_out = total_sampled;
    return bhseed_msc;
}

/* Descending comparator for doubles (largest first). */
static int cmp_double_desc(const void * a, const void * b)
{
    double x = *(const double *) a, y = *(const double *) b;
    return (x < y) - (x > y);
}

/* Per-cluster secFOF seeding (SecFOFseedsumover=0): see sfr_eff.h. Draws the same Poisson cluster
 * population and individual cluster masses as starcluster_combined_bhseed_msc (identical
 * RNG offsets and StarClusterICMFcutoff-aware mass function), but instead of summing the
 * masses > 1e4 Msun it returns the count of clusters >= min_seed_mass and the largest
 * min(n_qualify, cap) of those masses (descending). SCmasscapSecFOFstarmass applies here
 * too, as a draw-order truncation at Mcut (see struct msc_budget): the returned masses
 * are those of the surviving prefix, so n_qualify itself shrinks when the cap bites. */
int starcluster_combined_seed_masslist(double Mcut, double sum_mGamma,
                                       uint64_t rand_id, const RandTable * const rnd,
                                       double min_seed_mass, double * out_masses, int cap)
{
    if(Mcut <= 0 || sum_mGamma <= 0)
        return 0;
    double m_ave = msc_ave_from_cutoff(Mcut);
    if(m_ave <= 0)
        return 0;
    double lambda = sum_mGamma / m_ave;

    int N = msc_poisson_draw(lambda, rand_id, rnd, 7775);
    if(N <= 0)
        return 0;

    /* Sample N cluster masses (same inverse-CDF draw and RNG offsets +300+s as the
     * summed sampler), collecting those >= min_seed_mass. When collecting, all qualifying
     * masses are buffered and sorted once (O(N + n_qualify log n_qualify)); the count-only
     * path (out_masses == NULL or cap <= 0, used for the slot upper bound) allocates
     * nothing. The RNG draw sequence is identical either way, so both calls agree.
     * SCmasscapSecFOFstarmass truncates the draw in order at Mcut BEFORE the
     * min_seed_mass test and before the sort, so the cap deletes an unbiased prefix of
     * the population rather than only its lightest members. */
    struct msc_icmf icmf;
    msc_icmf_init(&icmf, Mcut);
    struct msc_budget budget;
    msc_budget_init(&budget, Mcut, 1);

    int collecting = (out_masses != NULL && cap > 0);
    double * qual = collecting ? (double *) mymalloc("SeedMassQual", (size_t)N * sizeof(double)) : NULL;

    int n_qualify = 0;
    for(int s = 0; s < N; s++) {
        double u_s = get_random_number(rand_id + 300 + (uint64_t)s, rnd);
        double mass = msc_budget_take(&budget, msc_icmf_sample(&icmf, u_s));
        if(mass <= 0)
            break;              /* stellar mass budget spent: the rest never forms */
        if(mass >= min_seed_mass) {
            if(collecting)
                qual[n_qualify] = mass;
            n_qualify++;
        }
    }

    if(collecting) {
        qsort(qual, n_qualify, sizeof(double), cmp_double_desc);
        int n_copy = (n_qualify < cap) ? n_qualify : cap;
        if(n_copy > 0)
            memcpy(out_masses, qual, (size_t)n_copy * sizeof(double));
        myfree(qual);
    }
    return n_qualify;
}

/* StarClusterDetails full-population pass (MinMscForSCdetail): see sfr_eff.h.
 * Redraws exactly the population of starcluster_combined_seed_masslist (same Poisson
 * count and same per-cluster inverse-CDF from the same rand_id) and streams every
 * cluster with mass_lo <= m < mass_hi to the callback, in draw order. Nothing is
 * buffered, so the (potentially very large) sub-seeding-threshold population costs no
 * memory. Serial only (uses the global GSL error handler). */
void starcluster_seed_masslist_detail(double Mcut, double sum_mGamma,
                                      uint64_t rand_id, const RandTable * const rnd,
                                      double mass_lo, double mass_hi,
                                      sc_detail_cb cb, void * cbdata)
{
    if(!cb || Mcut <= 0 || sum_mGamma <= 0 || mass_lo >= mass_hi)
        return;
    double m_ave = msc_ave_from_cutoff(Mcut);
    if(m_ave <= 0)
        return;

    int N = msc_poisson_draw(sum_mGamma / m_ave, rand_id, rnd, 7776);
    if(N <= 0)
        return;

    struct msc_icmf icmf;
    msc_icmf_init(&icmf, Mcut);
    /* The budget is charged for every drawn cluster, not just the ones inside
     * [mass_lo, mass_hi), so this pass truncates at exactly the same place as
     * starcluster_combined_seed_masslist whatever window the caller asked for. */
    struct msc_budget budget;
    msc_budget_init(&budget, Mcut, 1);

    for(int s = 0; s < N; s++) {
        double u_s = get_random_number(rand_id + 300 + (uint64_t)s, rnd);
        double mass = msc_budget_take(&budget, msc_icmf_sample(&icmf, u_s));
        if(mass <= 0)
            break;              /* stellar mass budget spent: the rest never forms */
        if(mass >= mass_lo && mass < mass_hi)
            cb(mass, s, cbdata);
    }
}

/* Whether the combined-sampled SC mass is capped at the group's unseeded stellar
 * mass. Exposed so fof.c can apply the cap on the group-summed tot_msc_fof in the
 * per-particle (SeedSecFOFcomSampleParticle) seeding mode. */
int get_scmasscap_secfof_starmass(void)
{
    return sfr_params.SCmasscapSecFOFstarmass;
}

/* Per-secFOF multi-seed mass threshold (1e8 Msun) in code mass units. Exposed so
 * fof.c can decide how many BHs to seed in a massive secondary-FOF group. */
double get_msc_multiseed_thresh_code(void)
{
    return sfr_params.msc_multiseed_thresh_code;
}

/* Lower mass limit of the cluster mass function (1e2 Msun) in code mass units.
 * Exposed so fof.c can express a solar-mass constant (the 200 Msun VMS-collapse
 * floor of MinBHSeedInSC) in code units with exactly the same conversion the
 * sampled cluster masses use. Only valid after set_units/init has run. */
double get_msc_min_code(void)
{
    return sfr_params.msc_min_code;
}

/* Effective radius (in pc) of a seeded star cluster of code-unit mass mcl_code.
 * If StarClusterFixReff > 0, that fixed radius (in pc) is returned for every cluster.
 * Otherwise the median follows the size-mass relation selected by StarClusterReffRelation,
 * R_eff = R4 pc * (M_cl/1e4 Msun)^beta (BG21: 2.365 pc, 0.180; GBF: 1.4 pc, 0.25), with a
 * lognormal scatter drawn via Box-Muller -- the relation's own (BG21: 0.32 dex; GBF: 0.5 dex)
 * unless StarClusterReffScatter >= 0 replaces it (StarClusterReffSigma holds the value in
 * use). The scatter is keyed on rand_id (the host star ID) so the radius is reproducible across
 * ranks/restarts; log10(R/pc) is clipped to [-1, 2] (0.1-100 pc). The cluster mass is
 * converted to solar masses with the SAME factor as the mass-function thresholds
 * (msc_min_code = 100 Msun) to stay consistent with the sampled cluster masses. */
double starcluster_sample_reff_pc(double mcl_code, uint64_t rand_id, const RandTable * const rnd)
{
    if(mcl_code <= 0 || sfr_params.msc_min_code <= 0)
        return 0;
    /* StarClusterFixReff > 0: bypass the size-mass relation and use one fixed
     * effective radius (in pc) for every star cluster. */
    if(sfr_params.StarClusterFixReff > 0)
        return sfr_params.StarClusterFixReff;
    /* code mass -> solar mass: msc_min_code corresponds to 100 Msun. */
    double mcl_solar = mcl_code / sfr_params.msc_min_code * 100.0;
    const int rel = sfr_params.StarClusterReffRelation;
    /* median log10(R/pc) = log10(R4) + beta * log10(M_cl / 1e4 Msun) */
    double logR = sc_reff_beta[rel] * (log10(mcl_solar) - 4.0) + log10(sc_reff_R4_pc[rel]);
    /* lognormal scatter of the relation: Box-Muller Gaussian. Mix the ID (as rs_star_key does)
     * then offset by +700/+701 to stay clear of the mass sampler's RNG streams. */
    uint64_t h = rand_id * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = get_random_number(h + 700, rnd);
    double u2 = get_random_number(h + 701, rnd);
    if(u1 < 1e-20)
        u1 = 1e-20;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double logR_err = logR + sfr_params.StarClusterReffSigma * z;
    if(logR_err < -1.0) logR_err = -1.0;
    if(logR_err >  2.0) logR_err =  2.0;
    return pow(10.0, logR_err);
}

static int make_particle_star(int child, int parent, int placement, double Time, const double GravInternal, const RandTable * const rnd)
{
    int retflag = 2;
    if(P[parent].Type != 0)
        endrun(7772, "Only gas forms stars, what's wrong?\n");

    /*Store the SPH particle slot properties, as the PI may be over-written
     *in slots_convert*/
    struct sph_particle_data oldslot = SPHP(parent);

    /*Convert the child slot to the new type.*/
    child = slots_convert(child, 4, placement, PartManager, SlotsManager);

    /*Set properties*/
    STARP(child).FormationTime = Time;
    STARP(child).LastEnrichmentMyr = 0;
    STARP(child).TotalMassReturned = 0;
    STARP(child).Seeded = 0;
    /* Unseeded, so no seeding time yet. */
    STARP(child).SeedBHTime = -1;
    /* Not in any secondary FOF group until a catalogue pass says otherwise. */
    STARP(child).Bounded = -1;
    STARP(child).BirthDensity = oldslot.Density;
    const double a3inv = 1.0 / (Time * Time * Time);
    STARP(child).BirthInternalEnergy = oldslot.Entropy * entropy_to_u(oldslot.Density, a3inv);

    if (sfr_params.StarClusterOn) {
        const double G = GravInternal;
        const double rho_phys = STARP(child).BirthDensity * a3inv;
        const double u = STARP(child).BirthInternalEnergy;
        const double Pressure = GAMMA_MINUS1 * rho_phys * u;

        /* Cluster formation efficiency: tabulated Gamma(P/k_B), or Kruijssen (2012) eq. 26 with the parent's measured dispersion (StarClusterCFEsigma) */
        double Pressure_over_kB = Pressure * sfr_params.pressure_to_pkb;
        double CFE = gas_cluster_formation_efficiency(&oldslot, rho_phys, Pressure_over_kB, Time, G);
        STARP(child).ClusterFormationEfficiency = CFE;
        STARP(child).ClusterMass = P[child].Mass * CFE;
        STARP(child).initClusterMass = STARP(child).ClusterMass;

        /* --- M_cstar: star cluster mass from Toomre mass model --- */

        /* Pressure correction factor phi_P from stellar/gas velocity dispersion */
        double phi_P = 1.0;
        const int Ncut = 5;
        if(oldslot.VDisp_Nstar > Ncut) {
            double m_total = oldslot.VDisp_mstar + oldslot.VDisp_mgas;
            if(m_total > 0 && oldslot.VDisp_mgas > 0 && oldslot.VDisp_star > 0) {
                double f_gas = oldslot.VDisp_mgas / m_total;
                phi_P = 1.0 + oldslot.VDisp_gas / oldslot.VDisp_star * (1.0 / f_gas - 1.0);
            }
        }

        /* Gas surface density: Sigma_gas = sqrt(2 * P / (pi * G * phi_P)) */
        double Sigma_gas = sqrt(2.0 * Pressure / (M_PI * G * phi_P));

        /* Epicyclic frequency squared from the tidal field eigenvalues (E-MOSAICS eq. A6).
         * E-MOSAICS uses T_ij = -d^2Phi/dx^2 and kappa^2 = -(sum_i lambda_i) - lambda_1 with
         * lambda_1 their largest eigenvalue. MP-Gadget stores T_ij = +d^2Phi/dx^2 sorted
         * descending, so their lambda_1 maps to -eig[2] and the formula becomes
         *   kappa^2 = trace + eig[2]   (eig[2] = smallest / most negative eigenvalue).
         * The eigenvalues are comoving (trace = 4 pi G rho_comoving); multiply by a3inv to get
         * the physical kappa^2 used together with the physical Sigma_gas and densities. */
        double trace = oldslot.TidalFieldEigenvalues[0] + oldslot.TidalFieldEigenvalues[1] + oldslot.TidalFieldEigenvalues[2];
        double kappa_sq = (trace + oldslot.TidalFieldEigenvalues[2]) * a3inv;

        /* Maximum cloud (GMC) mass = min(Toomre-limited, feedback-limited) mass.
         * This equals f_coll * M_T for kappa^2 > 0 (f_coll ~ kappa^4 and M_T ~ kappa^-4 cancel
         * in the feedback-limited regime), but stays finite for kappa^2 <= 0, where there is no
         * centrifugal support and the cloud is always feedback-limited. */
        double M_GMC = 0;
        if(rho_phys > 0) {
            const double esp_ff = 0.012;
            const double t_sn = sfr_params.t_sn_code;
            const double phi_fb = sfr_params.phi_fb_code;

            /* Cloud feedback timescale t_fbg (independent of kappa) */
            double sigma_loc = sqrt(Pressure / rho_phys);
            double t_ff = sqrt(3.0 * M_PI / (32.0 * G * rho_phys));
            double term_tmp = 4.0 * t_ff * sigma_loc * sigma_loc
                            / (phi_fb * esp_ff * t_sn * t_sn);
            double t_fbg = t_sn / 2.0 * (1.0 + sqrt(1.0 + term_tmp));

            /* Feedback-limited mass: M_fb = pi^3 G^2 Sigma_gas^3 t_fbg^4  (== f_coll * M_T) */
            double M_feedback = pow(M_PI, 3) * G * G
                              * Sigma_gas * Sigma_gas * Sigma_gas
                              * t_fbg * t_fbg * t_fbg * t_fbg;

            if(kappa_sq > 0) {
                /* Toomre mass: M_T = 4 * pi^5 * G^2 * Sigma_gas^3 / kappa^4 */
                double M_T = 4.0 * pow(M_PI, 5) * G * G
                           * Sigma_gas * Sigma_gas * Sigma_gas
                           / (kappa_sq * kappa_sq);
                M_GMC = M_T < M_feedback ? M_T : M_feedback;
            }
            else {
                /* kappa^2 <= 0: no centrifugal support, always feedback-limited */
                M_GMC = M_feedback;
            }
        }

        /* M_cstar = 0.1 * CFE * M_GMC  (0.1 = star formation efficiency per cloud) */
        double Mcstar = 0.1 * CFE * M_GMC;
        STARP(child).Mcstar = Mcstar;

        if(sfr_params.StarClusterSampling && !sfr_params.SeedSecFOFcomSample) {
            /* Average cluster mass <m> of n(m) ~ m^-2 exp(-m/Mcstar) over [m_min, m_max]
             * (shared helper, reused by the combined-sample seeder). */
            /* Flag to skip Poisson sampling and mass sampling if Msc_ave is invalid */
            int skip_sampling = 0;
            STARP(child).Msc_ave = msc_ave_from_cutoff(Mcstar);
            if(STARP(child).Msc_ave <= 0)
                skip_sampling = 1;

            /* Number of star clusters: N = Gamma*m_star/<m>. */
            if(STARP(child).Msc_ave > 0)
                STARP(child).NumStarCluster = STARP(child).ClusterMass / STARP(child).Msc_ave;
            else
                STARP(child).NumStarCluster = 0;

            /* Poisson sample from NumStarCluster.
             * Offsets ID+10..ID+11 are reserved for this sampling;
             * existing code uses ID+0..4 and ID+23. */
            double lambda = STARP(child).NumStarCluster;
            if(skip_sampling)
                lambda = 0;
            if(lambda > 30) {
                /* Normal approximation: N(lambda, lambda) */
                double u1 = get_random_number(P[child].ID + 10, rnd);
                double u2 = get_random_number(P[child].ID + 11, rnd);
                /* Guard against u1 == 0 (gsl_rng_uniform returns [0,1)) */
                if(u1 < 1e-20)
                    u1 = 1e-20;
                /* Box-Muller transform for a standard normal */
                double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                int sample = (int)(lambda + sqrt(lambda) * z + 0.5);
                STARP(child).Nsc_sample = sample > 0 ? sample : 0;
            }
            else if(lambda > 0) {
                /* Knuth's algorithm for small lambda (lambda <= 30) */
                const int max_iter = 200;
                double L = exp(-lambda);
                double p = 1.0;
                int k = 0;
                uint64_t seed = P[child].ID + 10;
                do {
                    k++;
                    p *= get_random_number(seed, rnd);
                    seed++;
                    if(k > max_iter)
                        endrun(7773, "Poisson sampling exceeded %d iterations for lambda=%g, particle ID=%ld\n",
                               max_iter, lambda, (long)P[child].ID);
                } while(p > L);
                STARP(child).Nsc_sample = k - 1;
            }
            else {
                STARP(child).Nsc_sample = 0;
            }

            /* Sample Nsc_sample cluster masses from the ICMF.
             * Random seed offsets: ID + 300 + s (s = 0..Nsc_sample-1).
             * StarClusterICMFcutoff = 1: n(m) ~ m^-2 exp(-m/Mcstar) on [m_min, m_max],
             *   inverse CDF with bisection (fixed iteration count, no while loop),
             *   CDF(x) = [e^{-x_min}/x_min - e^{-x}/x + E1(x) - E1(x_min)] / norm, x = m/Mcstar.
             * StarClusterICMFcutoff = 0: pure power law n(m) ~ m^-2 on [1e2,1e8]
             *   (closed-form inverse CDF, cutoff-independent). */
            double total_sample_mass = 0;
            int Nsc = STARP(child).Nsc_sample;
            if(Nsc > 0 && Mcstar > 0) {
                int use_cutoff = sfr_params.StarClusterICMFcutoff;
                double x_min_s = 0, x_max_s = 0, E1_xmin = 0, emxmin_over_xmin = 0, g_norm = 0;
                if(use_cutoff) {
                    x_min_s = sfr_params.msc_min_code / Mcstar;
                    x_max_s = sfr_params.msc_max_code / Mcstar;
                    E1_xmin = safe_expint_E1(x_min_s);
                    emxmin_over_xmin = exp(-x_min_s) / x_min_s;
                    /* CDF normalization */
                    g_norm = emxmin_over_xmin - exp(-x_max_s) / x_max_s
                           + safe_expint_E1(x_max_s) - E1_xmin;
                }

                for(int s = 0; s < Nsc; s++) {
                    double u_s = get_random_number(P[child].ID + 300 + (uint64_t)s, rnd);
                    if(use_cutoff) {
                        double target = u_s * g_norm;

                        /* Bisection: find x in [x_min_s, x_max_s] such that g(x) = target */
                        double lo = x_min_s, hi = x_max_s;
                        for(int iter = 0; iter < 50; iter++) {
                            double mid = 0.5 * (lo + hi);
                            double g_mid = emxmin_over_xmin - exp(-mid) / mid
                                         + safe_expint_E1(mid) - E1_xmin;
                            if(g_mid < target)
                                lo = mid;
                            else
                                hi = mid;
                        }
                        total_sample_mass += Mcstar * 0.5 * (lo + hi);
                    }
                    else {
                        total_sample_mass += msc_sample_powerlaw(u_s);
                    }
                }
            }
            STARP(child).StarClusterMass_sample = total_sample_mass;
            STARP(child).initStarClusterMass_sample = total_sample_mass;
        }
        else {
            /* StarClusterSampling off: skip sampling, set to initial values */
            STARP(child).Msc_ave = 0;
            STARP(child).NumStarCluster = 0;
            STARP(child).Nsc_sample = 0;
            STARP(child).StarClusterMass_sample = 0;
            STARP(child).initStarClusterMass_sample = 0;
        }
    }
    else {
        STARP(child).ClusterFormationEfficiency = 0;
        STARP(child).ClusterMass = 0;
        STARP(child).initClusterMass = 0;
        STARP(child).Mcstar = 0;
        STARP(child).Msc_ave = 0;
        STARP(child).NumStarCluster = 0;
        STARP(child).Nsc_sample = 0;
        STARP(child).StarClusterMass_sample = 0;
        STARP(child).initStarClusterMass_sample = 0;
    }

    STARP(child).VDisp = oldslot.VDisp;
    /*Copy metallicity*/
    STARP(child).Metallicity = oldslot.Metallicity;
    /*Record the parent gas metallicity at formation. Frozen hereafter: never modified again.*/
    STARP(child).BirthMetallicity = oldslot.Metallicity;
    int j;
    for(j = 0; j < NMETALS; j++)
        STARP(child).Metals[j] = oldslot.Metals[j];

    return retflag;
}

/* This function cools gas on the effective equation of state*/
static void
cooling_relaxed(int i, double dtime, struct UVBG * local_uvbg, const double redshift, const double a3inv, struct sfr_eeqos_data sfr_data, const struct UVBG * const GlobalUVBG)
{
    const double egyeff = sfr_params.EgySpecCold * sfr_data.cloudfrac + (1 - sfr_data.cloudfrac) * sfr_data.egyhot;
    const double densityfac = entropy_to_u(SPHP(i).Density, a3inv);
    double egycurrent = SPHP(i).Entropy * densityfac;
    double trelax = sfr_data.trelax;
    // const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * sfr_data.uu_in_cgs;
    /* SB: Added a check to cool on the cooling time when the gas is very hot.
     * Gas which is heated by the BH is capped at T=5e8 or U = 1e7.
     * However, the BH is not active each timestep, so rarely very dense gas can have enough internal energy that it doesn't cool in
     * one timestep and can somehow be pressurized to get even hotter. This can make the shortest timestep in the code even shorter,
     * and is unphysical for star forming gas anyway. For this reason, allow gas with U > 5e6 or T > 1e8 to cool using the cooling time. */
    if(sfr_params.BHFeedbackUseTcool == 3 || (sfr_params.BHFeedbackUseTcool == 1 && (P[i].BHHeated || egycurrent > 5e6)))
    {
        if(egycurrent > egyeff)
        {
            double ne = SPHP(i).Ne;
            /* In practice tcool << trelax*/
            double tcool = GetCoolingTime(redshift, egycurrent, SPHP(i).Density * a3inv, local_uvbg, &ne, SPHP(i).Metallicity);

            /* The point of the star-forming equation of state is to pressurize the gas. However,
             * when the gas has been heated above the equation of state it is pressurized and does not cool successfully.
             * This code uses the cooling time rather than the relaxation time.
             * This reduces the effect of black hole feedback marginally (a 5% reduction in star formation)
             * and dates from the earliest versions of this code available.
             * The main impact is on the high end of the black hole mass function: turning this off
             * removes most massive black holes. */
            if(tcool < trelax && tcool > 0)
                trelax = tcool;
        }
        P[i].BHHeated = 0;
    }

    SPHP(i).Entropy =  (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax)) /densityfac;
}

/*Forms stars according to the quick lyman alpha star formation criterion,
 * which forms stars with a constant probability (usually 1) if they are star forming.
 * Returns 1 if converted a particle to a star, 0 if not.*/
static int
quicklyastarformation(int i, const double a3inv, const RandTable * const rnd)
{
    if(SPHP(i).Density <= sfr_params.OverDensThresh)
        return 0;

    const double enttou = entropy_to_u(SPHP(i).Density, a3inv);
    double unew = SPHP(i).Entropy * enttou;

    const double meanweight = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)));
    double temp = unew * meanweight / sfr_params.temp_to_u;

    if(temp >= sfr_params.QuickLymanAlphaTempThresh)
        return 0;

    if(get_random_number(P[i].ID + 1, rnd) < sfr_params.QuickLymanAlphaProbability)
        return 1;

    return 0;
}

/* Forms stars and winds.
 * Returns -1 if no star formed, otherwise returns the index of the particle which is to be made a star.
 * The star slot is not actually created here, but a particle for it is.
 */
static int
starformation(int i, double *localsfr, MyFloat * sm_out, MyFloat * sum_sm, MyFloat * sum_dtime,MyFloat * GradRho, const double redshift, const double a3inv, const double hubble, const double GravInternal, const struct UVBG * const GlobalUVBG, const RandTable * const rnd)
{
    /*  the proper time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift);
    double dtime = dloga / hubble;
    *sum_dtime += dtime;
    int newstar = -1;
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  SPHP(i).local_J21;
    zreion = SPHP(i).zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, GlobalUVBG, P[i].Pos, PartManager->CurrentParticleOffset, localJ21, zreion);

    struct sfr_eeqos_data sfr_data = get_sfr_eeqos(&P[i], &SPHP(i), dtime, &uvbg, redshift, a3inv);

    double atime = 1/(1+redshift);
    double smr = get_starformation_rate_full(i, GradRho, sfr_data, atime, a3inv, hubble, GravInternal);

    double sm = smr * dtime;

    *sm_out = sm;
    double p = sm / P[i].Mass;

    double dM = P[i].Mass * (1 - exp(-p));
    *sum_sm += dM;

    /* convert to Solar per Year: this is dM_* / dt = p_* M_* / dt ~ smr when smr << 1 */
    if(dtime > 0) {
        SPHP(i).Sfr = dM / dtime * sfr_params.UnitSfr_in_solar_per_year;
    } else {
        /* At the final sync point Dloga_interval_ti==0 so dtime can be zero.
         * Use the dtime->0 limit to avoid NaNs in the last snapshot. */
        SPHP(i).Sfr = smr * sfr_params.UnitSfr_in_solar_per_year;
    }

    SPHP(i).Ne = sfr_data.ne;
    *localsfr += SPHP(i).Sfr;

    /* Update the gas ClusterFormationEfficiency based on current density and entropy */
    if (sfr_params.StarClusterOn) {
        double InternalEnergy = SPHP(i).Entropy * entropy_to_u(SPHP(i).Density, a3inv);
        double Pressure_over_kB = GAMMA_MINUS1 * SPHP(i).Density * a3inv
                                  * InternalEnergy * sfr_params.pressure_to_pkb;
        SPHP(i).ClusterFormationEfficiency = gas_cluster_formation_efficiency(&SPHP(i), SPHP(i).Density * a3inv, Pressure_over_kB, atime, GravInternal);
    }

    /* Accumulate SFR * dt and SFR * dt * CFE (in internal mass units) */
    SPHP(i).SumSFRdt += dM;
    SPHP(i).SumSFRdt_v2 += sm;
    SPHP(i).SumSFRdtCFE += dM * SPHP(i).ClusterFormationEfficiency;

    const double w = get_random_number(P[i].ID, rnd);
    const double frac = (1 - exp(-p));
    SPHP(i).Metallicity += w * METAL_YIELD * frac / sfr_params.Generations;

    /* upon start-up, we need to protect against dloga ==0 */
    if(dloga > 0 && P[i].TimeBinHydro)
        cooling_relaxed(i, dtime, &uvbg, redshift, a3inv, sfr_data, GlobalUVBG);

    double mass_of_star = find_star_mass(i, sfr_params.avg_baryon_mass);
    double prob = dM / mass_of_star;

    int form_star = (get_random_number(P[i].ID + 1, rnd) < prob);
    if(form_star) {
        /* ok, make a star */
        newstar = i;
        /* If we get a fraction of the mass we need to create
         * a new particle for the star and remove mass from i.*/
        if(P[i].Mass >= 1.1 * mass_of_star)
            newstar = slots_split_particle(i, mass_of_star, PartManager);
    }

    /* Add the rest of the metals if we didn't form a star.
     * If we did form a star, add winds to the star-forming particle
     * that formed it if it is still around*/
    if(!form_star || newstar != i) {
        SPHP(i).Metallicity += (1-w) * METAL_YIELD * frac / sfr_params.Generations;
    }
    return newstar;
}

/* Get the parameters of the basic effective
 * equation of state model for a particle.*/
struct sfr_eeqos_data get_sfr_eeqos(struct particle_data * part, struct sph_particle_data * sph, double dtime, struct UVBG *local_uvbg, const double redshift, const double a3inv)
{
    struct sfr_eeqos_data data;
    /* Initialise data to something, just in case.*/
    data.trelax = sfr_params.MaxSfrTimescale;
    data.tsfr = sfr_params.MaxSfrTimescale;
    data.egyhot = sfr_params.EgySpecCold;
    data.cloudfrac = 0;
    data.ne = 0;

    /* This shall never happen, but just in case*/
    if(!sfreff_on_eeqos(sph, a3inv))
        return data;

    data.ne = sph->Ne;
    data.tsfr = sqrt(sfr_params.PhysDensThresh / (sph->Density * a3inv)) * sfr_params.MaxSfrTimescale;
    if(sfr_params.BoostSFDenseGas && ((sph->Density * a3inv) / sfr_params.PhysDensThresh > sfr_params.BoostSFOverDenseFactor))
        data.tsfr = sfr_params.PhysDensThresh / (sph->Density * a3inv) * sfr_params.MaxSfrTimescale;
    /*
     * gadget-p doesn't have this cap.
     * without the cap sm can be bigger than cloudmass.
    */
    if(data.tsfr < dtime && dtime > 0)
        data.tsfr = dtime;

    double factorEVP = pow(sph->Density * a3inv / sfr_params.PhysDensThresh, -0.8) * sfr_params.FactorEVP;

    data.egyhot = sfr_params.EgySpecSN / (1 + factorEVP) + sfr_params.EgySpecCold;
    data.egycold = sfr_params.EgySpecCold;

    double tcool = GetCoolingTime(redshift, data.egyhot, sph->Density * a3inv, local_uvbg, &data.ne, sph->Metallicity);
    double y = data.tsfr / tcool * data.egyhot / (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold);

    data.cloudfrac = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    data.trelax = data.tsfr * (1 - data.cloudfrac) / data.cloudfrac / (sfr_params.FactorSN * (1 + factorEVP));
    return data;
}

static double get_starformation_rate_full(int i, MyFloat * GradRho, struct sfr_eeqos_data sfr_data, const double atime, const double a3inv, const double hubble, const double GravInternal)
{
    if(!sfreff_on_eeqos(&SPHP(i), a3inv)) {
        return 0;
    }

    double cloudmass = sfr_data.cloudfrac * P[i].Mass;

    double rateOfSF = (1 - sfr_params.FactorSN) * cloudmass / sfr_data.tsfr;

    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
        if(!GradRho)
            endrun(1, "GradRho not allocated but has SFR_CRITERION_MOLECULAR_H2. Should never happen!\n");
        rateOfSF *= get_sfr_factor_due_to_h2(i, GradRho, atime);
    }
    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_SELFGRAVITY)) {
        rateOfSF *= get_sfr_factor_due_to_selfgravity(i, atime, a3inv, hubble, GravInternal);
    }
    return rateOfSF;
}

/*Gets the effective energy*/
static double
get_egyeff(double redshift, double dens, struct UVBG * uvbg)
{
    double tsfr = sqrt(sfr_params.PhysDensThresh / (dens)) * sfr_params.MaxSfrTimescale;
    double factorEVP = pow(dens / sfr_params.PhysDensThresh, -0.8) * sfr_params.FactorEVP;
    double egyhot = sfr_params.EgySpecSN / (1 + factorEVP) + sfr_params.EgySpecCold;

    double ne = 0.5;
    double tcool = GetCoolingTime(redshift, egyhot, dens, uvbg, &ne, 0.0);

    double y = tsfr / tcool * egyhot / (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold);
    double x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
    return egyhot * (1 - x) + sfr_params.EgySpecCold * x;
}

/* Minimum temperature in internal energy*/
double get_MinEgySpec(void)
{
    /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
    const double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);
    /*Enforces a minimum internal energy in cooling. */
    return sfr_params.temp_to_u / meanweight * sfr_params.MinGasTemp;
}

void init_cooling_and_star_formation(int CoolingOn, int StarformationOn, Cosmology * CP, const double avg_baryon_mass, const double BoxSize, const struct UnitSystem units)
{
    struct cooling_units coolunits;
    coolunits.CoolingOn = CoolingOn;
    coolunits.density_in_phys_cgs = units.UnitDensity_in_cgs * CP->HubbleParam * CP->HubbleParam;
    coolunits.uu_in_cgs = units.UnitInternalEnergy_in_cgs;
    coolunits.tt_in_s = units.UnitTime_in_s / CP->HubbleParam;
    /* Get mean cosmic baryon density for photoheating rate from long mean free path photons */
    coolunits.rho_crit_baryon = 3 * pow(CP->HubbleParam * HUBBLE,2) * CP->OmegaBaryon / (8 * M_PI * GRAVITY);

    sfr_params.temp_to_u = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) / units.UnitInternalEnergy_in_cgs;

    sfr_params.UnitSfr_in_solar_per_year = (units.UnitMass_in_g / SOLAR_MASS) / (units.UnitTime_in_s / SEC_PER_YEAR);

    sfr_params.pressure_to_pkb = coolunits.density_in_phys_cgs * coolunits.uu_in_cgs / BOLTZMANN;  // ~22500 
    /* Precompute constants for M_cstar (star cluster mass) calculation */
    /* t_sn = 3 Myr in code time units: 3 * SEC_PER_MEGAYEAR / UnitTime_in_s (note UnitTime includes h) */
    sfr_params.t_sn_code = 3.0 * SEC_PER_MEGAYEAR / (units.UnitTime_in_s / CP->HubbleParam);
    /* phi_fb = 0.16 cm^2/s^3 in code units:
     * code_velocity^2 / code_time = UnitVelocity^2 / (UnitLength/UnitVelocity)
     * = UnitVelocity^3 / UnitLength. Divide by h factor from code_time. */
    sfr_params.phi_fb_code = 0.16 / (units.UnitVelocity_in_cm_per_s * units.UnitVelocity_in_cm_per_s
                                     / (units.UnitTime_in_s / CP->HubbleParam));
    /* Kruijssen (2012) model constants for StarClusterCFEsigma > 0, with the unit conventions of t_sn_code / phi_fb_code */
    sfr_params.cs_cold_code = 0.3e5 / units.UnitVelocity_in_cm_per_s;
    sfr_params.t_inc_code = 10.0 * SEC_PER_MEGAYEAR / (units.UnitTime_in_s / CP->HubbleParam);
    /* Mass limits for msc_ave: 1e2 and 1e8 solar masses in code mass units.
     * The code mass unit is 1e10 Msun/h (UnitMass_in_g converts to g/h), so a
     * PHYSICAL solar-mass constant carries an extra factor of h -- exactly as
     * t_sn_code above carries one for a physical time. Every code<->Msun
     * conversion in the star cluster model is a ratio against one of the four
     * yardsticks set here (starcluster_sample_reff_pc, cw_seed_mass_code, the
     * MinBHSeedInSC VMS floor), so all of them must carry it together. */
    sfr_params.msc_min_code = 1e2 * SOLAR_MASS * CP->HubbleParam / units.UnitMass_in_g;
    sfr_params.msc_max_code = 1e8 * SOLAR_MASS * CP->HubbleParam / units.UnitMass_in_g;
    /* Mean cluster mass of the pure power law (StarClusterICMFcutoff=0) is a fixed
     * constant (independent of any cutoff), so precompute it once here instead of
     * recomputing the log() per seeding event. */
    sfr_params.msc_ave_powerlaw_code = msc_ave_powerlaw();
    /* "Massive cluster" threshold for combined-sample seeding: 1e4 solar masses */
    sfr_params.msc_seed_thresh_code = 1e4 * SOLAR_MASS * CP->HubbleParam / units.UnitMass_in_g;
    /* Per-secFOF multi-seed threshold: 1e8 solar masses. When the seeding cluster
     * mass M_SC of a secondary-FOF group exceeds this, floor(M_SC/1e8) BHs are seeded. */
    sfr_params.msc_multiseed_thresh_code = 1e8 * SOLAR_MASS * CP->HubbleParam / units.UnitMass_in_g;

    init_cooling(sfr_params.TreeCoolFile, sfr_params.J21CoeffFile, sfr_params.MetalCoolFile, sfr_params.ReionHistFile, coolunits, CP);

    if(!CoolingOn)
        return;

    /*Initialize the uv fluctuation table*/
    init_uvf_table(sfr_params.UVFluctuationFile, sizeof(sfr_params.UVFluctuationFile), BoxSize, units.UnitLength_in_cm);

    sfr_params.StarformationOn = StarformationOn;

    if(!StarformationOn)
        return;

    sfr_params.avg_baryon_mass = avg_baryon_mass;

    sfr_params.tau_fmol_unit = units.UnitDensity_in_cgs*CP->HubbleParam*units.UnitLength_in_cm;
    sfr_params.OverDensThresh =
        sfr_params.CritOverDensity * CP->OmegaBaryon * CP->RhoCrit;

    sfr_params.PhysDensThresh = sfr_params.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / units.UnitDensity_in_cgs;

    /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
    double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);
    sfr_params.EgySpecCold = (sfr_params.temp_to_u/meanweight) * sfr_params.TempClouds;

    /* mean molecular weight assuming FULL ionization */
    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    sfr_params.EgySpecSN = sfr_params.temp_to_u/meanweight * sfr_params.TempSupernova;

    if(sfr_params.PhysDensThresh == 0)
    {
        double egyhot = sfr_params.EgySpecSN / sfr_params.FactorEVP;

        meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

        double u4 = sfr_params.temp_to_u/meanweight * 1.0e4;

        double dens = 1.0e6 * CP->RhoCrit;

        double ne = 1.0;

        struct UVBG uvbg = {0};
        /*XXX: We set the threshold without metal cooling
         * and with zero ionization at z=0.
         * It probably make sense to set the parameters with
         * a metalicity dependence.
         * */
        const double tcool = GetCoolingTime(0, egyhot, dens, &uvbg, &ne, 0.0);

        const double coolrate = egyhot / tcool / dens;

        const double x = (egyhot - u4) / (egyhot - sfr_params.EgySpecCold);

        sfr_params.PhysDensThresh = x / pow(1 - x, 2) *
                    (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold)
                    / (sfr_params.MaxSfrTimescale * coolrate);

        message(0, "A0= %g  \n", sfr_params.FactorEVP);
        message(0, "Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n", sfr_params.PhysDensThresh,
                sfr_params.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / units.UnitDensity_in_cgs));
        message(0, "EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n", x);
        message(0, "tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);

        dens = sfr_params.PhysDensThresh * 10;

        double neff;
        do
        {
            double egyeff = get_egyeff(0, dens, &uvbg);

            double peff = GAMMA_MINUS1 * dens * egyeff;

            const double fac = 1 / (log(dens * 1.025) - log(dens));
            neff = -log(peff) * fac;

            dens *= 1.025;
            egyeff = get_egyeff(0, dens, &uvbg);
            peff = GAMMA_MINUS1 * dens * egyeff;

            neff += log(peff) * fac;
        }
        while(neff > 4.0 / 3);

        message(0, "Run-away sets in for dens=%g\n", dens);
        message(0, "Dynamic range for quiescent star formation= %g\n", dens / sfr_params.PhysDensThresh);

        const double sigma = 10.0 / CP->Hubble * 1.0e-10 / pow(1.0e-3, 2);

        message(0, "Isotherm sheet central density: %g   z0=%g\n",
                M_PI * CP->GravInternal * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                GAMMA_MINUS1 * u4 / (2 * M_PI * CP->GravInternal * sigma));
    }

    if(sfr_params.WindOn) {
        init_winds(sfr_params.FactorSN, sfr_params.EgySpecSN, sfr_params.PhysDensThresh, units.UnitTime_in_s);
    }

}

static double
find_star_mass(int i, const double avg_baryon_mass)
{
    /*Quick Lyman Alpha always turns all of a particle into stars*/
    if(sfr_params.QuickLymanAlphaProbability > 0)
        return P[i].Mass;

    double mass_of_star =  avg_baryon_mass / sfr_params.Generations;
    if(mass_of_star > P[i].Mass) {
        /* if some mass has been stolen by BH, e.g */
        mass_of_star = P[i].Mass;
    }
    /* Conditions to turn the gas into a star. .
     * The mass check makes sure we never get a gas particle which is lighter
     * than the smallest star particle.
     * The Generations check (which can happen because of mass return)
     * ensures we never instantaneously enrich stars above solar. */
    if(P[i].Mass < 2 * mass_of_star  || P[i].Generation > sfr_params.Generations) {
        mass_of_star = P[i].Mass;
    }
    return mass_of_star;
}

/********************
 *
 * The follow functions are from Desika and Gadget-P.
 * We really are mostly concerned about H2 here.
 *
 * You may need a license to run with these modess.

 * */

int sfr_need_to_compute_sph_grad_rho(void)
{
    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
        return 1;
    }
    return 0;
}
static double ev_NH_from_GradRho(MyFloat gradrho_mag, double hsml, double rho, double include_h)
{
    /* column density from GradRho, copied from gadget-p; what is it
     * calculating? */
    if(rho<=0)
        return 0;
    double ev_NH = 0;
    if(gradrho_mag > 0)
        ev_NH = rho*rho/gradrho_mag;
    if(include_h > 0)
        ev_NH += rho*hsml;
    return ev_NH; // *(Z/Zsolar) add metallicity dependence
}

static double get_sfr_factor_due_to_h2(int i, MyFloat * GradRho_mag, const double atime) {
    /*  Krumholz & Gnedin fitting function for f_H2 as a function of local
     *  properties, from gadget-p; we return the enhancement on SFR in this
     *  function */
    double tau_fmol;
    const double a2 = atime * atime;
    double zoverzsun = SPHP(i).Metallicity/METAL_YIELD;
    double gradrho_mag = GradRho_mag[P[i].PI];
    //message(4, "GradRho %g rho %g hsml %g i %d\n", gradrho_mag, SPHP(i).Density, P[i].Hsml, i);
    tau_fmol = ev_NH_from_GradRho(gradrho_mag,P[i].Hsml,SPHP(i).Density,1) /a2;
    tau_fmol *= (0.1 + zoverzsun);
    if(tau_fmol>0) {
        tau_fmol *= 434.78*sfr_params.tau_fmol_unit;
        double y = 0.756*(1+3.1*pow(zoverzsun,0.365));
        y = log(1+0.6*y+0.01*y*y)/(0.6*tau_fmol);
        y = 1-0.75*y/(1+0.25*y);
        if(y<0) y=0;
        if(y>1) y=1;
        return y;

    } // if(tau_fmol>0)
    return 1.0;
}

static double get_sfr_factor_due_to_selfgravity(int i, const double atime, const double a3inv, const double hubble, const double GravInternal) {
    const double a2 = atime * atime;
    double divv = SPHP(i).DivVel / a2;

    divv += 3.0*hubble * a2; // hubble-flow correction

    if(HAS(sfr_params.StarformationCriterion, SFR_CRITERION_CONVERGENT_FLOW)) {
        if( divv>=0 ) return 0; // restrict to convergent flows (optional) //
    }

    double dv2abs = (divv*divv
            + (SPHP(i).CurlVel/a2)
            * (SPHP(i).CurlVel/a2)
           ); // all in physical units
    double alpha_vir = 0.2387 * dv2abs/(GravInternal * SPHP(i).Density * a3inv);

    double y = 1.0;

    if((alpha_vir < 1.0)
    || (SPHP(i).Density * a3inv > 100. * sfr_params.PhysDensThresh)
    )  {
        y = 66.7;
    } else {
        y = 0.1;
    }
    // PFH: note the latter flag is an arbitrary choice currently set
    // -by hand- to prevent runaway densities from this prescription! //

    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_CONTINUOUS_CUTOFF)) {
        // continuous cutoff w alpha_vir instead of sharp (optional) //
        y *= 1.0/(1.0 + alpha_vir);
    }
    return y;
}

/* Update the active particle list when a new star is formed.
 * if the parent is active the child should also be active.
 * Stars must always be (hydro) active on formation. Returns
 * whether particle is gravity active. */
static int
add_new_particle_to_active(const int parent, const int child, ActiveParticles * act)
{

    /* If gravity active, increment the counter*/
    int is_grav_active = is_timebin_active(P[parent].TimeBinGravity, P[parent].Ti_drift);
    /* If either is active, need to be in the active list. */
    if(is_grav_active || is_timebin_active(P[parent].TimeBinHydro, P[parent].Ti_drift)) {
        int64_t childactive = atomic_fetch_and_add_64(&act->NumActiveParticle, 1);
        if(act->ActiveParticle) {
            /* This should never happen because we allocate as much space for active particles as we have space
             * for particles, but just in case*/
            if(childactive >= act->MaxActiveParticle)
                endrun(5, "Tried to add %ld active particles, more than %ld allowed\n", childactive, act->MaxActiveParticle);
            act->ActiveParticle[childactive] = child;
        }
    }
    return is_grav_active;
}

/* Copy the gravitational acceleration if necessary for a new particle.*/
static int
copy_gravaccel_new_particle(const int parent, const int child, MyFloat (* GravAccel)[3], int64_t nstoredgravaccel)
{
    /* If gravity active, copy the grav accel to the new child*/
    int is_grav_active = is_timebin_active(P[parent].TimeBinGravity, P[parent].Ti_drift);
    if(is_grav_active && GravAccel) {
        if(child >= nstoredgravaccel)
            endrun(1, "Not enough space (%ld) in stored GravAccel to copy new star %d from parent %d\n", nstoredgravaccel, child, parent);
        int j;
        for(j=0; j < 3 ; j++)
            GravAccel[child][j] = GravAccel[parent][j];
    }
    return 0;
}
