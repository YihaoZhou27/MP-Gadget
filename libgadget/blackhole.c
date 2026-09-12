#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "physconst.h"
#include "gravity.h"
#include "densitykernel.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "blackhole.h"
#include "timestep.h"
#include "density.h"
#include "sfr_eff.h"
#include "winds.h"
#include "walltime.h"
#include "bhinfo.h"
#include "bhdynfric.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

struct BlackholeParams
{
    double BlackHoleAccretionFactor;	/*!< Fraction of BH bondi accretion rate */
    double BlackHoleFeedbackFactor;	/*!< Fraction of the black luminosity feed into thermal feedback */
    enum BlackHoleFeedbackMethod BlackHoleFeedbackMethod;	/*!< method of the feedback*/
    double BlackHoleEddingtonFactor;	/*! Factor above Eddington */

    int BlackHoleKineticOn; /*If 1, perform AGN kinetic feedback when the Eddington accretion rate is low */
    double BHKE_EddingtonThrFactor; /*Threshold of the Eddington rate for the kinetic feedback*/
    double BHKE_EddingtonMFactor; /* Factor for mbh-dependent Eddington threshold */
    double BHKE_EddingtonMPivot; /* Pivot MBH for mbh-dependent Eddington threshold */
    double BHKE_EddingtonMIndex; /* Powlaw index for mbh-dependent Eddington threshold */
    double BHKE_EffRhoFactor; /* Minimum kinetic feedback efficiency factor scales with BH density*/
    double BHKE_EffCap; /* Cap of the kinetic feedback efficiency factor */
    double BHKE_InjEnergyThr; /*Minimum injection of KineticFeedbackEnergy, controls the burstiness of kinetic feedback*/
    double BHKE_SfrCritOverDensity; /*for KE efficiency calculation, borrow from sfr.params */
    /**********************************************************************/
    int MergeGravBound; /*if 1, apply gravitational bound criteria for BH mergers */
    int BH_DRAG; /*Hydro drag force*/

    double SeedBHDynMass; /* The initial dynamic mass of BH particle */

    double SeedBlackHoleMass;	/*!< (minimum) Seed black hole mass */
    double MaxSeedBlackHoleMass; /* Maximum black hole seed mass*/
    double SeedBlackHoleMassIndex; /* Power law index for BH seed mass*/

    int StarClusterOn; /* If 1, enable star-cluster bh seeding formation */
    int StarClusterSampling; /* If 1, use sampled star cluster mass for BH seeding */
    int BHseedMassScaleMsc; /* When star-cluster bh seeding formation is enabled, whether the seed mass is scaled by the star cluster mass. If so, parameter SeedBlackHoleMass is in unit of Msc. If not, it is in mass unit. */
    double MinMscForBHseed; /* Minimum star cluster mass for BH seeding */
    int BlackholeSeedSCparticle; /* If 1, seed BH from individual star particles with SC mass >= MinMscForBHseed */
    int BHseedEveryTimestep; /* If 1, seed BH from SC particles every timestep (not just PM steps). Requires BlackholeSeedSCparticle=1. */
    int StarClusterBHDyn; /* 0: P.Mass = max(Mtrack, SeedBHDynMass).
                           * 1: include the evolving star cluster mass in the BH dynamical
                           *    mass, P.Mass = max(Mtrack + StarClusterMass, SeedBHDynMass);
                           *    BH+SC treated as one body for dynamics.
                           * 2: the star-cluster mass that seeded the BH (init_Msc) acts as a
                           *    FROZEN per-BH dynamical-mass floor replacing SeedBHDynMass:
                           *    P.Mass = max(Mtrack, init_Msc).  No SC payload is attached
                           *    (no SC evolution / merger SC transfer, like mode 0); the floor
                           *    never changes (mergers keep the accretor's own init_Msc);
                           *    non-star-cluster seeds keep the SeedBHDynMass floor. */
    int BlackholeTidalField; /* If 1, compute tidal field strength for BH particles every timestep */
    int GWRecoilVelocityKick; /* If 1, apply GW recoil velocity kick to BH merger remnants */
    int GWRecoilSCKick; /* If 1, check if GW kick ejects BH from star cluster and zero SC mass */
    int BHVorticity; /* If 1, compute SPH vorticity of surrounding gas for each BH */
    /************************************************************************/
} blackhole_params;

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Density;
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat Vel[3];
    MyFloat Accel[3];
    MyIDType ID;
    MyFloat Mtrack;
} TreeWalkQueryBHAccretion;

typedef struct {
    TreeWalkResultBase base;
    int encounter;
    int alignment;
    MyFloat FeedbackWeightSum;
    MyFloat SmoothedEntropy;
    MyFloat GasVel[3];
    /* used for AGN kinetic feedback */
    MyFloat V2sumDM;
    MyFloat V1sumDM[3];
    MyFloat NumDM;
    MyFloat MgasEnc;
    MyFloat Vorticity[3];
} TreeWalkResultBHAccretion;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
} TreeWalkNgbIterBHAccretion;


/*****************************************************************************/

/*Set the parameters of the BH module*/
void set_blackhole_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_params.BlackHoleAccretionFactor = param_get_double(ps, "BlackHoleAccretionFactor");
        blackhole_params.BlackHoleEddingtonFactor = param_get_double(ps, "BlackHoleEddingtonFactor");
        blackhole_params.BlackHoleFeedbackFactor = param_get_double(ps, "BlackHoleFeedbackFactor");

        blackhole_params.BlackHoleFeedbackMethod = (enum BlackHoleFeedbackMethod) param_get_enum(ps, "BlackHoleFeedbackMethod");

        blackhole_params.BlackHoleKineticOn = param_get_int(ps,"BlackHoleKineticOn");
        blackhole_params.BHKE_EddingtonThrFactor = param_get_double(ps, "BHKE_EddingtonThrFactor");
        blackhole_params.BHKE_EddingtonMFactor = param_get_double(ps, "BHKE_EddingtonMFactor");
        blackhole_params.BHKE_EddingtonMPivot = param_get_double(ps, "BHKE_EddingtonMPivot");
        blackhole_params.BHKE_EddingtonMIndex = param_get_double(ps, "BHKE_EddingtonMIndex");
        blackhole_params.BHKE_EffRhoFactor = param_get_double(ps, "BHKE_EffRhoFactor");
        blackhole_params.BHKE_EffCap = param_get_double(ps, "BHKE_EffCap");
        blackhole_params.BHKE_InjEnergyThr = param_get_double(ps, "BHKE_InjEnergyThr");
        blackhole_params.BHKE_SfrCritOverDensity = param_get_double(ps, "CritOverDensity");
        /***********************************************************************************/
        blackhole_params.BH_DRAG = param_get_int(ps, "BH_DRAG");
        blackhole_params.MergeGravBound = param_get_int(ps, "MergeGravBound");
        blackhole_params.SeedBHDynMass = param_get_double(ps,"SeedBHDynMass");


        blackhole_params.SeedBlackHoleMass = param_get_double(ps, "SeedBlackHoleMass");
        blackhole_params.MaxSeedBlackHoleMass = param_get_double(ps,"MaxSeedBlackHoleMass");
        blackhole_params.SeedBlackHoleMassIndex = param_get_double(ps,"SeedBlackHoleMassIndex");
        blackhole_params.StarClusterOn = param_get_int(ps, "StarClusterOn");
        blackhole_params.StarClusterSampling = param_get_int(ps, "StarClusterSampling");
        blackhole_params.BHseedMassScaleMsc = param_get_int(ps, "BHseedMassScaleMsc");
        blackhole_params.MinMscForBHseed = param_get_double(ps, "MinMscForBHseed");
        blackhole_params.BlackholeSeedSCparticle = param_get_int(ps, "BlackholeSeedSCparticle");
        blackhole_params.BHseedEveryTimestep = param_get_int(ps, "BHseedEveryTimestep");
        blackhole_params.StarClusterBHDyn = param_get_int(ps, "StarClusterBHDyn");
        if(blackhole_params.StarClusterBHDyn < 0 || blackhole_params.StarClusterBHDyn > 2)
            endrun(1, "StarClusterBHDyn must be 0, 1 or 2 (got %d).\n", blackhole_params.StarClusterBHDyn);
        if(!blackhole_params.StarClusterOn && blackhole_params.StarClusterBHDyn) {
            message(0, "StarClusterBHDyn=%d requires StarClusterOn=1. Forcing StarClusterBHDyn=0.\n",
                    blackhole_params.StarClusterBHDyn);
            blackhole_params.StarClusterBHDyn = 0;
        }
        blackhole_params.BlackholeTidalField = param_get_int(ps, "BlackholeTidalField");
        blackhole_params.GWRecoilVelocityKick = param_get_int(ps, "GWRecoilVelocityKick");
        blackhole_params.GWRecoilSCKick = param_get_int(ps, "GWRecoilSCKick");
        blackhole_params.BHVorticity = param_get_int(ps, "BHVorticity");
        if(blackhole_params.GWRecoilSCKick && !blackhole_params.StarClusterOn)
            endrun(1, "GWRecoilSCKick=1 requires StarClusterOn=1.\n");
        /* Hierarchical gravity: the sub-step trees hold only the active particles, so the
         * tidal tensor (grav_short_tree) is computed only on the full tree of a PM step and
         * the BH keeps that value until the next PM step (a zero-order hold). */
        if(blackhole_params.BlackholeTidalField && param_get_int(ps, "SplitGravityTimestepsOn"))
            message(0, "BlackholeTidalField with SplitGravityTimestepsOn=1: the BH tidal field is computed at PM steps only and held until the next PM step.\n");
        if(blackhole_params.BHseedEveryTimestep && !blackhole_params.BlackholeSeedSCparticle)
            endrun(1, "BHseedEveryTimestep requires BlackholeSeedSCparticle=1.\n");
        if(blackhole_params.BlackholeSeedSCparticle && blackhole_params.BHseedMassScaleMsc && blackhole_params.MinMscForBHseed <= 0)
            endrun(1, "MinMscForBHseed must be > 0 when BlackholeSeedSCparticle and BHseedMassScaleMsc are enabled.\n");
        /***********************************************************************************/
    }
    MPI_Bcast(&blackhole_params, sizeof(struct BlackholeParams), MPI_BYTE, 0, MPI_COMM_WORLD);

    set_blackhole_dynfric_params(ps);
}

int
get_bh_tidalfield_on(void)
{
    return blackhole_params.BlackholeTidalField;
}

double
get_bh_seed_dyn_mass(void)
{
    return blackhole_params.SeedBHDynMass;
}

double
get_bh_seed_mass(void)
{
    return blackhole_params.SeedBlackHoleMass;
}

int
get_starcluster_bhdyn_on(void)
{
    /* True only for mode 1: the evolving StarClusterMass payload is attached to
     * the BH and included in P.Mass.  Mode 2 attaches NO payload (like mode 0):
     * the seed cluster mass only sets a frozen per-BH dynamical-mass floor via
     * init_Msc (see blackhole_make_one). */
    return blackhole_params.StarClusterBHDyn == 1;
}

/* Warn (do NOT abort) when StarClusterBHDyn=2 is used with a minimum seeding
 * cluster mass below the dark matter particle mass.  In mode 2 the seed cluster
 * mass (>= MinMscForBHseed) sets the per-BH dynamical-mass floor, so a
 * MinMscForBHseed below the DM particle mass means BH seeds can be lighter than
 * the background DM particles.  dm_particle_mass is the header MassTable[1], in
 * the same internal mass units as MinMscForBHseed. */
void
blackhole_check_seed_dm_resolution(double dm_particle_mass)
{
    if(blackhole_params.StarClusterBHDyn == 2 &&
       blackhole_params.MinMscForBHseed < dm_particle_mass)
        message(0, "WARNING: StarClusterBHDyn=2 but MinMscForBHseed (%g) < dark matter particle mass (%g); "
                   "BH seeds may be lighter than the background dark matter particles.\n",
                blackhole_params.MinMscForBHseed, dm_particle_mass);
}

/* accretion routines */
static void
blackhole_accretion_postprocess(int n, TreeWalk * tw);

static void
blackhole_accretion_reduce(int place, TreeWalkResultBHAccretion * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

static void
blackhole_accretion_copy(int place, TreeWalkQueryBHAccretion * I, TreeWalk * tw);

static void
blackhole_accretion_ngbiter(TreeWalkQueryBHAccretion * I,
        TreeWalkResultBHAccretion * O,
        TreeWalkNgbIterBHAccretion * iter,
        LocalTreeWalk * lv);

/* Do the black hole feedback tree walk. Tree needs to have gas and BH.*/
static void
blackhole_feedback(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHPriv * priv);

/*************************************************************************************/

static double blackhole_soundspeed(double entropy, double rho, const double atime) {
    /* rho is comoving !*/
    if(rho <= 0)
        return 0;
    double cs = sqrt(GAMMA * entropy * pow(rho, GAMMA_MINUS1));

    cs *= pow(atime, -1.5 * GAMMA_MINUS1);

    return cs;
}

/* check if two BHs are gravitationally bounded, input dv, da, dx in code unit */
/* same as Bellovary2011, Tremmel2017 */
static int
check_grav_bound(double dx[3], double dv[3], double da[3], const double atime)
{
    int j;
    double KE = 0;
    double PE = 0;

    for(j = 0; j < 3; j++){
        KE += 0.5 * pow(dv[j], 2);
        PE += da[j] * dx[j];
    }

    KE /= (atime * atime); /* convert to proper velocity */
    PE /= atime; /* convert to proper unit */

    /* The gravitationally bound condition is PE + KE < 0.
     * Still merge if it is marginally bound so that we merge
     * particles at zero distance and velocity from each other.*/
    return (PE + KE <= 0);
}

/*******************************************************************/
static int
blackhole_haswork(int n, TreeWalk * tw){
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed);
}

/* Build a list of active black holes, done once and reused for all the later treewalks.*/
int
blackholes_active(const ActiveParticles * act, int ** ActiveBlackHoles, int64_t * NumActiveBlackHoles)
{
    TreeWalk tw_bh[1] = {{0}};
    tw_bh->haswork = blackhole_haswork;

    /* Build the queue once, since it is really 'all black holes' and similar for all treewalks.*/
    treewalk_build_queue(tw_bh, act->ActiveParticle, act->NumActiveParticle, 0);
    /* If this queue is empty, nothing to do.*/
    int64_t totbh;
    MPI_Allreduce(&tw_bh->WorkSetSize, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Now we have a BH queue and we can re-use it. Create a new variable so that
     * treewalk_run does not mess with the pointer. */
    /* Move the working set high so we can keep the tree we build after making the list and still free this active set.*/
    *NumActiveBlackHoles = tw_bh->WorkSetSize;
    if(totbh > 0) {
        *ActiveBlackHoles = (int*)mymalloc2("activeBH", tw_bh->WorkSetSize * sizeof(int));
        memcpy(*ActiveBlackHoles, tw_bh->WorkSet, tw_bh->WorkSetSize * sizeof(int));
    }
    myfree(tw_bh->WorkSet);
    return totbh;
}

void
blackhole(const ActiveParticles * act, double atime, Cosmology * CP, ForceTree * tree, DomainDecomp * ddecomp, DriftKickTimes * times, RandTable * rnd, const struct UnitSystem units, FILE * FdBlackHoles, FILE * FdBlackholeDetails, size_t * bhdetailswritten, int is_PM)
{
    /* Do nothing if no black holes*/
    int64_t totbh;
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totbh == 0)
        return;

    walltime_measure("/Misc");
    /* Build the queue once, since it is really 'all black holes' and similar for all treewalks.*/
    int * ActiveBlackHoles = NULL;
    int64_t NumActiveBlackHoles = 0;
    totbh = blackholes_active(act, &ActiveBlackHoles, &NumActiveBlackHoles);
    if(totbh == 0) {
        return;
    }

    /* Types used in treewalks:
     * accretion uses: gas + black holes (to flag mergers).
     * feedback uses: gas + black holes (to flag mergers).
     */
    if(!tree->tree_allocated_flag || !(tree->mask & GASMASK) ||  !(tree->mask & BHMASK) )
        endrun(5, "Blackhole called with bad tree allocated %d mask %d want %d\n", tree->tree_allocated_flag, tree->mask, GASMASK | BHMASK);

    struct kick_factor_data kf;
    init_kick_factor_data(&kf, times, CP);

    /*************************************************************************/
    /*  Dynamical Friction Treewalk */
    /*************************************************************************/
    struct BHDynFricPriv dynpriv[1] = {0};
    dynpriv->kf = &kf;
    dynpriv->Ti_Current = times->Ti_Current;
    /* Update the kernel quantities for dynamic friction, if required.
     * This takes place on a longer timestep than the hydro acceleration
     * to avoid extra treebuilds. Note this includes the potential minimum.
     * If black hole repositioning is on, the treewalk to reposition
     * to the local potential minimum is run.*/
    blackhole_dynfric(ActiveBlackHoles, NumActiveBlackHoles, ddecomp, dynpriv);
    /* Compute the DF acceleration for all active black holes*/
    blackhole_dfaccel(ActiveBlackHoles, NumActiveBlackHoles, atime, CP->GravInternal);

    walltime_measure("/BH/DynFric");

    struct BHPriv priv[1] = {0};
    priv->units = units;
    priv->rnd = rnd;
    priv->is_PM = is_PM;
    /*************************************************************************/
    priv->atime = atime;
    priv->a3inv = 1./(atime * atime * atime);
    priv->hubble = hubble_function(CP, atime);
    priv->CP = CP;
    priv->kf = &kf;

    /* Let's determine which gas particles may be swallowed and calculate total feedback weights */
    priv->SPH_SwallowID = (MyIDType *) mymalloc("SPH_SwallowID", SlotsManager->info[0].size * sizeof(MyIDType));
    memset(priv->SPH_SwallowID, 0, SlotsManager->info[0].size * sizeof(MyIDType));
    /* Let's determine which BHs may be swallowed and calculate total feedback weights */
    priv->BH_SwallowID = (MyIDType *) mymalloc("BH_SwallowID", SlotsManager->info[5].size * sizeof(MyIDType));
    memset(priv->BH_SwallowID, 0, SlotsManager->info[5].size * sizeof(MyIDType));

    /* Computed in accretion, used in feedback*/
    priv->BH_FeedbackWeightSum = (MyFloat *) mymalloc("BH_FeedbackWeightSum", SlotsManager->info[5].size * sizeof(MyFloat));

    /* Local to this treewalk*/
    priv->BH_Entropy = (MyFloat *) mymalloc("BH_Entropy", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_SurroundingGasVel = (MyFloat (*) [3]) mymalloc("BH_SurroundVel", 3* SlotsManager->info[5].size * sizeof(priv->BH_SurroundingGasVel[0]));

    /* For AGN kinetic feedback */
    priv->NumDM = (MyFloat *) mymalloc("NumDM", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->MgasEnc = (MyFloat *) mymalloc("MgasEnc", SlotsManager->info[5].size * sizeof(MyFloat));
    /* mark the state of AGN kinetic feedback */
    priv->KEflag = (int *) mymalloc("KEflag", SlotsManager->info[5].size * sizeof(int));

    /* Dimensionless vorticity of surrounding gas for each BH */
    priv->BH_Vorticity = (MyFloat *) mymalloc("BH_Vorticity", SlotsManager->info[5].size * sizeof(MyFloat));
    memset(priv->BH_Vorticity, 0, SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_VorticityVec = (MyFloat (*) [3]) mymalloc("BH_VorticityVec", 3 * SlotsManager->info[5].size * sizeof(MyFloat));
    memset(priv->BH_VorticityVec, 0, 3 * SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_SoundSpeed = (MyFloat *) mymalloc("BH_SoundSpeed", SlotsManager->info[5].size * sizeof(MyFloat));
    memset(priv->BH_SoundSpeed, 0, SlotsManager->info[5].size * sizeof(MyFloat));

    /* Need hmax for the symmetric BH merger treewalk*/
    if(!tree->hmax_computed_flag)
        force_tree_calc_moments(tree, ddecomp);

    walltime_measure("/BH/Init");

    /*************************************************************************/
    TreeWalk tw_accretion[1] = {{0}};

    tw_accretion->ev_label = "BH_ACCRETION";
    tw_accretion->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_accretion->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHAccretion);
    tw_accretion->ngbiter = (TreeWalkNgbIterFunction) blackhole_accretion_ngbiter;
    tw_accretion->haswork = NULL;
    tw_accretion->postprocess = (TreeWalkProcessFunction) blackhole_accretion_postprocess;
    tw_accretion->preprocess = NULL;
    tw_accretion->fill = (TreeWalkFillQueryFunction) blackhole_accretion_copy;
    tw_accretion->reduce = (TreeWalkReduceResultFunction) blackhole_accretion_reduce;
    tw_accretion->query_type_elsize = sizeof(TreeWalkQueryBHAccretion);
    tw_accretion->result_type_elsize = sizeof(TreeWalkResultBHAccretion);
    tw_accretion->tree = tree;
    tw_accretion->priv = priv;

    /* This treewalk marks all black holes and gas which can be swallowed with a SwllowID of a potential swallower.
     * The treewalk is symmetric. A swallower needs to be active, the black holes must be within each other's
     * smoothing radius and optionally gravitationally bound. In case a black hole can be swallowed by multiple  */
    treewalk_run(tw_accretion, ActiveBlackHoles, NumActiveBlackHoles);

    /*************************************************************************/

    walltime_measure("/BH/Accretion");

    priv->BH_accreted_Mass = (MyFloat *) mymalloc("BH_accretedmass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_BHMass = (MyFloat *) mymalloc("BH_accreted_BHMass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_momentum = (MyFloat (*) [3]) mymalloc("BH_accretemom", 3* SlotsManager->info[5].size * sizeof(priv->BH_accreted_momentum[0]));
    priv->BH_accreted_StarClusterMass = (MyFloat *) mymalloc("BH_accreted_SCMass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->BH_accreted_SCmax = (struct bh_sc_state *) mymalloc("BH_accreted_SCmax", SlotsManager->info[5].size * sizeof(struct bh_sc_state));
    priv->BH_GWRecoilKick = (MyFloat (*) [3]) mymalloc("BH_GWRecoilKick", 3 * SlotsManager->info[5].size * sizeof(MyFloat));
    memset(priv->BH_GWRecoilKick, 0, 3 * SlotsManager->info[5].size * sizeof(MyFloat));

    /* Now do the swallowing of particles and dump feedback energy */
    /* We also merge BHs here. Only BHs which are not themselves
     * being swallowed are eligible to swallow. If there are multiple BHs within a search radius,
     * the BH with the largest ID will be selected by the SwallowID search and will swallow all the
     * surrounding BHs.
     * Example 1:
     * We have BHs A,B,C, where A and B are close and B and C are close. B.ID < C.ID and A.ID < B.ID.
     * B will be swallowed by C. A will not be swallowed by B as B is itself being swallowed (A will
     * have a non-zero encounter).
     * Example 2:
     * We have BHs A,B,C, where A and C are both close to B. B.ID > C.ID and A.ID < B.ID.
     * In this case B will swallow C and A.
     * Example 3:
     * We have BHs A,B,C, where A and B are close and B and C are close. B.ID < C.ID and A.ID > B.ID.
     * In this case B will be swallowed by whichever of A and C has the larger ID.
    */
    blackhole_feedback(ActiveBlackHoles, NumActiveBlackHoles, tree, priv);

    walltime_measure("/BH/Feedback");

    if(FdBlackholeDetails){
        *bhdetailswritten += collect_BH_info(ActiveBlackHoles, NumActiveBlackHoles, priv, PartManager, (struct bh_particle_data*) SlotsManager->info[5].ptr, FdBlackholeDetails);
    }

    myfree(priv->BH_GWRecoilKick);
    myfree(priv->BH_accreted_SCmax);
    myfree(priv->BH_accreted_StarClusterMass);
    myfree(priv->BH_accreted_momentum);
    myfree(priv->BH_accreted_BHMass);
    myfree(priv->BH_accreted_Mass);

    /*****************************************************************/
    myfree(priv->BH_SoundSpeed);
    myfree(priv->BH_VorticityVec);
    myfree(priv->BH_Vorticity);
    myfree(priv->KEflag);
    myfree(priv->MgasEnc);
    myfree(priv->NumDM);

    myfree(priv->BH_SurroundingGasVel);
    myfree(priv->BH_Entropy);

    myfree(priv->BH_FeedbackWeightSum);
    myfree(priv->BH_SwallowID);
    myfree(priv->SPH_SwallowID);

    myfree(ActiveBlackHoles);

    write_blackhole_txt(FdBlackHoles, units, atime);
    walltime_measure("/BH/Info");
}

static void
blackhole_accretion_postprocess(int i, TreeWalk * tw)
{
    int k;
    int PI = P[i].PI;
    double mdot = 0;    /* if no accretion model is enabled, we have mdot=0 */
    /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
    const double meddington = (4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS / (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * BHP(i).Mass
            * BH_GET_PRIV(tw)->units.UnitTime_in_s / BH_GET_PRIV(tw)->CP->HubbleParam;

    if(BHP(i).Density > 0)
    {
        BH_GET_PRIV(tw)->BH_Entropy[PI] /= BHP(i).Density;
        for(k = 0; k < 3; k++)
            BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k] /= BHP(i).Density;

        double bhvel = 0;
        for(k = 0; k < 3; k++)
            bhvel += pow(P[i].Vel[k] - BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k], 2);

        bhvel = sqrt(bhvel);
        bhvel /= BH_GET_PRIV(tw)->atime;
        double rho = BHP(i).Density;
        double rho_proper = rho * BH_GET_PRIV(tw)->a3inv;

        double soundspeed = blackhole_soundspeed(BH_GET_PRIV(tw)->BH_Entropy[PI], rho, BH_GET_PRIV(tw)->atime);
        BH_GET_PRIV(tw)->BH_SoundSpeed[PI] = soundspeed;

        double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

        if(norm > 0)
            mdot = 4. * M_PI * blackhole_params.BlackHoleAccretionFactor * BH_GET_PRIV(tw)->CP->GravInternal * BH_GET_PRIV(tw)->CP->GravInternal *
                BHP(i).Mass * BHP(i).Mass * rho_proper / norm;

        /* Compute dimensionless vorticity: omega_star = omega_phys * G * M_BH / (c_s^2 + v_rel^2)^1.5
         * where R_bondi = G * M_BH / (c_s^2 + v_rel^2).
         * The SPH curl in comoving code units gives omega_code = a^2 * omega_phys,
         * because v_code = a * v_phys and nabla_comov W = a^4 * nabla_phys W_phys,
         * so after dividing by rho_comov: omega_code = a^2 * omega_phys.
         * soundspeed and bhvel are already physical. */
        if(blackhole_params.BHVorticity && norm > 0) {
            double omega = 0;
            for(k = 0; k < 3; k++) {
                double vort_k = BH_GET_PRIV(tw)->BH_VorticityVec[PI][k] / rho;
                omega += vort_k * vort_k;
            }
            omega = sqrt(omega);
            /* Convert from comoving to physical: omega_phys = omega_code / a^2 */
            double atime = BH_GET_PRIV(tw)->atime;
            omega /= (atime * atime);
            double G = BH_GET_PRIV(tw)->CP->GravInternal;
            BH_GET_PRIV(tw)->BH_Vorticity[PI] = omega * G * BHP(i).Mass / norm;
        } else if(blackhole_params.BHVorticity) {
            BH_GET_PRIV(tw)->BH_Vorticity[PI] = 0;
        }
    }

    if(blackhole_params.BlackHoleEddingtonFactor > 0.0 &&
        mdot > blackhole_params.BlackHoleEddingtonFactor * meddington) {
        mdot = blackhole_params.BlackHoleEddingtonFactor * meddington;
    }
    BHP(i).Mdot = mdot;

    double dtime = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift) / BH_GET_PRIV(tw)->hubble;

    BHP(i).Mass += BHP(i).Mdot * dtime;

    /*************************************************************************/

    if(blackhole_params.BH_DRAG > 0){
        /* a_BH = (v_gas - v_BH) Mdot/M_BH                                   */
        /* motivated by BH gaining momentum from the accreted gas            */
        /*c.f.section 3.2,in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
        double fac = 0;
        if (blackhole_params.BH_DRAG == 1) {
            fac = BHP(i).Mdot / P[i].Mass;
        }
        if (blackhole_params.BH_DRAG == 2) fac = blackhole_params.BlackHoleEddingtonFactor * meddington/BHP(i).Mass;
        fac *= BH_GET_PRIV(tw)->atime; /* dv = acc * kick_fac = acc * a^{-1}dt, therefore acc = a*dv/dt  */
        for(k = 0; k < 3; k++) {
            BHP(i).DragAccel[k] = -(P[i].Vel[k] - BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k])*fac;
        }
    }
    else{
        for(k = 0; k < 3; k++){
            BHP(i).DragAccel[k] = 0;
        }
    }
    /*************************************************************************/

    if(blackhole_params.BlackHoleKineticOn == 1){
        /* accumulate kenetic feedback energy by dE = epsilon x mdot x c^2 */
        /* epsilon = Min(rho_BH/(BHKE_EffRhoFactor*rho_sfr),BHKE_EffCap)   */
        /* KE is released when exceeding injection energy threshold        */
        BH_GET_PRIV(tw)->KEflag[PI] = 0;
        double Edd_ratio = BHP(i).Mdot/meddington;
        double lam_thresh = blackhole_params.BHKE_EddingtonThrFactor;
        double x = blackhole_params.BHKE_EddingtonMFactor * pow(BHP(i).Mass/blackhole_params.BHKE_EddingtonMPivot, blackhole_params.BHKE_EddingtonMIndex);
        if (lam_thresh > x)
            lam_thresh = x;
        if (Edd_ratio < lam_thresh){
            /* mark this timestep is accumulating KE feedback energy */
            BH_GET_PRIV(tw)->KEflag[PI] = 1;
            const double rho_crit_baryon = BH_GET_PRIV(tw)->CP->OmegaBaryon * 3 * pow(BH_GET_PRIV(tw)->CP->Hubble, 2) / (8 * M_PI * BH_GET_PRIV(tw)->CP->GravInternal);
            const double rho_sfr = blackhole_params.BHKE_SfrCritOverDensity * rho_crit_baryon;
            double epsilon = (BHP(i).Density/rho_sfr)/blackhole_params.BHKE_EffRhoFactor;
            if (epsilon > blackhole_params.BHKE_EffCap){
                epsilon = blackhole_params.BHKE_EffCap;
            }

            BHP(i).KineticFdbkEnergy += epsilon * (BHP(i).Mdot * dtime * pow(LIGHTCGS / BH_GET_PRIV(tw)->units.UnitVelocity_in_cm_per_s, 2));
        }
        /* decide whether to release KineticFdbkEnergy*/
        double KE_thresh = 0.5 * BHP(i).VDisp * BHP(i).VDisp * BH_GET_PRIV(tw)->MgasEnc[PI];
        KE_thresh *= blackhole_params.BHKE_InjEnergyThr;

        if (BHP(i).VDisp > 0 && BHP(i).KineticFdbkEnergy > KE_thresh){
            /* mark KineticFdbkEnergy is ready to be released in the feedback treewalk */
            BH_GET_PRIV(tw)->KEflag[PI] = 2;
        }
    }
}

static void
blackhole_accretion_ngbiter(TreeWalkQueryBHAccretion * I,
        TreeWalkResultBHAccretion * O,
        TreeWalkNgbIterBHAccretion * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {
        O->encounter = 0;
        iter->base.mask = GASMASK + BHMASK;
        iter->base.Hsml = I->Hsml;
        /* Symmetric for the BH mergers*/
        iter->base.symmetric = NGB_TREEFIND_SYMMETRIC;

        density_kernel_init(&iter->kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;

    if(P[other].Mass < 0) return;

     /* BH does not accrete wind */
    if(winds_is_particle_decoupled(other)) return;

    /* Accretion / merger doesn't do self interaction */
    if(P[other].ID == I->ID) return;

    /* we have a black hole merger. Now we use 2 times GravitationalSoftening as merging criteria.
     * Note there is another condition: they must also be closer than the SPH smoothing length,
     * as enforced by the tree search above. */
    if(P[other].Type == 5 && r < (2*FORCE_SOFTENING()/2.8))
    {
        O->encounter = 1; // mark the event when two BHs encounter each other

        int flag = 0; // the flag for BH merge

        if(BHGetRepositionEnabled() == 1) // directly merge if reposition is enabled
            flag = 1;
        if(blackhole_params.MergeGravBound == 0)
            flag = 1;
        /* apply Grav Bound check only when Reposition is disabled, otherwise BHs would be repositioned to the same location but not merge */
        if(blackhole_params.MergeGravBound == 1 && BHGetRepositionEnabled() == 0){

            double dx[3];
            double dv[3];
            double da[3];
            int d;
            MyFloat VelPred[3];
            DM_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
            for(d = 0; d < 3; d++){
                dx[d] = NEAREST(I->base.Pos[d] - P[other].Pos[d], PartManager->BoxSize);
                dv[d] = I->Vel[d] - VelPred[d];
                /* we include long range PM force, short range force from the last long timestep and DF */
                da[d] = (I->Accel[d] - P[other].FullTreeGravAccel[d] - P[other].GravPM[d] - BHP(other).DFAccel[d]);
            }
            flag = check_grav_bound(dx,dv,da, BH_GET_PRIV(lv->tw)->atime);
            /*if(flag == 0)
                message(0, "dx %g %g %g dv %g %g %g da %g %g %g\n",dx[0], dx[1], dx[2], dv[0], dv[1], dv[2], da[0], da[1], da[2]);*/
        }

        /* Mark the BH via SwallowID.*/
        if(flag == 1)
        {
            MyIDType readid, newswallowid;

            int PI = P[other].PI;
            MyIDType * swal = BH_GET_PRIV(lv->tw)->BH_SwallowID + PI;
            #pragma omp atomic read
            readid = *swal;

            /* Here we mark the black hole as "ready to be swallowed" using the SwallowID.
             * The actual swallowing is done in the feedback treewalk by setting Swallowed = 1
             * and merging the masses.*/
            do {
                /* Already marked, prefer to be swallowed by a bigger ID */
                if(readid != 0 && readid < I->ID ) {
                    /* Already marked, prefer to be swallowed by a bigger ID */
                    newswallowid = I->ID + 1;
                } else if(readid == 0 && (P[other].ID < I->ID || !is_timebin_active(P[other].TimeBinHydro, P[other].Ti_drift))) {
                    /* Unmarked, the BH with bigger ID swallows This avoids two BHs trying to swallow each other.
                     * (in which case neither would merge). If only one is active, only one can swallow.*/
                    newswallowid = I->ID + 1;
                }
                else
                    break;
            /* Swap in the new id only if the old one hasn't changed:
             * in principle an extension, but supported on at least clang >= 9, gcc >= 5 and icc >= 18.*/
            } while(!__atomic_compare_exchange_n(swal, &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        }
    }


    if(P[other].Type == 0) {
        if(r2 < iter->kernel.HH) {
            double u = r * iter->kernel.Hinv;
            double wk = density_kernel_wk(&iter->kernel, u);
            double mass_j = P[other].Mass;

            O->SmoothedEntropy += (mass_j * wk * SPHP(other).Entropy);
            MyFloat VelPred[3];
            SPH_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
            O->GasVel[0] += (mass_j * wk * VelPred[0]);
            O->GasVel[1] += (mass_j * wk * VelPred[1]);
            O->GasVel[2] += (mass_j * wk * VelPred[2]);

            /* Accumulate SPH curl of velocity: Σ m_j (v_j - v_BH) × (dwk/r * dist) */
            if(blackhole_params.BHVorticity && r > 0) {
                double dwk = density_kernel_dwk(&iter->kernel, r * iter->kernel.Hinv);
                double fac = mass_j * dwk / r;
                double dv[3];
                const double * dist = iter->base.dist;
                int d;
                for(d = 0; d < 3; d++)
                    dv[d] = VelPred[d] - I->Vel[d];
                double rot[3];
                crossproduct(dv, dist, rot);
                for(d = 0; d < 3; d++)
                    O->Vorticity[d] += fac * rot[d];
            }

            /* here we have a gas particle; check for swallowing */

            /* compute accretion probability */
            double p = 0;

            /* Mtrack traces mass conservation via stochastic gas swallowing.
             * Compare Mtrack with BH_Mass to determine swallowing probability. */
            MyFloat BHPartMass = I->Mtrack;

            /* This is an averaged Mdot, because Mdot increases BH_Mass but not Mass.
             * So if the total accretion is significantly above the dynamical mass,
             * a particle is swallowed. */
            if((I->BH_Mass - BHPartMass) > 0 && I->Density > 0)
                p = (I->BH_Mass - BHPartMass) * wk / I->Density;

            /* compute random number, uniform in [0,1] */
            const double w = get_random_number(P[other].ID, BH_GET_PRIV(lv->tw)->rnd);
            if(w < p)
            {
                MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;
                MyIDType readid, newswallowid;
                #pragma omp atomic read
                readid = SPH_SwallowID[P[other].PI];
                do {
                    /* Already marked, prefer to be swallowed by a bigger ID.
                     * Not marked, the SwallowID is 0 */
                    if(readid < I->ID + 1) {
                        newswallowid = I->ID + 1;
                    }
                    else
                        break;
                    /* Swap in the new id only if the old one hasn't changed*/
                } while(!__atomic_compare_exchange_n(&SPH_SwallowID[P[other].PI], &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
            }
        }

        if(r2 < iter->kernel.HH) {
            /* update the feedback weighting */
            double mass_j;
            if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_OPTTHIN)) {
                double redshift = 1./BH_GET_PRIV(lv->tw)->atime - 1;
                double nh0 = get_neutral_fraction_sfreff(redshift, BH_GET_PRIV(lv->tw)->hubble, &P[other], &SPHP(other));
                if(r2 > 0)
                    O->FeedbackWeightSum += (P[other].Mass * nh0) / r2;
            } else {
                if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                    mass_j = P[other].Mass;
                } else {
                    mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
                }
                if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE)) {
                    double u = r * iter->kernel.Hinv;
                    O->FeedbackWeightSum += (mass_j *
                          density_kernel_wk(&iter->kernel, u)
                           );
                } else {
                    O->FeedbackWeightSum += (mass_j);
                }
            }
        }
    }

    /* collect info for sigmaDM and Menc for kinetic feedback */
    if(blackhole_params.BlackHoleKineticOn == 1 &&
        P[other].Type == 0 &&
        r2 < iter->kernel.HH ){
            O->MgasEnc += P[other].Mass;
        }
}

static void
blackhole_accretion_reduce(int place, TreeWalkResultBHAccretion * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;

    int PI = P[place].PI;

    /* Set encounter to true if it is true on any remote*/
    if (mode == TREEWALK_PRIMARY || BHP(place).encounter < remote->encounter) {
        BHP(place).encounter = remote->encounter;
    }

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI], remote->FeedbackWeightSum);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_Entropy[PI], remote->SmoothedEntropy);
    for (k = 0; k < 3; k++){
        TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_SurroundingGasVel[PI][k], remote->GasVel[k]);
    }
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->NumDM[PI], remote->NumDM);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->MgasEnc[PI], remote->MgasEnc);
    if(blackhole_params.BHVorticity) {
        for (k = 0; k < 3; k++)
            TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_VorticityVec[PI][k], remote->Vorticity[k]);
    }
}

static void
blackhole_accretion_copy(int place, TreeWalkQueryBHAccretion * I, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Vel[k] = P[place].Vel[k];
        I->Accel[k] = P[place].FullTreeGravAccel[k] + P[place].GravPM[k] + BHP(place).DFAccel[k];
    }
    I->Hsml = P[place].Hsml;
    I->Mass = P[place].Mass;
    I->BH_Mass = BHP(place).Mass;
    I->Density = BHP(place).Density;
    I->ID = P[place].ID;
    I->Mtrack = BHP(place).Mtrack;
}

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Mtrack;
    MyFloat BH_Mass;
    MyIDType ID;
    MyFloat Density;
    MyFloat FeedbackEnergy;
    MyFloat FeedbackWeightSum;
    MyFloat KEFeedbackEnergy;
    int FdbkChannel; /* 0 thermal, 1 kinetic */
    int alignment; /* Ensure alignment*/
    MyFloat Vel[3]; /* Velocity of the swallower BH, for GW recoil kick direction */
} TreeWalkQueryBHFeedback;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Mass; /* the accreted Mdyn */
    MyFloat AccretedMomentum[3];
    MyFloat BH_Mass;
    MyFloat StarClusterMass; /* summed star cluster mass of the swallowed BHs (for Mtrack) */
    struct bh_sc_state SCmax; /* heaviest star cluster among the swallowed BHs (Mass 0: none) */
    int BH_CountProgs;
    int BH_minTimeBin;
    MyFloat GWRecoilKick[3]; /* Accumulated GW recoil kick velocity from BH mergers */
} TreeWalkResultBHFeedback;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
} TreeWalkNgbIterBHFeedback;

/* Adds the injected black hole energy to an internal energy and caps it at a maximum temperature*/
static double
add_injected_BH_energy(double unew, double injected_BH_energy, double mass, double uu_in_cgs)
{
    unew += injected_BH_energy / mass;
    const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * uu_in_cgs;
    /* Cap temperature*/
    if(unew > 5.0e8 / u_to_temp_fac)
        unew = 5.0e8 / u_to_temp_fac;

    return unew;
}

static int
get_random_dir(int i, double dir[3], const RandTable * const rnd)
{
    double theta = acos(2 * get_random_number(P[i].ID + 3, rnd) - 1);
    double phi = 2 * M_PI * get_random_number(P[i].ID + 4, rnd);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
    return 0;
}

/* Star cluster at a BH merger: the remnant keeps the heaviest of the merging clusters.
 * Cluster (mass ma, carried by BH ida) beats (mb, idb) if it is heavier, ties going to the
 * larger BH ID so the choice does not depend on the order the merger is evaluated in. */
static int
sc_state_heavier(const MyFloat ma, const MyIDType ida, const MyFloat mb, const MyIDType idb)
{
    return ma > mb || (ma == mb && ida > idb);
}

/* Copy the full star-cluster state of BH particle i into s */
static void
sc_state_get(struct bh_sc_state * s, const int i)
{
    s->ID = P[i].ID;
    s->Mass = BHP(i).StarClusterMass;
    s->FormationTime = BHP(i).StarClusterFormationTime;
    s->Metallicity = BHP(i).StarClusterMetallicity;
    s->TotalMassReturned = BHP(i).StarClusterTotalMassReturned;
    s->InitReff = BHP(i).SC_initReff;
    s->Reff = BHP(i).SC_Reff;
    s->RlxPendingMyr = BHP(i).SC_RlxPendingMyr;
    memcpy(s->Metals, BHP(i).StarClusterMetals, sizeof(s->Metals));
    s->LastEnrichmentMyr = BHP(i).StarClusterLastEnrichmentMyr;
}

/* Give BH particle i the star cluster described by s (replacing its own) */
static void
sc_state_set(const int i, const struct bh_sc_state * s)
{
    BHP(i).StarClusterMass = s->Mass;
    BHP(i).StarClusterFormationTime = s->FormationTime;
    BHP(i).StarClusterMetallicity = s->Metallicity;
    BHP(i).StarClusterTotalMassReturned = s->TotalMassReturned;
    BHP(i).SC_initReff = s->InitReff;
    BHP(i).SC_Reff = s->Reff;
    BHP(i).SC_RlxPendingMyr = s->RlxPendingMyr;
    memcpy(BHP(i).StarClusterMetals, s->Metals, sizeof(s->Metals));
    BHP(i).StarClusterLastEnrichmentMyr = s->LastEnrichmentMyr;
}

/**
 * perform blackhole swallow / merger;
 */
static void
blackhole_feedback_ngbiter(TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        TreeWalkNgbIterBHFeedback * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {
        O->BH_minTimeBin = TIMEBINS;
        iter->base.mask = GASMASK + BHMASK;
        iter->base.Hsml = I->Hsml;
        /* Needs to be symmetric because the BH mergers should be symmetric*/
        iter->base.symmetric = NGB_TREEFIND_SYMMETRIC;
        density_kernel_init(&iter->kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;
    double r = iter->base.r;
    /* Exclude self interaction */

    if(P[other].ID == I->ID) return;

     /* BH does not accrete wind */
    if(winds_is_particle_decoupled(other))
        return;


    /* we have a black hole merger! */
    int PI = P[other].PI;
    if(P[other].Type == 5 && BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] != 0)
    {
        if(BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] != I->ID + 1) return;

        /* Swallow the particle*/
        /* A note on Swallowed vs SwallowID: black hole particles which have been completely swallowed
         * (ie, their mass has been added to another particle) have Swallowed = 1.
         * These particles are ignored in future tree walks. SwallowID is set to the swallowing particle.
         */
        BHP(other).SwallowID = BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] - 1;
        BHP(other).SwallowTime = BH_GET_PRIV(lv->tw)->atime;
        P[other].Swallowed = 1;
        /* Set encounter to zero when we merge*/
        BHP(other).encounter = 0;
        O->BH_CountProgs += BHP(other).CountProgs;
        O->BH_Mass += (BHP(other).Mass);
        O->StarClusterMass += BHP(other).StarClusterMass;
        /* Keep the heaviest of the swallowed clusters, with its full state */
        if(BHP(other).StarClusterMass > 0 &&
           sc_state_heavier(BHP(other).StarClusterMass, P[other].ID, O->SCmax.Mass, O->SCmax.ID))
            sc_state_get(&O->SCmax, other);

        /* Use the true physical mass (Mtrack + SC) for merger bookkeeping,
         * not the possibly inflated SeedBHDynMass stored in P.Mass.
         * This ensures dynaccmass = othermass - SC = Mtrack_other in
         * postprocess, so the remnant Mtrack = sum of Mtracks. */
        double othermass = BHP(other).Mtrack + BHP(other).StarClusterMass;
        /* Add the accreted mass tracer to the total accreted Mass.
         * We will decide in postprocess if it goes into Mtrack or Mass*/
        O->Mass += othermass;

        MyFloat VelPred[3];
        DM_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (othermass * VelPred[d]);

        /* GW recoil kick for non-spinning BH mergers.
         * When multiple BHs are swallowed in one step, each kick is computed
         * against the original swallower mass (I->BH_Mass is fixed in the
         * query and not updated during the neighbor loop).  This is equivalent
         * to assuming the swallowed progenitors first merge among themselves,
         * then the combined remnant merges with the swallower — the individual
         * kicks are vector-summed into GWRecoilKick and applied once in
         * postprocess. */
        if(blackhole_params.GWRecoilVelocityKick || blackhole_params.GWRecoilSCKick) {
            /* Mass ratio q = m_small / m_large, with q <= 1 */
            double m1 = I->BH_Mass;  /* swallower BH mass */
            double m2 = BHP(other).Mass;  /* swallowed BH mass */
            double q = (m1 < m2) ? m1 / m2 : m2 / m1;

            /* Kick magnitude from Fitchett 1983 / Gonzalez+ 2007 fit */
            const double A_gw = 1.2e4;   /* km/s */
            const double B_gw = -0.93;
            double eta = q / ((1.0 + q) * (1.0 + q));
            double v_kick = A_gw * eta * eta * (1.0 - q) / (1.0 + q) * (1.0 + B_gw * eta);

            /* Kick direction: random in orbital plane (perpendicular to L).
             * L = (r1 - r2) x (v1 - v2) */
            double dx[3], dv[3];
            for(d = 0; d < 3; d++) {
                dx[d] = NEAREST(I->base.Pos[d] - P[other].Pos[d], PartManager->BoxSize);
                dv[d] = I->Vel[d] - VelPred[d];
            }
            double Lx = dx[1]*dv[2] - dx[2]*dv[1];
            double Ly = dx[2]*dv[0] - dx[0]*dv[2];
            double Lz = dx[0]*dv[1] - dx[1]*dv[0];
            double Lmag = sqrt(Lx*Lx + Ly*Ly + Lz*Lz);

            double nx, ny, nz;
            if(Lmag > 0) {
                /* Normalize L */
                Lx /= Lmag; Ly /= Lmag; Lz /= Lmag;

                /* Find a vector not parallel to L to construct orthonormal basis */
                double ax, ay, az;
                if(fabs(Lx) <= fabs(Ly) && fabs(Lx) <= fabs(Lz)) {
                    ax = 1; ay = 0; az = 0;
                } else if(fabs(Ly) <= fabs(Lz)) {
                    ax = 0; ay = 1; az = 0;
                } else {
                    ax = 0; ay = 0; az = 1;
                }
                /* e1 = a x L (normalized) */
                double e1x = ay*Lz - az*Ly;
                double e1y = az*Lx - ax*Lz;
                double e1z = ax*Ly - ay*Lx;
                double e1mag = sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
                e1x /= e1mag; e1y /= e1mag; e1z /= e1mag;
                /* e2 = L x e1 */
                double e2x = Ly*e1z - Lz*e1y;
                double e2y = Lz*e1x - Lx*e1z;
                double e2z = Lx*e1y - Ly*e1x;

                /* Random angle in the orbital plane */
                double phi = 2 * M_PI * get_random_number(P[other].ID + 7, BH_GET_PRIV(lv->tw)->rnd);
                nx = cos(phi)*e1x + sin(phi)*e2x;
                ny = cos(phi)*e1y + sin(phi)*e2y;
                nz = cos(phi)*e1z + sin(phi)*e2z;
            } else {
                /* Head-on merger (L = 0): pick isotropic random direction */
                double theta = acos(2 * get_random_number(P[other].ID + 5, BH_GET_PRIV(lv->tw)->rnd) - 1);
                double phi = 2 * M_PI * get_random_number(P[other].ID + 7, BH_GET_PRIV(lv->tw)->rnd);
                nx = sin(theta) * cos(phi);
                ny = sin(theta) * sin(phi);
                nz = cos(theta);
            }

            /* Convert v_kick from km/s to internal velocity units.
             * Internal velocity = physical velocity * atime (scale factor).
             * 1 km/s = 1e5 cm/s; divide by UnitVelocity_in_cm_per_s to get
             * code velocity units, then multiply by atime. */
            double atime = BH_GET_PRIV(lv->tw)->atime;
            double v_kick_internal = v_kick * (1.0e5 / BH_GET_PRIV(lv->tw)->units.UnitVelocity_in_cm_per_s) * atime;

            O->GWRecoilKick[0] += v_kick_internal * nx;
            O->GWRecoilKick[1] += v_kick_internal * ny;
            O->GWRecoilKick[2] += v_kick_internal * nz;
        }

        if(BHP(other).SwallowTime < BH_GET_PRIV(lv->tw)->atime)
            endrun(2, "Encountered BH %i swallowed at earlier time %g\n", other, BHP(other).SwallowTime);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_BH_swallowed[tid]++;

    }

    MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;

    /* perform thermal or kinetic feedback into non-swallowed particles. */
    if(P[other].Type == 0 && SPH_SwallowID[P[other].PI] == 0 &&
        (r2 < iter->kernel.HH))
    {
        /* For accretion stability, set the BH timestep to the smallest gas timestep.
         * Ignore swallowed or wind gas particles.*/
        if (O->BH_minTimeBin > P[other].TimeBinHydro)
            O->BH_minTimeBin = P[other].TimeBinHydro;

        double u = r * iter->kernel.Hinv;
        double wk = 1.0;
        double mass_j;

        if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
            mass_j = P[other].Mass;
        } else {
            mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
        }
        if(HAS(blackhole_params.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE))
            wk = density_kernel_wk(&iter->kernel, u);

        /* thermal feedback */
        if(I->FeedbackWeightSum > 0 && I->FeedbackEnergy > 0 && I->FdbkChannel == 0 && P[other].Mass > 0){
            const double injected_BH = I->FeedbackEnergy * mass_j * wk / I->FeedbackWeightSum;
            /* Set a flag for star-forming particles:
                * we want these to cool to the EEQOS via
                * tcool rather than trelax.*/
            if(sfreff_on_eeqos(&SPHP(other), BH_GET_PRIV(lv->tw)->a3inv)) {
                /* We cannot atomically set a bitfield.
                 * This flag is never read in this thread loop, and we are careful not to
                 * do this with a swallowed particle (as this can race with IsGarbage being set).
                 * So lack of atomicity is (I think) not a problem.*/
                //#pragma omp atomic write
                P[other].BHHeated = 1;
            }
            const double enttou = pow(SPHP(other).Density * BH_GET_PRIV(lv->tw)->a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
            const double uu_in_cgs = BH_GET_PRIV(lv->tw)->units.UnitEnergy_in_cgs / BH_GET_PRIV(lv->tw)->units.UnitMass_in_g;

            double entold, entnew;
            double * entptr = &(SPHP(other).Entropy);
            #pragma omp atomic read
            entold = *entptr;
            do {
                entnew = add_injected_BH_energy(entold * enttou, injected_BH, P[other].Mass, uu_in_cgs) / enttou;
                /* Swap in the new gas entropy only if the old one hasn't changed.*/
            } while(!__atomic_compare_exchange(entptr, &entold, &entnew, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        }

        /* kinetic feedback */
        if(I->KEFeedbackEnergy > 0 && I->FdbkChannel == 1 && I->Density > 0){
            /* Kick the gas particle*/
            double dvel = sqrt(2 * I->KEFeedbackEnergy * wk / I->Density);
            double dir[3];
            get_random_dir(other, dir, BH_GET_PRIV(lv->tw)->rnd);
            int j;
            for(j = 0; j < 3; j++){
                #pragma omp atomic update
                P[other].Vel[j] += (dvel*dir[j]);
            }
        }
    }

    /* Swallowing a gas */
    /* This will only be true on one thread so we do not need a lock here*/
    /* Note that it will rarely happen that gas is swallowed by a BH which is itself swallowed.
     * In that case we do not swallow this particle: all swallowing changes before this are temporary*/
    if(P[other].Type == 0 && SPH_SwallowID[P[other].PI] == I->ID+1)
    {
        O->Mass += P[other].Mass;
        MyFloat VelPred[3];
        SPH_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (P[other].Mass * VelPred[d]);

        slots_mark_garbage(other, PartManager, SlotsManager);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_sph_swallowed[tid]++;
    }
}
static int
blackhole_feedback_haswork(int n, TreeWalk * tw)
{
    /*Black hole not being swallowed*/
    int PI = P[n].PI;
    return (P[n].Type == 5) && (!P[n].Swallowed) && (BH_GET_PRIV(tw)->BH_SwallowID[PI] == 0);
}

static void
blackhole_feedback_copy(int i, TreeWalkQueryBHFeedback * I, TreeWalk * tw)
{
    I->Hsml = P[i].Hsml;
    I->BH_Mass = BHP(i).Mass;
    I->ID = P[i].ID;
    I->Density = BHP(i).Density;
    int PI = P[i].PI;
    int k;
    for(k = 0; k < 3; k++)
        I->Vel[k] = P[i].Vel[k];

    I->FeedbackWeightSum = BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI];
    I->FdbkChannel = 0; /* thermal feedback mode */

    double dtime = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift) / BH_GET_PRIV(tw)->hubble;

    I->FeedbackEnergy = blackhole_params.BlackHoleFeedbackFactor * 0.1 * BHP(i).Mdot * dtime *
                pow(LIGHTCGS / BH_GET_PRIV(tw)->units.UnitVelocity_in_cm_per_s, 2);
    I->KEFeedbackEnergy = 0;
    if (blackhole_params.BlackHoleKineticOn == 1 && BH_GET_PRIV(tw)->KEflag[PI] > 0){
        I->FdbkChannel = 1; /* kinetic feedback mode, (no thermal feedback for this timestep) */
        /* KEflag = 1: KEFeedbackEnergy is accumulating; KEflag = 2: KEFeedbackEnergy is released. */
        if (BH_GET_PRIV(tw)->KEflag[PI] == 2){
            I->KEFeedbackEnergy = BHP(i).KineticFdbkEnergy;
        }
    }
}

static void
blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    int PI = P[place].PI;

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_Mass[PI], remote->Mass);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_BHMass[PI], remote->BH_Mass);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_StarClusterMass[PI], remote->StarClusterMass);
    for(k = 0; k < 3; k++) {
        TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_momentum[PI][k], remote->AccretedMomentum[k]);
        TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_GWRecoilKick[PI][k], remote->GWRecoilKick[k]);
    }
    /* Arg-max reduce: the heaviest swallowed cluster over all ranks */
    struct bh_sc_state * scmax = &BH_GET_PRIV(tw)->BH_accreted_SCmax[PI];
    if(mode == TREEWALK_PRIMARY || sc_state_heavier(remote->SCmax.Mass, remote->SCmax.ID, scmax->Mass, scmax->ID))
        *scmax = remote->SCmax;
    if (mode == TREEWALK_PRIMARY || BHP(place).minTimeBin > remote->BH_minTimeBin) {
        BHP(place).minTimeBin = remote->BH_minTimeBin;
    }
    TREEWALK_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
}

static void
blackhole_feedback_postprocess(int n, TreeWalk * tw)
{
    const int PI = P[n].PI;
    if(BH_GET_PRIV(tw)->BH_accreted_BHMass[PI] > 0){
       BHP(n).Mass += BH_GET_PRIV(tw)->BH_accreted_BHMass[PI];
    }
    /* Star clusters at the merger: the remnant carries ONE cluster, the heaviest of the merging
     * clusters (its own or a swallowed one, compared by cluster mass rather than inherited with
     * the surviving BH ID), with that cluster's full state -- mass, radii, metallicity, formation
     * time and stellar-evolution bookkeeping.  The lighter clusters are removed: their mass is
     * not added to the remnant (with StarClusterBHDyn=1 it leaves P.Mass, which is recomputed
     * from Mtrack + StarClusterMass below). */
    {
        const struct bh_sc_state * scmax = &BH_GET_PRIV(tw)->BH_accreted_SCmax[PI];
        if(BH_GET_PRIV(tw)->BH_accreted_StarClusterMass[PI] > 0 && scmax->Mass > 0 &&
           sc_state_heavier(scmax->Mass, scmax->ID, BHP(n).StarClusterMass, P[n].ID))
            sc_state_set(n, scmax);
    }
    if(BH_GET_PRIV(tw)->BH_accreted_Mass[PI] > 0)
    {
        /* velocity feedback due to accretion; momentum conservation. */
        const MyFloat accmass = BH_GET_PRIV(tw)->BH_accreted_Mass[PI];
        int k;
        for(k = 0; k < 3; k++)
            P[n].Vel[k] = (P[n].Vel[k] * P[n].Mass + BH_GET_PRIV(tw)->BH_accreted_momentum[PI][k]) / (P[n].Mass + accmass);

        /* Apply GW recoil kick to merger remnant velocity */
        if(blackhole_params.GWRecoilVelocityKick) {
            for(k = 0; k < 3; k++)
                P[n].Vel[k] += BH_GET_PRIV(tw)->BH_GWRecoilKick[PI][k];
        }

        /* Check if the kick ejects the BH from its host star cluster.
         * If v_kick > v_esc of the remnant's (kept) star cluster, zero the SC mass. */
        if(blackhole_params.GWRecoilSCKick && BHP(n).StarClusterMass > 0) {
            /* Kick magnitude in physical km/s */
            double v_kick_sq = 0;
            for(k = 0; k < 3; k++)
                v_kick_sq += BH_GET_PRIV(tw)->BH_GWRecoilKick[PI][k]
                           * BH_GET_PRIV(tw)->BH_GWRecoilKick[PI][k];
            double atime = BH_GET_PRIV(tw)->atime;
            double unit_vel = BH_GET_PRIV(tw)->units.UnitVelocity_in_cm_per_s;
            double v_kick_kms = sqrt(v_kick_sq) / atime * (unit_vel / 1.0e5);

            /* Escape velocity using BG21 (Brown & Gnedin 2021) half-mass radius
             * with age="all" (full LEGUS sample):
             * Reff = 2.55 * (M_sc / 1e4)^0.242  [pc]
             * rh = (4/3) * Reff  (projected -> 3D half-mass radius)
             * v_esc = 33.4 * sqrt(M_sc / 1e5) * rh^(-0.5)  [km/s] */
            double M_sc_solar = BHP(n).StarClusterMass
                              * BH_GET_PRIV(tw)->units.UnitMass_in_g / SOLAR_MASS;
            double Reff = 2.55 * pow(M_sc_solar / 1.0e4, 0.242);
            double rh = (4.0 / 3.0) * Reff;
            double v_esc = 33.4 * sqrt(M_sc_solar / 1.0e5) * pow(rh, -0.5);

            if(v_kick_kms > v_esc) {
                message(0, "BH %ld: GW kick %.1f km/s exceeds v_esc %.1f km/s "
                        "(M_sc = %.3g Msun), ejecting from star cluster\n",
                        (long) P[n].ID, v_kick_kms, v_esc, M_sc_solar);
                /* Reset all star cluster state: the BH carries no cluster.
                 * A cluster it acquires at a later merger brings its own
                 * full state (the heaviest-cluster rule above). */
                BHP(n).StarClusterMass = 0;
                BHP(n).StarClusterMetallicity = 0;
                memset(BHP(n).StarClusterMetals, 0, sizeof(BHP(n).StarClusterMetals));
                BHP(n).StarClusterTotalMassReturned = 0;
                /* FormationTime=0 follows the existing "no cluster" convention
                 * (SC evolution guard: FormationTime <= 0 → skip). */
                BHP(n).StarClusterLastEnrichmentMyr = -1;
                BHP(n).StarClusterFormationTime = 0;
                /* the radii describe the cluster, which is gone */
                BHP(n).SC_initReff = 0;
                BHP(n).SC_Reff = 0;
                BHP(n).SC_RlxPendingMyr = 0;
            }
        }

        /* Mtrack accumulates the non-SC portion of swallowed mass for
         * mass conservation. P.Mass is derived from Mtrack below. */
        const MyFloat dynaccmass = accmass - BH_GET_PRIV(tw)->BH_accreted_StarClusterMass[PI];
        BHP(n).Mtrack += dynaccmass;
    }
    /* P.Mass = max(Mtrack [+ StarClusterMass], floor).
     * StarClusterMass is included only in mode StarClusterBHDyn=1.
     * Mode 2: the floor is the accretor's OWN frozen seed cluster mass
     * (init_Msc, never modified by mergers) instead of SeedBHDynMass; the
     * victim's full dynamical mass entered Mtrack above (no SC payloads exist
     * in mode 2 from the secFOF paths, so accreted_StarClusterMass = 0).
     * When the floor is 0, the max just returns Mtrack [+ SC]. */
    {
        double target = BHP(n).Mtrack;
        double dynfloor = blackhole_params.SeedBHDynMass;
        if(blackhole_params.StarClusterBHDyn == 1)
            target += BHP(n).StarClusterMass;
        else if(blackhole_params.StarClusterBHDyn == 2 && BHP(n).init_Msc > 0)
            dynfloor = BHP(n).init_Msc;
        if(target < dynfloor)
            target = dynfloor;
        P[n].Mass = target;
    }

    /* Reset KineticFdbkEnerg to 0 after released */
    if(BH_GET_PRIV(tw)->KEflag[PI] == 2){
        BHP(n).KineticFdbkEnergy = 0;
    }
}

/* Do the black hole feedback tree walk. Tree needs to have gas and BH.*/
static void
blackhole_feedback(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHPriv * priv)
{
    if(!(tree->mask & GASMASK) || !(tree->mask & BHMASK))
        endrun(5, "Error: BH tree types GAS: %d BH %d\n", tree->mask & GASMASK, tree->mask & BHMASK);

    TreeWalk tw_feedback[1] = {{0}};
    tw_feedback->ev_label = "BH_FEEDBACK";
    tw_feedback->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_feedback->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHFeedback);
    tw_feedback->ngbiter = (TreeWalkNgbIterFunction) blackhole_feedback_ngbiter;
    tw_feedback->haswork = blackhole_feedback_haswork;
    tw_feedback->fill = (TreeWalkFillQueryFunction) blackhole_feedback_copy;
    tw_feedback->postprocess = (TreeWalkProcessFunction) blackhole_feedback_postprocess;
    tw_feedback->reduce = (TreeWalkReduceResultFunction) blackhole_feedback_reduce;
    tw_feedback->query_type_elsize = sizeof(TreeWalkQueryBHFeedback);
    tw_feedback->result_type_elsize = sizeof(TreeWalkResultBHFeedback);
    tw_feedback->tree = tree;
    tw_feedback->priv = priv;

    /* Ionization counters*/
    priv[0].N_sph_swallowed = ta_malloc("n_sph_swallowed", int64_t, omp_get_max_threads());
    priv[0].N_BH_swallowed = ta_malloc("n_BH_swallowed", int64_t, omp_get_max_threads());
    memset(priv[0].N_sph_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());
    memset(priv[0].N_BH_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());

    treewalk_run(tw_feedback, ActiveBlackHoles, NumActiveBlackHoles);

    int i;
    int64_t Ntot_gas_swallowed, Ntot_BH_swallowed;
    int64_t N_sph_swallowed = 0, N_BH_swallowed = 0;
    for(i = 0; i < omp_get_max_threads(); i++) {
        N_sph_swallowed += priv[0].N_sph_swallowed[i];
        N_BH_swallowed += priv[0].N_BH_swallowed[i];
    }
    ta_free(priv[0].N_BH_swallowed);
    ta_free(priv[0].N_sph_swallowed);

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

    message(0, "Accretion done: %ld gas particles swallowed, %ld BH particles swallowed\n", Ntot_gas_swallowed, Ntot_BH_swallowed);
}

/* Sample from a power law to get the initial BH mass*/
static double
bh_powerlaw_seed_mass(const MyIDType ID, const RandTable * const rnd)
{
    /* compute random number, uniform in [0,1] */
    const double w = get_random_number(ID+23, rnd);
    /* Normalisation for this power law index*/
    double norm = pow(blackhole_params.MaxSeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex)
                - pow(blackhole_params.SeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex);
    /* Sample from the CDF:
     * w  = [M^(1+x) - M_0^(1+x)]/[M_1^(1+x) - M_0^(1+x)]
     * w [M_1^(1+x) - M_0^(1+x)] + M_0^(1+x) = M^(1+x)
     * M = pow((w [M_1^(1+x) - M_0^(1+x)] + M_0^(1+x)), 1/(1+x))*/
    double mass = pow(w * norm + pow(blackhole_params.SeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex),
                      1./(1+blackhole_params.SeedBlackHoleMassIndex));
    return mass;
}

void
blackhole_make_one(int index, const double atime, const RandTable * const rnd, int seeded_by_starcluster, MyFloat StarClusterMass, MyFloat ScalingMass, MyFloat init_Msc, MyFloat init_Msc_sample, MyFloat CappedStarMass, int BHNgbAtSeeding, MyFloat StarClusterMetallicity, const float * StarClusterMetals, MyFloat SeedMassOverride, MyFloat SC_initReff) {
    int child;

    /* Convert the parent particle in-place into a black hole, keeping its ID
     * and full mass.  Both gas (BlackHoleSeedHaloBased / gas-based seeding) and
     * star (star-cluster seeding: BlackholeSeedSCparticle / SeedInSecFOFasStarCluster)
     * are consumed the same way: the parent particle disappears as its original
     * type and its full P.Mass is carried over (so we don't leave low-mass
     * tracers, and the mass is conserved into BHP.Mtrack below). */
    if(P[index].Type != 0 && P[index].Type != 4)
        endrun(7772, "Only Gas or Star turns into blackholes, got type %d.\n", P[index].Type);
    child = slots_convert(index, 5, -1, PartManager, SlotsManager);

    /* The accretion mass should always be the seed black hole mass,
     * irrespective of the gravitational mass of the particle. */
    if(SeedMassOverride > 0) {
        /* MbhMscRelationCWmodel: seed mass fixed by the caller (M_VMS of the
         * Williams et al. 2026 collision model, cwmodel.c). */
        BHP(child).Mass = SeedMassOverride;
    } else {
        if(blackhole_params.MaxSeedBlackHoleMass > 0)
            BHP(child).Mass = bh_powerlaw_seed_mass(P[child].ID, rnd);
        else
            BHP(child).Mass = blackhole_params.SeedBlackHoleMass;

        if(seeded_by_starcluster && blackhole_params.BHseedMassScaleMsc)
            BHP(child).Mass *= ScalingMass;
    }

    BHP(child).Mseed = BHP(child).Mass;
    BHP(child).Mdot = 0;
    BHP(child).FormationTime = atime;
    BHP(child).SwallowID = (MyIDType) -1;
    BHP(child).Density = 0;
    BHP(child).TimeBinDynFric = P[child].TimeBinHydro;

    if(blackhole_params.StarClusterOn && seeded_by_starcluster) {
        BHP(child).StarClusterMass = StarClusterMass;
    } else {
        BHP(child).StarClusterMass = 0;
    }
    if(blackhole_params.StarClusterOn && StarClusterMass > 0) {
        BHP(child).StarClusterFormationTime = atime;
        BHP(child).StarClusterMetallicity = StarClusterMetallicity;
        int k;
        for(k = 0; k < NMETALS; k++)
            BHP(child).StarClusterMetals[k] = StarClusterMetals ? StarClusterMetals[k] : 0;
    } else {
        BHP(child).StarClusterFormationTime = -1;
        BHP(child).StarClusterMetallicity = 0;
        int k;
        for(k = 0; k < NMETALS; k++)
            BHP(child).StarClusterMetals[k] = 0;
    }
    BHP(child).StarClusterLastEnrichmentMyr = 0;
    BHP(child).StarClusterTotalMassReturned = 0;
    /* Initial effective radius of the cluster (physical pc): the drawn one, or, for a BH that
     * carries a cluster from a path that draws none, the size-mass median at its mass.
     * SC_Reff starts from it and evolves only with StarClusterSizeEvolution=1. */
    BHP(child).SC_initReff = SC_initReff;
    if(BHP(child).SC_initReff <= 0 && BHP(child).StarClusterMass > 0)
        BHP(child).SC_initReff = starcluster_median_reff_pc(BHP(child).StarClusterMass);
    BHP(child).SC_Reff = BHP(child).SC_initReff;
    BHP(child).SC_RlxPendingMyr = 0;

    /* Record the star-cluster mass that seeded this BH. Frozen at creation:
     * never modified by mergers, so an accretor keeps its own seed value. */
    BHP(child).init_Msc = init_Msc;
    BHP(child).init_Msc_sample = init_Msc_sample;
    /* SeedSecFOFcomSample only (0 otherwise): total mass of the host secFOF's
     * unseeded star particles (= Mcut, the SCmasscapSecFOFstarmass cap value).
     * Debug-only output; frozen at creation like init_Msc. */
    BHP(child).CappedStarMass = CappedStarMass;
    /* SeedInSecFOFasStarCluster only (0 otherwise): number of BH particles already
     * present in the host secFOF (or FOF halo) when this BH was seeded, excluding
     * the seed itself.  Debug-only output; frozen at creation like init_Msc. */
    BHP(child).BHNgbAtSeeding = BHNgbAtSeeding;

    /* Initialize MinPotPos to the current position to avoid drifting
     * to unknown locations (0,0,0) immediately after creation. */
    int j;
    for(j = 0; j < 3; j++) {
        BHP(child).MinPotPos[j] = P[child].Pos[j];
        BHP(child).DFAccel[j] = 0;
        BHP(child).DF_SurroundingVel[j] = 0;
        BHP(child).DragAccel[j] = 0;
    }
    BHP(child).DF_SurroundingRmsVel = 0;
    BHP(child).DF_SurroundingDensity = 0;
    memset(BHP(child).TidalTensorPM, 0, sizeof(BHP(child).TidalTensorPM));
    memset(BHP(child).TidalFieldEigenvalues, 0, sizeof(BHP(child).TidalFieldEigenvalues));
    BHP(child).TidalFieldStrength = 0;
    BHP(child).TidalFieldAtime = 0;     /* no field until the next full-tree gravity step */
    BHP(child).JumpToMinPot = 0;
    BHP(child).CountProgs = 1;

    /* Mtrack always tracks mass conservation.  The BH is an in-place conversion
     * of its parent particle, so it starts at the parent mass: the gas mass for
     * gas-based seeding, the parent star mass for star-cluster seeding. */
    BHP(child).Mtrack = P[child].Mass;
    /* Unless the seed mass is at or above the parent mass: then Mtrack would
     * start below BH_Mass and the BH would stochastically swallow neighbouring
     * gas to close the gap.  That deficit is typically a fraction of a gas
     * particle while the smallest bite is a whole one, so the catch-up
     * overshoots badly; start Mtrack at the seed mass instead.  This credits the
     * BH with mass never removed from the simulation, hence the warning. */
    if(BHP(child).Mass >= BHP(child).Mtrack) {
        message(1, "WARNING: BH seed mass (%g) for ID %ld is >= its parent particle mass (%g); "
                   "raising Mtrack to the seed mass (this mass is not taken from the gas).\n",
                BHP(child).Mass, (long) P[child].ID, BHP(child).Mtrack);
        BHP(child).Mtrack = BHP(child).Mass;
    }
    /* P.Mass: mode 1 = max(Mtrack + StarClusterMass, SeedBHDynMass);
     * mode 2 = max(Mtrack, init_Msc) for star-cluster seeds (the seed cluster
     * mass is a frozen per-BH floor replacing SeedBHDynMass; non-SC seeds have
     * init_Msc = 0 and keep the SeedBHDynMass floor);
     * mode 0 = max(Mtrack, SeedBHDynMass). */
    {
        double target = BHP(child).Mtrack;
        double dynfloor = blackhole_params.SeedBHDynMass;
        if(blackhole_params.StarClusterBHDyn == 1)
            target += BHP(child).StarClusterMass;
        else if(blackhole_params.StarClusterBHDyn == 2 && init_Msc > 0)
            dynfloor = init_Msc;
        if(target < dynfloor)
            target = dynfloor;
        P[child].Mass = target;
    }

    double sc_in_dyn = (blackhole_params.StarClusterBHDyn == 1) ? BHP(child).StarClusterMass : 0;
    if(P[child].Mass - sc_in_dyn < BHP(child).Mass ||
       P[child].Mass - sc_in_dyn < BHP(child).Mtrack)
        message(1, "WARNING: BH Mass (%g) for ID %ld is larger than particle mass (%g) or mtrack (%g)\n",
                BHP(child).Mass, P[child].ID, P[child].Mass - sc_in_dyn, BHP(child).Mtrack);

    BHP(child).KineticFdbkEnergy = 0;
    BHP(child).VDisp = 0;
}

/* Seed black holes from individual star particles whose star cluster mass
 * exceeds MinMscForBHseed.  Called every PM step when BlackholeSeedSCparticle
 * is enabled, or every timestep (after star formation) when BHseedEveryTimestep=1.
 * If NewStars/NumNewStar are provided (non-NULL, > 0), only those particle
 * indices are checked (newly formed stars from this timestep).  Otherwise
 * falls back to a full scan of all type-4 particles (PM-step path). */
void
blackhole_seed_sc_particle(ActiveParticles * act, ForceTree * tree, double atime,
                           const RandTable * const rnd, MPI_Comm Comm,
                           int * NewStars, int64_t NumNewStar)
{
    if(!blackhole_params.BlackholeSeedSCparticle)
        return;

    int64_t i;
    double MinMsc = blackhole_params.MinMscForBHseed;
    const int use_newstars = (NewStars != NULL && NumNewStar > 0);

    /* First pass: count how many stars qualify for seeding on this rank. */
    int Nseed = 0;
    if(use_newstars) {
        /* Fast path: only check newly formed stars. */
        for(i = 0; i < NumNewStar; i++) {
            int pi = NewStars[i];
            if(P[pi].Type != 4 || STARP(pi).Seeded)
                continue;
            MyFloat sc_mass = blackhole_params.StarClusterSampling ?
                STARP(pi).StarClusterMass_sample : STARP(pi).ClusterMass;
            if(sc_mass >= MinMsc)
                Nseed++;
        }
    } else {
        /* Fallback: full scan over all particles. */
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].Type != 4 || STARP(i).Seeded)
                continue;
            MyFloat sc_mass = blackhole_params.StarClusterSampling ?
                STARP(i).StarClusterMass_sample : STARP(i).ClusterMass;
            if(sc_mass >= MinMsc)
                Nseed++;
        }
    }

    int Nseed_total;
    MPI_Allreduce(&Nseed, &Nseed_total, 1, MPI_INT, MPI_SUM, Comm);
    message(0, "BlackholeSeedSCparticle: seeding %d new black holes from star particles.\n", Nseed_total);

    if(Nseed_total == 0)
        return;

    /* Ensure enough BH slots. */
    if(Nseed + SlotsManager->info[5].size > SlotsManager->info[5].maxsize) {
        /* The live force tree (gasTree from run.c) and act->ActiveParticle both sit
         * on the MAIN bottom stack ABOVE SlotsBase.  slots_reserve grows SlotsBase via
         * myrealloc, which requires SlotsBase to be the top of the bottom stack, so
         * relocate them to the top stack first and restore afterwards (strict LIFO).
         * Mirrors the pattern in sfr_eff.c.  The tree stays valid because seeding only
         * converts particles in place (no position/mass change, just type). */
        struct NODE * nodes_base_tmp = NULL;
        int * Father_tmp = NULL;
        int * ActiveParticle_tmp = NULL;
        if(tree && force_tree_allocated(tree)) {
            nodes_base_tmp = (struct NODE *) mymalloc2("nodesbasetmp", tree->numnodes * sizeof(struct NODE));
            memmove(nodes_base_tmp, tree->Nodes_base, tree->numnodes * sizeof(struct NODE));
            myfree(tree->Nodes_base);
            Father_tmp = (int *) mymalloc2("Father_tmp", PartManager->MaxPart * sizeof(int));
            memmove(Father_tmp, tree->Father, PartManager->MaxPart * sizeof(int));
            myfree(tree->Father);
        }
        if(act->ActiveParticle) {
            ActiveParticle_tmp = (int *) mymalloc2("ActiveParticle_tmp",
                                    act->NumActiveParticle * sizeof(int));
            memmove(ActiveParticle_tmp, act->ActiveParticle,
                    act->NumActiveParticle * sizeof(int));
            myfree(act->ActiveParticle);
        }
        int64_t atleast[6];
        int64_t k;
        for(k = 0; k < 6; k++)
            atleast[k] = SlotsManager->info[k].maxsize;
        atleast[5] += Nseed_total * 1.1;
        slots_reserve(1, atleast, SlotsManager);
        /* Restore in reverse allocation order. */
        if(ActiveParticle_tmp) {
            act->ActiveParticle = (int *) mymalloc("ActiveParticle",
                sizeof(int) * (act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
            memmove(act->ActiveParticle, ActiveParticle_tmp,
                    act->NumActiveParticle * sizeof(int));
            myfree(ActiveParticle_tmp);
        }
        if(tree && force_tree_allocated(tree)) {
            tree->Father = (int *) mymalloc("Father", PartManager->MaxPart * sizeof(int));
            memmove(tree->Father, Father_tmp, PartManager->MaxPart * sizeof(int));
            myfree(Father_tmp);
            tree->Nodes_base = (struct NODE *) mymalloc("Nodes_base", tree->numnodes * sizeof(struct NODE));
            memmove(tree->Nodes_base, nodes_base_tmp, tree->numnodes * sizeof(struct NODE));
            myfree(nodes_base_tmp);
            /* Don't forget to update the Nodes pointer as well as Nodes_base! */
            tree->Nodes = tree->Nodes_base - tree->firstnode;
        }
    }

    /* Second pass: seed BHs from qualifying stars.  Each qualifying star is
     * converted in-place into a BH (consumed); NumPart does not grow, so no
     * extra base-particle capacity is needed.  Iterate over the original range. */
    int64_t NumPart_before = PartManager->NumPart;
    int n_seeded = 0;

    /* Determine iteration range: NewStars list or full particle array. */
    const int64_t niter = use_newstars ? NumNewStar : NumPart_before;
    for(i = 0; i < niter; i++) {
        int pi = use_newstars ? NewStars[i] : (int) i;
        if(P[pi].Type != 4 || STARP(pi).Seeded)
            continue;
        MyFloat sc_mass = blackhole_params.StarClusterSampling ?
            STARP(pi).StarClusterMass_sample : STARP(pi).ClusterMass;
        if(sc_mass < MinMsc)
            continue;

        /* Compute mass-weighted metallicity from the star particle's own metals. */
        MyFloat sc_metallicity = STARP(pi).Metallicity;
        float sc_metals[NMETALS];
        int j;
        for(j = 0; j < NMETALS; j++)
            sc_metals[j] = STARP(pi).Metals[j];

        blackhole_make_one(pi, atime, rnd, 1, sc_mass, sc_mass,
                           STARP(pi).ClusterMass, STARP(pi).StarClusterMass_sample,
                           0, 0, sc_metallicity, sc_metals, 0, 0);

        /* The parent star has been converted in-place into the BH (consumed):
         * it is now type 5, so it no longer participates in any seeding scan or
         * FOF star-cluster sum and needs no Seeded flag.  Do NOT touch STARP(pi)
         * here — that slot is now the BH slot. */
        n_seeded++;
    }

    int n_seeded_total;
    MPI_Allreduce(&n_seeded, &n_seeded_total, 1, MPI_INT, MPI_SUM, Comm);
    message(0, "BlackholeSeedSCparticle: created %d black holes from star particles.\n", n_seeded_total);
}
