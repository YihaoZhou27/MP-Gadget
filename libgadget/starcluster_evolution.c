/*! \file starcluster_evolution.c
 *  \brief Stellar evolution (mass and metal return) for star clusters attached to BH particles.
 *
 *  Star clusters are SSPs (simple stellar populations) that live on BH particles.
 *  This module computes their AGB + SNII + Sn1a mass/metal return using the same
 *  yield tables and Chabrier IMF as metal_return.c, and deposits the returned
 *  mass and metals into neighboring gas particles via an SPH-kernel-weighted treewalk.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_integration.h>

#include "physconst.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "treewalk.h"
#include "metal_return.h"
#include "starcluster_evolution.h"
#include "blackhole.h"
#include "densitykernel.h"
#include "density.h"
#include "cosmology.h"
#include "utils/spinlocks.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "metal_tables.h"

struct SCMetalPriv {
    gsl_integration_workspace ** gsl_work;
    MyFloat * SC_StellarAges;    /* Age of star cluster in Myr */
    MyFloat * SC_MassReturn;     /* Mass to return this timestep */
    MyFloat * SC_LowDyingMass;
    MyFloat * SC_HighDyingMass;
    MyFloat * SC_VolumeSPH;      /* SPH volume weight sum */
    double imf_norm;
    double hub;
    double MaxGasMass;
    Cosmology *CP;
    struct interps interp;
    struct SpinLocks * spin;
};

#define SC_GET_PRIV(tw) ((struct SCMetalPriv*) ((tw)->priv))

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Metallicity;
    MyFloat Mass;
    MyFloat Hsml;
    MyFloat VolumeSPH;
    MyFloat MetalSpeciesGenerated[NMETALS];
    MyFloat MassGenerated;
    MyFloat MetalGenerated;
} TreeWalkQuerySCMetals;

typedef struct {
    TreeWalkResultBase base;
    MyFloat MassReturn;
} TreeWalkResultSCMetals;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
} TreeWalkNgbIterSCMetals;

/* Forward declarations */
static int sc_metal_haswork(int n, TreeWalk * tw);
static void sc_metal_ngbiter(TreeWalkQuerySCMetals * I, TreeWalkResultSCMetals * O, TreeWalkNgbIterSCMetals * iter, LocalTreeWalk * lv);
static void sc_metal_copy(int place, TreeWalkQuerySCMetals * input, TreeWalk * tw);
static void sc_metal_postprocess(int place, TreeWalk * tw);
static void sc_metal_reduce(int place, TreeWalkResultSCMetals * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

/* ---- Volume SPH computation for BH particles ---- */
/* We use the BH's existing Hsml (from the BH accretion density computation)
 * rather than running the multi-probe stellar density convergence loop.
 * Note: the BH's Hsml was converged for all neighbor types, but this walk
 * only counts gas neighbors (GASMASK). The effective neighbor count within
 * Hsml may differ, but the normalization is self-consistent because both
 * the volume sum here and the deposition kernel in sc_metal_ngbiter use
 * the same Hsml and gas-only mask. */

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
} TreeWalkQuerySCDensity;

typedef struct {
    TreeWalkResultBase base;
    MyFloat VolumeSPH;
} TreeWalkResultSCDensity;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
} TreeWalkNgbIterSCDensity;

static int sc_density_haswork(int i, TreeWalk * tw)
{
    if(P[i].Type != 5)
        return 0;
    int pi = P[i].PI;
    if(SC_GET_PRIV(tw)->SC_MassReturn[pi] <= 0)
        return 0;
    return 1;
}

static void
sc_density_copy(int place, TreeWalkQuerySCDensity * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;
}

static void
sc_density_reduce(int place, TreeWalkResultSCDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(SC_GET_PRIV(tw)->SC_VolumeSPH[P[place].PI], remote->VolumeSPH);
}

static void
sc_density_ngbiter(
    TreeWalkQuerySCDensity * I,
    TreeWalkResultSCDensity * O,
    TreeWalkNgbIterSCDensity * iter,
    LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.mask = GASMASK;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        O->VolumeSPH = 0;
        density_kernel_init(&iter->kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    const int other = iter->base.other;
    const double r2 = iter->base.r2;
    const double r = iter->base.r;

    if(r2 > 0 && r2 < iter->kernel.HH) {
        const double u = r * iter->kernel.Hinv;
        double wk = density_kernel_wk(&iter->kernel, u);
        double volume = P[other].Mass / SPHP(other).Density;
        O->VolumeSPH += wk * volume;
    }
}

static void
sc_compute_volume_sph(const ActiveParticles * act, struct SCMetalPriv * priv, const ForceTree * gasTree)
{
    TreeWalk tw[1] = {{0}};

    tw->ev_label = "SC_DENSITY";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) sc_density_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSCDensity);
    tw->haswork = sc_density_haswork;
    tw->fill = (TreeWalkFillQueryFunction) sc_density_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sc_density_reduce;
    tw->postprocess = NULL;
    tw->query_type_elsize = sizeof(TreeWalkQuerySCDensity);
    tw->result_type_elsize = sizeof(TreeWalkResultSCDensity);
    tw->tree = gasTree;
    tw->priv = priv;

    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
}

/* ---- Init: compute mass return for each BH with a star cluster ---- */
static int64_t
sc_metal_return_init(const ActiveParticles * act, Cosmology * CP, struct SCMetalPriv * priv, const double atime)
{
    int nthread = omp_get_max_threads();
    priv->gsl_work = ta_malloc("sc_gsl_work", gsl_integration_workspace *, nthread);
    int i;
    for(i = 0; i < nthread; i++)
        priv->gsl_work[i] = gsl_integration_workspace_alloc(GSL_WORKSPACE);
    priv->hub = CP->HubbleParam;
    priv->CP = CP;

    setup_metal_table_interp(&priv->interp);
    priv->SC_StellarAges = (MyFloat *) mymalloc("SC_StellarAges", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->SC_MassReturn = (MyFloat *) mymalloc("SC_MassReturn", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->SC_LowDyingMass = (MyFloat *) mymalloc("SC_LowDyingMass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->SC_HighDyingMass = (MyFloat *) mymalloc("SC_HighDyingMass", SlotsManager->info[5].size * sizeof(MyFloat));
    priv->SC_VolumeSPH = (MyFloat *) mymalloc("SC_VolumeSPH", SlotsManager->info[5].size * sizeof(MyFloat));

    priv->imf_norm = compute_imf_norm(priv->gsl_work[0]);
    double maxmassfrac = mass_yield(0, 1/(CP->HubbleParam*HUBBLE * SEC_PER_MEGAYEAR), snii_metallicities[SNII_NMET-1], CP->HubbleParam, &priv->interp, priv->imf_norm, priv->gsl_work[0], agb_masses[0], MAXMASS);

    int64_t haswork = 0;
    #pragma omp parallel for reduction(+: haswork)
    for(i = 0; i < act->NumActiveParticle; i++)
    {
        int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p_i].Type != 5)
            continue;
        const int slot = P[p_i].PI;
        /* Skip BH particles without star clusters */
        if(BHP(p_i).StarClusterMass <= 0 || BHP(p_i).StarClusterFormationTime <= 0) {
            priv->SC_MassReturn[slot] = 0;
            continue;
        }

        int tid = omp_get_thread_num();
        priv->SC_StellarAges[slot] = atime_to_myr(CP, BHP(p_i).StarClusterFormationTime, atime, priv->gsl_work[tid]);

        double initialmass = BHP(p_i).StarClusterMass + BHP(p_i).StarClusterTotalMassReturned;
        find_mass_bin_limits(&priv->SC_LowDyingMass[slot], &priv->SC_HighDyingMass[slot],
            BHP(p_i).StarClusterLastEnrichmentMyr, priv->SC_StellarAges[slot],
            BHP(p_i).StarClusterMetallicity, priv->interp.lifetime_interp);

        priv->SC_MassReturn[slot] = initialmass * mass_yield(
            BHP(p_i).StarClusterLastEnrichmentMyr, priv->SC_StellarAges[slot],
            BHP(p_i).StarClusterMetallicity, CP->HubbleParam,
            &priv->interp, priv->imf_norm, priv->gsl_work[tid],
            priv->SC_LowDyingMass[slot], priv->SC_HighDyingMass[slot]);

        /* Guard against returning more mass than the star cluster has */
        if(BHP(p_i).StarClusterTotalMassReturned + priv->SC_MassReturn[slot] > initialmass * maxmassfrac) {
            priv->SC_MassReturn[slot] = initialmass * maxmassfrac - BHP(p_i).StarClusterTotalMassReturned;
            if(priv->SC_MassReturn[slot] < 0)
                priv->SC_MassReturn[slot] = 0;
        }

        /* Only count significant enrichment */
        if(priv->SC_MassReturn[slot] >= 1e-3 * initialmass)
            haswork++;
        else {
            /* Update last enrichment time even if skipping */
            if(priv->SC_MassReturn[slot] > 0)
                BHP(p_i).StarClusterLastEnrichmentMyr = priv->SC_StellarAges[slot];
            priv->SC_MassReturn[slot] = 0;
        }
    }
    return haswork;
}

static void
sc_metal_return_priv_free(struct SCMetalPriv * priv)
{
    int nthread = omp_get_max_threads();
    myfree(priv->SC_VolumeSPH);
    myfree(priv->SC_HighDyingMass);
    myfree(priv->SC_LowDyingMass);
    myfree(priv->SC_MassReturn);
    myfree(priv->SC_StellarAges);
    int i;
    for(i = nthread - 1; i >= 0; i--)
        gsl_integration_workspace_free(priv->gsl_work[i]);
    ta_free(priv->gsl_work);
}

/* ---- Main entry point ---- */
void
starcluster_metal_return(const ActiveParticles * act, ForceTree * gasTree, Cosmology * CP, const double atime, const double AvgGasMass)
{
    /* Do nothing if no BHs yet */
    int64_t totbh;
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totbh == 0)
        return;

    struct SCMetalPriv priv[1];
    int64_t nwork = sc_metal_return_init(act, CP, priv, atime);

    priv->MaxGasMass = 4 * AvgGasMass;

    int64_t totwork;
    MPI_Allreduce(&nwork, &totwork, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    walltime_measure("/BH/SCMetals/Init");

    if(totwork == 0) {
        sc_metal_return_priv_free(priv);
        return;
    }

    if(!gasTree->tree_allocated_flag || !(gasTree->mask & GASMASK))
        endrun(5, "starcluster_metal_return called with bad tree allocated %d mask %d\n",
               gasTree->tree_allocated_flag, gasTree->mask);

    /* Compute SPH volume weights for BH particles with star clusters */
    sc_compute_volume_sph(act, priv, gasTree);

    walltime_measure("/BH/SCMetals/Density");

    /* Do the metal return treewalk */
    TreeWalk tw[1] = {{0}};

    tw->ev_label = "SC_METALS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) sc_metal_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSCMetals);
    tw->haswork = sc_metal_haswork;
    tw->fill = (TreeWalkFillQueryFunction) sc_metal_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sc_metal_reduce;
    tw->postprocess = (TreeWalkProcessFunction) sc_metal_postprocess;
    tw->query_type_elsize = sizeof(TreeWalkQuerySCMetals);
    tw->result_type_elsize = sizeof(TreeWalkResultSCMetals);
    tw->tree = gasTree;
    tw->priv = priv;

    priv->spin = init_spinlocks(SlotsManager->info[0].size);
    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
    free_spinlocks(priv->spin);

    sc_metal_return_priv_free(priv);

    walltime_measure("/BH/SCMetals/Yield");
}

/* ---- Treewalk callbacks ---- */
static int
sc_metal_haswork(int n, TreeWalk * tw)
{
    if(P[n].Type != 5)
        return 0;
    int pi = P[n].PI;
    if(SC_GET_PRIV(tw)->SC_MassReturn[pi] <= 0)
        return 0;
    return 1;
}

static void
sc_metal_copy(int place, TreeWalkQuerySCMetals * input, TreeWalk * tw)
{
    input->Metallicity = BHP(place).StarClusterMetallicity;
    input->Mass = BHP(place).StarClusterMass;
    input->Hsml = P[place].Hsml;
    int pi = P[place].PI;
    input->VolumeSPH = SC_GET_PRIV(tw)->SC_VolumeSPH[pi];

    double InitialMass = BHP(place).StarClusterMass + BHP(place).StarClusterTotalMassReturned;
    double dtmyrend = SC_GET_PRIV(tw)->SC_StellarAges[pi];
    double dtmyrstart = BHP(place).StarClusterLastEnrichmentMyr;
    int tid = omp_get_thread_num();

    input->MassGenerated = SC_GET_PRIV(tw)->SC_MassReturn[pi];

    double total_z_yield = metal_yield(dtmyrstart, dtmyrend, input->Metallicity,
        SC_GET_PRIV(tw)->hub, &SC_GET_PRIV(tw)->interp, input->MetalSpeciesGenerated,
        SC_GET_PRIV(tw)->imf_norm, SC_GET_PRIV(tw)->gsl_work[tid],
        SC_GET_PRIV(tw)->SC_LowDyingMass[pi], SC_GET_PRIV(tw)->SC_HighDyingMass[pi]);

    input->MetalGenerated = InitialMass * total_z_yield;
    if(input->MetalGenerated < 0)
        input->MetalGenerated = 0;

    int i;
    for(i = 0; i < NMETALS; i++) {
        input->MetalSpeciesGenerated[i] *= InitialMass;
        if(input->MetalSpeciesGenerated[i] < 0)
            input->MetalSpeciesGenerated[i] = 0;
    }
}

static void
sc_metal_reduce(int place, TreeWalkResultSCMetals * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(SC_GET_PRIV(tw)->SC_MassReturn[P[place].PI], remote->MassReturn);
}

static void
sc_metal_postprocess(int place, TreeWalk * tw)
{
    int pi = P[place].PI;
    MyFloat returned = SC_GET_PRIV(tw)->SC_MassReturn[pi];

    /* Decrease star cluster mass. Recompute P.Mass only when
     * StarClusterBHDyn=1 (SC mass is part of the dynamical mass). */
    BHP(place).StarClusterMass -= returned;
    if(get_starcluster_bhdyn_on()) {
        double target = BHP(place).Mtrack + BHP(place).StarClusterMass;
        double SeedBHDynMass = get_bh_seed_dyn_mass();
        if(target < SeedBHDynMass)
            target = SeedBHDynMass;
        P[place].Mass = target;
    }
    BHP(place).StarClusterTotalMassReturned += returned;

    /* Update last enrichment time */
    BHP(place).StarClusterLastEnrichmentMyr = SC_GET_PRIV(tw)->SC_StellarAges[pi];
}

static void
sc_metal_ngbiter(
    TreeWalkQuerySCMetals * I,
    TreeWalkResultSCMetals * O,
    TreeWalkNgbIterSCMetals * iter,
    LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.mask = GASMASK;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        O->MassReturn = 0;
        density_kernel_init(&iter->kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    const int other = iter->base.other;
    const double r2 = iter->base.r2;
    const double r = iter->base.r;

    if(r2 > 0 && r2 < iter->kernel.HH)
    {
        const double u = r * iter->kernel.Hinv;
        double wk = density_kernel_wk(&iter->kernel, u);

        if(I->VolumeSPH == 0)
            endrun(3, "SC VolumeSPH %g hsml %g\n", I->VolumeSPH, I->Hsml);

        int pi = P[other].PI;
        lock_spinlock(pi, SC_GET_PRIV(lv->tw)->spin);

        double volume = P[other].Mass / SPHP(other).Density;
        double returnfraction = wk * volume / I->VolumeSPH;
        double thismass = returnfraction * I->MassGenerated;

        /* Cap gas particle mass */
        if(P[other].Mass + thismass > SC_GET_PRIV(lv->tw)->MaxGasMass) {
            unlock_spinlock(pi, SC_GET_PRIV(lv->tw)->spin);
            return;
        }

        /* Add metals weighted by SPH kernel */
        double thismetal = returnfraction * I->MetalGenerated;
        int i;
        for(i = 0; i < NMETALS; i++) {
            double thisspecies = returnfraction * I->MetalSpeciesGenerated[i];
            SPHP(other).Metals[i] = (SPHP(other).Metals[i] * P[other].Mass + thisspecies) / (P[other].Mass + thismass);
        }
        SPHP(other).Metallicity = (SPHP(other).Metallicity * P[other].Mass + thismetal) / (P[other].Mass + thismass);

        /* Update mass */
        double massfrac = (P[other].Mass + thismass) / P[other].Mass;
        P[other].Mass *= massfrac;
        SPHP(other).Density *= massfrac;

        O->MassReturn += thismass;
        double newmass = P[other].Mass;
        unlock_spinlock(pi, SC_GET_PRIV(lv->tw)->spin);

        if(newmass <= 0)
            endrun(3, "SC metal return: new mass %g in particle %d id %ld from BH mass %g\n",
                   newmass, other, P[other].ID, I->Mass);
    }
}
