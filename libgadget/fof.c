#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <inttypes.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <omp.h>

#include "utils/endrun.h"
#include "utils/mpsort.h"

#include "walltime.h"
#include "blackhole.h"
#include "domain.h"
#include "winds.h"

#include "forcetree.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "densitykernel.h"
#include "utils/mymalloc.h"
#include "utils/openmpsort.h"
#include "utils/spinlocks.h"
#include "utils/string.h"
#include "physconst.h"
#include "sfr_eff.h"
#include "gravity.h"
#include "cosmology.h"
/*! \file fof.c
 *  \brief parallel FoF group finder
 */

#include "fof.h"
#include "secondfof.h"   /* get_seed_in_secfof(): gate BHNgbAtSeeding to the secFOF seed path */
#include "scinfo.h"      /* scinfo_record_seed(): per-seeded-star-cluster detail records */
#include "cwmodel.h"     /* cw_final_vms_mass_msun(): Williams et al. 2026 VMS seed-mass model */

#define LARGE 1e29
#define MAXITER 400

/* CWmodelMetallicity modes: how the per-cluster metallicity fed to the CW
 * seed-mass model is chosen (see cw_sample_cluster_met). */
#define CW_MET_AVE       0   /* host group's unseeded-star metal mass ratio (default) */
#define CW_MET_LOGNORMAL 1   /* per-cluster Gaussian draw in log10(Z) (equal-weight mean/std) */
#define CW_MET_UNIFORM   2   /* per-cluster uniform draw in log10(Z) on [min,max] */

struct FOFParams
{
    int FOFSaveParticles ; /* saving particles in the fof group */
    double MinFoFMassForNewSeed;	/* Halo mass required before new seed is put in */
    double MinMStarForNewSeed; /* Minimum stellar mass required before new seed */
    double FOFHaloLinkingLength;
    double FOFHaloComovingLinkingLength; /* in code units */
    double BlackHoleSeedsfmpGas;
    double BlackHoleseedsMetalThres;


    int FOFHaloMinLength;
    /* Minimum number of primary-linking-type particles per group.
     * Groups below this are eliminated in fof_compile_base. 0 disables it.
     * Primary FOF reads it from "FOFMinPrimaryLength"; the second FOF overrides
     * it with "SecondFOFMinPrimaryLength" via fof_set_params. */
    int FOFMinPrimaryLength;
    int FOFPrimaryLinkTypes;
    int FOFSecondaryLinkTypes;
    int ExcursionSetReionOn;
    int BlackHoleSeedGasBased;

    int BlackHoleSeedHaloBased;
    int BlackHoleSeedStarCluster;
    int StarClusterOn;
    int StarClusterSampling;
    int SeedSecFOFcomSample; /* combined per-secFOF star-cluster sampling for BH seeding */
    int SeedSecFOFcomSampleParticle; /* per-star-particle sampling variant of SeedSecFOFcomSample */
    int SeedInSecFOFMultipleSeeds; /* if 1, seed multiple BHs in a secFOF group with M_SC > 1e8 Msun */
    /* SeedSecFOFcomSample seed aggregation: 1 = the combined per-secFOF draw is
     * summed into ONE BH seed per group; 0 = per-cluster seeding (every sampled
     * cluster >= MinMscForBHseed seeds its own BH on a distinct unseeded star). */
    int SecFOFseedsumover;
    /* Host-star choice of the per-cluster mode (SecFOFseedsumover=0): 1 = random
     * unseeded stars; 0 = the unseeded stars with the largest f(Z)-scaled
     * cluster-forming mass f(Z)*ClusterMass. Ignored when SecFOFseedsumover=1. */
    int SeedInSecFOFRandomStarParticle;
    /* if 1, the per-cluster (SecFOFseedsumover=0) seed mass is M_VMS from the
     * Williams et al. 2026 stellar-collision model (cwmodel.c).
     * BHseedMassScaleMsc is ignored; SeedBlackHoleMass is the lower seed-mass
     * limit: clusters with M_VMS < SeedBlackHoleMass seed no BH. */
    int MbhMscRelationCWmodel;
    /* density power-law index alpha (rho ~ r^-alpha) for the CW seed-mass model;
     * only used when MbhMscRelationCWmodel=1 (default 1.2) */
    double CWmodelAlpha;
    /* CWmodelMetallicity: per-cluster metallicity mode of the CW seed-mass model
     * (CW_MET_* below); -1 = unrecognised string (an error when
     * MbhMscRelationCWmodel=1). Only used when MbhMscRelationCWmodel=1. */
    int CWmodelMetallicity;
    /* if 1, a secFOF group with unseeded Sum(m*Gamma) > 1e8 Msun is restricted to
     * the gravitationally bound unseeded stars before SeedSecFOFcomSample seeding */
    int SeedSeedFOFMassiveBoundStar;
    /* BHseedSecFOFbound: 0 = off, 1 = keep only the stars bound to (member stars +
     * DM inside Rmax), 2 = the same with the DM sphere capped at 2*R50.  Applies to
     * EVERY secFOF group and to every secFOF seeding variant. */
    int BHseedSecFOFbound;
    int BHseedMassScaleMsc;
    double MinMscForBHseed;
    /* Lowest sampled cluster mass written to the StarClusterDetails files. Resolved at
     * startup: the -1 default becomes MinMscForBHseed (only seed-capable clusters are
     * recorded, the historical behavior); anything smaller additionally records the
     * clusters that cannot seed. Per-cluster mode (SecFOFseedsumover=0) only. */
    double MinMscForSCdetail;
    /* Lowest BH seed mass a star cluster can physically produce (code units), as read
     * from the parameter file; 0 disables the compensation.  The value actually used is
     * max(200 Msun, this), see min_bh_seed_in_sc_code() -- the clamp cannot be applied
     * here because the code<->solar conversion is only set up once the units are known,
     * well after set_fof_params runs.  Only used when MbhMscRelationCWmodel=1. */
    double MinBHSeedInSC;
    int FOFPotentialMin;
    /* If 1, seeded star particles (Type==4 && STARP.Seeded) are excluded from
     * the primary-linking set. Set transiently by secondfof_run when
     * SecFOFUnseededPart=1. Not read from the parameter file directly. */
    int FOFPrimaryUnseededStarsOnly;
} fof_params;

/*Set the parameters of the BH module*/
void set_fof_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        fof_params.FOFSaveParticles = param_get_int(ps, "FOFSaveParticles");
        fof_params.FOFHaloLinkingLength = param_get_double(ps, "FOFHaloLinkingLength");
        fof_params.FOFHaloMinLength = param_get_int(ps, "FOFHaloMinLength");
        fof_params.FOFMinPrimaryLength = param_get_int(ps, "FOFMinPrimaryLength");
        fof_params.MinFoFMassForNewSeed = param_get_double(ps, "MinFoFMassForNewSeed");
        fof_params.MinMStarForNewSeed = param_get_double(ps, "MinMStarForNewSeed");
        fof_params.BlackHoleSeedsfmpGas = param_get_double(ps, "BlackHoleSeedsfmpGas");
        fof_params.BlackHoleseedsMetalThres = param_get_double(ps, "BlackHoleseedsMetalThres");
        fof_params.FOFPrimaryLinkTypes = param_get_int(ps, "FOFPrimaryLinkTypes");
        fof_params.FOFSecondaryLinkTypes = param_get_int(ps, "FOFSecondaryLinkTypes");
        fof_params.ExcursionSetReionOn = param_get_int(ps, "ExcursionSetReionOn");
        fof_params.BlackHoleSeedGasBased = param_get_int(ps, "BlackHoleSeedGasBased");
        fof_params.BlackHoleSeedHaloBased = param_get_int(ps, "BlackHoleSeedHaloBased");
        fof_params.BlackHoleSeedStarCluster = param_get_int(ps, "BlackHoleSeedStarCluster");
        fof_params.StarClusterOn = param_get_int(ps, "StarClusterOn");
        fof_params.StarClusterSampling = param_get_int(ps, "StarClusterSampling");
        fof_params.SeedSecFOFcomSample = param_get_int(ps, "SeedSecFOFcomSample");
        fof_params.SeedSecFOFcomSampleParticle = param_get_int(ps, "SeedSecFOFcomSampleParticle");
        fof_params.SeedInSecFOFMultipleSeeds = param_get_int(ps, "SeedInSecFOFMultipleSeeds");
        fof_params.SecFOFseedsumover = param_get_int(ps, "SecFOFseedsumover");
        fof_params.SeedInSecFOFRandomStarParticle = param_get_int(ps, "SeedInSecFOFRandomStarParticle");
        fof_params.MbhMscRelationCWmodel = param_get_int(ps, "MbhMscRelationCWmodel");
        fof_params.CWmodelAlpha = param_get_double(ps, "CWmodelAlpha");
        /* CWmodelMetallicity: string -> mode; -1 keeps the unrecognised value an
         * error below when the CW model is actually enabled. */
        const char * cwmet = param_get_string(ps, "CWmodelMetallicity");
        if(strcmp(cwmet, "ave") == 0)
            fof_params.CWmodelMetallicity = CW_MET_AVE;
        else if(strcmp(cwmet, "lognormal") == 0)
            fof_params.CWmodelMetallicity = CW_MET_LOGNORMAL;
        else if(strcmp(cwmet, "uniform") == 0)
            fof_params.CWmodelMetallicity = CW_MET_UNIFORM;
        else
            fof_params.CWmodelMetallicity = -1;
        fof_params.SeedSeedFOFMassiveBoundStar = param_get_int(ps, "SeedSeedFOFMassiveBoundStar");
        fof_params.BHseedSecFOFbound = param_get_int(ps, "BHseedSecFOFbound");
        if(fof_params.BHseedSecFOFbound < 0 || fof_params.BHseedSecFOFbound > 2)
            endrun(1, "BHseedSecFOFbound must be 0, 1 or 2 (got %d).\n", fof_params.BHseedSecFOFbound);
        /* Both features restrict the seeding budget by overwriting the same two
         * fields (StarClusterMassUnseeded, SCcomMcut); running them together would
         * make the result depend on their order. */
        if(fof_params.BHseedSecFOFbound && fof_params.SeedSeedFOFMassiveBoundStar)
            endrun(1, "BHseedSecFOFbound=%d is incompatible with SeedSeedFOFMassiveBoundStar=1 "
                      "(both restrict the seeding budget to bound stars).\n", fof_params.BHseedSecFOFbound);
        /* The bound restriction acts at GROUP level: it rewrites the group's summed
         * budget (StarClusterMassUnseeded / StarClusterMassSampleUnseeded / SCcomMcut)
         * and repoints the single seed host to a bound star.  It carries no per-star
         * bound flag, so any mode that afterwards re-scans the group's unseeded stars
         * individually would still see the unbound ones and defeat (or mis-normalise)
         * the restriction.  These are exactly the modes SeedSeedFOFMassiveBoundStar
         * excludes for the same reason:
         *
         *  - SeedSecFOFcomSampleParticle: fof_secfof_particle_sample() draws a cluster
         *    population for EVERY unseeded star, so unbound stars keep contributing to
         *    tot_msc_fof; worse, secondfof_seed then redistributes that total as
         *    totmsc * m_star / SCcomMcut over EVERY unseeded star while SCcomMcut is
         *    now only the BOUND stellar mass, inflating the group sum by
         *    M_unseeded / M_bound instead of conserving it.
         *  - SeedInSecFOFMultipleSeeds: seeds 2..N are placed on the next-largest
         *    unseeded stars without a boundedness test, so only seed 1 is guaranteed
         *    bound.
         *  - SecFOFseedsumover = 0 (per-cluster seeding): each cluster's host is drawn
         *    from all unseeded stars, again with no boundedness test.
         *
         * Supporting these needs a per-star bound flag persisted from
         * fof_secfof_bound_restrict() to those loops; until then, refuse the
         * combination rather than silently produce a restriction that does not hold. */
        if(fof_params.BHseedSecFOFbound) {
            if(fof_params.SeedSecFOFcomSampleParticle)
                endrun(1, "BHseedSecFOFbound=%d is incompatible with SeedSecFOFcomSampleParticle=1: "
                          "the per-star sampler draws for every unseeded star (unbound included) and "
                          "the redistribution would divide by the bound stellar mass while summing "
                          "over all unseeded stars, inflating the group cluster mass.\n",
                          fof_params.BHseedSecFOFbound);
            if(fof_params.SeedInSecFOFMultipleSeeds)
                endrun(1, "BHseedSecFOFbound=%d is incompatible with SeedInSecFOFMultipleSeeds=1: "
                          "only the primary seed is placed on a bound star; the extra hosts are not "
                          "boundedness-tested.\n", fof_params.BHseedSecFOFbound);
            if(!fof_params.SecFOFseedsumover)
                endrun(1, "BHseedSecFOFbound=%d is incompatible with SecFOFseedsumover=0: "
                          "the per-cluster hosts are drawn from all unseeded stars with no "
                          "boundedness test.\n", fof_params.BHseedSecFOFbound);
        }
        fof_params.BHseedMassScaleMsc = param_get_int(ps, "BHseedMassScaleMsc");
        fof_params.MinMscForBHseed = param_get_double(ps, "MinMscForBHseed");

        if(fof_params.BlackHoleSeedStarCluster && fof_params.BHseedMassScaleMsc && fof_params.MinMscForBHseed <= 0)
            endrun(1, "MinMscForBHseed must be > 0 when BlackHoleSeedStarCluster and BHseedMassScaleMsc are enabled.\n");
        /* SecFOFseedsumover=0 is a self-contained per-cluster multi-seed mechanism on
         * the combined per-secFOF draw; it is mutually exclusive with the other secFOF
         * sampling-variant / multi-seed flags. */
        if(!fof_params.SecFOFseedsumover) {
            if(!fof_params.SeedSecFOFcomSample)
                endrun(1, "SecFOFseedsumover=0 requires SeedSecFOFcomSample=1.\n");
            if(fof_params.SeedSecFOFcomSampleParticle)
                endrun(1, "SecFOFseedsumover=0 is incompatible with SeedSecFOFcomSampleParticle=1.\n");
            if(fof_params.SeedInSecFOFMultipleSeeds)
                endrun(1, "SecFOFseedsumover=0 is incompatible with SeedInSecFOFMultipleSeeds=1.\n");
            if(fof_params.SeedSeedFOFMassiveBoundStar)
                endrun(1, "SecFOFseedsumover=0 is incompatible with SeedSeedFOFMassiveBoundStar=1.\n");
        }
        else if(fof_params.SeedInSecFOFRandomStarParticle)
            message(0, "SecFOFseedsumover=1: SeedInSecFOFRandomStarParticle is ignored (it now only "
                       "selects the host stars of the per-cluster mode SecFOFseedsumover=0).\n");
        if(fof_params.MbhMscRelationCWmodel &&
           (fof_params.SecFOFseedsumover || !fof_params.SeedSecFOFcomSample))
            endrun(1, "MbhMscRelationCWmodel=1 requires SeedSecFOFcomSample=1 and SecFOFseedsumover=0.\n");
        if(fof_params.MbhMscRelationCWmodel &&
           (fof_params.CWmodelAlpha <= 0 || fof_params.CWmodelAlpha >= 3))
            endrun(1, "CWmodelAlpha must be in (0, 3); got %g.\n", fof_params.CWmodelAlpha);
        if(fof_params.MbhMscRelationCWmodel && fof_params.CWmodelMetallicity < 0)
            endrun(1, "CWmodelMetallicity must be 'ave', 'lognormal' or 'uniform'; got '%s'.\n", cwmet);
        if(fof_params.MbhMscRelationCWmodel && fof_params.BHseedMassScaleMsc)
            message(0, "MbhMscRelationCWmodel=1: BHseedMassScaleMsc=1 is ignored; the seed mass is "
                       "M_VMS from the CW model, with SeedBlackHoleMass as the lower seed-mass limit.\n");
        /* StarClusterDetails recording floor. The -1 default means "record only what can
         * seed", i.e. MinMscForBHseed. Lowering it below that adds one record per sampled
         * sub-threshold cluster, which the m^-2 ICMF makes far more numerous than the
         * seed-capable ones, so warn with the expected inflation factor. */
        fof_params.MinMscForSCdetail = param_get_double(ps, "MinMscForSCdetail");
        if(fof_params.MinMscForSCdetail < 0)
            fof_params.MinMscForSCdetail = fof_params.MinMscForBHseed;
        if(fof_params.MinMscForSCdetail < fof_params.MinMscForBHseed) {
            if(fof_params.SecFOFseedsumover || !fof_params.SeedSecFOFcomSample)
                endrun(1, "MinMscForSCdetail below MinMscForBHseed requires the per-cluster secFOF "
                          "seeding mode (SeedSecFOFcomSample=1 and SecFOFseedsumover=0).\n");
            if(!param_get_int(ps, "StarClusterDetails"))
                endrun(1, "MinMscForSCdetail below MinMscForBHseed has no effect unless "
                          "StarClusterDetails=1.\n");
            if(fof_params.MinMscForSCdetail > 0)
                message(0, "MinMscForSCdetail=%g (< MinMscForBHseed=%g): StarClusterDetails will also "
                           "record sampled clusters that cannot seed a BH, roughly %.3g times more "
                           "records than seeding clusters alone.\n",
                        fof_params.MinMscForSCdetail, fof_params.MinMscForBHseed,
                        fof_params.MinMscForBHseed / fof_params.MinMscForSCdetail);
            else
                message(0, "MinMscForSCdetail=0 (< MinMscForBHseed=%g): StarClusterDetails will record "
                           "the whole sampled cluster population down to the 100 Msun mass-function "
                           "limit, which is orders of magnitude more records than seeding clusters alone.\n",
                        fof_params.MinMscForBHseed);
        }
        /* MinBHSeedInSC: compensate the BH seeds the run cannot make.  Meaningless
         * without the CW model, which is what supplies the M_VMS being compared. */
        fof_params.MinBHSeedInSC = param_get_double(ps, "MinBHSeedInSC");
        if(fof_params.MinBHSeedInSC > 0) {
            if(!fof_params.MbhMscRelationCWmodel)
                endrun(1, "MinBHSeedInSC > 0 requires MbhMscRelationCWmodel=1 (it thresholds the "
                          "CW-model M_VMS).\n");
            if(param_get_double(ps, "SeedBlackHoleMass") <= 0)
                endrun(1, "MinBHSeedInSC > 0 requires SeedBlackHoleMass > 0: it is the mass of each "
                          "compensating BH seed and the quantum the missed mass is divided by.\n");
        }
        fof_params.FOFPotentialMin = param_get_int(ps, "FOFPotentialMin");
    }
    MPI_Bcast(&fof_params, sizeof(struct FOFParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* Set parameters for the tests*/
void set_fof_testpar(int FOFSaveParticles, double FOFHaloLinkingLength, int FOFHaloMinLength)
{
    fof_params.FOFSaveParticles = FOFSaveParticles;
    fof_params.FOFPrimaryLinkTypes = 2;
    fof_params.FOFSecondaryLinkTypes = 1+16+32;
    fof_params.FOFHaloLinkingLength = FOFHaloLinkingLength;
    fof_params.FOFHaloMinLength = FOFHaloMinLength;
    /* For seeding (not yet tested)*/
    fof_params.MinFoFMassForNewSeed = 2;
    fof_params.MinMStarForNewSeed = 5e-4;

    fof_params.BlackHoleSeedGasBased = 0;
    fof_params.BlackHoleSeedsfmpGas = 1e-3;
    fof_params.BlackHoleseedsMetalThres = 1e-4;

    fof_params.FOFPotentialMin = 0;
    fof_params.FOFMinPrimaryLength = 0;
}

void fof_init(double DMMeanSeparation)
{
    fof_params.FOFHaloComovingLinkingLength = fof_params.FOFHaloLinkingLength * DMMeanSeparation;
}

void fof_get_params(int *PrimaryLinkTypes, int *SecondaryLinkTypes,
                    double *ComovingLinkingLength, int *MinLength,
                    int *PotentialMin, int *MinPrimaryLength)
{
    *PrimaryLinkTypes = fof_params.FOFPrimaryLinkTypes;
    *SecondaryLinkTypes = fof_params.FOFSecondaryLinkTypes;
    *ComovingLinkingLength = fof_params.FOFHaloComovingLinkingLength;
    *MinLength = fof_params.FOFHaloMinLength;
    *PotentialMin = fof_params.FOFPotentialMin;
    *MinPrimaryLength = fof_params.FOFMinPrimaryLength;
}

void fof_set_params(int PrimaryLinkTypes, int SecondaryLinkTypes,
                    double ComovingLinkingLength, int MinLength,
                    int PotentialMin, int MinPrimaryLength)
{
    fof_params.FOFPrimaryLinkTypes = PrimaryLinkTypes;
    fof_params.FOFSecondaryLinkTypes = SecondaryLinkTypes;
    fof_params.FOFHaloComovingLinkingLength = ComovingLinkingLength;
    fof_params.FOFHaloMinLength = MinLength;
    fof_params.FOFPotentialMin = PotentialMin;
    fof_params.FOFMinPrimaryLength = MinPrimaryLength;
}

/* Toggle the "unseeded stars only" restriction on the primary-linking set.
 * Set on every MPI rank (like fof_set_params); secondfof_run enables it
 * around the second-FOF fof_fof() call and resets it to 0 afterwards. */
void fof_set_primary_unseeded_only(int flag)
{
    fof_params.FOFPrimaryUnseededStarsOnly = flag;
}

void fof_get_seed_params(int *BlackHoleSeedStarCluster, int *BlackHoleSeedHaloBased,
                         int *BlackHoleSeedGasBased)
{
    *BlackHoleSeedStarCluster = fof_params.BlackHoleSeedStarCluster;
    *BlackHoleSeedHaloBased = fof_params.BlackHoleSeedHaloBased;
    *BlackHoleSeedGasBased = fof_params.BlackHoleSeedGasBased;
}

void fof_set_seed_params(int BlackHoleSeedStarCluster, int BlackHoleSeedHaloBased,
                         int BlackHoleSeedGasBased)
{
    fof_params.BlackHoleSeedStarCluster = BlackHoleSeedStarCluster;
    fof_params.BlackHoleSeedHaloBased = BlackHoleSeedHaloBased;
    fof_params.BlackHoleSeedGasBased = BlackHoleSeedGasBased;
}

static double fof_periodic_wrap(double x, double BoxSize)
{
    while(x >= BoxSize)
        x -= BoxSize;
    while(x < 0)
        x += BoxSize;
    return x;
}

struct fof_particle_list
{
    MyIDType MinID;
    int MinIDTask;
    int Pindex;
};

static void fof_label_secondary(struct fof_particle_list * HaloLabel, ForceTree * tree);
static int fof_compare_HaloLabel_MinID(const void *a, const void *b);
static int _fof_compare_Group_MinIDTask_ThisTask;
static int fof_compare_Group_MinIDTask(const void *a, const void *b);
static int fof_compare_Group_OriginalIndex(const void *a, const void *b);
static int fof_compare_Group_MinID(const void *a, const void *b);
static void fof_reduce_groups(
    void * groups,
    int nmemb,
    size_t elsize,
    void (*reduce_group)(void * gdst, void * gsrc), MPI_Comm Comm);

static void fof_finish_group_properties(FOFGroups * fof, double BoxSize);

static int fof_compile_base(struct BaseGroup * base, int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm);
static void fof_compile_catalogue(FOFGroups * fof, const int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm);

static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm);

void fof_label_primary(struct fof_particle_list * HaloLabel, ForceTree * tree, MPI_Comm Comm);

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyIDType MinID;
    int MinIDTask;
    int pad;
} TreeWalkQueryFOF;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Distance;
    MyIDType MinID;
    int MinIDTask;
    int pad;
} TreeWalkResultFOF;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterFOF;


static MPI_Datatype MPI_TYPE_GROUP = MPI_DATATYPE_NULL;

/*
 * The FOF finder will produce Group[], which is allocated to the top side of the
 * main heap.
 *
 **/

FOFGroups
fof_fof(DomainDecomp * ddecomp, const int StoreGrNr, MPI_Comm Comm)
{
    int i;

    message(0, "Begin to compute FoF group catalogues. (allocated: %g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    message(0, "Comoving linking length: %g\n", fof_params.FOFHaloComovingLinkingLength);

    struct fof_particle_list * HaloLabel = (struct fof_particle_list *) mymalloc("HaloLabel", PartManager->NumPart * sizeof(struct fof_particle_list));

    /* HaloLabel stores the MinID and MinIDTask of particles, this pair serves as a halo label. */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        HaloLabel[i].Pindex = i;
    }

    /* We only need a tree containing primary linking particles only. No moments*/
    ForceTree dmtree = {0};
    force_tree_rebuild_mask(&dmtree, ddecomp, fof_params.FOFPrimaryLinkTypes, NULL);
    walltime_measure("/FOF/Build");

    /* Fill FOFP_List of primary */
    fof_label_primary(HaloLabel, &dmtree, Comm);
    walltime_measure("/FOF/Primary");

    /* Fill FOFP_List of secondary */
    fof_label_secondary(HaloLabel, &dmtree);
    force_tree_free(&dmtree);

    message(0, "Attached gas and star particles to nearest dm particles.\n");

    walltime_measure("/FOF/Secondary");

    /* sort HaloLabel according to MinID, because we need that for compiling catalogues */
    qsort_openmp(HaloLabel, PartManager->NumPart, sizeof(struct fof_particle_list), fof_compare_HaloLabel_MinID);

    int NgroupsExt = 0;

    for(i = 0; i < PartManager->NumPart; i ++) {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) NgroupsExt ++;
    }

    /* The first round is to eliminate groups that are too short. */
    /* We create the smaller 'BaseGroup' data set for this. */
    struct BaseGroup * base = (struct BaseGroup *) mymalloc("BaseGroup", sizeof(struct BaseGroup) * NgroupsExt);

    NgroupsExt = fof_compile_base(base, NgroupsExt, HaloLabel, Comm);

    message(0, "Compiled local group data and catalogue.\n");

    fof_assign_grnr(base, NgroupsExt, Comm);

    /*Store the group number in the particle struct*/
    if(StoreGrNr) {
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
            P[i].GrNr = -1;	/* will mark particles that are not in any group */

        int64_t start = 0;
        for(i = 0; i < NgroupsExt; i++)
        {
            for(;start < PartManager->NumPart; start++) {
                if (HaloLabel[start].MinID >= base[i].MinID)
                    break;
            }

            for(;start < PartManager->NumPart; start++) {
                if (HaloLabel[start].MinID != base[i].MinID)
                    break;
                P[HaloLabel[start].Pindex].GrNr = base[i].GrNr;
            }
        }
    }

    /*Initialise the Group object from the BaseGroup*/
    FOFGroups fof;
    /* Free any previous MPI_TYPE_GROUP to avoid handle leak when
     * fof_fof() is called multiple times (e.g. primary + secondary FOF). */
    if(MPI_TYPE_GROUP != MPI_DATATYPE_NULL)
        MPI_Type_free(&MPI_TYPE_GROUP);
    MPI_Type_contiguous(sizeof(fof.Group[0]), MPI_BYTE, &MPI_TYPE_GROUP);
    MPI_Type_commit(&MPI_TYPE_GROUP);

    fof.Group = fof_alloc_group(base, NgroupsExt);

    myfree(base);

    fof_compile_catalogue(&fof, NgroupsExt, HaloLabel, Comm);

    MPIU_Barrier(Comm);
    message(0, "Finished FoF. Group properties are now allocated.. (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/FOF/Compile");

    myfree(HaloLabel);

    return fof;
}

void
fof_finish(FOFGroups * fof)
{
    myfree(fof->Group);

    message(0, "Finished computing FoF groups.  (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    if(MPI_TYPE_GROUP != MPI_DATATYPE_NULL) {
        MPI_Type_free(&MPI_TYPE_GROUP);
        MPI_TYPE_GROUP = MPI_DATATYPE_NULL;
    }
}

struct FOFPrimaryPriv {
    int * Head;
    struct SpinLocks * spin;
    char * PrimaryActive;
    MyIDType * OldMinID;
    struct fof_particle_list * HaloLabel;
};
#define FOF_PRIMARY_GET_PRIV(tw) ((struct FOFPrimaryPriv *) (tw->priv))

/* This function walks the particle tree starting at particle i until it reaches
 * a particle which has Head[i] = i, the root node (particles are initialised in
 * this state, so this is equivalent to finding a particle which has yet to be merged).
 * Once it reaches a root, it returns that particle number.
 * Arguments:
 *
 * stop: When this particle is reached, return -1. We use this to find an already merged tree.
 *
 * Returns:
 *      root particle if found
 *      -1 if stop particle reached
 */
static int
HEADl(int stop, int i, const int * const Head)
{
    int next = i;

    do {
        i = next;
        /* Reached stop, return*/
        if(i == stop)
            return -1;
        /* atomic read because we may change
         * this in update_root: not necessary on x86_64, but avoids tears elsewhere*/
        #pragma omp atomic read
        next = Head[i];
    } while(next != i);

    /* return unmerged particle*/
    return i;
}

/* Rewrite a tree so that all values in it point directly to the true root.
 * This means that the trees are O(1) deep and speeds up future accesses.
 * See https://arxiv.org/abs/1607.03224 */
static void
update_root(int i, const int r, int * Head)
{
    int t = i;
    do {
        i = t;
        #pragma omp atomic capture
        {
            t = Head[i];
            Head[i]= r;
        }
        /* Stop if we reached the top (new head is the same as the old)
         * or if the new head is less than or equal to the desired head, indicating
         * another thread changed us*/
    } while(t != i && (t > r));
}

/* Find the current head particle by walking the tree. No updates are done
 * so this can be performed from a threaded context. */
static int
HEAD(int i, const int * const Head)
{
    int r = i;
    while(Head[r] != r) {
        r = Head[r];
    }
    return r;
}

static void fof_primary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {
    /* The copied data is *only* used for the
     * secondary treewalk, so fill up garbage for the primary treewalk.
     * The copy is a technical race otherwise. */
    if(I->base.NodeList[0] == tw->tree->firstnode) {
        I->MinID = IDTYPE_MAX;
        I->MinIDTask = -1;
        return;
    }
    /* Secondary treewalk, no need for locking here*/
    int head = HEAD(place, FOF_PRIMARY_GET_PRIV(tw)->Head);
    I->MinID = FOF_PRIMARY_GET_PRIV(tw)->HaloLabel[head].MinID;
    I->MinIDTask = FOF_PRIMARY_GET_PRIV(tw)->HaloLabel[head].MinIDTask;
}

/* True if particle i acts as a primary-linking particle for the current FOF run.
 * This is the type-mask test, plus the FOFPrimaryUnseededStarsOnly restriction
 * (second FOF with SecFOFUnseededPart=1) which drops seeded star particles
 * (Type==4 && STARP.Seeded) from the primary set. Used in the primary/secondary
 * neighbour iterators and the LenPrimary / PotMin accounting so a seeded star is
 * consistently treated as not-primary even though it stays in the FOF tree. */
static inline int
fof_is_primary_link(int i)
{
    if(!((1 << P[i].Type) & fof_params.FOFPrimaryLinkTypes))
        return 0;
    if(fof_params.FOFPrimaryUnseededStarsOnly && P[i].Type == 4 && STARP(i).Seeded)
        return 0;
    return 1;
}

static int fof_primary_haswork(int n, TreeWalk * tw) {
    if(P[n].IsGarbage || P[n].Swallowed)
        return 0;
    return fof_is_primary_link(n) && FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[n];
}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv);

void fof_label_primary(struct fof_particle_list * HaloLabel, ForceTree * tree, MPI_Comm Comm)
{
    int i;
    int64_t link_across_tot;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    message(0, "Start linking particles (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    TreeWalk tw[1] = {{0}};
    tw->ev_label = "FOF_FIND_GROUPS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_primary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);

    tw->haswork = fof_primary_haswork;
    tw->fill = (TreeWalkFillQueryFunction) fof_primary_copy;
    tw->reduce = NULL;
    tw->type = TREEWALK_ALL;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);
    tw->tree = tree;
    struct FOFPrimaryPriv priv[1];
    tw->priv = priv;

    FOF_PRIMARY_GET_PRIV(tw)->Head = (int*) mymalloc("FOF_Links", PartManager->NumPart * sizeof(int));
    FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive = (char*) mymalloc("FOFActive", PartManager->NumPart * sizeof(char));
    FOF_PRIMARY_GET_PRIV(tw)->OldMinID = (MyIDType *) mymalloc("FOFActive", PartManager->NumPart * sizeof(MyIDType));
    FOF_PRIMARY_GET_PRIV(tw)->HaloLabel = HaloLabel;
    /* allocate buffers to arrange communication */

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        FOF_PRIMARY_GET_PRIV(tw)->Head[i] = i;
        FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i]= P[i].ID;
        FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 1;

        HaloLabel[i].MinID = P[i].ID;
        HaloLabel[i].MinIDTask = ThisTask;
    }

    /* The lock is used to protect MinID*/
    priv[0].spin = init_spinlocks(PartManager->NumPart);
    do
    {
        double t0 = second();

        treewalk_run(tw, NULL, PartManager->NumPart);

        double t1 = second();
        /* This sets the MinID of the head particle to the minimum ID
         * of the child particles. We set this inside the treewalk,
         * but the locking allows a race, where the particle with MinID set
         * is no longer the one which is the true Head of the group.
         * So we must check it again here.*/
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++) {
            int head = HEAD(i, FOF_PRIMARY_GET_PRIV(tw)->Head);
            /* Don't check against ourself*/
            if(head == i)
                continue;
            MyIDType headminid;
            #pragma omp atomic read
            headminid = HaloLabel[head].MinID;
            /* No atomic needed for i as this is not a head*/
            if(headminid > HaloLabel[i].MinID) {
                lock_spinlock(head, priv->spin);
                if(HaloLabel[head].MinID > HaloLabel[i].MinID) {
                    #pragma omp atomic write
                    HaloLabel[head].MinID = HaloLabel[i].MinID;
                    HaloLabel[head].MinIDTask = HaloLabel[i].MinIDTask;
                }
                unlock_spinlock(head, priv->spin);
            }
        }
        /* let's check out which particles have changed their MinID,
         * mark them for next round. */
        int64_t link_across = 0;
#pragma omp parallel for reduction(+: link_across)
        for(i = 0; i < PartManager->NumPart; i++) {
            int head = HEAD(i, FOF_PRIMARY_GET_PRIV(tw)->Head);
            /* This loop sets the MinID of the children to the minID of the head.
             * The minID of the head is set above and is stable at this point.*/
            if(i != head) {
                HaloLabel[i].MinID = HaloLabel[head].MinID;
                HaloLabel[i].MinIDTask = HaloLabel[head].MinIDTask;
            }
            MyIDType newMinID = HaloLabel[head].MinID;
            if(newMinID != FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i]) {
                FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 1;
                FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i] = newMinID;
                link_across ++;
            } else {
                FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 0;
            }
        }
        double t2 = second();

        MPI_Allreduce(&link_across, &link_across_tot, 1, MPI_INT64, MPI_SUM, Comm);
        message(0, "Linked %ld particles %g seconds postproc was %g seconds\n", link_across_tot, t1 - t0, t2 - t1);
    }
    while(link_across_tot > 0);

    free_spinlocks(priv[0].spin);

    message(0, "Local groups found.\n");

    myfree(FOF_PRIMARY_GET_PRIV(tw)->OldMinID);
    myfree(FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive);
    myfree(FOF_PRIMARY_GET_PRIV(tw)->Head);
}

static void
fofp_merge(int target, int other, TreeWalk * tw)
{
    /* this will lock h1 */
    int * Head = FOF_PRIMARY_GET_PRIV(tw)->Head;
    int h1, h2;
    do {
        h1 = HEADl(-1, target, Head);
        /* Done if we find h1 along the path
         * (because other is already in the same halo) */
        h2 = HEADl(h1, other, Head);
        if(h2 < 0)
            return;
        /* Ensure that we always merge to the lower entry.
         * This avoids circular loops in the Head entries:
         * a -> b -> a */
        if(h1 > h2) {
            int tmp = h2;
            h2 = h1;
            h1 = tmp;
        }
     /* Atomic compare exchange to make h2 a subtree of h1.
      * Set Head[h2] = h1 iff Head[h2] is still h2. Otherwise loop.*/
    } while(!__atomic_compare_exchange(&Head[h2], &h2, &h1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

    struct SpinLocks * spin = FOF_PRIMARY_GET_PRIV(tw)->spin;

    /* update MinID of h1: h2 is now just another child of h1
     * so we don't need to check that h2 changes its head.
     * It might happen that h1 is added to another halo at this point
     * and the addition gets the wrong MinID.
     * For this reason we recompute the MinIDs after the main treewalk.
     * We also lock h2 for a copy in case it is the h1 in another thread,
     * and may have inconsistent MinID and MinIDTask.*/

    /* Get a copy of h2 under the lock, which ensures
     * that MinID and MinIDTask do not change independently. */
    struct fof_particle_list * HaloLabel = FOF_PRIMARY_GET_PRIV(tw)->HaloLabel;
    struct fof_particle_list h2label;
    lock_spinlock(h2, spin);
    h2label.MinID = HaloLabel[h2].MinID;
    h2label.MinIDTask = HaloLabel[h2].MinIDTask;
    unlock_spinlock(h2, spin);

    /* Now lock h1 so we don't change MinID but not MinIDTask.*/
    lock_spinlock(h1, spin);
    if(HaloLabel[h1].MinID > h2label.MinID)
    {
        HaloLabel[h1].MinID = h2label.MinID;
        HaloLabel[h1].MinIDTask = h2label.MinIDTask;
    }
    unlock_spinlock(h1, spin);

    /* h1 must be the root of other and target both:
     * do the splay to speed up future accesses.
     * We do not need to have h2 locked, because h2 is
     * now just another child of h1: these do not change the root,
     * they make the tree shallow.*/
    update_root(target, h1, Head);
    update_root(other, h1, Head);

}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv)
{
    TreeWalk * tw = lv->tw;
    if(iter->base.other == -1) {
        iter->base.Hsml = fof_params.FOFHaloComovingLinkingLength;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        iter->base.mask = fof_params.FOFPrimaryLinkTypes;
        return;
    }
    int other = iter->base.other;

    /* The neighbour search only filters by particle type, so seeded stars are
     * still returned when FOFPrimaryUnseededStarsOnly is set. Skip them so they
     * are never linked as primary particles (the initiating target is already
     * filtered out by fof_primary_haswork). */
    if(!fof_is_primary_link(other))
        return;

    if(lv->mode == TREEWALK_PRIMARY) {
        /* Local FOF */
        if(lv->target <= other) {
            // printf("locked merge %d %d by %d\n", lv->target, other, omp_get_thread_num());
            fofp_merge(lv->target, other, tw);
        }
    }
    else /* mode is 1, target is a ghost */
    {
        int head = HEAD(other, FOF_PRIMARY_GET_PRIV(tw)->Head);
        struct fof_particle_list * HaloLabel = FOF_PRIMARY_GET_PRIV(tw)->HaloLabel;
        struct SpinLocks * spin = FOF_PRIMARY_GET_PRIV(tw)->spin;
//        printf("locking %d by %d in ngbiter\n", other, omp_get_thread_num());
        lock_spinlock(head, spin);
        if(HaloLabel[head].MinID > I->MinID)
        {
            HaloLabel[head].MinID = I->MinID;
            HaloLabel[head].MinIDTask = I->MinIDTask;
        }
//        printf("unlocking %d by %d in ngbiter\n", other, omp_get_thread_num());
        unlock_spinlock(head, spin);
    }
}

static void fof_reduce_base_group(void * pdst, void * psrc) {
    struct BaseGroup * gdst = (struct BaseGroup *) pdst;
    struct BaseGroup * gsrc = (struct BaseGroup *) psrc;
    gdst->Length += gsrc->Length;
    gdst->LenPrimary += gsrc->LenPrimary;
    /* preserve the dst FirstPos so all other base group gets the same FirstPos */
}

static void fof_reduce_group(void * pdst, void * psrc) {
    struct Group * gdst = (struct Group *) pdst;
    struct Group * gsrc = (struct Group *) psrc;
    int j;
    gdst->Length += gsrc->Length;
    gdst->Mass += gsrc->Mass;

    for(j = 0; j < 6; j++)
    {
        gdst->LenType[j] += gsrc->LenType[j];
        gdst->MassType[j] += gsrc->MassType[j];
    }

    gdst->Sfr += gsrc->Sfr;
    gdst->sfmp_mass += gsrc->sfmp_mass;
    gdst->StarClusterMass += gsrc->StarClusterMass;
    gdst->StarClusterMetallicity += gsrc->StarClusterMetallicity;
    for(j = 0; j < NMETALS; j++)
        gdst->StarClusterMetalElemMass[j] += gsrc->StarClusterMetalElemMass[j];
    gdst->StarClusterMassSample += gsrc->StarClusterMassSample;
    gdst->NscSample += gsrc->NscSample;
    gdst->StarClusterMassUnseeded += gsrc->StarClusterMassUnseeded;
    gdst->StarClusterMassSampleUnseeded += gsrc->StarClusterMassSampleUnseeded;
    gdst->SCMass_seeded += gsrc->SCMass_seeded;
    gdst->NStarUnseeded += gsrc->NStarUnseeded;
    gdst->SCMetalMassUnseeded += gsrc->SCMetalMassUnseeded;
    gdst->SCClusterMassUnseededInit += gsrc->SCClusterMassUnseededInit;
    /* Unseeded-star metallicity distribution: min/max combine, sums/hist add. */
    if(gsrc->SCMetUnseededMin < gdst->SCMetUnseededMin)
        gdst->SCMetUnseededMin = gsrc->SCMetUnseededMin;
    if(gsrc->SCMetUnseededMax > gdst->SCMetUnseededMax)
        gdst->SCMetUnseededMax = gsrc->SCMetUnseededMax;
    gdst->SCMetUnseededSum  += gsrc->SCMetUnseededSum;
    gdst->SCMetUnseededSum2 += gsrc->SCMetUnseededSum2;
    gdst->SCMetUnseededLogSum  += gsrc->SCMetUnseededLogSum;
    gdst->SCMetUnseededLogSum2 += gsrc->SCMetUnseededLogSum2;
    for(j = 0; j < SC_MET_HIST_NBIN; j++)
        gdst->SCMetUnseededHist[j] += gsrc->SCMetUnseededHist[j];
    gdst->SCcomMcut += gsrc->SCcomMcut;
    gdst->GasMetalMass += gsrc->GasMetalMass;
    gdst->StellarMetalMass += gsrc->StellarMetalMass;
    gdst->MassHeIonized += gsrc->MassHeIonized;
    for(j = 0; j < NMETALS; j++) {
        gdst->GasMetalElemMass[j] += gsrc->GasMetalElemMass[j];
        gdst->StellarMetalElemMass[j] += gsrc->StellarMetalElemMass[j];
    }
    gdst->BH_Mdot += gsrc->BH_Mdot;
    gdst->BH_Mass += gsrc->BH_Mass;
    if(gsrc->MaxDens > gdst->MaxDens)
    {
        gdst->MaxDens = gsrc->MaxDens;
        gdst->seed_index = gsrc->seed_index;
        gdst->seed_task = gsrc->seed_task;
    }
    if(fof_params.FOFPotentialMin && gsrc->PotMin < gdst->PotMin)
    {
        gdst->PotMin = gsrc->PotMin;
        int d;
        for(d = 0; d < 3; d++)
            gdst->PotMinPos[d] = gsrc->PotMinPos[d];
    }
    if(gsrc->MaxStarClusterMass > gdst->MaxStarClusterMass)
    {
        gdst->MaxStarClusterMass = gsrc->MaxStarClusterMass;
        gdst->seed_index_star = gsrc->seed_index_star;
        gdst->seed_task_star = gsrc->seed_task_star;
        gdst->SeedStarID = gsrc->SeedStarID;
    }

    int d1, d2;
    for(d1 = 0; d1 < 3; d1++)
    {
        gdst->CM[d1] += gsrc->CM[d1];
        gdst->Vel[d1] += gsrc->Vel[d1];
        gdst->Jmom[d1] += gsrc->Jmom[d1];
        for(d2 = 0; d2 < 3; d2 ++) {
            gdst->Imom[d1][d2] += gsrc->Imom[d1][d2];
        }
    }

}

/* Map an unseeded-star metallicity Z (absolute mass fraction) to its bin in the
 * per-group SCMetUnseededHist.  Z<=0 (pristine) and Z below the floor land in the
 * underflow bin 0; Z above the ceiling in the overflow bin NBIN-1. */
static int sc_met_hist_bin(double Z)
{
    if(Z <= 0)
        return 0;
    double lz = log10(Z);
    if(lz <= SC_MET_HIST_LOGMIN)
        return 0;
    if(lz >= SC_MET_HIST_LOGMAX)
        return SC_MET_HIST_NBIN - 1;
    const double dlog = (SC_MET_HIST_LOGMAX - SC_MET_HIST_LOGMIN) / (SC_MET_HIST_NBIN - 2);
    int b = 1 + (int) ((lz - SC_MET_HIST_LOGMIN) / dlog);
    if(b < 1) b = 1;
    if(b > SC_MET_HIST_NBIN - 2) b = SC_MET_HIST_NBIN - 2;
    return b;
}

/* Percentile p in [0,1] of the unseeded-star metallicity from the fixed log10(Z)
 * histogram (N total stars); zmin/zmax are the exact tracked extrema used for the
 * under/overflow bins.  Interior bins interpolate linearly in log-Z. */
static double sc_met_hist_percentile(const float * hist, int64_t N,
                                     double zmin, double zmax, double p)
{
    if(N <= 0)
        return 0;
    const double dlog = (SC_MET_HIST_LOGMAX - SC_MET_HIST_LOGMIN) / (SC_MET_HIST_NBIN - 2);
    double target = p * (double) N;
    double cum = 0;
    int b;
    for(b = 0; b < SC_MET_HIST_NBIN; b++) {
        double c = hist[b];
        if(cum + c >= target || b == SC_MET_HIST_NBIN - 1) {
            if(b == 0)                      /* underflow (incl. pristine): exact min */
                return zmin;
            if(b == SC_MET_HIST_NBIN - 1)   /* overflow: exact max */
                return zmax;
            double frac = c > 0 ? (target - cum) / c : 0;   /* within-bin position */
            double z = pow(10.0, SC_MET_HIST_LOGMIN + (b - 1 + frac) * dlog);
            if(z < zmin) z = zmin;
            if(z > zmax) z = zmax;
            return z;
        }
        cum += c;
    }
    return zmax;
}

/* Fill the StarClusterDetails unseeded-star metallicity distribution (equal
 * weight per star) from a fully reduced host group: exact min/max, standard
 * deviation from the running sums, and 25/50/75 percentiles from the histogram.
 * All zero when the group has no unseeded star. */
static void sc_met_unseeded_stats(const struct Group * g, struct SCmetdist * md)
{
    int64_t N = g->NStarUnseeded;
    if(N <= 0) {
        memset(md, 0, sizeof(*md));
        return;
    }
    md->min = g->SCMetUnseededMin;
    md->max = g->SCMetUnseededMax;
    double mean = g->SCMetUnseededSum / (double) N;
    double var = g->SCMetUnseededSum2 / (double) N - mean * mean;
    md->std = var > 0 ? sqrt(var) : 0;
    md->p25    = sc_met_hist_percentile(g->SCMetUnseededHist, N, md->min, md->max, 0.25);
    md->median = sc_met_hist_percentile(g->SCMetUnseededHist, N, md->min, md->max, 0.50);
    md->p75    = sc_met_hist_percentile(g->SCMetUnseededHist, N, md->min, md->max, 0.75);
}

/* Fill the StarClusterDetails bound-star summary from a (reduced, owned) group.
 * All zero when BHseedSecFOFbound is off, since nothing filled the SCBound* fields. */
static void sc_bound_stats(const struct Group * g, struct SCboundinfo * b)
{
    memset(b, 0, sizeof(*b));
    b->mode = fof_params.BHseedSecFOFbound;
    if(!b->mode)
        return;
    b->mass          = g->SCBoundStarMass;
    b->mass_unseeded = g->SCBoundStarMassUnseeded;
    b->scmass        = g->SCBoundClusterMass;
    b->rdm           = g->SCBoundRdm;
    b->mdm           = g->SCBoundMdm;
    b->num           = g->NStarBound;
}

static void add_particle_to_group(struct Group * gdst, int i, int ThisTask) {

    /* My local number of particles contributing to the full catalogue. */
    const int index = i;
    if(gdst->Length == 0) {
        struct BaseGroup base = gdst->base;
        memset(gdst, 0, sizeof(gdst[0]));
        gdst->base = base;
        gdst->seed_index = gdst->seed_task = -1;
        gdst->seed_index_star = gdst->seed_task_star = -1;
        gdst->MaxStarClusterMass = 0;
        gdst->PotMin = 1e30;
        gdst->SCMetUnseededMin = 1e30;   /* running min sentinel (Z>=0, so max starts at 0) */
    }

    gdst->Length ++;
    gdst->Mass += P[index].Mass;
    gdst->LenType[P[index].Type]++;
    gdst->MassType[P[index].Type] += P[index].Mass;

    if(P[index].Type == 0) {
        gdst->MassHeIonized += P[index].Mass * P[index].HeIIIionized;
        gdst->Sfr += SPHP(index).Sfr;
        gdst->GasMetalMass += SPHP(index).Metallicity * P[index].Mass;
        int j;
        for(j = 0; j < NMETALS; j++)
            gdst->GasMetalElemMass[j] += SPHP(index).Metals[j] * P[index].Mass;

        if (SPHP(index).Sfr > 0 && SPHP(index).Metallicity < (fof_params.BlackHoleseedsMetalThres * SOLAR_METAL)){
            gdst->sfmp_mass += P[index].Mass;
        }

    }
    if(P[index].Type == 4) {
        int j;
        gdst->StellarMetalMass += STARP(index).Metallicity * P[index].Mass;
        for(j = 0; j < NMETALS; j++)
            gdst->StellarMetalElemMass[j] += STARP(index).Metals[j] * P[index].Mass;

        /* Totals over ALL hosted stars (seeded + unseeded): catalogue output and
         * the (intensive) mass-weighted metallicity used for the seed payload. */
        gdst->StarClusterMass += STARP(index).ClusterMass;
        gdst->StarClusterMetallicity += STARP(index).Metallicity * STARP(index).ClusterMass;
        for(j = 0; j < NMETALS; j++)
            gdst->StarClusterMetalElemMass[j] += STARP(index).Metals[j] * STARP(index).ClusterMass;
        gdst->StarClusterMassSample += STARP(index).StarClusterMass_sample;
        gdst->NscSample += STARP(index).Nsc_sample;

        /* Stars that have already contributed to a BH seed (Seeded==1) are
         * excluded from everything that drives seeding: the unseeded cluster-mass
         * sums, the M_cut cutoff, and the seed-particle pick. Their cluster mass is
         * recorded separately in SCMass_seeded for the catalogue. */
        if(STARP(index).Seeded) {
            gdst->SCMass_seeded += STARP(index).ClusterMass;
        } else {
            /* Metallicity-dependent seeding factor f(Z) scales the per-star
             * cluster mass that drives seeding (ClusterMass = Gamma*m_star is kept
             * raw). The sampled mass StarClusterMass_sample already carries f(Z)
             * (its Poisson rate was scaled at formation), so it is not rescaled. */
            double fseed = get_seed_metallicity_factor(STARP(index).BirthMetallicity);
            gdst->StarClusterMassUnseeded += fseed * STARP(index).ClusterMass;
            gdst->StarClusterMassSampleUnseeded += STARP(index).StarClusterMass_sample;
            gdst->SCcomMcut += P[index].Mass;
            gdst->NStarUnseeded++;

            /* Unseeded-star metallicity for StarClusterDetails: metal mass (frozen
             * BirthMetallicity * frozen initClusterMass) and the raw Gamma*m_star
             * denominator (initClusterMass), both over unseeded stars only. */
            gdst->SCMetalMassUnseeded += STARP(index).BirthMetallicity * STARP(index).initClusterMass;
            gdst->SCClusterMassUnseededInit += STARP(index).initClusterMass;

            /* Per-particle (equal-weight) BirthMetallicity distribution of the
             * unseeded stars for StarClusterDetails: exact min/max, sums for the
             * standard deviation, and a log10(Z) histogram for the quartiles. */
            {
                double zb = STARP(index).BirthMetallicity;
                if(zb < gdst->SCMetUnseededMin) gdst->SCMetUnseededMin = zb;
                if(zb > gdst->SCMetUnseededMax) gdst->SCMetUnseededMax = zb;
                gdst->SCMetUnseededSum  += zb;
                gdst->SCMetUnseededSum2 += zb * zb;
                gdst->SCMetUnseededHist[sc_met_hist_bin(zb)] += 1;
                /* log10(Z) sums (equal weight) for the CW-model 'lognormal'
                 * metallicity draw; pristine/tiny Z floored like the histogram. */
                double lzb = (zb > 0) ? log10(zb) : SC_MET_HIST_LOGMIN;
                if(lzb < SC_MET_HIST_LOGMIN) lzb = SC_MET_HIST_LOGMIN;
                gdst->SCMetUnseededLogSum  += lzb;
                gdst->SCMetUnseededLogSum2 += lzb * lzb;
            }

            /* Track the unseeded star with the largest (f(Z)-scaled) ClusterMass
             * (or StarClusterMass_sample when StarClusterSampling=1) as the seed
             * particle for star-cluster BH seeding in secondary FOF. For
             * SeedSecFOFcomSample use f(Z)*ClusterMass and record its ID as the RNG
             * seed for the combined draw. */
            MyFloat scm = (fof_params.StarClusterSampling && !fof_params.SeedSecFOFcomSample) ?
                STARP(index).StarClusterMass_sample : fseed * STARP(index).ClusterMass;
            if(scm > gdst->MaxStarClusterMass) {
                gdst->MaxStarClusterMass = scm;
                gdst->seed_index_star = index;
                gdst->seed_task_star = ThisTask;
                gdst->SeedStarID = P[index].ID;
            }
        }
    }

    if(P[index].Type == 5)
    {
        gdst->BH_Mdot += BHP(index).Mdot;
        gdst->BH_Mass += BHP(index).Mass;
    }
    /*This used to depend on black holes being enabled, but I do not see why.
     * I think because it is only useful for seeding*/
    /* Don't make bh in wind.*/
    if(P[index].Type == 0 && !winds_is_particle_decoupled(index))
        if(SPHP(index).Density > gdst->MaxDens)
        {
            gdst->MaxDens = SPHP(index).Density;
            gdst->seed_index = index;
            gdst->seed_task = ThisTask;
        }

    /* Track minimum potential among primary-linked particles.
     * Used by second FOF for group center; only when FOFPotentialMin is enabled. */
    if(fof_params.FOFPotentialMin && fof_is_primary_link(index)) {
        if(P[index].Potential < gdst->PotMin) {
            gdst->PotMin = P[index].Potential;
            int d;
            for(d = 0; d < 3; d++)
                gdst->PotMinPos[d] = P[index].Pos[d];
        }
    }

    int d1, d2;
    double xyz[3];
    double rel[3];
    double vel[3];
    double jmom[3];

    for(d1 = 0; d1 < 3; d1++)
    {
        double first = gdst->base.FirstPos[d1];
        rel[d1] = NEAREST(P[index].Pos[d1] - first, PartManager->BoxSize) ;
        xyz[d1] = rel[d1] + first;
        vel[d1] = P[index].Vel[d1];
    }

    crossproduct(rel, vel, jmom);

    for(d1 = 0; d1 < 3; d1++) {
        gdst->CM[d1] += P[index].Mass * xyz[d1];
        gdst->Vel[d1] += P[index].Mass * vel[d1];
        gdst->Jmom[d1] += P[index].Mass * jmom[d1];

        for(d2 = 0; d2 < 3; d2++) {
            gdst->Imom[d1][d2] += P[index].Mass * rel[d1] * rel[d2];
        }
    }
}

static void
fof_finish_group_properties(struct FOFGroups * fof, double BoxSize)
{
    int i;

    for(i = 0; i < fof->Ngroups; i++)
    {
        int d1, d2;
        double cm[3];
        double rel[3];
        double jcm[3];
        double vcm[3];

        struct Group * gdst = &fof->Group[i];
        for(d1 = 0; d1 < 3; d1++)
        {
            gdst->Vel[d1] /= gdst->Mass;
            vcm[d1] = gdst->Vel[d1];
            cm[d1] = gdst->CM[d1] / gdst->Mass;

            rel[d1] = NEAREST(cm[d1] - gdst->base.FirstPos[d1], BoxSize);

            cm[d1] = fof_periodic_wrap(cm[d1], BoxSize);
            gdst->CM[d1] = cm[d1];

        }
        crossproduct(rel, vcm, jcm);

        for(d1 = 0; d1 < 3; d1 ++) {
            gdst->Jmom[d1] -= jcm[d1] * gdst->Mass;
        }

        for(d1 = 0; d1 < 3; d1 ++) {
            for(d2 = 0; d2 < 3; d2++) {
                /* Parallel Axis theorem:
                 * https://en.wikipedia.org/wiki/Parallel_axis_theorem ;
                 * J was relative to FirstPos, I is relative to CM.
                 *
                 * Note that our definition of Imom follows the astronomy one,
                 *
                 * I_ij = sum x_i x_j (where x_i x_j is relative displacement)
                 * */

                double diff = rel[d1] * rel[d2];

                gdst->Imom[d1][d2] -= gdst->Mass * diff;
            }
        }
    }

}

static int
fof_compile_base(struct BaseGroup * base, int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm)
{
    memset(base, 0, sizeof(base[0]) * NgroupsExt);

    int i;
    int start;

    start = 0;
    for(i = 0; i < PartManager->NumPart; i++)
    {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) {
            base[start].MinID = HaloLabel[i].MinID;
            base[start].MinIDTask = HaloLabel[i].MinIDTask;
            int d;
            for(d = 0; d < 3; d ++) {
                base[start].FirstPos[d] = P[HaloLabel[i].Pindex].Pos[d];
            }
            start ++;
        }
    }

    /* count local lengths */
    /* This works because base is sorted by MinID by construction. */
    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= base[i].MinID) break;
        }
        /* count particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != base[i].MinID) {
                break;
            }
            base[i].Length ++;
            if(fof_is_primary_link(HaloLabel[start].Pindex))
                base[i].LenPrimary ++;
        }
    }

    /* update global attributes */
    fof_reduce_groups(base, NgroupsExt, sizeof(base[0]), fof_reduce_base_group, Comm);

    /* eliminate all groups that are too small. The FOFMinPrimaryLength filter
     * (0 = off, set only by the second FOF) additionally drops groups with too
     * few primary-linking-type particles. */
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].Length < fof_params.FOFHaloMinLength
           || base[i].LenPrimary < fof_params.FOFMinPrimaryLength)
        {
            base[i] = base[NgroupsExt - 1];
            NgroupsExt--;
            i--;
        }
    }
    return NgroupsExt;
}

/* Allocate memory for and initialise a Group object
 * from a BaseGroup object.*/
static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt)
{
    int i;
    struct Group * Group = (struct Group *) mymalloc2("Group", sizeof(struct Group) * NgroupsExt);
    memset(Group, 0, sizeof(Group[0]) * NgroupsExt);

    /* copy in the base properties */
    /* at this point base group shall be sorted by MinID */
    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i ++) {
        Group[i].base = base[i];
    }
    return Group;
}

/* TODO: It would be a good idea to generalise this to arbitrary fof/particle properties */
#ifdef EXCUR_REION
static void fof_set_escapefraction(struct FOFGroups * fof, const int NgroupsExt, struct fof_particle_list * HaloLabel)
{
    int i = 0;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++){
        if(P[i].Type == 0){
            SPHP(i).EscapeFraction = 0.;
        }
        if(P[i].Type == 4){
            STARP(i).EscapeFraction = 0.;	/* will mark particles that are not in any group */
        }
    }

    int start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= fof->Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != fof->Group[i].base.MinID) {
                break;
            }
            int pi = HaloLabel[start].Pindex;

            /* putting halo mass in escape fraction for now, converted before uvbg calculation */
            //TODO: switch this off for gas particles if we are smoothing the star formation rate
            if(P[pi].Type == 0){
                SPHP(pi).EscapeFraction = fof->Group[i].Mass;
            }
            else if(P[pi].Type == 4){
                STARP(pi).EscapeFraction = fof->Group[i].Mass;
            }
        }
    }
}
#endif

static void
fof_compile_catalogue(struct FOFGroups * fof, const int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm)
{
    int i, start, ThisTask;

    MPI_Comm_rank(Comm, &ThisTask);

    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= fof->Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != fof->Group[i].base.MinID) {
                break;
            }
            add_particle_to_group(&fof->Group[i], HaloLabel[start].Pindex, ThisTask);
        }
    }

    /* collect global properties */
    fof_reduce_groups(fof->Group, NgroupsExt, sizeof(fof->Group[0]), fof_reduce_group, Comm);

    /* count Groups and number of particles hosted by me */
    fof->Ngroups = 0;
    int64_t Nids = 0;
    for(i = 0; i < NgroupsExt; i ++) {
        if(fof->Group[i].base.MinIDTask != ThisTask) continue;

        fof->Ngroups++;
        Nids += fof->Group[i].base.Length;

        if(fof->Group[i].base.Length != fof->Group[i].Length) {
            /* These two shall be consistent */
            endrun(3333, "i=%d Group base Length %d != Group Length %d\n", i, fof->Group[i].base.Length, fof->Group[i].Length);
        }
    }

    fof_finish_group_properties(fof, PartManager->BoxSize);
#ifdef EXCUR_REION
    /* feed group property back to each particle. */
    if(fof_params.ExcursionSetReionOn)
        fof_set_escapefraction(fof, NgroupsExt, HaloLabel);
#endif
    int64_t TotNids;
    MPI_Allreduce(&fof->Ngroups, &fof->TotNgroups, 1, MPI_INT64, MPI_SUM, Comm);
    MPI_Allreduce(&Nids, &TotNids, 1, MPI_INT64, MPI_SUM, Comm);

    /* report some statistics */
    int largestloc_tot = 0;
    double largestmass_tot= 0;
    if(fof->TotNgroups > 0)
    {
        double largestmass = 0;
        int largestlength = 0;

        for(i = 0; i < NgroupsExt; i++)
            if(fof->Group[i].Length > largestlength) {
                largestlength = fof->Group[i].Length;
                largestmass = fof->Group[i].Mass;
            }
        MPI_Allreduce(&largestlength, &largestloc_tot, 1, MPI_INT, MPI_MAX, Comm);
        MPI_Allreduce(&largestmass, &largestmass_tot, 1, MPI_DOUBLE, MPI_MAX, Comm);
    }

    message(0, "Total number of groups with at least %d particles: %ld\n", fof_params.FOFHaloMinLength, fof->TotNgroups);
    if(fof->TotNgroups > 0)
    {
        message(0, "Largest group has %d particles, mass %g.\n", largestloc_tot, largestmass_tot);
        message(0, "Total number of particles in groups: %012ld\n", TotNids);
    }
}


static void fof_reduce_groups(
    void * groups,
    int nmemb,
    size_t elsize,
    void (*reduce_group)(void * gdst, void * gsrc), MPI_Comm Comm)
{

    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    /* slangs:
     *   prime: groups hosted by ThisTask
     *   ghosts: groups that spans into ThisTask but not hosted by ThisTask;
     *           part of the local catalogue
     *   images: ghosts that are sent from another rank.
     *           images are reduced to prime, then the prime attributes
     *           are copied to images, and sent back to the ghosts.
     *
     *   in the begining, prime and ghosts contains local group attributes.
     *   in the end, prime and ghosts all contain full group attributes.
     **/
    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    void * images = NULL;
    void * ghosts = NULL;
    int i;
    int start;

    MPI_Datatype dtype;

    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    /*Set global data for the comparison*/
    _fof_compare_Group_MinIDTask_ThisTask = ThisTask;
    /* local groups will be moved to the beginning, we skip them with offset */
    qsort_openmp(groups, nmemb, elsize, fof_compare_Group_MinIDTask);
    /* count how many we have of each task */
    memset(Send_count, 0, sizeof(int) * NTask);

    for(i = 0; i < nmemb; i++) {
        struct BaseGroup * gi = (struct BaseGroup *) (((char*) groups) + i * elsize);
        Send_count[gi->MinIDTask]++;
    }

    /* Skip local groups */
    int Nmine = Send_count[ThisTask];
    Send_count[ThisTask] = 0;

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int nimport = 0;
    for(i = 0; i < NTask; i ++) {
        nimport += Recv_count[i];
    }

    images = mymalloc("images", nimport * elsize);
    ghosts = ((char*) groups) + elsize * Nmine;

    MPI_Alltoallv_smart(ghosts, Send_count, NULL, dtype,
                        images, Recv_count, NULL, dtype, Comm);

    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        gi->OriginalIndex = i;
    }

    /* sort the groups according to MinID */
    qsort_openmp(groups, Nmine, elsize, fof_compare_Group_MinID);
    qsort_openmp(images, nimport, elsize, fof_compare_Group_MinID);

    /* merge the imported ones with the local ones */
    start = 0;
    for(i = 0; i < Nmine; i++) {
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start  * elsize);
            if(image->MinID >= prime->MinID) {
                break;
            }
        }
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID != prime->MinID) {
                break;
            }
            reduce_group(prime, image);
        }
    }

    /* update the images, such that they can be send back to the ghosts */
    start = 0;
    for(i = 0; i < Nmine; i++)
    {
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID >= prime->MinID) {
                break;
            }
        }
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID != prime->MinID) {
                break;
            }
            int save = image->OriginalIndex;
            memcpy(image, prime, elsize);
            image->OriginalIndex = save;
        }
    }

    /* reset the ordering of imported list, such that it can be properly returned */
    qsort_openmp(images, nimport, elsize, fof_compare_Group_OriginalIndex);
#ifdef DEBUG
    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        if(gi->MinIDTask != ThisTask) {
            endrun(5, "Error in basegroup import: minidtask %d != ThisTask %d\n", gi->MinIDTask, ThisTask);
        }
    }
#endif
    void * ghosts2 = mymalloc("TMP", nmemb * elsize);

    MPI_Alltoallv_smart(images, Recv_count, NULL, dtype,
                        ghosts2, Send_count, NULL, dtype,
                        Comm);
    for(i = 0; i < nmemb - Nmine; i ++) {
        struct BaseGroup * g1 = (struct BaseGroup*) ((char*) ghosts + i * elsize);
        struct BaseGroup * g2 = (struct BaseGroup*) ((char*) ghosts2 + i* elsize);
        if(g1->MinID != g2->MinID) {
            endrun(2, "g1 minID %lu, g2 minID %lu\n", g1->MinID, g2->MinID);
        }
        if(g1->MinIDTask != g2->MinIDTask) {
            endrun(2, "g1 minIDTask %d, g2 minIDTask %d\n", g1->MinIDTask, g2->MinIDTask);
        }
    }
    memcpy(ghosts, ghosts2, elsize * (nmemb - Nmine));
    myfree(ghosts2);

    myfree(images);

    MPI_Type_free(&dtype);

    /* At this point, each Group entry has the reduced attribute of the full group */
    /* And the local groups (MinIDTask == ThisTask) are placed at the begining of the list*/
    ta_free(Recv_count);
    ta_free(Send_count);
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg);
static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm)
{
    int i, j, NTask, ThisTask;
    int64_t ngr;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);

    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i++)
    {
        base[i].OriginalTask = ThisTask;	/* original task */
    }

    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_TotalCountTaskDiffMinID, 24, NULL, Comm);

    /* assign group numbers
     * at this point, both Group are is sorted by length,
     * and the every time OriginalTask == MinIDTask, a list of ghost base is stored.
     * they shall get the same GrNr.
     * */
    ngr = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].OriginalTask == base[i].MinIDTask) {
            ngr++;
        }
        base[i].GrNr = ngr;
    }

    int64_t * ngra = ta_malloc("NGRA", int64_t, NTask);

    MPI_Allgather(&ngr, 1, MPI_INT64, ngra, 1, MPI_INT64, Comm);

    /* shift to the global grnr. */
    int64_t groffset = 0;
    #pragma omp parallel for reduction(+: groffset)
    for(j = 0; j < ThisTask; j++)
        groffset += ngra[j];
    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i++)
        base[i].GrNr += groffset;

    ta_free(ngra);

    /* bring the group list back into the original task, sorted by MinID */
    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_OriginalTaskMinID, 16, NULL, Comm);
}

int
fof_save_groups(FOFGroups * fof, const char * OutputDir, const char * FOFFileBase, int num, Cosmology * CP, double atime, const double * MassTable, int MetalReturnOn, const int OutputDebugFields, MPI_Comm Comm)
{
    char * fname = fastpm_strdup_printf("%s/%s_%03d", OutputDir, FOFFileBase, num);
    message(0, "Saving particle groups into %s\n", fname);

    return fof_save_particles(fof, fname, fof_params.FOFSaveParticles, CP, atime, MassTable, MetalReturnOn, OutputDebugFields, Comm);
}

/* FIXME: these shall goto the private member of secondary tree walk */
struct FOFSecondaryPriv {
    float *distance;
    float *hsml;
    int64_t *npleft;
    struct fof_particle_list * HaloLabel;
};

#define FOF_SECONDARY_GET_PRIV(tw) ((struct FOFSecondaryPriv *) (tw->priv))

static void fof_secondary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {

    I->Hsml = FOF_SECONDARY_GET_PRIV(tw)->hsml[place];
    I->MinID = FOF_SECONDARY_GET_PRIV(tw)->HaloLabel[place].MinID;
    I->MinIDTask = FOF_SECONDARY_GET_PRIV(tw)->HaloLabel[place].MinIDTask;
}

static int fof_secondary_haswork(int n, TreeWalk * tw) {
    if(P[n].IsGarbage || P[n].Swallowed)
        return 0;
    /* Exclude particles where we already found a neighbour*/
    if(FOF_SECONDARY_GET_PRIV(tw)->distance[n] < 0.5 * LARGE)
        return 0;
    return ((1 << P[n].Type) & fof_params.FOFSecondaryLinkTypes);
}
static void fof_secondary_reduce(int place, TreeWalkResultFOF * O, enum TreeWalkReduceMode mode, TreeWalk * tw) {
    if(O->Distance < FOF_SECONDARY_GET_PRIV(tw)->distance[place] && O->Distance >= 0 && O->Distance < 0.5 * LARGE)
    {
        FOF_SECONDARY_GET_PRIV(tw)->distance[place] = O->Distance;
        FOF_SECONDARY_GET_PRIV(tw)->HaloLabel[place].MinID = O->MinID;
        FOF_SECONDARY_GET_PRIV(tw)->HaloLabel[place].MinIDTask = O->MinIDTask;
    }
}

static void
fof_secondary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        O->Distance = LARGE;
        O->MinID = I->MinID;
        O->MinIDTask = I->MinIDTask;
        iter->base.Hsml = I->Hsml;
        iter->base.mask = fof_params.FOFPrimaryLinkTypes;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    int other = iter->base.other;
    double r = iter->base.r;
    /* Don't attach a secondary particle to a seeded star left in the tree when
     * FOFPrimaryUnseededStarsOnly is set: seeded stars are not primary anchors. */
    if(!fof_is_primary_link(other))
        return;
    if(r < O->Distance)
    {
        O->Distance = r;
        O->MinID = FOF_SECONDARY_GET_PRIV(lv->tw)->HaloLabel[other].MinID;
        O->MinIDTask = FOF_SECONDARY_GET_PRIV(lv->tw)->HaloLabel[other].MinIDTask;
    }
    /* No need to search nodes at a greater distance
     * now that we have a neighbour.*/
    iter->base.Hsml = iter->base.r;
}

static void
fof_secondary_postprocess(int p, TreeWalk * tw)
{
    /* More work needed: add this particle to the redo queue*/
    const int tid = omp_get_thread_num();

    if(FOF_SECONDARY_GET_PRIV(tw)->distance[p] > 0.5 * LARGE)
    {
        if(FOF_SECONDARY_GET_PRIV(tw)->hsml[p] < 4 * fof_params.FOFHaloComovingLinkingLength)  /* we only search out to a maximum distance */
        {
            /* need to redo this particle */
            FOF_SECONDARY_GET_PRIV(tw)->npleft[tid]++;
            FOF_SECONDARY_GET_PRIV(tw)->hsml[p] *= 2.0;
/*
            if(iter >= MAXITER - 10)
            {
                endrun(1, "i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)\n",
                        p, ThisTask, P[p].ID, FOF_SECONDARY_GET_PRIV(tw)->hsml[p],
                        P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
            }
*/
        } else {
            FOF_SECONDARY_GET_PRIV(tw)->distance[p] = -1;  /* we not continue to search for this particle */
        }
    }
}

static void fof_label_secondary(struct fof_particle_list * HaloLabel, ForceTree * tree)
{
    int n;

    TreeWalk tw[1] = {{0}};
    tw->ev_label = "FOF_FIND_NEAREST";
    tw->visit = treewalk_visit_nolist_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_secondary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);
    tw->haswork = fof_secondary_haswork;
    tw->fill = (TreeWalkFillQueryFunction) fof_secondary_copy;
    tw->reduce = (TreeWalkReduceResultFunction) fof_secondary_reduce;
    tw->postprocess = (TreeWalkProcessFunction) fof_secondary_postprocess;
    tw->type = TREEWALK_ALL;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);
    tw->tree = tree;
    struct FOFSecondaryPriv priv[1];

    tw->priv = priv;

    message(0, "Start finding nearest dm-particle (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    FOF_SECONDARY_GET_PRIV(tw)->distance = (float *) mymalloc("FOF_SECONDARY->distance", sizeof(float) * PartManager->NumPart);
    FOF_SECONDARY_GET_PRIV(tw)->hsml = (float *) mymalloc("FOF_SECONDARY->hsml", sizeof(float) * PartManager->NumPart);
    FOF_SECONDARY_GET_PRIV(tw)->HaloLabel = HaloLabel;

    #pragma omp parallel for
    for(n = 0; n < PartManager->NumPart; n++)
    {
        FOF_SECONDARY_GET_PRIV(tw)->distance[n] = LARGE;
        FOF_SECONDARY_GET_PRIV(tw)->hsml[n] = 0.4 * fof_params.FOFHaloComovingLinkingLength;

        if((P[n].Type == 0 || P[n].Type == 4 || P[n].Type == 5) && FOF_SECONDARY_GET_PRIV(tw)->hsml[n] < 0.5 * P[n].Hsml) {
            /* use gas sml as a hint (faster convergence than 0.1 fof_params.FOFHaloComovingLinkingLength at high-z */
            FOF_SECONDARY_GET_PRIV(tw)->hsml[n] = 0.5 * P[n].Hsml;
        }
    }

    int64_t ntot;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

    message(0, "fof-nearest iteration started\n");
    const int NumThreads = omp_get_max_threads();
    FOF_SECONDARY_GET_PRIV(tw)->npleft = ta_malloc("NPLeft", int64_t, NumThreads);

    do
    {
        memset(FOF_SECONDARY_GET_PRIV(tw)->npleft, 0, sizeof(int64_t) * NumThreads);

        treewalk_run(tw, NULL, PartManager->NumPart);

        for(n = 1; n < NumThreads; n++) {
            FOF_SECONDARY_GET_PRIV(tw)->npleft[0] += FOF_SECONDARY_GET_PRIV(tw)->npleft[n];
        }
        MPI_Allreduce(&FOF_SECONDARY_GET_PRIV(tw)->npleft[0], &ntot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

        if(ntot < 0 || (ntot > 0 && tw->Niteration > MAXITER))
            endrun(1159, "Failed to converge in fof-nearest: ntot %ld", ntot);
    }
    while(ntot > 0);

    ta_free(FOF_SECONDARY_GET_PRIV(tw)->npleft);
    myfree(FOF_SECONDARY_GET_PRIV(tw)->hsml);
    myfree(FOF_SECONDARY_GET_PRIV(tw)->distance);
}

/*
 * Deal with seeding of particles At each FOF stage,
 * if seed_index is >= 0,  then that particle on seed_task
 * will be converted to a seed.
 *
 * */
static int cmp_seed_task(const void * c1, const void * c2) {
    const struct Group * g1 = (const struct Group *) c1;
    const struct Group * g2 = (const struct Group *) c2;

    return g1->seed_task - g2->seed_task;
}

/* Per-secFOF multi-seed count.  N_seed = floor(M_SC/thresh) (thresh = 1e8 Msun),
 * plus one extra seed when the surplus mass (M_SC - floor(M_SC/thresh)*thresh) is
 * itself seedable (>= MinMscForBHseed).  This avoids a single BH growing past the
 * threshold mass when thresh < M_SC < 2*thresh.  N_seed is capped by the number of
 * unseeded stars and is at least 1 (groups below the threshold return 1 = ordinary
 * single seed).  The seeds are NOT equal mass (see secfof_seed_scaling); only
 * meaningful for SC seeds. */
static int secfof_compute_nseed(const struct Group * g)
{
    if(!fof_params.SeedInSecFOFMultipleSeeds)
        return 1;
    double Msc;
    if(fof_params.SeedSecFOFcomSample)
        Msc = g->BHSeedMsc;
    else
        Msc = fof_params.StarClusterSampling ? g->StarClusterMassSampleUnseeded
                                             : g->StarClusterMassUnseeded;
    double thresh = get_msc_multiseed_thresh_code();
    if(thresh <= 0)
        return 1;
    int n = (int) floor(Msc / thresh);
    if(n < 1)
        return 1;
    /* Add one more seed if the leftover above n*thresh is itself seedable. */
    double rem = Msc - (double) n * thresh;
    if(rem >= fof_params.MinMscForBHseed)
        n += 1;
    if(g->NStarUnseeded > 0 && n > g->NStarUnseeded)
        n = g->NStarUnseeded;
    if(n < 1) n = 1;
    return n;
}

/* Seed-mass scaling (the M_SC-equivalent passed as ScalingMass to blackhole_make_one)
 * for the rank-th seed of a multi-seed group, rank>=1.  "First capped, rest equal":
 * seed 1 carries exactly the threshold mass (1e8 Msun in code units) so its BH mass is
 * SeedBlackHoleMass*thresh; the remaining nseed-1 seeds split the surplus (M_SC-thresh)
 * equally.  Groups that do not actually multi-seed (floor(M_SC/thresh) < 1, or nseed<=1)
 * keep the ordinary full-M_SC scaling.  Mass is conserved: thresh + (nseed-1)*extra = M_SC. */
static double secfof_seed_scaling(double Msc, int nseed, int rank, double thresh)
{
    if(thresh <= 0 || nseed <= 1)
        return Msc;
    if((int) floor(Msc / thresh) < 1)
        return Msc;
    if(rank <= 1)
        return thresh;
    return (Msc - thresh) / (nseed - 1);
}

/* Fraction of the group's StarClusterMass payload (and init_Msc / init_Msc_sample
 * records) carried by the rank-th seed.  "Proportional to seed mass": with
 * BHseedMassScaleMsc the seed mass scales with M_SC, so the share = scaling/M_SC;
 * otherwise all seeds are equal mass and the payload splits evenly (1/nseed). */
static double secfof_seed_massfrac(double Msc, int nseed, int rank, double thresh)
{
    if(nseed <= 1)
        return 1.0;
    if(fof_params.BHseedMassScaleMsc) {
        if(Msc <= 0)
            return 1.0 / nseed;
        return secfof_seed_scaling(Msc, nseed, rank, thresh) / Msc;
    }
    return 1.0 / nseed;
}

static void fof_seed_make_one(struct Group * g, int ThisTask, const double atime, const RandTable * const rnd) {
   if(g->seed_task != ThisTask) {
        endrun(7771, "Seed does not belong to the right task");
    }
    int index = g->seed_index;

    /* scaling_mass: scales the seed mass when BHseedMassScaleMsc=1.
     * payload_mass: attached to the BH as BHP.StarClusterMass. */
    MyFloat scaling_mass, payload_mass;
    /* Star-cluster mass that seeded the BH, recorded on the BH slot (frozen
     * across mergers). init_msc = cluster-forming mass (Sum star_mass*Gamma);
     * init_msc_sample = the mass sampled from the cluster mass function. Both 0
     * for non star-cluster (gas/halo) seeds. */
    MyFloat init_msc = 0, init_msc_sample = 0;
    /* Debug-only record: total mass of the host secFOF's unseeded star particles
     * (= Mcut, the SCmasscapSecFOFstarmass cap value). Only meaningful in the
     * combined-sample mode; 0 for all other seeding paths. */
    MyFloat capped_star_mass = 0;
    int seeded_by_starcluster;
    if(fof_params.SeedSecFOFcomSample) {
        /* Combined-sample seeding: Gate 2 already passed in fof_seed.
         * Seed mass from bhseed_msc; attach the full unseeded cluster-forming
         * mass only when StarClusterBHDyn is on. */
        seeded_by_starcluster = 1;
        scaling_mass = g->BHSeedMsc;
        payload_mass = get_starcluster_bhdyn_on() ? g->StarClusterMassUnseeded : 0;
        /* init_Msc = cluster-forming mass of ALL consumed stars; init_Msc_sample =
         * the full sampled cluster mass for those stars (not just the >1e4 part
         * that sets the seed mass). */
        init_msc = g->StarClusterMassUnseeded;
        init_msc_sample = g->BHSampledMscTotal;
        capped_star_mass = g->SCcomMcut;
    } else {
        /* Select which star cluster mass to use based on StarClusterSampling.
         * Must use the same (unseeded) mass variable as the marking code in fof_seed. */
        MyFloat sc_mass = fof_params.StarClusterSampling ? g->StarClusterMassSampleUnseeded : g->StarClusterMassUnseeded;
        seeded_by_starcluster = fof_params.BlackHoleSeedStarCluster
            && (sc_mass >= fof_params.MinMscForBHseed);
        scaling_mass = sc_mass;
        payload_mass = sc_mass;
        if(seeded_by_starcluster) {
            init_msc = g->StarClusterMassUnseeded;
            init_msc_sample = g->StarClusterMassSampleUnseeded;
        }
    }
    /* Per-secFOF multi-seeding: a massive group (M_SC > 1e8 Msun) seeds N_seed BHs.
     * This is seed 1 (the largest-m*Gamma star).  "First capped, rest equal": seed 1
     * carries exactly the threshold mass (1e8 Msun-equivalent); the remaining N_seed-1
     * extras (placed by fof_secfof_extra_seeds() after the single-seed loop) split the
     * surplus.  The attached cluster mass / init records follow the seed-mass share. */
    if(seeded_by_starcluster) {
        int nseed = secfof_compute_nseed(g);
        if(nseed > 1) {
            double thresh = get_msc_multiseed_thresh_code();
            double Msc = scaling_mass;     /* full seeding cluster mass M_SC */
            double frac = secfof_seed_massfrac(Msc, nseed, 1, thresh);
            scaling_mass = secfof_seed_scaling(Msc, nseed, 1, thresh);
            payload_mass *= frac;
            init_msc *= frac;
            init_msc_sample *= frac;
        }
    }

    /* Compute mass-weighted average metallicity for star cluster */
    MyFloat sc_metallicity = 0;
    float sc_metals[NMETALS] = {0};
    if(g->StarClusterMass > 0) {
        sc_metallicity = g->StarClusterMetallicity / g->StarClusterMass;
        int j;
        for(j = 0; j < NMETALS; j++)
            sc_metals[j] = g->StarClusterMetalElemMass[j] / g->StarClusterMass;
    }
    /* Debug-only record: number of BH particles already in the host secFOF group
     * at seeding (LenType[5], excludes this seed).  Only meaningful for the secFOF
     * seed path (SeedInSecFOFasStarCluster); 0 for the primary-FOF seed path. */
    int bh_ngb_at_seeding = get_seed_in_secfof() ? g->LenType[5] : 0;
    blackhole_make_one(index, atime, rnd, seeded_by_starcluster, payload_mass, scaling_mass, init_msc, init_msc_sample, capped_star_mass, bh_ngb_at_seeding, sc_metallicity, sc_metals, 0);

    /* StarClusterDetails: one record per star-cluster seed (no-op unless enabled).
     * scaling_mass is this seed's cluster mass (com: bhseed_msc; else the mode's
     * seeding SC mass), already carrying the multi-seed share for seed 1. */
    if(seeded_by_starcluster) {
        double sc_met = g->SCClusterMassUnseededInit > 0 ?
            g->SCMetalMassUnseeded / g->SCClusterMassUnseededInit : 0;
        struct SCmetdist md;
        sc_met_unseeded_stats(g, &md);
        struct SCboundinfo bd;
        sc_bound_stats(g, &bd);
        /* Reff = 0: not the per-cluster (SecFOFseedsumover=0) path.
         * Mbh_seed = the just-made BH's subgrid mass (the particle at index was
         * converted in place by blackhole_make_one). */
        scinfo_record_seed(index, atime, scaling_mass, g->StarClusterMass,
                           g->MassType[4], g->SCMass_seeded, sc_met, 0, BHP(index).Mass,
                           g->LenType[5], g->base.GrNr, &md, &bd, SC_FLAG_SEEDED);
    }
}

/* ===================== Per-secFOF multi-seeding (M_SC > 1e8 Msun) =====================
 * A secondary-FOF group whose seeding star-cluster mass M_SC exceeds 1e8 Msun seeds
 * N_seed = floor(M_SC/1e8) equal-mass BHs (capped by the number of unseeded stars).
 * Seed 1 is the largest-m*Gamma star (handled by fof_seed_make_one).  Seeds 2..N are
 * the next-largest-m*Gamma unseeded stars lying farther than 2*GravitySoftening from
 * seed 1; each is converted in place into an equal-mass BH.  Stars are considered in
 * descending m*Gamma order; if too few are far enough, the missing seeds are skipped
 * (logged).  The separation criterion is measured from seed 1 only. */

/* Whether this (owned, reduced) group seeds a BH via the star-cluster path from a
 * star particle (the case that supports multi-seeding).  Mirrors the SC gate in
 * fof_seed and requires the chosen seed to be the star (not a gas particle). */
static int secfof_group_sc_seeds(const struct Group * g)
{
    if(!fof_params.BlackHoleSeedStarCluster)
        return 0;
    if(g->seed_index_star < 0)
        return 0;
    /* Seed 1 must be the star itself (in secFOF there is no gas, so this always holds;
     * in a mixed group it excludes a gas-particle seed). */
    if(g->seed_index != g->seed_index_star || g->seed_task != g->seed_task_star)
        return 0;
    if(fof_params.SeedSecFOFcomSample)
        return g->BHSeedMsc >= fof_params.MinMscForBHseed;
    MyFloat sc_mass = fof_params.StarClusterSampling ? g->StarClusterMassSampleUnseeded
                                                     : g->StarClusterMassUnseeded;
    return sc_mass >= fof_params.MinMscForBHseed;
}

static int cmp_int64_asc(const void * a, const void * b)
{
    int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return (x > y) - (x < y);
}

/* One multi-seed group, gathered to every rank. */
struct ms_group {
    int64_t  GrNr;
    int      Nseed;             /* target number of seeds (>= 2) */
    int      seed_task_star;    /* rank owning seed 1 */
    int      seed_index_star;   /* local index of seed 1 on seed_task_star */
    uint64_t SeedStarID;        /* ID of seed 1 (excluded from extra candidates) */
    double   seed1pos[3];       /* position of seed 1 (filled by its owner) */
    double   Msc;               /* full seeding cluster mass M_SC (for logging) */
    double   per_scaling;       /* seed-mass scaling for each extra seed (rank>=2) */
    double   per_payload;       /* StarClusterMass payload per extra seed */
    double   per_init_msc;      /* init_Msc per extra seed */
    double   per_init_msc_sample; /* init_Msc_sample per extra seed */
    double   capped;            /* SCcomMcut (group; recorded as-is) */
    int      bh_ngb;            /* BHNgbAtSeeding: BH count in the host secFOF (LenType[5]) */
    double   metallicity;       /* group mass-weighted metallicity */
    float    metals[NMETALS];
    /* StarClusterDetails record fields (per host group). */
    double   sc_mass_total;     /* group total Sum(Gamma*m_star) over all stars (StarClusterMass) */
    double   stellar_mass_total; /* group total Sum(m_star) over all stars (MassType[4]) */
    double   scmass_seeded;     /* group consumed SC mass (SCMass_seeded, as-is) */
    double   met_unseeded;      /* unseeded-star metal mass ratio (StarClusterDetails metallicity) */
    struct SCmetdist metdist;   /* unseeded-star metallicity distribution (min/max/median/quartiles/std) */
    struct SCboundinfo boundinfo; /* BHseedSecFOFbound bound-star summary (all zero when off) */
};

static int cmp_ms_group_grnr(const void * a, const void * b)
{
    int64_t x = ((const struct ms_group *) a)->GrNr, y = ((const struct ms_group *) b)->GrNr;
    return (x > y) - (x < y);
}

/* Binary search the (GrNr-sorted) MSG array; returns index or -1. */
static int ms_find(const struct ms_group * msg, int n, int64_t gr)
{
    int lo = 0, hi = n;
    while(lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if(msg[mid].GrNr == gr) return mid;
        else if(msg[mid].GrNr < gr) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

/* One eligible extra-seed candidate star (far enough from seed 1). */
struct ms_cand {
    int64_t  GrNr;
    double   mGamma;       /* ClusterMass = m_star*Gamma (ranking key) */
    uint64_t ID;           /* tiebreak / identity */
    double   pos[3];       /* position (for logging the seed locations) */
    int      owner_task;
    int      local_index;
};

/* Order: GrNr asc, then m*Gamma desc, then ID asc (deterministic across ranks). */
static int cmp_ms_cand(const void * a, const void * b)
{
    const struct ms_cand * x = (const struct ms_cand *) a;
    const struct ms_cand * y = (const struct ms_cand *) b;
    if(x->GrNr != y->GrNr) return (x->GrNr > y->GrNr) - (x->GrNr < y->GrNr);
    if(x->mGamma != y->mGamma) return (x->mGamma < y->mGamma) - (x->mGamma > y->mGamma);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* (GrNr, Nseed) for one multi-seed group, gathered to every rank so the local
 * extra-seed upper bound can be computed with the per-group (Nseed-1) cap. */
struct ms_seed_cap {
    int64_t GrNr;
    int     Nseed;      /* target number of seeds (>= 2) */
    int     placed;     /* running local count, capped at Nseed-1 */
};
static int cmp_ms_seed_cap_grnr(const void * a, const void * b)
{
    int64_t x = ((const struct ms_seed_cap *) a)->GrNr, y = ((const struct ms_seed_cap *) b)->GrNr;
    return (x > y) - (x < y);
}

/* Upper bound on the number of EXTRA (2nd..N_seed) seeds this rank will convert,
 * used to pre-reserve BH slots while the ActiveParticle/tree juggling is still safe.
 * Builds the global set of multi-seed groups (GrNr, Nseed) and counts local
 * unseeded member stars in those groups, capped at (Nseed-1) per group: a group
 * cannot receive more than Nseed-1 extra seeds regardless of how many stars it
 * holds, so without the cap a single massive group can inflate the request by
 * thousands of slots and force an unnecessary slots grow.  All scratch is freed
 * before returning (LIFO). */
static int64_t secfof_count_extra_seed_ub(FOFGroups * fof, MPI_Comm Comm)
{
    if(!fof_params.SeedInSecFOFMultipleSeeds || !fof_params.BlackHoleSeedStarCluster)
        return 0;
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int64_t i;
    int t;

    int n_local = 0;
    for(i = 0; i < fof->Ngroups; i++)
        if(secfof_group_sc_seeds(&fof->Group[i]) && secfof_compute_nseed(&fof->Group[i]) >= 2)
            n_local++;

    int * rc = (int *) mymalloc("MSubRC", NTask * sizeof(int));
    MPI_Allgather(&n_local, 1, MPI_INT, rc, 1, MPI_INT, Comm);
    int n_tot = 0;
    for(t = 0; t < NTask; t++) n_tot += rc[t];
    if(n_tot == 0) { myfree(rc); return 0; }

    int * bc = (int *) mymalloc("MSubBC", NTask * sizeof(int));
    int * bd = (int *) mymalloc("MSubBD", NTask * sizeof(int));
    int boff = 0;
    for(t = 0; t < NTask; t++) { bc[t] = rc[t] * (int) sizeof(struct ms_seed_cap); bd[t] = boff; boff += bc[t]; }

    struct ms_seed_cap * local_g = (struct ms_seed_cap *) mymalloc("MSubLocal",
            (n_local > 0 ? n_local : 1) * sizeof(struct ms_seed_cap));
    int k = 0;
    for(i = 0; i < fof->Ngroups; i++)
        if(secfof_group_sc_seeds(&fof->Group[i]) && secfof_compute_nseed(&fof->Group[i]) >= 2) {
            local_g[k].GrNr   = fof->Group[i].base.GrNr;
            local_g[k].Nseed  = secfof_compute_nseed(&fof->Group[i]);
            local_g[k].placed = 0;
            k++;
        }
    struct ms_seed_cap * all_g = (struct ms_seed_cap *) mymalloc("MSubAll", n_tot * sizeof(struct ms_seed_cap));
    MPI_Allgatherv(local_g, n_local * (int) sizeof(struct ms_seed_cap), MPI_BYTE,
                   all_g, bc, bd, MPI_BYTE, Comm);
    qsort(all_g, n_tot, sizeof(struct ms_seed_cap), cmp_ms_seed_cap_grnr);

    int64_t cnt = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].Type != 4 || P[i].GrNr < 0 || STARP(i).Seeded) continue;
        int lo = 0, hi = n_tot, found = -1;
        int64_t key = P[i].GrNr;
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if(all_g[mid].GrNr == key) { found = mid; break; }
            else if(all_g[mid].GrNr < key) lo = mid + 1;
            else hi = mid;
        }
        if(found >= 0 && all_g[found].placed < all_g[found].Nseed - 1) {
            all_g[found].placed++;
            cnt++;
        }
    }
    myfree(all_g);
    myfree(local_g);
    myfree(bd);
    myfree(bc);
    myfree(rc);
    return cnt;
}

/* Place the 2nd..N_seed seeds for massive secondary-FOF groups.  BH slots were
 * pre-reserved by the caller (no reservation/ActiveParticle juggling here).  Called
 * after the single-seed loop, while ImportGroups (mymalloc2/high stack) is alive;
 * all scratch here is mymalloc (low stack) and freed in LIFO order. */
static void fof_secfof_extra_seeds(FOFGroups * fof, double atime, const RandTable * const rnd, MPI_Comm Comm)
{
    if(!fof_params.SeedInSecFOFMultipleSeeds || !fof_params.BlackHoleSeedStarCluster)
        return;

    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    int64_t i;
    int t;
    const int bhdyn = get_starcluster_bhdyn_on();

    /* ---- Phase 1: build the global multi-seed group list (MSG). ---- */
    int n_local = 0;
    for(i = 0; i < fof->Ngroups; i++)
        if(secfof_group_sc_seeds(&fof->Group[i]) && secfof_compute_nseed(&fof->Group[i]) >= 2)
            n_local++;

    int * rc = (int *) mymalloc("MSrc", NTask * sizeof(int));
    MPI_Allgather(&n_local, 1, MPI_INT, rc, 1, MPI_INT, Comm);
    int n_msg = 0;
    for(t = 0; t < NTask; t++) n_msg += rc[t];
    if(n_msg == 0) { myfree(rc); return; }

    int * bc = (int *) mymalloc("MSbc", NTask * sizeof(int));
    int * bd = (int *) mymalloc("MSbd", NTask * sizeof(int));
    int boff = 0;
    for(t = 0; t < NTask; t++) { bc[t] = rc[t] * (int) sizeof(struct ms_group); bd[t] = boff; boff += bc[t]; }

    struct ms_group * local_msg = (struct ms_group *) mymalloc("MSlocal",
            (n_local > 0 ? n_local : 1) * sizeof(struct ms_group));
    int k = 0;
    for(i = 0; i < fof->Ngroups; i++) {
        struct Group * g = &fof->Group[i];
        if(!(secfof_group_sc_seeds(g) && secfof_compute_nseed(g) >= 2)) continue;
        int nseed = secfof_compute_nseed(g);
        struct ms_group * m = &local_msg[k++];
        memset(m, 0, sizeof(*m));
        m->GrNr = g->base.GrNr;
        m->Nseed = nseed;
        m->seed_task_star = g->seed_task_star;
        m->seed_index_star = g->seed_index_star;
        m->SeedStarID = (uint64_t) g->SeedStarID;
        /* BH count already in the host secFOF (excludes the seeds placed here).
         * Always the secFOF path here (multi-seed requires SeedInSecFOFasStarCluster);
         * gated for consistency with fof_seed_make_one. */
        m->bh_ngb = get_seed_in_secfof() ? g->LenType[5] : 0;
        /* Per-extra-seed (rank>=2) masses; all extras are uniform under the
         * "first capped, rest equal" scheme (seed 1 is handled separately). */
        double Msc, payload, init_msc, init_msc_sample;
        if(fof_params.SeedSecFOFcomSample) {
            Msc = g->BHSeedMsc;
            payload = bhdyn ? g->StarClusterMassUnseeded : 0;
            init_msc = g->StarClusterMassUnseeded;
            init_msc_sample = g->BHSampledMscTotal;
            m->capped = g->SCcomMcut;
        } else {
            double sc_mass = fof_params.StarClusterSampling ? g->StarClusterMassSampleUnseeded
                                                            : g->StarClusterMassUnseeded;
            Msc = sc_mass;
            payload = sc_mass;
            init_msc = g->StarClusterMassUnseeded;
            init_msc_sample = g->StarClusterMassSampleUnseeded;
            m->capped = 0;
        }
        double thresh = get_msc_multiseed_thresh_code();
        double frac = secfof_seed_massfrac(Msc, nseed, 2, thresh);
        m->Msc = Msc;
        m->per_scaling = secfof_seed_scaling(Msc, nseed, 2, thresh);
        m->per_payload = payload * frac;
        m->per_init_msc = init_msc * frac;
        m->per_init_msc_sample = init_msc_sample * frac;
        if(g->StarClusterMass > 0) {
            m->metallicity = g->StarClusterMetallicity / g->StarClusterMass;
            int j;
            for(j = 0; j < NMETALS; j++)
                m->metals[j] = g->StarClusterMetalElemMass[j] / g->StarClusterMass;
        }
        /* StarClusterDetails record fields (per host group). */
        m->sc_mass_total = g->StarClusterMass;
        m->stellar_mass_total = g->MassType[4];
        m->scmass_seeded = g->SCMass_seeded;
        m->met_unseeded = g->SCClusterMassUnseededInit > 0 ?
            g->SCMetalMassUnseeded / g->SCClusterMassUnseededInit : 0;
        sc_met_unseeded_stats(g, &m->metdist);
        sc_bound_stats(g, &m->boundinfo);
    }
    struct ms_group * msg = (struct ms_group *) mymalloc("MSG", n_msg * sizeof(struct ms_group));
    MPI_Allgatherv(local_msg, n_local * (int) sizeof(struct ms_group), MPI_BYTE,
                   msg, bc, bd, MPI_BYTE, Comm);
    qsort(msg, n_msg, sizeof(struct ms_group), cmp_ms_group_grnr);

    /* ---- Phase 2: fill seed-1 positions (owner of seed_index_star contributes). ---- */
    double * pos = (double *) mymalloc("MSpos", 3 * n_msg * sizeof(double));
    memset(pos, 0, 3 * n_msg * sizeof(double));
    for(i = 0; i < n_msg; i++) {
        if(msg[i].seed_task_star == ThisTask) {
            int si = msg[i].seed_index_star;
            pos[3 * i + 0] = P[si].Pos[0];
            pos[3 * i + 1] = P[si].Pos[1];
            pos[3 * i + 2] = P[si].Pos[2];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, pos, 3 * n_msg, MPI_DOUBLE, MPI_SUM, Comm);
    for(i = 0; i < n_msg; i++) {
        msg[i].seed1pos[0] = pos[3 * i + 0];
        msg[i].seed1pos[1] = pos[3 * i + 1];
        msg[i].seed1pos[2] = pos[3 * i + 2];
    }
    myfree(pos);

    /* ---- Phase 3: collect local eligible candidates (> 2*GravitySoftening from seed 1). ---- */
    const double eps = FORCE_SOFTENING() / 2.8;   /* GravitySoftening (Plummer-equivalent) */
    const double sep2 = (2.0 * eps) * (2.0 * eps);
    const double box = PartManager->BoxSize;

    int n_elig = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].Type != 4 || P[i].GrNr < 0 || STARP(i).Seeded) continue;
        int c = ms_find(msg, n_msg, P[i].GrNr);
        if(c < 0) continue;
        if((uint64_t) P[i].ID == msg[c].SeedStarID) continue;
        double dx = NEAREST(P[i].Pos[0] - msg[c].seed1pos[0], box);
        double dy = NEAREST(P[i].Pos[1] - msg[c].seed1pos[1], box);
        double dz = NEAREST(P[i].Pos[2] - msg[c].seed1pos[2], box);
        if(dx * dx + dy * dy + dz * dz <= sep2) continue;
        n_elig++;
    }
    struct ms_cand * elig = (struct ms_cand *) mymalloc("MSelig",
            (n_elig > 0 ? n_elig : 1) * sizeof(struct ms_cand));
    int e = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].Type != 4 || P[i].GrNr < 0 || STARP(i).Seeded) continue;
        int c = ms_find(msg, n_msg, P[i].GrNr);
        if(c < 0) continue;
        if((uint64_t) P[i].ID == msg[c].SeedStarID) continue;
        double dx = NEAREST(P[i].Pos[0] - msg[c].seed1pos[0], box);
        double dy = NEAREST(P[i].Pos[1] - msg[c].seed1pos[1], box);
        double dz = NEAREST(P[i].Pos[2] - msg[c].seed1pos[2], box);
        if(dx * dx + dy * dy + dz * dz <= sep2) continue;
        elig[e].GrNr = P[i].GrNr;
        /* Rank extra-seed candidates by the f(Z)-scaled cluster mass. */
        elig[e].mGamma = get_seed_metallicity_factor(STARP(i).BirthMetallicity) * STARP(i).ClusterMass;
        elig[e].ID = (uint64_t) P[i].ID;
        elig[e].pos[0] = P[i].Pos[0];
        elig[e].pos[1] = P[i].Pos[1];
        elig[e].pos[2] = P[i].Pos[2];
        elig[e].owner_task = ThisTask;
        elig[e].local_index = (int) i;
        e++;
    }
    /* Pre-truncate locally to at most (Nseed-1) per group (descending m*Gamma). */
    qsort(elig, n_elig, sizeof(struct ms_cand), cmp_ms_cand);
    int n_keep = 0;
    {
        int c = 0;
        while(c < n_elig) {
            int64_t gr = elig[c].GrNr;
            int gi = ms_find(msg, n_msg, gr);
            int limit = (gi >= 0) ? (msg[gi].Nseed - 1) : 0;
            int taken = 0;
            while(c < n_elig && elig[c].GrNr == gr) {
                if(taken < limit) { elig[n_keep++] = elig[c]; taken++; }
                c++;
            }
        }
    }

    /* ---- Phase 4: gather candidates and globally select top (Nseed-1) per group. ---- */
    int * crc = (int *) mymalloc("MScrc", NTask * sizeof(int));
    MPI_Allgather(&n_keep, 1, MPI_INT, crc, 1, MPI_INT, Comm);
    int n_call = 0;
    for(t = 0; t < NTask; t++) n_call += crc[t];
    int * cbc = (int *) mymalloc("MScbc", NTask * sizeof(int));
    int * cbd = (int *) mymalloc("MScbd", NTask * sizeof(int));
    int coff = 0;
    for(t = 0; t < NTask; t++) { cbc[t] = crc[t] * (int) sizeof(struct ms_cand); cbd[t] = coff; coff += cbc[t]; }
    struct ms_cand * allc = (struct ms_cand *) mymalloc("MSallc",
            (n_call > 0 ? n_call : 1) * sizeof(struct ms_cand));
    MPI_Allgatherv(elig, n_keep * (int) sizeof(struct ms_cand), MPI_BYTE,
                   allc, cbc, cbd, MPI_BYTE, Comm);
    qsort(allc, n_call, sizeof(struct ms_cand), cmp_ms_cand);

    /* The selection is identical on every rank (msg and allc are global+sorted), so
     * each rank converts only the candidates it owns while the per-group counts and
     * logging are consistent.  Loop over msg (ALL multi-seed groups, including ones
     * with no far-enough star) and consume each group's contiguous block of eligible
     * candidates in allc (sorted by GrNr, then m*Gamma desc), taking the top
     * (Nseed-1).  M_SC, N_seed and every seed position are logged on rank 0. */
    const double thresh_code = get_msc_multiseed_thresh_code();
    int64_t n_placed = 0, n_short = 0, n_conv_local = 0;
    int ac = 0;
    for(i = 0; i < n_msg; i++) {
        struct ms_group * m = &msg[i];
        int limit = m->Nseed - 1;
        /* Advance to this group's contiguous candidate block in allc. */
        while(ac < n_call && allc[ac].GrNr < m->GrNr) ac++;
        int bstart = ac;
        while(ac < n_call && allc[ac].GrNr == m->GrNr) ac++;
        int navail = ac - bstart;
        int nsel = (navail < limit) ? navail : limit;

        /* thresh_code = 1e8 Msun in code units, so M_SC[Msun] = M_SC_code / thresh_code * 1e8. */
        double Msc_code = m->Msc;
        double Msc_solar = (thresh_code > 0) ? Msc_code / thresh_code * 1e8 : 0;
        message(0, "secFOF multi-seed group GrNr=%ld: M_SC=%.4g Msun (%.4g code), "
                   "N_seed=%d, placed=%d; seed 1 ID=%lu pos=(%.5g, %.5g, %.5g)\n",
                (long) m->GrNr, Msc_solar, Msc_code, m->Nseed, 1 + nsel,
                (unsigned long) m->SeedStarID,
                m->seed1pos[0], m->seed1pos[1], m->seed1pos[2]);

        int s;
        for(s = 0; s < nsel; s++) {
            struct ms_cand * cc = &allc[bstart + s];
            if(cc->owner_task == ThisTask) {
                blackhole_make_one(cc->local_index, atime, rnd, 1,
                                   (MyFloat) m->per_payload, (MyFloat) m->per_scaling,
                                   (MyFloat) m->per_init_msc, (MyFloat) m->per_init_msc_sample,
                                   (MyFloat) m->capped, m->bh_ngb, (MyFloat) m->metallicity, m->metals, 0);
                /* StarClusterDetails: this extra seed's cluster mass is per_scaling.
                 * Reff = 0: not the per-cluster (SecFOFseedsumover=0) path. */
                scinfo_record_seed(cc->local_index, atime, m->per_scaling, m->sc_mass_total,
                                   m->stellar_mass_total, m->scmass_seeded, m->met_unseeded, 0,
                                   BHP(cc->local_index).Mass, m->bh_ngb, m->GrNr, &m->metdist,
                                   &m->boundinfo, SC_FLAG_SEEDED);
                n_conv_local++;
            }
            message(0, "    seed %d ID=%lu pos=(%.5g, %.5g, %.5g)\n",
                    s + 2, (unsigned long) cc->ID, cc->pos[0], cc->pos[1], cc->pos[2]);
            n_placed++;
        }
        if(nsel < limit) n_short++;
    }

    message(0, "secFOF multi-seed: %d massive group(s); placed %ld extra BH seed(s); "
               "%ld group(s) short of target (too few stars >2*softening from seed 1).\n",
            n_msg, n_placed, n_short);
    (void) n_conv_local;

    /* Free all scratch in reverse allocation order (msg/local_msg/bd/bc/rc are the
     * deepest, freed last). */
    myfree(allc);
    myfree(cbd);
    myfree(cbc);
    myfree(crc);
    myfree(elig);
    myfree(msg);
    myfree(local_msg);
    myfree(bd);
    myfree(bc);
    myfree(rc);
}

/* ============== per-cluster secFOF seeding (SecFOFseedsumover=0) ==============
 * Combined per-secFOF draw, but every sampled cluster with mass >= MinMscForBHseed
 * seeds its OWN BH (mass SeedBlackHoleMass*m_sc when BHseedMassScaleMsc=1, else
 * SeedBlackHoleMass), each hosted on a distinct unseeded star of the group chosen by
 * SeedInSecFOFRandomStarParticle: 1 = randomly sampled; 0 = the stars with the
 * largest f(Z)-scaled cluster-forming mass f(Z)*ClusterMass.  Mutually exclusive
 * with the other secFOF multi-seed / sampling-variant flags (checked in
 * set_fof_params).  Distributed exactly like fof_secfof_extra_seeds: the
 * per-group cluster-mass lists and the unseeded candidate stars are gathered to every
 * rank, the selection is identical everywhere, and each rank converts only its own
 * stars. */

/* Per-cluster secFOF seeding active: the combined draw is NOT summed into one
 * seed per group (SeedSecFOFcomSample=1 with SecFOFseedsumover=0). */
static int secfof_percluster_seeding(void)
{
    return fof_params.BlackHoleSeedStarCluster && fof_params.SeedSecFOFcomSample
        && !fof_params.SecFOFseedsumover;
}

/* ---------------- MinBHSeedInSC: compensating BH seeds ----------------
 * The run only creates a BH for a sampled cluster that is BOTH at or above
 * MinMscForBHseed (a resolution limit, not a physical one) and whose CW-model M_VMS
 * reaches SeedBlackHoleMass.  Every other cluster whose M_VMS is at or above the
 * physical VMS-collapse limit MinBHSeedInSC would really have made a BH seed, and the
 * simulation misses it.  Their M_VMS is summed per secFOF group and paid back as
 * N = floor(missed / SeedBlackHoleMass) extra BH particles of SeedBlackHoleMass each.
 *
 * Effective threshold in code units: 0 when the feature is off, otherwise the parameter
 * raised to at least 200 Msun, since only a VMS above 200 Msun collapses to a BH seed
 * in the Williams et al. 2026 model.  get_msc_min_code() is 100 physical Msun in code
 * units (the h is already in it, see init_cooling_and_star_formation) -- the same
 * code<->solar conversion the sampled cluster masses and the CW model use -- so the
 * floor is exactly twice it, and MinBHSeedInSC itself is read in those same code units.
 * Evaluated lazily (not in set_fof_params) because the units are only established at
 * init. */
static double min_bh_seed_in_sc_code(void)
{
    if(!fof_params.MbhMscRelationCWmodel || fof_params.MinBHSeedInSC <= 0)
        return 0;
    const double floor_200msun = 2.0 * get_msc_min_code();
    return (fof_params.MinBHSeedInSC > floor_200msun) ? fof_params.MinBHSeedInSC : floor_200msun;
}

/* Summing callback for starcluster_seed_masslist_detail: total sampled cluster mass in
 * the requested window.  Used only for the BH-slot upper bound below. */
static void sc_mass_sum_cb(double mass, int draw_index, void * data)
{
    (void) draw_index;
    *(double *) data += mass;
}

/* A group eligible for per-cluster multi-seeding: passes Gate 1 and holds >= 1 unseeded
 * star.  Evaluated on the group's owner (reduced properties are complete there). */
static int secfof_random_group_eligible(const struct Group * g)
{
    if(!secfof_percluster_seeding())
        return 0;
    if(g->NStarUnseeded < 1)
        return 0;
    if(g->StarClusterMassUnseeded < fof_params.MinMscForBHseed)  /* Gate 1 */
        return 0;
    return 1;
}

/* Whether particle i can HOST a per-cluster BH seed: an unseeded type-4 star (in a group)
 * with a positive metallicity-dependent seeding factor f(Z). The group sampling mass is
 * Sum(f(Z)*ClusterMass), so f(Z)=0 stars contribute nothing and must not host a seed; the
 * host pool, seed-cap and Seeded-flagging are all restricted to f(Z)>0 stars to match. */
static int secfof_random_seedable_star(int64_t i)
{
    return P[i].Type == 4 && P[i].GrNr >= 0 && !STARP(i).Seeded
        && get_seed_metallicity_factor(STARP(i).BirthMetallicity) > 0;
}

/* One random-seed group, gathered to every rank. */
struct rs_group {
    int64_t  GrNr;
    int      n_request;     /* seeds to place = min(n_qualify, NStarUnseeded) */
    int      n_qualify;     /* uncapped count of clusters >= MinMscForBHseed (for the shortage message) */
    /* MinBHSeedInSC compensating seeds owed to this group, floor(missed / SeedBlackHoleMass).
     * Not communicated: it is computed after the gather from the Allreduced missed mass,
     * so every rank fills it with the same number. */
    int      n_comp;
    double   comp_sc_share; /* StarClusterMass each of those seeds carries */
    int      nstar_unseeded;/* NStarUnseeded: the group's whole host-star supply */
    int      mass_offset;   /* offset into the gathered mass array (filled after gather) */
    uint64_t SeedStarID;    /* RNG seed for the combined draw */
    double   capped;        /* SCcomMcut (group unseeded stellar mass) = mass-function cutoff */
    double   sum_mgamma;    /* StarClusterMassUnseeded: the draw's Sum(f(Z)*ClusterMass).
                             * With capped and SeedStarID this fully determines the group's
                             * cluster population, so any rank can redraw it identically
                             * (used for the MinMscForSCdetail records). */
    int      bh_ngb;        /* BHNgbAtSeeding = host secFOF LenType[5] */
    /* StarClusterDetails record fields (per host group). */
    double   sc_mass_total; /* group total Sum(Gamma*m_star) over all stars (StarClusterMass) */
    double   stellar_mass_total; /* group total Sum(m_star) over all stars (MassType[4]) */
    double   scmass_seeded; /* group consumed SC mass (SCMass_seeded, as-is) */
    double   met_unseeded;  /* unseeded-star metal mass ratio (StarClusterDetails metallicity) */
    struct SCmetdist metdist; /* unseeded-star metallicity distribution (min/max/median/quartiles/std) */
    struct SCboundinfo boundinfo; /* BHseedSecFOFbound bound-star summary (all zero when off) */
    /* Equal-weight mean/std of the unseeded stars' log10(BirthMetallicity)
     * (floored at SC_MET_HIST_LOGMIN), for the CWmodelMetallicity 'lognormal'
     * per-cluster draw; the 'uniform' bounds come from metdist.min/max. */
    double   met_logmean;
    double   met_logstd;
};
static int cmp_rs_group_grnr(const void * a, const void * b)
{
    int64_t x = ((const struct rs_group *) a)->GrNr, y = ((const struct rs_group *) b)->GrNr;
    return (x > y) - (x < y);
}

/* One unseeded candidate star for per-cluster seeding. */
struct rs_cand {
    int64_t  GrNr;
    double   key;               /* ordering key: reproducible random in [0,1)
                                 * (SeedInSecFOFRandomStarParticle=1) or
                                 * -f(Z)*ClusterMass (=0, largest first) */
    uint64_t ID;
    float    metallicity;       /* host star's frozen BirthMetallicity */
    float    metals[NMETALS];   /* species mass fractions, rescaled to sum to metallicity */
    int      owner_task;
    int      local_index;
};
/* Order: GrNr asc, then random key asc, then ID asc (deterministic across ranks). */
static int cmp_rs_cand(const void * a, const void * b)
{
    const struct rs_cand * x = (const struct rs_cand *) a;
    const struct rs_cand * y = (const struct rs_cand *) b;
    if(x->GrNr != y->GrNr) return (x->GrNr > y->GrNr) - (x->GrNr < y->GrNr);
    if(x->key  != y->key)  return (x->key  > y->key)  - (x->key  < y->key);
    return (x->ID > y->ID) - (x->ID < y->ID);
}

/* Reproducible random ordering key for a star (mix the ID, then draw from the table). */
static double rs_star_key(uint64_t id, const RandTable * const rnd)
{
    uint64_t h = id * 6364136223846793005ULL + 1442695040888963407ULL;
    return get_random_number(h, rnd);
}

/* Per-cluster metallicity fed to the CW seed-mass model (CWmodelMetallicity):
 *   'ave'       the host group's unseeded-star metal mass ratio (every cluster);
 *   'lognormal' log10(Z) ~ Normal(met_logmean, met_logstd) of the group's
 *               unseeded stars (equal weight per star), Box-Muller, clipped to
 *               the group's [min,max] log10(Z);
 *   'uniform'   log10(Z) ~ Uniform[log10(Zmin), log10(Zmax)] of the group.
 * log10 bounds are floored at SC_MET_HIST_LOGMIN, matching the accumulated log
 * sums (covers pristine Z=0).  The draw mixes the host star ID (as rs_star_key)
 * with stream offsets +800/+801, clear of the mass sampler's and the reff
 * sampler's (+700/+701) streams, so every rank computes the same Z for the same
 * cluster. */
static double cw_sample_cluster_met(const struct rs_group * m, uint64_t star_id,
                                    const RandTable * const rnd)
{
    if(fof_params.CWmodelMetallicity == CW_MET_AVE)
        return m->met_unseeded;
    double lzmin = (m->metdist.min > 0) ? log10(m->metdist.min) : SC_MET_HIST_LOGMIN;
    if(lzmin < SC_MET_HIST_LOGMIN) lzmin = SC_MET_HIST_LOGMIN;
    double lzmax = (m->metdist.max > 0) ? log10(m->metdist.max) : SC_MET_HIST_LOGMIN;
    if(lzmax < lzmin) lzmax = lzmin;
    uint64_t h = star_id * 6364136223846793005ULL + 1442695040888963407ULL;
    double lz;
    if(fof_params.CWmodelMetallicity == CW_MET_LOGNORMAL) {
        double u1 = get_random_number(h + 800, rnd);
        double u2 = get_random_number(h + 801, rnd);
        if(u1 < 1e-20)
            u1 = 1e-20;
        double gauss = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        lz = m->met_logmean + m->met_logstd * gauss;
        if(lz < lzmin) lz = lzmin;      /* clip to the group's actual Z range */
        if(lz > lzmax) lz = lzmax;
    }
    else {                              /* CW_MET_UNIFORM */
        lz = lzmin + (lzmax - lzmin) * get_random_number(h + 800, rnd);
    }
    return pow(10.0, lz);
}

/* Upper bound on the number of per-cluster seeds this rank will convert, used to
 * pre-reserve BH slots.  Builds the global (GrNr, n_request) set (n_request from the
 * same deterministic draw used at placement) and counts local unseeded member stars in
 * those groups, capped at n_request per group.  All scratch freed in LIFO order. */
static int64_t secfof_count_random_seed_ub(FOFGroups * fof, const RandTable * const rnd, MPI_Comm Comm)
{
    if(!secfof_percluster_seeding())
        return 0;
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int64_t i;
    int t;

    int n_local = 0;
    for(i = 0; i < fof->Ngroups; i++)
        if(secfof_random_group_eligible(&fof->Group[i]))
            n_local++;

    int * rc = (int *) mymalloc("RSubRC", NTask * sizeof(int));
    MPI_Allgather(&n_local, 1, MPI_INT, rc, 1, MPI_INT, Comm);
    int n_tot = 0;
    for(t = 0; t < NTask; t++) n_tot += rc[t];
    if(n_tot == 0) { myfree(rc); return 0; }

    int * bc = (int *) mymalloc("RSubBC", NTask * sizeof(int));
    int * bd = (int *) mymalloc("RSubBD", NTask * sizeof(int));
    int boff = 0;
    for(t = 0; t < NTask; t++) { bc[t] = rc[t] * (int) sizeof(struct ms_seed_cap); bd[t] = boff; boff += bc[t]; }

    struct ms_seed_cap * local_g = (struct ms_seed_cap *) mymalloc("RSubLocal",
            (n_local > 0 ? n_local : 1) * sizeof(struct ms_seed_cap));
    /* MinBHSeedInSC also converts stars, so its seeds need slots reserved here too.
     * Their exact count is only known after the CW model has run on every sampled
     * cluster (which in turn needs the host stars picked in the seeding pass), so bound
     * it instead: cw_seed_mass_code caps M_VMS at the cluster mass, hence a cluster can
     * only contribute if its own mass reaches MinBHSeedInSC, and it contributes at most
     * that mass.  Summing the sampled clusters above MinBHSeedInSC therefore bounds the
     * missed mass -- with no CW-model evaluation at all, just the inverse-CDF redraw. */
    const double comp_min_code = min_bh_seed_in_sc_code();
    const double comp_unit_code = get_bh_seed_mass();
    const int comp_on = (comp_min_code > 0) && (comp_unit_code > 0);
    int k = 0;
    for(i = 0; i < fof->Ngroups; i++)
        if(secfof_random_group_eligible(&fof->Group[i])) {
            struct Group * g = &fof->Group[i];
            int n_qual = starcluster_combined_seed_masslist(
                    g->SCcomMcut, g->StarClusterMassUnseeded,
                    (uint64_t) g->SeedStarID, rnd, fof_params.MinMscForBHseed, NULL, 0);
            int nreq = (n_qual > g->NStarUnseeded) ? g->NStarUnseeded : n_qual;
            if(comp_on) {
                double m_ub = 0;
                starcluster_seed_masslist_detail(g->SCcomMcut, g->StarClusterMassUnseeded,
                                                 (uint64_t) g->SeedStarID, rnd,
                                                 comp_min_code, DBL_MAX, sc_mass_sum_cb, &m_ub);
                /* +1 so a last-bit rounding difference between this sum and the M_VMS
                 * sum it bounds (M_VMS is capped at the cluster mass, but through a
                 * Msun round trip) can never make the reservation one slot short. */
                double n_ub = floor(m_ub / comp_unit_code) + 1;
                if(n_ub > (double) g->NStarUnseeded)
                    n_ub = (double) g->NStarUnseeded;
                nreq += (int) n_ub;
                if(nreq > g->NStarUnseeded)     /* both kinds share one host-star pool */
                    nreq = g->NStarUnseeded;
            }
            local_g[k].GrNr   = g->base.GrNr;
            local_g[k].Nseed  = nreq;   /* reuse field: n_request (NOT Nseed-1; all seeds are "extra" here) */
            local_g[k].placed = 0;
            k++;
        }
    struct ms_seed_cap * all_g = (struct ms_seed_cap *) mymalloc("RSubAll", n_tot * sizeof(struct ms_seed_cap));
    MPI_Allgatherv(local_g, n_local * (int) sizeof(struct ms_seed_cap), MPI_BYTE,
                   all_g, bc, bd, MPI_BYTE, Comm);
    qsort(all_g, n_tot, sizeof(struct ms_seed_cap), cmp_ms_seed_cap_grnr);

    int64_t cnt = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(!secfof_random_seedable_star(i)) continue;   /* host pool: unseeded & f(Z)>0 */
        int lo = 0, hi = n_tot, found = -1;
        int64_t key = P[i].GrNr;
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if(all_g[mid].GrNr == key) { found = mid; break; }
            else if(all_g[mid].GrNr < key) lo = mid + 1;
            else hi = mid;
        }
        if(found >= 0 && all_g[found].placed < all_g[found].Nseed) {
            all_g[found].placed++;
            cnt++;
        }
    }
    myfree(all_g);
    myfree(local_g);
    myfree(bd);
    myfree(bc);
    myfree(rc);
    return cnt;
}

/* CW-model seed mass M_VMS of one sampled cluster, in code units (0 when the model is
 * off).  m_sc and the return value are code masses; reff_pc the cluster's effective
 * radius; z_cw the metallicity fed to the model; t_uni_sec the age of the universe at
 * seeding; thresh_code the 1e8 Msun code-mass yardstick used for the code<->Msun
 * conversion.  Capped at the cluster mass -- the VMS cannot outweigh its host cluster.
 * Shared by the seeding path and the StarClusterDetails records so the recorded mass is
 * by construction the same number the seeder would use for that cluster. */
static double cw_seed_mass_code(double m_sc, double reff_pc, double z_cw,
                                double t_uni_sec, double thresh_code)
{
    if(!fof_params.MbhMscRelationCWmodel || thresh_code <= 0)
        return 0;
    double m_sc_msun = m_sc / thresh_code * 1e8;
    double mvms_msun = cw_final_vms_mass_msun(m_sc_msun, 1.4 * reff_pc, z_cw, t_uni_sec,
                                              fof_params.CWmodelAlpha);
    if(mvms_msun > m_sc_msun)         /* the VMS cannot exceed its host cluster */
        mvms_msun = m_sc_msun;
    return mvms_msun / 1e8 * thresh_code;
}

/* ---- StarClusterDetails records for the clusters that cannot seed (MinMscForSCdetail) ----
 * These clusters are drawn like any other but never reach MinMscForBHseed, so they get no
 * host star and no BH.  They are written with the host group's reference star as ID/Pos
 * (shared by the whole group draw) and Flag = SC_FLAG_BELOWSEED.  Their Mbh_seed is the
 * CW model's M_VMS, evaluated exactly as for a seeding cluster: it is what the model
 * predicts for that cluster, NOT a BH in the simulation -- no particle is created and the
 * SeedBlackHoleMass floor is deliberately not applied, since the point of the record is
 * the model mass itself, including where it falls below the floor. */
struct sc_detail_ctx {
    const struct rs_group * m;
    const RandTable * rnd;
    double atime;
    MyIDType refid;             /* reference star: ID ... */
    const double * refpos;      /* ... and position (translated frame, as P[].Pos) */
    double t_uni_sec;           /* CW model inputs shared by every cluster of this call */
    double thresh_code;
    int64_t count;              /* records written for this group */
    /* The pass is shared by two consumers with independent mass windows, so it runs over
     * the union of the two and each consumer re-tests its own lower bound. */
    int    record_on;           /* write StarClusterDetails records ... */
    double record_lo;           /* ... for clusters at or above MinMscForSCdetail */
    int    comp_on;             /* accumulate the MinBHSeedInSC missed mass ... */
    double comp_min;            /* ... for clusters whose M_VMS reaches MinBHSeedInSC */
    double missed_bh;           /* summed M_VMS of those clusters (code mass) */
    double missed_sc;           /* summed CLUSTER mass of the same clusters (code mass) */
};

/* One sampled sub-seeding-threshold cluster: a StarClusterDetails record, a contribution
 * to the group's missed BH seed mass, or both. */
static void sc_detail_record_cluster(double mass, int draw_index, void * data)
{
    struct sc_detail_ctx * c = (struct sc_detail_ctx *) data;
    const struct rs_group * m = c->m;
    /* Per-cluster effective radius and CW metallicity, drawn exactly as for the seeding
     * clusters but keyed on the cluster's position in the group draw mixed into the
     * reference star ID: these clusters all share one host, so keying on the host alone
     * would collapse the whole group onto a single radius and metallicity. */
    uint64_t key = c->refid + 0x9E3779B97F4A7C15ULL * (uint64_t)(draw_index + 1);
    double reff_pc = starcluster_sample_reff_pc(mass, key, c->rnd);
    double z_record = cw_sample_cluster_met(m, key, c->rnd);
    double mvms_code = cw_seed_mass_code(mass, reff_pc, z_record, c->t_uni_sec, c->thresh_code);
    if(c->record_on && mass >= c->record_lo) {
        scinfo_record_cluster(c->refid, c->refpos, c->atime, mass, m->sc_mass_total,
                              m->stellar_mass_total, m->scmass_seeded, z_record, reff_pc,
                              mvms_code, m->bh_ngb,
                              m->GrNr, &m->metdist, &m->boundinfo, SC_FLAG_BELOWSEED);
        c->count++;
    }
    /* MinBHSeedInSC: this cluster is below MinMscForBHseed, so it never gets a BH no
     * matter what SeedBlackHoleMass is -- if the CW model says it makes a VMS heavy
     * enough to collapse, that seed is missed and is owed compensation. */
    if(c->comp_on && mvms_code >= c->comp_min) {
        c->missed_bh += mvms_code;
        c->missed_sc += mass;
    }
}

/* Place the per-cluster seeds.  BH slots were pre-reserved by the caller.  Collective:
 * every rank participates.  All scratch is mymalloc (low stack), freed in LIFO order. */
static void fof_secfof_random_seeds(FOFGroups * fof, double atime, const RandTable * const rnd, Cosmology * CP, MPI_Comm Comm)
{
    if(!secfof_percluster_seeding())
        return;
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    int64_t i;
    int t;
    const int bhdyn = get_starcluster_bhdyn_on();
    /* MinMscForSCdetail: also record the sampled clusters below MinMscForBHseed, i.e.
     * the ones that can never seed a BH.  Decided from parameters alone so every rank
     * agrees (it steers the collective candidate gather below); whether this rank
     * actually writes anything is a local question left to scinfo. */
    const double sc_detail_lo = fof_params.MinMscForSCdetail;
    const double sc_detail_hi = fof_params.MinMscForBHseed;
    const int sc_detail_on = sc_detail_lo < sc_detail_hi;
    /* MinBHSeedInSC: compensate the BH seeds this run cannot make (see
     * min_bh_seed_in_sc_code).  comp_min is the physical VMS-collapse floor the model
     * M_VMS is tested against; comp_unit is SeedBlackHoleMass, both the mass of each
     * compensating BH and the quantum the missed mass is divided into.  Parameter-derived
     * on every rank, as it steers the collective candidate gather below. */
    const double comp_min = min_bh_seed_in_sc_code();
    const double comp_unit = get_bh_seed_mass();
    const int comp_on = secfof_percluster_seeding() && comp_min > 0 && comp_unit > 0;
    /* The sub-MinMscForBHseed redraw serves the detail records and the compensation sum;
     * run it over the union of their windows.  A cluster lighter than comp_min cannot
     * contribute, because cw_seed_mass_code caps M_VMS at the cluster mass. */
    const int sc_pass_rec = sc_detail_on && scinfo_enabled();
    double sc_pass_lo = sc_detail_hi;
    if(sc_pass_rec && sc_detail_lo < sc_pass_lo)
        sc_pass_lo = sc_detail_lo;
    if(comp_on && comp_min < sc_pass_lo)
        sc_pass_lo = comp_min;
    const int sc_pass_on = sc_pass_lo < sc_detail_hi;
    /* Parameter-only twin of sc_pass_on: whether a group needs a reference star gathered
     * even when it seeds nothing.  Must not depend on scinfo_enabled(), which is a
     * per-rank file handle, because it steers the collective gather. */
    const int need_refstar = sc_detail_on || comp_on;
    /* MbhMscRelationCWmodel: age of the universe at seeding (flat matter+Lambda
     * closed form, simulation cosmology), the code<->Msun conversion, and the
     * lower seed-mass limit, all shared by every seed of this call.
     * thresh_code = 1e8 Msun in code units (same conversion convention as the
     * sampled cluster masses).  In this mode SeedBlackHoleMass is not the seed
     * mass (and BHseedMassScaleMsc is ignored) but the FLOOR: clusters whose
     * model M_VMS falls below it seed no BH. */
    const double thresh_code = get_msc_multiseed_thresh_code();
    const double seed_mass_floor_code = get_bh_seed_mass();
    double t_uni_sec = 0;
    if(fof_params.MbhMscRelationCWmodel) {
        const double H0_cgs = CP->HubbleParam * HUBBLE;   /* s^-1 */
        t_uni_sec = 2.0 / (3.0 * H0_cgs * sqrt(CP->OmegaLambda))
            * asinh(sqrt(CP->OmegaLambda / CP->Omega0) * pow(atime, 1.5));
    }

    /* ---- Phase 1: per-owned-group draw -> (rs_group, descending mass list). ---- */
    int n_local = 0;
    int64_t ub_mass = 0;    /* upper bound on local masses = sum of NStarUnseeded */
    for(i = 0; i < fof->Ngroups; i++)
        if(secfof_random_group_eligible(&fof->Group[i])) {
            n_local++;
            ub_mass += fof->Group[i].NStarUnseeded;
        }

    int * rc = (int *) mymalloc("RSrc", NTask * sizeof(int));
    MPI_Allgather(&n_local, 1, MPI_INT, rc, 1, MPI_INT, Comm);
    int n_msg = 0;
    for(t = 0; t < NTask; t++) n_msg += rc[t];
    if(n_msg == 0) { myfree(rc); return; }

    int * gbc = (int *) mymalloc("RSgbc", NTask * sizeof(int));
    int * gbd = (int *) mymalloc("RSgbd", NTask * sizeof(int));
    int boff = 0;
    for(t = 0; t < NTask; t++) { gbc[t] = rc[t] * (int) sizeof(struct rs_group); gbd[t] = boff; boff += gbc[t]; }

    struct rs_group * local_rsg = (struct rs_group *) mymalloc("RSlocal",
            (n_local > 0 ? n_local : 1) * sizeof(struct rs_group));
    double * masses_local = (double *) mymalloc("RSmassloc",
            (ub_mass > 0 ? ub_mass : 1) * sizeof(double));
    int k = 0;
    int64_t off = 0;
    for(i = 0; i < fof->Ngroups; i++) {
        struct Group * g = &fof->Group[i];
        if(!secfof_random_group_eligible(g)) continue;
        int cap_stars = g->NStarUnseeded;
        int n_qual = starcluster_combined_seed_masslist(
                g->SCcomMcut, g->StarClusterMassUnseeded, (uint64_t) g->SeedStarID,
                rnd, fof_params.MinMscForBHseed, &masses_local[off], cap_stars);
        int n_req = (n_qual > cap_stars) ? cap_stars : n_qual;   /* = number of masses filled */
        struct rs_group * m = &local_rsg[k++];
        m->GrNr = g->base.GrNr;
        m->n_request = n_req;
        m->n_qualify = n_qual;
        m->n_comp = 0;          /* set after gather, from the Allreduced missed mass */
        m->comp_sc_share = 0;
        m->nstar_unseeded = cap_stars;
        m->mass_offset = 0;     /* set after gather */
        m->SeedStarID = (uint64_t) g->SeedStarID;
        m->capped = g->SCcomMcut;
        m->sum_mgamma = g->StarClusterMassUnseeded;
        m->bh_ngb = get_seed_in_secfof() ? g->LenType[5] : 0;
        /* StarClusterDetails record fields (per host group). */
        m->sc_mass_total = g->StarClusterMass;
        m->stellar_mass_total = g->MassType[4];
        m->scmass_seeded = g->SCMass_seeded;
        m->met_unseeded = g->SCClusterMassUnseededInit > 0 ?
            g->SCMetalMassUnseeded / g->SCClusterMassUnseededInit : 0;
        sc_met_unseeded_stats(g, &m->metdist);
        sc_bound_stats(g, &m->boundinfo);
        /* Equal-weight log10(Z) mean/std of the unseeded stars for the
         * CWmodelMetallicity 'lognormal' per-cluster draw. */
        if(g->NStarUnseeded > 0) {
            double lmean = g->SCMetUnseededLogSum / (double) g->NStarUnseeded;
            double lvar = g->SCMetUnseededLogSum2 / (double) g->NStarUnseeded - lmean * lmean;
            m->met_logmean = lmean;
            m->met_logstd = lvar > 0 ? sqrt(lvar) : 0;
        } else {
            m->met_logmean = SC_MET_HIST_LOGMIN;
            m->met_logstd = 0;
        }
        off += n_req;
    }
    int64_t nmass_local = off;

    struct rs_group * all_rsg = (struct rs_group *) mymalloc("RSG", n_msg * sizeof(struct rs_group));
    MPI_Allgatherv(local_rsg, n_local * (int) sizeof(struct rs_group), MPI_BYTE,
                   all_rsg, gbc, gbd, MPI_BYTE, Comm);

    /* Gather the per-group mass blocks (same rank/group order as all_rsg). */
    int * mrc = (int *) mymalloc("RSmrc", NTask * sizeof(int));
    int nmass_local_i = (int) nmass_local;
    MPI_Allgather(&nmass_local_i, 1, MPI_INT, mrc, 1, MPI_INT, Comm);
    int n_mass_tot = 0;
    for(t = 0; t < NTask; t++) n_mass_tot += mrc[t];
    int * mbc = (int *) mymalloc("RSmbc", NTask * sizeof(int));
    int * mbd = (int *) mymalloc("RSmbd", NTask * sizeof(int));
    int moff = 0;
    for(t = 0; t < NTask; t++) { mbc[t] = mrc[t] * (int) sizeof(double); mbd[t] = moff; moff += mbc[t]; }
    double * all_masses = (double *) mymalloc("RSmass",
            (n_mass_tot > 0 ? n_mass_tot : 1) * sizeof(double));
    MPI_Allgatherv(masses_local, nmass_local_i * (int) sizeof(double), MPI_BYTE,
                   all_masses, mbc, mbd, MPI_BYTE, Comm);

    /* Mass offsets: prefix-sum n_request over the gathered (pre-sort) order, which is
     * aligned with all_masses.  The offset then travels with each group through the sort. */
    {
        int acc = 0;
        for(i = 0; i < n_msg; i++) { all_rsg[i].mass_offset = acc; acc += all_rsg[i].n_request; }
    }
    qsort(all_rsg, n_msg, sizeof(struct rs_group), cmp_rs_group_grnr);

    /* ---- Phase 2: gather unseeded candidate stars of the random-seed groups. ----
     * Single NumPart pass: size elig at the O(1) upper bound (local type-4 count) and
     * shrink to the exact candidate count afterwards (the allocate-UB-then-myrealloc
     * idiom used for NewStars in sfr_eff.c), avoiding a second pass just to pre-count. */
    int64_t n_star_local = SlotsManager->info[4].size;
    struct rs_cand * elig = (struct rs_cand *) mymalloc("RScand",
            (n_star_local > 0 ? n_star_local : 1) * sizeof(struct rs_cand));
    int e = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(!secfof_random_seedable_star(i)) continue;   /* host pool: unseeded & f(Z)>0 */
        int lo = 0, hi = n_msg, found = -1; int64_t key = P[i].GrNr;
        while(lo < hi) { int mid = lo + (hi - lo) / 2; if(all_rsg[mid].GrNr == key) { found = mid; break; } else if(all_rsg[mid].GrNr < key) lo = mid + 1; else hi = mid; }
        if(found < 0) continue;
        elig[e].GrNr = P[i].GrNr;
        /* Host-star ordering: SeedInSecFOFRandomStarParticle=1 draws a reproducible
         * random key; =0 ranks by descending f(Z)-scaled cluster-forming mass
         * (negated so the ascending key sort takes the largest first, generalizing
         * the sum-over mode's largest-f(Z)*ClusterMass host pick). */
        if(fof_params.SeedInSecFOFRandomStarParticle)
            elig[e].key = rs_star_key((uint64_t) P[i].ID, rnd);
        else
            elig[e].key = -get_seed_metallicity_factor(STARP(i).BirthMetallicity)
                * STARP(i).ClusterMass;
        elig[e].ID = (uint64_t) P[i].ID;
        /* Host-star metallicity: frozen BirthMetallicity for the scalar; species mass
         * fractions taken from the star's current Metals[] but rescaled so they sum to
         * BirthMetallicity (keeps the scalar/array pair consistent). */
        float zbirth = STARP(i).BirthMetallicity;
        double zcur = STARP(i).Metallicity;
        double mass = P[i].Mass;
        elig[e].metallicity = zbirth;
        int j;
        for(j = 0; j < NMETALS; j++) {
            double frac = (mass > 0) ? STARP(i).Metals[j] / mass : 0;
            elig[e].metals[j] = (zcur > 0) ? (float)(frac * (zbirth / zcur)) : 0;
        }
        elig[e].owner_task = ThisTask;
        elig[e].local_index = (int) i;
        e++;
    }
    /* Pre-truncate locally to at most n_request per group (lowest random key first):
     * the global top-n_request by key is a subset of the per-rank top-n_request. */
    qsort(elig, e, sizeof(struct rs_cand), cmp_rs_cand);
    int n_keep = 0;
    {
        int c = 0;
        while(c < e) {
            int64_t gr = elig[c].GrNr;
            int gi = -1;
            { int lo = 0, hi = n_msg; while(lo < hi) { int mid = lo + (hi - lo) / 2; if(all_rsg[mid].GrNr == gr) { gi = mid; break; } else if(all_rsg[mid].GrNr < gr) lo = mid + 1; else hi = mid; } }
            int limit = (gi >= 0) ? all_rsg[gi].n_request : 0;
            /* MinMscForSCdetail / MinBHSeedInSC: a group whose draw produced no seeding
             * cluster still needs one candidate gathered, to serve as the reference star
             * of its sub-threshold pass.  Seeding is unaffected: nsel = min(n_request=0,
             * navail) is still 0 for those groups. */
            if(need_refstar && gi >= 0 && limit < 1)
                limit = 1;
            int taken = 0;
            while(c < e && elig[c].GrNr == gr) {
                if(taken < limit) { elig[n_keep++] = elig[c]; taken++; }
                c++;
            }
        }
    }
    /* Shrink the candidate buffer (still the top of the bottom stack) to the kept count,
     * reclaiming the upper-bound slack before the gather/select allocations below. */
    elig = (struct rs_cand *) myrealloc(elig, (n_keep > 0 ? n_keep : 1) * sizeof(struct rs_cand));

    int * crc = (int *) mymalloc("RScrc", NTask * sizeof(int));
    MPI_Allgather(&n_keep, 1, MPI_INT, crc, 1, MPI_INT, Comm);
    int n_call = 0;
    for(t = 0; t < NTask; t++) n_call += crc[t];
    int * cbc = (int *) mymalloc("RScbc", NTask * sizeof(int));
    int * cbd = (int *) mymalloc("RScbd", NTask * sizeof(int));
    int coff = 0;
    for(t = 0; t < NTask; t++) { cbc[t] = crc[t] * (int) sizeof(struct rs_cand); cbd[t] = coff; coff += cbc[t]; }
    struct rs_cand * allc = (struct rs_cand *) mymalloc("RSallc",
            (n_call > 0 ? n_call : 1) * sizeof(struct rs_cand));
    MPI_Allgatherv(elig, n_keep * (int) sizeof(struct rs_cand), MPI_BYTE,
                   allc, cbc, cbd, MPI_BYTE, Comm);
    qsort(allc, n_call, sizeof(struct rs_cand), cmp_rs_cand);

    /* ---- Phase 3: identical selection on every rank; each converts its own stars. ---- */
    int64_t n_placed = 0, n_short = 0, n_conv_local = 0, n_novms = 0, n_detail_local = 0;
    /* MinBHSeedInSC per-group missed mass: [0, n_msg) the missed BH seed mass, [n_msg,
     * 2*n_msg) the cluster mass of the same clusters.  Accumulated on the group's
     * reference-star owner only (which also runs the loop below, so it sees the
     * candidate clusters too) and summed to every rank in one Allreduce, so that the
     * compensating-seed counts are identical everywhere without a second collective. */
    double * miss = (double *) mymalloc("RSmiss", 2 * (n_msg > 0 ? n_msg : 1) * sizeof(double));
    memset(miss, 0, 2 * (n_msg > 0 ? n_msg : 1) * sizeof(double));
    int ac = 0;
    for(i = 0; i < n_msg; i++) {
        struct rs_group * m = &all_rsg[i];
        while(ac < n_call && allc[ac].GrNr < m->GrNr) ac++;
        int bstart = ac;
        while(ac < n_call && allc[ac].GrNr == m->GrNr) ac++;
        int navail = ac - bstart;
        int nsel = (m->n_request < navail) ? m->n_request : navail;
        /* The group's reference star: host of its most massive cluster, and the single
         * writer of everything that belongs to the group rather than to one cluster. */
        const int is_ref_owner = (navail > 0) && (allc[bstart].owner_task == ThisTask);

        int s;
        for(s = 0; s < nsel; s++) {
            struct rs_cand * cc = &allc[bstart + s];
            double m_sc = all_masses[m->mass_offset + s];   /* descending; arbitrary->random star */
            /* Effective radius from the size-mass relation (0.5 dex scatter,
             * reproducibly keyed on the host star ID): recorded in the
             * StarClusterDetails seed record, and the cluster size input of the
             * CW seed-mass model.  Deterministic, so every rank agrees. */
            double reff_pc = starcluster_sample_reff_pc(m_sc, (uint64_t) cc->ID, rnd);
            /* MbhMscRelationCWmodel: seed mass = M_VMS of the Williams et al.
             * 2026 collision model for this cluster (mass, virial radius
             * r_max=1.4*Reff, host-group unseeded-star metal mass ratio, age of
             * the universe), capped at the cluster mass.  Clusters at or above
             * the model's mean-density cap (rho_mean >= 6e7 Msun/pc^3 inside
             * r_max) bypass the collision model and get 0.01*M_sc instead
             * (handled inside cw_final_vms_mass_msun).  BHseedMassScaleMsc is
             * ignored in this mode; SeedBlackHoleMass instead acts as the LOWER
             * seed-mass limit: a cluster whose M_VMS < SeedBlackHoleMass (which
             * covers M_VMS=0, no net inflow -> no VMS forms) seeds no BH, but
             * is consumed like the seeded ones (its host star stays in the
             * group's Seeded=1 marking below).  Computed identically on every
             * rank. */
            MyFloat seed_mass_override = 0;
            /* Metallicity written to this cluster's StarClusterDetails record:
             * the Z actually fed to the CW model (= met_unseeded in 'ave' mode
             * and outside the CW model). */
            double z_record = m->met_unseeded;
            if(fof_params.MbhMscRelationCWmodel) {
                /* Per-cluster CW-model metallicity (CWmodelMetallicity mode);
                 * keyed on the host star ID, so every rank agrees. */
                double z_cw = cw_sample_cluster_met(m, cc->ID, rnd);
                z_record = z_cw;
                double mvms_code = cw_seed_mass_code(m_sc, reff_pc, z_cw, t_uni_sec, thresh_code);
                if(mvms_code < seed_mass_floor_code) {
                    n_novms++;
                    /* MinBHSeedInSC: the cluster is heavy enough to be a candidate but
                     * its VMS is too light for the run's seed-mass floor.  If the VMS
                     * still reaches the physical collapse limit, the seed is real and
                     * missed, so it is owed compensation. */
                    if(comp_on && is_ref_owner && mvms_code >= comp_min) {
                        miss[i]         += mvms_code;
                        miss[n_msg + i] += m_sc;
                    }
                    /* StarClusterDetails: record the skipped cluster too.  No BH is
                     * created (the candidate star stays a star and supplies the
                     * record's ID/Pos), but Mbh_seed still carries the model M_VMS
                     * that fell short of SeedBlackHoleMass -- Flag, not Mbh_seed,
                     * is what says no BH exists. */
                    if(cc->owner_task == ThisTask)
                        scinfo_record_seed(cc->local_index, atime, m_sc, m->sc_mass_total,
                                           m->stellar_mass_total, m->scmass_seeded,
                                           z_record, reff_pc,
                                           mvms_code, m->bh_ngb, m->GrNr, &m->metdist,
                                           &m->boundinfo, SC_FLAG_NOVMS);
                    continue;
                }
                seed_mass_override = (MyFloat) mvms_code;
            }
            if(cc->owner_task == ThisTask) {
                MyFloat payload = bhdyn ? (MyFloat) m_sc : 0;   /* StarClusterMass (dynamics + evolution) */
                blackhole_make_one(cc->local_index, atime, rnd, 1,
                                   payload, (MyFloat) m_sc,
                                   (MyFloat) m_sc, (MyFloat) m_sc,
                                   (MyFloat) m->capped, m->bh_ngb,
                                   (MyFloat) cc->metallicity, cc->metals,
                                   seed_mass_override);
                /* StarClusterDetails: this seed's cluster mass is the sampled m_sc;
                 * Mbh_seed is the just-made BH's subgrid mass. */
                scinfo_record_seed(cc->local_index, atime, m_sc, m->sc_mass_total,
                                   m->stellar_mass_total, m->scmass_seeded, z_record, reff_pc,
                                   BHP(cc->local_index).Mass, m->bh_ngb, m->GrNr, &m->metdist,
                                   &m->boundinfo, SC_FLAG_SEEDED);
                n_conv_local++;
            }
            n_placed++;
        }
        /* StarClusterDetails (MinMscForSCdetail): record the sampled clusters that are
         * too light to ever seed a BH.  Having no host of their own, they all carry the
         * group's reference star -- allc[bstart], the host of the group's most massive
         * cluster -- as ID/Pos, and only that star's owner writes them, so each cluster
         * is recorded exactly once.  The population is redrawn locally from (SCcomMcut,
         * sum_mgamma, SeedStarID), which is deterministic and reproduces the draw above
         * exactly, so this costs no communication and buffers nothing. */
        if(sc_pass_on && is_ref_owner) {
            int ri = allc[bstart].local_index;
            struct sc_detail_ctx ctx = {m, rnd, atime, P[ri].ID, P[ri].Pos,
                                        t_uni_sec, thresh_code, 0,
                                        sc_pass_rec, sc_detail_lo, comp_on, comp_min, 0, 0};
            starcluster_seed_masslist_detail(m->capped, m->sum_mgamma, m->SeedStarID, rnd,
                                             sc_pass_lo, sc_detail_hi,
                                             sc_detail_record_cluster, &ctx);
            n_detail_local += ctx.count;
            miss[i]         += ctx.missed_bh;
            miss[n_msg + i] += ctx.missed_sc;
        }
        /* Shortage: more eligible clusters than unseeded stars available. */
        if(m->n_qualify > nsel) {
            n_short++;
            message(0, "secFOF per-cluster seeding group GrNr=%ld: %d eligible cluster(s) >= MinMscForBHseed "
                       "but only %d unseeded f(Z)>0 host star(s); seeded %d, skipped %d.\n",
                    (long) m->GrNr, m->n_qualify, navail, nsel, m->n_qualify - nsel);
        }
    }

    if(fof_params.MbhMscRelationCWmodel)
        message(0, "secFOF per-cluster seeding: %d group(s); placed %ld BH seed(s) (CW-model VMS masses); "
                   "%ld cluster(s) skipped with M_VMS < SeedBlackHoleMass; %ld group(s) short of unseeded stars.\n",
                n_msg, n_placed, n_novms, n_short);
    else
        message(0, "secFOF per-cluster seeding: %d group(s); placed %ld BH seed(s); %ld group(s) short of unseeded stars.\n",
                n_msg, n_placed, n_short);
    (void) n_conv_local;

    /* MinMscForSCdetail: the sub-threshold records come in large batches, so scinfo
     * leaves them unflushed; flush once here, after every group has been written. */
    if(sc_detail_on) {
        scinfo_flush();
        int64_t n_detail_tot = 0;
        MPI_Allreduce(&n_detail_local, &n_detail_tot, 1, MPI_INT64, MPI_SUM, Comm);
        message(0, "secFOF per-cluster seeding: recorded %ld sub-seeding-threshold cluster(s) "
                   "(MinMscForSCdetail=%g) to StarClusterDetails.\n",
                n_detail_tot, sc_detail_lo);
    }

    /* ---- Phase 4 (MinBHSeedInSC): compensating BH seeds. ----------------------------
     * Every rank now knows each group's missed BH seed mass, so N = floor(missed /
     * SeedBlackHoleMass) and the cluster mass each compensating seed carries are computed
     * identically everywhere.  Placing them needs a SECOND candidate gather: the round-A
     * list was pre-truncated to n_request per group and is by now largely converted, so
     * there is nothing left in it to host these seeds.  Re-gathering also gets the host
     * pool right for free -- the stars converted above are type 5 and so no longer
     * seedable, while the candidates left standing (the hosts of the no-VMS clusters and
     * the reference stars) are still unseeded and stay in the running, in the same
     * f(Z)*ClusterMass order.
     * NOTE clusters dropped for want of a host star (the shortage message above) are not
     * compensated: their M_VMS is never evaluated, and a group short of host stars has
     * none to spare for compensating seeds either. */
    struct rs_cand * allc_c = NULL;
    int * crc_c = NULL, * cbc_c = NULL, * cbd_c = NULL;
    struct rs_cand * elig_c = NULL;
    int n_call_c = 0;
    int64_t n_comp_placed = 0;
    if(comp_on) {
        MPI_Allreduce(MPI_IN_PLACE, miss, 2 * n_msg, MPI_DOUBLE, MPI_SUM, Comm);
        double miss_bh_tot = 0, comp_mass_tot = 0;
        int64_t n_comp_want = 0, n_comp_groups = 0, n_comp_capped = 0;
        for(i = 0; i < n_msg; i++) {
            miss_bh_tot += miss[i];
            double nd = floor(miss[i] / comp_unit);
            /* One host star per compensating seed, so the group's whole unseeded-star
             * supply bounds them; the exact count still comes from the gather below,
             * since the ordinary seeds have already taken some of that supply. */
            if(nd > (double) all_rsg[i].nstar_unseeded) {
                nd = (double) all_rsg[i].nstar_unseeded;
                n_comp_capped++;
            }
            int nc = (nd > 0) ? (int) nd : 0;
            all_rsg[i].n_comp = nc;
            /* The whole missed cluster mass is shared over however many seeds are
             * actually made, so it is conserved even where the count is cut short. */
            all_rsg[i].comp_sc_share = (nc > 0) ? miss[n_msg + i] / nc : 0;
            if(nc > 0) { n_comp_want += nc; n_comp_groups++; }
        }

        /* Gather the still-unseeded candidate stars of the groups owed seeds, exactly as
         * in phase 2 but with n_comp as the per-group limit. */
        int64_t n_star_local_c = SlotsManager->info[4].size;
        elig_c = (struct rs_cand *) mymalloc("RSCcand",
                (n_star_local_c > 0 ? n_star_local_c : 1) * sizeof(struct rs_cand));
        int ec = 0;
        for(i = 0; i < PartManager->NumPart; i++) {
            if(!secfof_random_seedable_star(i)) continue;
            int lo = 0, hi = n_msg, found = -1; int64_t key = P[i].GrNr;
            while(lo < hi) { int mid = lo + (hi - lo) / 2; if(all_rsg[mid].GrNr == key) { found = mid; break; } else if(all_rsg[mid].GrNr < key) lo = mid + 1; else hi = mid; }
            if(found < 0 || all_rsg[found].n_comp < 1) continue;
            elig_c[ec].GrNr = P[i].GrNr;
            if(fof_params.SeedInSecFOFRandomStarParticle)
                elig_c[ec].key = rs_star_key((uint64_t) P[i].ID, rnd);
            else
                elig_c[ec].key = -get_seed_metallicity_factor(STARP(i).BirthMetallicity)
                    * STARP(i).ClusterMass;
            elig_c[ec].ID = (uint64_t) P[i].ID;
            float zbirth = STARP(i).BirthMetallicity;
            double zcur = STARP(i).Metallicity;
            double mass = P[i].Mass;
            elig_c[ec].metallicity = zbirth;
            int j;
            for(j = 0; j < NMETALS; j++) {
                double frac = (mass > 0) ? STARP(i).Metals[j] / mass : 0;
                elig_c[ec].metals[j] = (zcur > 0) ? (float)(frac * (zbirth / zcur)) : 0;
            }
            elig_c[ec].owner_task = ThisTask;
            elig_c[ec].local_index = (int) i;
            ec++;
        }
        qsort(elig_c, ec, sizeof(struct rs_cand), cmp_rs_cand);
        int n_keep_c = 0;
        {
            int c = 0;
            while(c < ec) {
                int64_t gr = elig_c[c].GrNr;
                int gi = -1;
                { int lo = 0, hi = n_msg; while(lo < hi) { int mid = lo + (hi - lo) / 2; if(all_rsg[mid].GrNr == gr) { gi = mid; break; } else if(all_rsg[mid].GrNr < gr) lo = mid + 1; else hi = mid; } }
                int limit = (gi >= 0) ? all_rsg[gi].n_comp : 0;
                int taken = 0;
                while(c < ec && elig_c[c].GrNr == gr) {
                    if(taken < limit) { elig_c[n_keep_c++] = elig_c[c]; taken++; }
                    c++;
                }
            }
        }
        elig_c = (struct rs_cand *) myrealloc(elig_c, (n_keep_c > 0 ? n_keep_c : 1) * sizeof(struct rs_cand));

        crc_c = (int *) mymalloc("RSCcrc", NTask * sizeof(int));
        MPI_Allgather(&n_keep_c, 1, MPI_INT, crc_c, 1, MPI_INT, Comm);
        for(t = 0; t < NTask; t++) n_call_c += crc_c[t];
        cbc_c = (int *) mymalloc("RSCcbc", NTask * sizeof(int));
        cbd_c = (int *) mymalloc("RSCcbd", NTask * sizeof(int));
        int coff_c = 0;
        for(t = 0; t < NTask; t++) { cbc_c[t] = crc_c[t] * (int) sizeof(struct rs_cand); cbd_c[t] = coff_c; coff_c += cbc_c[t]; }
        allc_c = (struct rs_cand *) mymalloc("RSCallc",
                (n_call_c > 0 ? n_call_c : 1) * sizeof(struct rs_cand));
        MPI_Allgatherv(elig_c, n_keep_c * (int) sizeof(struct rs_cand), MPI_BYTE,
                       allc_c, cbc_c, cbd_c, MPI_BYTE, Comm);
        qsort(allc_c, n_call_c, sizeof(struct rs_cand), cmp_rs_cand);

        /* Identical selection on every rank; each converts its own stars. */
        int64_t n_comp_short = 0;
        int acc = 0;
        for(i = 0; i < n_msg; i++) {
            struct rs_group * m = &all_rsg[i];
            if(m->n_comp < 1) continue;
            while(acc < n_call_c && allc_c[acc].GrNr < m->GrNr) acc++;
            int bstart = acc;
            while(acc < n_call_c && allc_c[acc].GrNr == m->GrNr) acc++;
            int navail = acc - bstart;
            int nsel = (m->n_comp < navail) ? m->n_comp : navail;
            if(nsel < m->n_comp)
                n_comp_short++;
            int s;
            for(s = 0; s < nsel; s++) {
                struct rs_cand * cc = &allc_c[bstart + s];
                if(cc->owner_task == ThisTask) {
                    /* Seed mass is exactly SeedBlackHoleMass (the compensation quantum);
                     * the cluster payload is the group's missed cluster mass shared out
                     * evenly, i.e. each seed's proportional share of it. */
                    MyFloat share = (MyFloat) m->comp_sc_share;
                    MyFloat payload = bhdyn ? share : 0;
                    blackhole_make_one(cc->local_index, atime, rnd, 1,
                                       payload, share, share, share,
                                       (MyFloat) m->capped, m->bh_ngb,
                                       (MyFloat) cc->metallicity, cc->metals,
                                       (MyFloat) comp_unit);
                    scinfo_record_seed(cc->local_index, atime, m->comp_sc_share,
                                       m->sc_mass_total, m->stellar_mass_total,
                                       m->scmass_seeded, m->met_unseeded, 0,
                                       BHP(cc->local_index).Mass, m->bh_ngb, m->GrNr,
                                       &m->metdist, &m->boundinfo, SC_FLAG_COMPENSATE);
                }
                n_comp_placed++;
                comp_mass_tot += comp_unit;
            }
        }
        message(0, "secFOF per-cluster seeding (MinBHSeedInSC=%g): %ld group(s) missed %g in BH seed "
                   "mass; placed %ld of %ld compensating seed(s) of %g each (%g total); "
                   "%ld group(s) short of unseeded stars at the gather, %ld capped at NStarUnseeded.\n",
                comp_min, n_comp_groups, miss_bh_tot, n_comp_placed, n_comp_want,
                comp_unit, comp_mass_tot, n_comp_short, n_comp_capped);
    }

    /* Flag every remaining seedable star (unseeded & f(Z)>0) of a seeded group (n_request >= 1,
     * or n_comp >= 1 for a group that only got compensating seeds) as Seeded=1. The group's
     * Sum(f(Z)*ClusterMass) fed the combined draw, so its seedable
     * population is consumed regardless of how many BHs were placed (matching the single-seed
     * combined path, which marks these via secondfof_seed's seeded_grnr_out list; random-mode
     * groups are absent from that list, so they are marked here). f(Z)=0 stars never participate
     * and are left untouched; stars already converted to BHs are type 5 and skipped. */
    int64_t n_flag = 0;
    #pragma omp parallel for reduction(+:n_flag)
    for(i = 0; i < PartManager->NumPart; i++) {
        if(!secfof_random_seedable_star(i)) continue;   /* unseeded & f(Z)>0 */
        int64_t key = P[i].GrNr;
        int lo = 0, hi = n_msg, found = -1;
        while(lo < hi) { int mid = lo + (hi - lo) / 2; if(all_rsg[mid].GrNr == key) { found = mid; break; } else if(all_rsg[mid].GrNr < key) lo = mid + 1; else hi = mid; }
        if(found >= 0 && (all_rsg[found].n_request >= 1 || all_rsg[found].n_comp >= 1)) { STARP(i).Seeded = 1; n_flag++; }
    }
    int64_t n_flag_tot = 0;
    MPI_Allreduce(&n_flag, &n_flag_tot, 1, MPI_INT64, MPI_SUM, Comm);
    message(0, "secFOF per-cluster seeding: flagged Seeded=1 for %ld star(s) in seeded groups.\n", n_flag_tot);

    /* Free all scratch in reverse allocation order.  The phase-4 buffers sit on top of
     * miss, which sits on top of the phase-2 ones, so they go first. */
    if(comp_on) {
        myfree(allc_c);
        myfree(cbd_c);
        myfree(cbc_c);
        myfree(crc_c);
        myfree(elig_c);
    }
    myfree(miss);
    myfree(allc);
    myfree(cbd);
    myfree(cbc);
    myfree(crc);
    myfree(elig);
    myfree(all_masses);
    myfree(mbd);
    myfree(mbc);
    myfree(mrc);
    myfree(all_rsg);
    myfree(masses_local);
    myfree(local_rsg);
    myfree(gbd);
    myfree(gbc);
    myfree(rc);
}

/* --- SeedSecFOFcomSampleParticle: per-star-particle star-cluster sampling ---
 * One candidate secFOF group (passed Gate 1), broadcast to all ranks so each rank
 * can sample the local member stars it holds. */
struct sc_particle_cand {
    int64_t GrNr;
    double Mcut;   /* SCcomMcut: group total unseeded stellar mass */
};

static int cmp_sc_particle_cand(const void * a, const void * b)
{
    int64_t ga = ((const struct sc_particle_cand *)a)->GrNr;
    int64_t gb = ((const struct sc_particle_cand *)b)->GrNr;
    return (ga > gb) - (ga < gb);
}

/* Per-particle replacement for the single combined per-group draw. For every
 * secFOF group passing Gate 1 (StarClusterMassUnseeded >= MinMscForBHseed with a
 * valid seed star), each UNSEEDED member star draws its own cluster population
 * (mass function n(m) ~ m^-2 exp(-m/m_cut), cutoff m_cut = min(M_cstar, group
 * unseeded stellar mass)); the clusters above 1e4 Msun are summed per star and
 * over the group into tot_msc_fof. Optionally capped at the group unseeded
 * stellar mass, the result is written into Group.BHSeedMsc (full sampled sum into
 * BHSampledMscTotal). Gate 2 (seed iff BHSeedMsc >= MinMscForBHseed) and the
 * actual seeding remain in the fof_seed serial pass below.
 *
 * Member stars are distributed across ranks, so this runs collectively: Allgather
 * the candidate groups, sample local stars into per-candidate partial sums,
 * Allreduce, cap, then scatter tot_msc_fof back into locally-owned groups.
 * Serial per rank (the sampler uses the global GSL error path), reproducible via
 * per-star P.ID RNG seeds (independent of the domain decomposition). */
static void fof_secfof_particle_sample(FOFGroups * fof, const RandTable * const rnd, MPI_Comm Comm)
{
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int64_t i;
    int t;

    /* Every owned group starts at 0 so non-candidates fail Gate 2 below. */
    for(i = 0; i < fof->Ngroups; i++) {
        fof->Group[i].BHSeedMsc = 0;
        fof->Group[i].BHSampledMscTotal = 0;
    }

    /* Local candidates (Gate 1: cluster-forming mass over threshold, valid seed). */
    int n_local_cand = 0;
    for(i = 0; i < fof->Ngroups; i++) {
        if(fof->Group[i].seed_index_star >= 0 &&
           fof->Group[i].StarClusterMassUnseeded >= fof_params.MinMscForBHseed)
            n_local_cand++;
    }

    int * recv_counts = (int *) mymalloc("CandRecvCounts", NTask * sizeof(int));
    MPI_Allgather(&n_local_cand, 1, MPI_INT, recv_counts, 1, MPI_INT, Comm);
    int n_cand = 0;
    for(t = 0; t < NTask; t++)
        n_cand += recv_counts[t];

    if(n_cand == 0) {
        myfree(recv_counts);
        return;
    }

    /* Byte counts/displacements for the struct Allgatherv. */
    int * byte_counts = (int *) mymalloc("CandByteCounts", NTask * sizeof(int));
    int * byte_displs = (int *) mymalloc("CandByteDispls", NTask * sizeof(int));
    int boff = 0;
    for(t = 0; t < NTask; t++) {
        byte_counts[t] = recv_counts[t] * (int) sizeof(struct sc_particle_cand);
        byte_displs[t] = boff;
        boff += byte_counts[t];
    }

    /* Pack and gather the candidate list. */
    struct sc_particle_cand * local_cand = (struct sc_particle_cand *)
        mymalloc("LocalCand", (n_local_cand > 0 ? n_local_cand : 1) * sizeof(struct sc_particle_cand));
    int k = 0;
    for(i = 0; i < fof->Ngroups; i++) {
        if(fof->Group[i].seed_index_star >= 0 &&
           fof->Group[i].StarClusterMassUnseeded >= fof_params.MinMscForBHseed) {
            local_cand[k].GrNr = fof->Group[i].base.GrNr;
            local_cand[k].Mcut = fof->Group[i].SCcomMcut;
            k++;
        }
    }
    struct sc_particle_cand * cand = (struct sc_particle_cand *)
        mymalloc("Cand", n_cand * sizeof(struct sc_particle_cand));
    MPI_Allgatherv(local_cand, n_local_cand * (int) sizeof(struct sc_particle_cand), MPI_BYTE,
                   cand, byte_counts, byte_displs, MPI_BYTE, Comm);
    qsort(cand, n_cand, sizeof(struct sc_particle_cand), cmp_sc_particle_cand);

    /* Sample local unseeded stars into per-candidate partial sums. */
    double * part_tot = (double *) mymalloc("CandPartTot", n_cand * sizeof(double));
    double * part_full = (double *) mymalloc("CandPartFull", n_cand * sizeof(double));
    memset(part_tot, 0, n_cand * sizeof(double));
    memset(part_full, 0, n_cand * sizeof(double));

    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].Type != 4 || P[i].GrNr < 0 || STARP(i).Seeded)
            continue;
        int64_t key = P[i].GrNr;
        int lo = 0, hi = n_cand, c = -1;
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if(cand[mid].GrNr == key) { c = mid; break; }
            else if(cand[mid].GrNr < key) lo = mid + 1;
            else hi = mid;
        }
        if(c < 0)
            continue;
        double Mcstar = STARP(i).Mcstar;
        double m_cut = (Mcstar < cand[c].Mcut) ? Mcstar : cand[c].Mcut;
        /* allow_cap = 0: the cap is applied below on the group-summed tot_msc_fof,
         * not on each per-star draw (whose cutoff is the per-star m_cut). */
        double full = 0;
        /* Scale the per-star Poisson rate (sum_mGamma = Gamma*m_star) by f(Z). */
        double fseed = get_seed_metallicity_factor(STARP(i).BirthMetallicity);
        double msc_star = starcluster_combined_bhseed_msc(
                m_cut, fseed * STARP(i).ClusterMass, (uint64_t) P[i].ID, rnd, &full, 0);
        part_tot[c] += msc_star;
        part_full[c] += full;
    }

    /* Sum the per-rank partials, then cap on the group total (if enabled). */
    MPI_Allreduce(MPI_IN_PLACE, part_tot, n_cand, MPI_DOUBLE, MPI_SUM, Comm);
    MPI_Allreduce(MPI_IN_PLACE, part_full, n_cand, MPI_DOUBLE, MPI_SUM, Comm);
    if(get_scmasscap_secfof_starmass()) {
        int c;
        for(c = 0; c < n_cand; c++) {
            if(part_tot[c] > cand[c].Mcut)
                part_tot[c] = cand[c].Mcut;
            if(part_full[c] > cand[c].Mcut)
                part_full[c] = cand[c].Mcut;
        }
    }

    /* Scatter tot_msc_fof back into each locally-owned candidate group. */
    for(i = 0; i < fof->Ngroups; i++) {
        if(!(fof->Group[i].seed_index_star >= 0 &&
             fof->Group[i].StarClusterMassUnseeded >= fof_params.MinMscForBHseed))
            continue;
        int64_t key = fof->Group[i].base.GrNr;
        int lo = 0, hi = n_cand, c = -1;
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if(cand[mid].GrNr == key) { c = mid; break; }
            else if(cand[mid].GrNr < key) lo = mid + 1;
            else hi = mid;
        }
        if(c < 0)
            continue;
        fof->Group[i].BHSeedMsc = part_tot[c];
        fof->Group[i].BHSampledMscTotal = part_full[c];
    }

    myfree(part_full);
    myfree(part_tot);
    myfree(cand);
    myfree(local_cand);
    myfree(byte_displs);
    myfree(byte_counts);
    myfree(recv_counts);
}

/* One gathered secFOF member record used by fof_secfof_bound_massive_restrict.
 * ALL member types are gathered (the softened potential uses every member); only
 * unseeded stars carry mGamma > 0 / is_unseeded_star = 1. */
struct bound_member {
    int64_t  GrNr;
    double   Pos[3];
    double   Vel[3];
    double   Mass;
    double   mGamma;            /* STARP(i).ClusterMass if unseeded star, else 0 */
    MyIDType ID;                /* particle ID (becomes SeedStarID if chosen as seed) */
    int      OrigTask;          /* rank that owns this particle */
    int      OrigIndex;         /* local index of this particle on OrigTask */
    int      is_unseeded_star;  /* 1 if Type==4 && !STARP.Seeded, else 0 */
};

static int cmp_bound_member_grnr(const void * a, const void * b)
{
    const struct bound_member * pa = (const struct bound_member *) a;
    const struct bound_member * pb = (const struct bound_member *) b;
    if(pa->GrNr < pb->GrNr) return -1;
    if(pa->GrNr > pb->GrNr) return 1;
    return 0;
}

/* Binary search: is key present in the sorted unique array arr[0..n)? */
static int bm_grnr_in_set(const int64_t * arr, int64_t n, int64_t key)
{
    int64_t lo = 0, hi = n;
    while(lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if(arr[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return (lo < n && arr[lo] == key);
}

/* SeedSeedFOFMassiveBoundStar: for every secFOF group whose UNSEEDED star-cluster
 * mass Sum(m*Gamma) (= StarClusterMassUnseeded) exceeds the 1e8-Msun threshold,
 * restrict the group to the unseeded stars that are gravitationally bound to the
 * secFOF, and overwrite StarClusterMassUnseeded (-> bound Sum(m*Gamma)) and
 * SCcomMcut (-> bound Sum(m_star)) in place so the downstream combined sampler,
 * the seed decision, and the seed mass all use the bound subset.
 *
 * The seed location is also moved to the largest-m*Gamma BOUND star: seed_index_star,
 * seed_task_star and SeedStarID (location + RNG seed) are repointed to that one bound
 * particle so the BH is never seeded at an unbound star. If no unseeded star is bound,
 * the seed is dropped (seed_index_star = seed_task_star = -1); Gate 1 also drops the
 * group since StarClusterMassUnseeded becomes 0.
 *
 * Boundedness (matching check_grav_bound / stats.c conventions): the softened
 * potential magnitude pot_mag(i) = Sum_{j!=i} m_j / sqrt(r_ij^2 + eps^2) is summed
 * over ALL members (comoving separations, code masses); the rest frame is the
 * velocity of the deepest-potential member; star i is bound iff
 *   0.5*|Vel_i - Vref|^2 <= atime * G * pot_mag(i).
 *
 * Distributed via the Allgatherv pattern from secondfof_compute_sizes, keyed on
 * P[i].GrNr (= secondary group id at this point, written by fof_fof in
 * secondfof_seed). Each group is overwritten only by its owning rank. The gather
 * and the O(N^2) potential are restricted to the (rare) > 1e8-Msun groups.
 *
 * TODO(perf): the per-group potential is O(N^2); a tree/Barnes-Hut potential is
 * the natural optimization for very large complexes (deferred, correctness-first). */
static void
fof_secfof_bound_massive_restrict(FOFGroups * fof, double atime, Cosmology * CP, MPI_Comm Comm)
{
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    const double BoxSize = PartManager->BoxSize;
    const double thresh = get_msc_multiseed_thresh_code();
    const double G = CP->GravInternal;
    const double eps = FORCE_SOFTENING() / 2.8;   /* Plummer-equivalent, comoving */
    const double eps2 = eps * eps;
    int i, g;

    if(thresh <= 0)
        return;   /* no threshold -> feature is a no-op */

    /* ---- Step 0: global set of qualifying group numbers (Sum(m*Gamma) > thresh). ---- */
    int n_local_qual = 0;
    for(g = 0; g < fof->Ngroups; g++)
        if(fof->Group[g].StarClusterMassUnseeded > thresh)
            n_local_qual++;

    int * q_counts = (int *) mymalloc2("BMqc", sizeof(int) * NTask);
    MPI_Allgather(&n_local_qual, 1, MPI_INT, q_counts, 1, MPI_INT, Comm);
    int64_t total_qual = 0;
    int * q_displs = (int *) mymalloc2("BMqd", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        q_displs[i] = total_qual;
        total_qual += q_counts[i];
    }
    if(total_qual == 0) {   /* no massive group anywhere: nothing to do (LIFO frees) */
        myfree(q_displs);
        myfree(q_counts);
        return;
    }

    int64_t * local_qual = (int64_t *) mymalloc2("BMlq",
            sizeof(int64_t) * (n_local_qual > 0 ? n_local_qual : 1));
    {
        int k = 0;
        for(g = 0; g < fof->Ngroups; g++)
            if(fof->Group[g].StarClusterMassUnseeded > thresh)
                local_qual[k++] = fof->Group[g].base.GrNr;
    }
    int64_t * qual_grnr = (int64_t *) mymalloc2("BMqg", sizeof(int64_t) * total_qual);
    int * qc_bytes = (int *) mymalloc2("BMqcb", sizeof(int) * NTask);
    int * qd_bytes = (int *) mymalloc2("BMqdb", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        qc_bytes[i] = q_counts[i] * (int) sizeof(int64_t);
        qd_bytes[i] = q_displs[i] * (int) sizeof(int64_t);
    }
    MPI_Allgatherv(local_qual, n_local_qual * (int) sizeof(int64_t), MPI_BYTE,
                   qual_grnr, qc_bytes, qd_bytes, MPI_BYTE, Comm);
    myfree(qd_bytes);
    myfree(qc_bytes);
    /* GrNr are globally unique per group -> qual_grnr is duplicate-free; sort for search. */
    qsort(qual_grnr, total_qual, sizeof(int64_t), cmp_int64_asc);

    /* ---- Step 1: pack local members (ALL types) of qualifying groups. ---- */
    int64_t n_pack = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].GrNr < 0) continue;
        if(!bm_grnr_in_set(qual_grnr, total_qual, P[i].GrNr)) continue;
        n_pack++;
    }
    struct bound_member * bm_local = (struct bound_member *) mymalloc2("BMlocal",
            sizeof(struct bound_member) * (n_pack > 0 ? n_pack : 1));
    {
        int64_t k = 0;
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].GrNr < 0) continue;
            if(!bm_grnr_in_set(qual_grnr, total_qual, P[i].GrNr)) continue;
            struct bound_member * m = &bm_local[k++];
            m->GrNr = P[i].GrNr;
            int d;
            for(d = 0; d < 3; d++) {
                m->Pos[d] = P[i].Pos[d];
                m->Vel[d] = P[i].Vel[d];
            }
            m->Mass = P[i].Mass;
            m->ID = P[i].ID;
            m->OrigTask = ThisTask;
            m->OrigIndex = i;
            if(P[i].Type == 4 && !STARP(i).Seeded) {
                /* f(Z)-scaled cluster mass: the bound sum overwrites
                 * StarClusterMassUnseeded below, so apply f(Z) here too. */
                m->mGamma = get_seed_metallicity_factor(STARP(i).BirthMetallicity) * STARP(i).ClusterMass;
                m->is_unseeded_star = 1;
            } else {
                m->mGamma = 0;
                m->is_unseeded_star = 0;
            }
        }
    }

    /* ---- Step 2: Allgatherv the member records to all ranks. ---- */
    int n_pack_int = (int) n_pack;
    int * p_counts = (int *) mymalloc2("BMpc", sizeof(int) * NTask);
    MPI_Allgather(&n_pack_int, 1, MPI_INT, p_counts, 1, MPI_INT, Comm);
    int64_t total_members = 0;
    int * p_displs = (int *) mymalloc2("BMpd", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        p_displs[i] = total_members;
        total_members += p_counts[i];
    }
    struct bound_member * bm_global = (struct bound_member *) mymalloc2("BMglobal",
            sizeof(struct bound_member) * (total_members > 0 ? total_members : 1));
    int * pc_bytes = (int *) mymalloc2("BMpcb", sizeof(int) * NTask);
    int * pd_bytes = (int *) mymalloc2("BMpdb", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        pc_bytes[i] = p_counts[i] * (int) sizeof(struct bound_member);
        pd_bytes[i] = p_displs[i] * (int) sizeof(struct bound_member);
    }
    MPI_Allgatherv(bm_local, n_pack_int * (int) sizeof(struct bound_member), MPI_BYTE,
                   bm_global, pc_bytes, pd_bytes, MPI_BYTE, Comm);
    myfree(pd_bytes);
    myfree(pc_bytes);

    /* ---- Step 3: sort by GrNr; process each owned qualifying group. ---- */
    qsort(bm_global, total_members, sizeof(struct bound_member), cmp_bound_member_grnr);

    for(g = 0; g < fof->Ngroups; g++) {
        if(fof->Group[g].StarClusterMassUnseeded <= thresh)
            continue;   /* not qualifying: leave whole-structure values untouched */
        int64_t grNr = fof->Group[g].base.GrNr;

        /* Binary search for this group's contiguous block [start, end). */
        int64_t lo = 0, hi = total_members;
        while(lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if(bm_global[mid].GrNr < grNr) lo = mid + 1;
            else hi = mid;
        }
        int64_t start = lo;
        while(lo < total_members && bm_global[lo].GrNr == grNr)
            lo++;
        int64_t end = lo;
        int64_t N = end - start;
        if(N <= 0) {
            /* No members gathered (should not happen for a qualifying group). */
            fof->Group[g].StarClusterMassUnseeded = 0;
            fof->Group[g].SCcomMcut = 0;
            fof->Group[g].seed_index_star = -1;
            fof->Group[g].seed_task_star  = -1;
            continue;
        }

        /* Pass A: softened potential magnitude at each member. The members are
         * read-only and each iteration writes only its own potmag[a], so the outer
         * loop is parallelized; the inner sum order is fixed, so potmag[] is bitwise
         * identical regardless of thread scheduling (reproducible). */
        double * potmag = (double *) mymalloc2("BMpot", sizeof(double) * N);
        int64_t a, b;
        #pragma omp parallel for schedule(dynamic) private(b)
        for(a = 0; a < N; a++) {
            const struct bound_member * pa = &bm_global[start + a];
            double sum = 0;
            for(b = 0; b < N; b++) {
                if(b == a) continue;
                const struct bound_member * pb = &bm_global[start + b];
                double dx = NEAREST(pa->Pos[0] - pb->Pos[0], BoxSize);
                double dy = NEAREST(pa->Pos[1] - pb->Pos[1], BoxSize);
                double dz = NEAREST(pa->Pos[2] - pb->Pos[2], BoxSize);
                sum += pb->Mass / sqrt(dx * dx + dy * dy + dz * dz + eps2);
            }
            potmag[a] = sum;
        }
        /* Deepest-potential member -> rest frame (serial argmax: cheap, deterministic). */
        int64_t kstar = 0;
        double potmax = -1.0;
        for(a = 0; a < N; a++)
            if(potmag[a] > potmax) { potmax = potmag[a]; kstar = a; }
        const double Vref0 = bm_global[start + kstar].Vel[0];
        const double Vref1 = bm_global[start + kstar].Vel[1];
        const double Vref2 = bm_global[start + kstar].Vel[2];

        /* Pass B: over UNSEEDED stars, sum the bound ones and track the bound star
         * with the largest m*Gamma (ties broken by smaller ID for reproducibility)
         * as the new seed location.
         * Bound iff 0.5*|Vel - Vref|^2 <= atime * G * pot_mag (E <= 0 marginally bound). */
        double M_bound_mGamma = 0, M_bound_starmass = 0;
        double best_mGamma = -1.0;
        const struct bound_member * best = NULL;
        for(a = 0; a < N; a++) {
            const struct bound_member * pa = &bm_global[start + a];
            if(!pa->is_unseeded_star) continue;
            double dvx = pa->Vel[0] - Vref0;
            double dvy = pa->Vel[1] - Vref1;
            double dvz = pa->Vel[2] - Vref2;
            double ke = 0.5 * (dvx * dvx + dvy * dvy + dvz * dvz);
            double pe = atime * G * potmag[a];
            if(ke <= pe) {
                M_bound_mGamma += pa->mGamma;
                M_bound_starmass += pa->Mass;
                if(pa->mGamma > best_mGamma ||
                   (best && pa->mGamma == best_mGamma && pa->ID < best->ID)) {
                    best_mGamma = pa->mGamma;
                    best = pa;
                }
            }
        }
        fof->Group[g].StarClusterMassUnseeded = M_bound_mGamma;
        fof->Group[g].SCcomMcut = M_bound_starmass;
        /* Repoint the seed to the largest-m*Gamma BOUND star (location + RNG seed +
         * ID stay one coherent particle). If no unseeded star is bound, drop the seed
         * so the group cannot seed at an unbound star (Gate 1 also drops it since
         * StarClusterMassUnseeded is now 0). */
        if(best) {
            fof->Group[g].seed_index_star = best->OrigIndex;
            fof->Group[g].seed_task_star  = best->OrigTask;
            fof->Group[g].SeedStarID      = best->ID;
        } else {
            fof->Group[g].seed_index_star = -1;
            fof->Group[g].seed_task_star  = -1;
        }
        myfree(potmag);   /* topmost on the stack each iteration */
    }

    /* ---- Step 4: strict LIFO frees (reverse allocation order). ---- */
    myfree(bm_global);
    myfree(p_displs);
    myfree(p_counts);
    myfree(bm_local);
    myfree(qual_grnr);
    myfree(local_qual);
    myfree(q_displs);
    myfree(q_counts);
}

/* ================= BHseedSecFOFbound: bound-star restriction of the budget =================
 *
 * For every secondary-FOF group, flag the member star particles that are
 * gravitationally bound to (all member stars + the local dark matter), and — when
 * `apply` is set — feed only the bound UNSEEDED stars to the seeding machinery by
 * overwriting StarClusterMassUnseeded (-> bound Sum(f(Z)*Gamma*m_star)) and
 * SCcomMcut (-> bound Sum(m_star)) in place, exactly as
 * fof_secfof_bound_massive_restrict does for its own threshold-gated variant.
 *
 * The DM sphere is centred on PotMinPos with radius
 *     mode 1 : Rdm = Rmax                     (Williams et al. 2025 Eq. 2)
 *     mode 2 : Rdm = min(2*R50, Rmax)
 * where Rmax / R50 are the maximum / half-mass radii of the member stars about that
 * centre.  Rmax is set by the single most distant member star, so once the FoF
 * percolates it tracks a filament rather than the object (measured: Rmax of the
 * dominant group grows 11x between linking lengths 0.244 and 7.32 ckpc/h while R50
 * grows 1.4x, and M_DM(<Rmax)/M_star goes 2 -> 24).  Mode 2 keeps the test on the
 * object.  The min() form is deliberate: for marginal few-star groups 2*R50 can
 * exceed Rmax, and enlarging the sphere there would make the test *more* permissive
 * exactly where it should be strict.
 *
 * POTENTIAL: spherically averaged about the group centre rather than the exact
 * O(N^2) double sum.  With member stars sorted by softened radius s = sqrt(r^2+eps^2),
 *     potmag(i) = [M_star(<r_i) + M_dm(<r_i)] / s_i
 *               + Sum_{stars j: r_j > r_i} m_j / s_j
 *               + Sum_{DM shells beyond r_i} m_shell / s_shell
 * (self-term excluded from both star sums), which is exact for a spherical mass
 * distribution and costs O(N log N) instead of O(N^2).  Validated against the exact
 * sum on this project's z=9 star-primary catalogues at all six linking lengths:
 * per-star flag agreement 97.9-98.4%, and the bound stellar mass is systematically
 * low by 1.7-2.0% (a stable bias, not a drift with configuration).  The exact sum
 * is not affordable here: 93-96% of Sum(N_star^2) sits in ONE group, and that group
 * alone is 3.4e10 pairs already at z=9 / Ng=512, growing as (stellar mass)^2.
 *
 * Boundedness convention matches check_grav_bound() and
 * fof_secfof_bound_massive_restrict(): with the internal velocity Vel = a^2 dx/dt,
 *     0.5*|Vel_i - Vcom|^2  <=  atime * G * potmag(i)
 * (E <= 0, i.e. marginally bound counts as bound).  Vcom is the mass-weighted
 * centre-of-mass velocity of the stars *and* the DM in the sphere, so the test is
 * evaluated in the frame of the system it is testing.
 *
 * DISTRIBUTION: the group geometry and the member stars are replicated on every rank
 * with the Allgatherv pattern of secondfof_compute_sizes; the DM never moves — each
 * rank bins its own local DM into per-group radial histograms which are then summed
 * with one Allreduce.  Every rank therefore evaluates the same profile and each group
 * is written only by its owning rank, so the result is independent of the rank count.
 *
 * MEMORY: the DM histograms are SB_NBIN doubles per group, replicated
 * (TotNgroups * SB_NBIN * 8 bytes on every rank).
 */

/* Radial bins of the per-group DM profile.  The profile is only ever consumed
 * through cumulative sums and the bin width stays below the gravitational
 * softening for every group measured here, so linear bins cost no accuracy. */
#define SB_NBIN 128

/* One member star, replicated on all ranks. */
struct sb_star {
    int64_t  GrNr;
    double   Pos[3];
    double   Vel[3];
    double   Mass;
    double   mGamma;            /* f(Z)*ClusterMass for an UNSEEDED star, else 0 */
    double   msample;           /* StarClusterMass_sample for an UNSEEDED star, else 0 */
    double   scm;               /* host-selection key, identical to the `scm` of
                                 * add_particle_to_group so the bound seed host is
                                 * picked by the same rule as the unrestricted one */
    MyIDType ID;
    int      OrigTask;          /* rank owning this particle */
    int      OrigIndex;         /* local index on OrigTask */
    int      is_unseeded;       /* 1 if STARP.Seeded == 0 */
};

/* Sort by GrNr so each group is a contiguous block; ties broken by ID so the
 * within-group order (and hence every partial sum below) is reproducible. */
static int cmp_sb_star(const void * a, const void * b)
{
    const struct sb_star * pa = (const struct sb_star *) a;
    const struct sb_star * pb = (const struct sb_star *) b;
    if(pa->GrNr != pb->GrNr) return (pa->GrNr > pb->GrNr) - (pa->GrNr < pb->GrNr);
    return (pa->ID > pb->ID) - (pa->ID < pb->ID);
}

/* Per-group geometry, replicated so every rank can bin its own DM. */
struct sb_group {
    int64_t GrNr;
    double  cen[3];     /* PotMinPos */
    double  Rmax;       /* max member-star radius about cen [comoving] */
    double  Rdm;        /* radius of the DM sphere (mode 1: Rmax, mode 2: min(2*R50,Rmax)) */
};

static int cmp_sb_group(const void * a, const void * b)
{
    int64_t va = ((const struct sb_group *)a)->GrNr;
    int64_t vb = ((const struct sb_group *)b)->GrNr;
    return (va > vb) - (va < vb);
}

/* (cell id, group index) pair of the DM lookup grid.  Each group is registered
 * into the 27 cells around its centre, so a DM particle only has to search its
 * own cell (the cell size is >= the largest Rdm). */
struct sb_cell {
    int64_t cell;
    int64_t grp;
};

static int cmp_sb_cell(const void * a, const void * b)
{
    const struct sb_cell * pa = (const struct sb_cell *) a;
    const struct sb_cell * pb = (const struct sb_cell *) b;
    if(pa->cell != pb->cell) return (pa->cell > pb->cell) - (pa->cell < pb->cell);
    return (pa->grp > pb->grp) - (pa->grp < pb->grp);
}

/* Comoving radius of a member star about its group centre (unsoftened; the
 * softening is applied where the potential is evaluated). */
static inline double sb_radius(const double * pos, const double * cen, double BoxSize)
{
    double dx = NEAREST(pos[0] - cen[0], BoxSize);
    double dy = NEAREST(pos[1] - cen[1], BoxSize);
    double dz = NEAREST(pos[2] - cen[2], BoxSize);
    return sqrt(dx * dx + dy * dy + dz * dz);
}

/* (radius, mass, index) triple, sorted by radius to build the profile.  Ties are
 * broken by the index, which is the position in the ID-sorted member block, so the
 * ordering — and hence every partial sum taken from it — is reproducible. */
struct sb_rm {
    double  r;
    double  m;
    int64_t k;
};

static int cmp_sb_rm(const void * a, const void * b)
{
    const struct sb_rm * pa = (const struct sb_rm *) a;
    const struct sb_rm * pb = (const struct sb_rm *) b;
    if(pa->r != pb->r) return (pa->r > pb->r) - (pa->r < pb->r);
    return (pa->k > pb->k) - (pa->k < pb->k);
}

void
fof_secfof_bound_restrict(FOFGroups * fof, int mode, int apply,
                          double atime, Cosmology * CP, MPI_Comm Comm)
{
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    const double BoxSize = PartManager->BoxSize;
    const double G = CP->GravInternal;
    const double eps = FORCE_SOFTENING() / 2.8;   /* Plummer-equivalent, comoving */
    const double eps2 = eps * eps;
    int i, g;
    int64_t k;

    if(mode != 1 && mode != 2)
        return;

    /* ---- Step 0: replicate (GrNr, PotMinPos) of every group on every rank. ---- */
    int local_ngroups = (int) fof->Ngroups;
    int * gc_counts = (int *) mymalloc2("SBgc", sizeof(int) * NTask);
    MPI_Allgather(&local_ngroups, 1, MPI_INT, gc_counts, 1, MPI_INT, Comm);
    int64_t total_groups = 0;
    int * gc_displs = (int *) mymalloc2("SBgd", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        gc_displs[i] = (int) total_groups;
        total_groups += gc_counts[i];
    }
    if(total_groups == 0) {
        myfree(gc_displs);
        myfree(gc_counts);
        return;
    }
    /* Checked up front: it also bounds the `int` group loop counters below. */
    if(total_groups > (int64_t) INT_MAX / SB_NBIN)
        endrun(1, "BHseedSecFOFbound: %ld secFOF groups x %d radial bins overflows the "
                  "MPI_Allreduce count. Raise SecondFOFMinPrimaryLength or lower SB_NBIN.\n",
                  total_groups, SB_NBIN);

    struct sb_group * sbg = (struct sb_group *)
        mymalloc2("SBgrp", sizeof(struct sb_group) * total_groups);
    {
        struct sb_group * loc = (struct sb_group *)
            mymalloc2("SBgrpl", sizeof(struct sb_group) * (local_ngroups > 0 ? local_ngroups : 1));
        for(g = 0; g < local_ngroups; g++) {
            loc[g].GrNr = fof->Group[g].base.GrNr;
            int d;
            for(d = 0; d < 3; d++)
                loc[g].cen[d] = fof->Group[g].PotMinPos[d];
            loc[g].Rmax = 0;
            loc[g].Rdm = 0;
        }
        int * cb = (int *) mymalloc2("SBgcb", sizeof(int) * NTask);
        int * db = (int *) mymalloc2("SBgdb", sizeof(int) * NTask);
        for(i = 0; i < NTask; i++) {
            cb[i] = gc_counts[i] * (int) sizeof(struct sb_group);
            db[i] = gc_displs[i] * (int) sizeof(struct sb_group);
        }
        MPI_Allgatherv(loc, local_ngroups * (int) sizeof(struct sb_group), MPI_BYTE,
                       sbg, cb, db, MPI_BYTE, Comm);
        myfree(db);
        myfree(cb);
        myfree(loc);
    }
    qsort(sbg, total_groups, sizeof(struct sb_group), cmp_sb_group);

    /* ---- Step 1: replicate every group's member STARS on every rank. ---- */
    int64_t n_local_star = 0;
    for(i = 0; i < PartManager->NumPart; i++)
        if(P[i].Type == 4 && P[i].GrNr >= 0)
            n_local_star++;

    int * sc_counts = (int *) mymalloc2("SBsc", sizeof(int) * NTask);
    int n_local_star_int = (int) n_local_star;
    MPI_Allgather(&n_local_star_int, 1, MPI_INT, sc_counts, 1, MPI_INT, Comm);
    int64_t total_stars = 0;
    int * sc_displs = (int *) mymalloc2("SBsd", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        sc_displs[i] = (int) total_stars;
        total_stars += sc_counts[i];
    }

    struct sb_star * sbs = (struct sb_star *)
        mymalloc2("SBstar", sizeof(struct sb_star) * (total_stars > 0 ? total_stars : 1));
    {
        struct sb_star * loc = (struct sb_star *)
            mymalloc2("SBstarl", sizeof(struct sb_star) * (n_local_star > 0 ? n_local_star : 1));
        int64_t n = 0;
        for(i = 0; i < PartManager->NumPart; i++) {
            if(!(P[i].Type == 4 && P[i].GrNr >= 0)) continue;
            struct sb_star * m = &loc[n++];
            m->GrNr = P[i].GrNr;
            int d;
            for(d = 0; d < 3; d++) {
                m->Pos[d] = P[i].Pos[d];
                m->Vel[d] = P[i].Vel[d];
            }
            m->Mass = P[i].Mass;
            m->ID = P[i].ID;
            m->OrigTask = ThisTask;
            m->OrigIndex = i;
            m->is_unseeded = !STARP(i).Seeded;
            /* f(Z)-scaled cluster mass: the bound sum replaces
             * StarClusterMassUnseeded, which carries f(Z) too. */
            m->mGamma = m->is_unseeded ?
                get_seed_metallicity_factor(STARP(i).BirthMetallicity) * STARP(i).ClusterMass : 0;
            m->msample = m->is_unseeded ? STARP(i).StarClusterMass_sample : 0;
            m->scm = (fof_params.StarClusterSampling && !fof_params.SeedSecFOFcomSample) ?
                m->msample : m->mGamma;
        }
        int * cb = (int *) mymalloc2("SBscb", sizeof(int) * NTask);
        int * db = (int *) mymalloc2("SBsdb", sizeof(int) * NTask);
        for(i = 0; i < NTask; i++) {
            cb[i] = sc_counts[i] * (int) sizeof(struct sb_star);
            db[i] = sc_displs[i] * (int) sizeof(struct sb_star);
        }
        MPI_Allgatherv(loc, n_local_star_int * (int) sizeof(struct sb_star), MPI_BYTE,
                       sbs, cb, db, MPI_BYTE, Comm);
        myfree(db);
        myfree(cb);
        myfree(loc);
    }
    qsort(sbs, total_stars, sizeof(struct sb_star), cmp_sb_star);

    /* Block [gstart[g], gend[g]) of each group's stars in sbs.  Both arrays are
     * sorted by GrNr, so one merge pass suffices; a group with no member star gets
     * an empty block.  Every rank does this for EVERY group (it has all the stars),
     * because the DM binning below needs the geometry of groups it does not own. */
    int64_t * gstart = (int64_t *) mymalloc2("SBgs", sizeof(int64_t) * total_groups);
    int64_t * gend   = (int64_t *) mymalloc2("SBge", sizeof(int64_t) * total_groups);
    {
        int64_t pos = 0;
        for(g = 0; g < total_groups; g++) {
            while(pos < total_stars && sbs[pos].GrNr < sbg[g].GrNr)
                pos++;
            gstart[g] = pos;
            while(pos < total_stars && sbs[pos].GrNr == sbg[g].GrNr)
                pos++;
            gend[g] = pos;
        }
    }

    /* Rmax and the stellar half-mass radius R50 about the group centre, hence the
     * DM sphere radius Rdm.  Sorting the (radius, mass) pairs once gives both
     * exactly and is cheaper than bisecting on the half-mass condition. */
    double Rdm_max = 0;
    for(g = 0; g < total_groups; g++) {
        const int64_t s = gstart[g], e = gend[g], ns = e - s;
        if(ns <= 0) { sbg[g].Rmax = 0; sbg[g].Rdm = 0; continue; }
        struct sb_rm * rm = (struct sb_rm *) mymalloc2("SBrm0", sizeof(struct sb_rm) * ns);
        double mtot = 0;
        for(k = 0; k < ns; k++) {
            rm[k].r = sb_radius(sbs[s + k].Pos, sbg[g].cen, BoxSize);
            rm[k].m = sbs[s + k].Mass;
            rm[k].k = k;
            mtot += rm[k].m;
        }
        qsort(rm, ns, sizeof(struct sb_rm), cmp_sb_rm);
        double rmax = rm[ns - 1].r;
        double r50 = rmax, acc = 0;
        for(k = 0; k < ns; k++) {
            acc += rm[k].m;
            if(acc >= 0.5 * mtot) { r50 = rm[k].r; break; }
        }
        myfree(rm);
        sbg[g].Rmax = rmax;
        sbg[g].Rdm = (mode == 2 && 2.0 * r50 < rmax) ? 2.0 * r50 : rmax;
        if(sbg[g].Rdm > Rdm_max) Rdm_max = sbg[g].Rdm;
    }

    /* ---- Step 2: bin the local DM into per-group radial histograms. ---- */
    /* hist[g*SB_NBIN + b] = DM mass in shell b of group g; mom[g*4 + 0..2] = DM
     * momentum inside Rdm, mom[g*4+3] = DM mass inside Rdm. */
    double * dmhist = (double *) mymalloc2("SBhist", sizeof(double) * total_groups * SB_NBIN);
    double * dmmom  = (double *) mymalloc2("SBmom", sizeof(double) * total_groups * 4);
    memset(dmhist, 0, sizeof(double) * total_groups * SB_NBIN);
    memset(dmmom, 0, sizeof(double) * total_groups * 4);

    if(Rdm_max > 0) {
        /* Lookup grid: cell size >= Rdm_max so registering each group in its 27
         * neighbouring cells lets a DM particle search only its own cell. */
        int ncell = (int) (BoxSize / Rdm_max);
        if(ncell < 1) ncell = 1;
        if(ncell > 512) ncell = 512;
        const double lcell = BoxSize / ncell;

        int64_t ncell_ent = 0;
        struct sb_cell * cells = (struct sb_cell *)
            mymalloc2("SBcell", sizeof(struct sb_cell) * total_groups * 27);
        for(g = 0; g < total_groups; g++) {
            if(sbg[g].Rdm <= 0) continue;
            int ci = (int) floor(sbg[g].cen[0] / lcell);
            int cj = (int) floor(sbg[g].cen[1] / lcell);
            int ck = (int) floor(sbg[g].cen[2] / lcell);
            int di, dj, dk;
            for(di = -1; di <= 1; di++)
            for(dj = -1; dj <= 1; dj++)
            for(dk = -1; dk <= 1; dk++) {
                int a = ((ci + di) % ncell + ncell) % ncell;
                int b = ((cj + dj) % ncell + ncell) % ncell;
                int c = ((ck + dk) % ncell + ncell) % ncell;
                cells[ncell_ent].cell = ((int64_t) a * ncell + b) * ncell + c;
                cells[ncell_ent].grp = g;
                ncell_ent++;
            }
        }
        qsort(cells, ncell_ent, sizeof(struct sb_cell), cmp_sb_cell);
        /* Drop duplicate (cell, group) pairs.  With ncell < 3 the 27 neighbour
         * offsets alias onto the same cell (ncell == 1 maps all of them to cell 0),
         * so without this a DM particle would be binned into the same group up to
         * 27 times.  cells is sorted by (cell, grp), so duplicates are adjacent. */
        {
            int64_t w = 0, t;
            for(t = 0; t < ncell_ent; t++) {
                if(w > 0 && cells[w - 1].cell == cells[t].cell
                         && cells[w - 1].grp == cells[t].grp)
                    continue;
                cells[w++] = cells[t];
            }
            ncell_ent = w;
        }

        /* Deliberately serial.  Threading this would need either per-thread copies of
         * the histogram (total_groups*SB_NBIN doubles each) or atomic adds, and the
         * latter would make the bin sums depend on the thread schedule.  The loop is
         * one binary search per local DM particle into a table of 27*Ngroups entries
         * (~1.7e7 operations for a 512^3 run on 80 ranks), which is negligible next
         * to the Allgatherv above. */
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].Type != 1) continue;
            int ci = (int) floor(P[i].Pos[0] / lcell);
            int cj = (int) floor(P[i].Pos[1] / lcell);
            int ck = (int) floor(P[i].Pos[2] / lcell);
            ci = (ci % ncell + ncell) % ncell;
            cj = (cj % ncell + ncell) % ncell;
            ck = (ck % ncell + ncell) % ncell;
            int64_t key = ((int64_t) ci * ncell + cj) * ncell + ck;
            /* first entry with this cell id */
            int64_t lo = 0, hi = ncell_ent;
            while(lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                if(cells[mid].cell < key) lo = mid + 1;
                else hi = mid;
            }
            int64_t t;
            for(t = lo; t < ncell_ent && cells[t].cell == key; t++) {
                int64_t gg = cells[t].grp;
                double Rdm = sbg[gg].Rdm;
                double dx = NEAREST(P[i].Pos[0] - sbg[gg].cen[0], BoxSize);
                double dy = NEAREST(P[i].Pos[1] - sbg[gg].cen[1], BoxSize);
                double dz = NEAREST(P[i].Pos[2] - sbg[gg].cen[2], BoxSize);
                double r2 = dx * dx + dy * dy + dz * dz;
                if(r2 > Rdm * Rdm) continue;
                double r = sqrt(r2);
                int b = (int) (r / Rdm * SB_NBIN);
                if(b < 0) b = 0;
                if(b >= SB_NBIN) b = SB_NBIN - 1;
                dmhist[gg * SB_NBIN + b] += P[i].Mass;
                dmmom[gg * 4 + 0] += P[i].Mass * P[i].Vel[0];
                dmmom[gg * 4 + 1] += P[i].Mass * P[i].Vel[1];
                dmmom[gg * 4 + 2] += P[i].Mass * P[i].Vel[2];
                dmmom[gg * 4 + 3] += P[i].Mass;
            }
        }
        myfree(cells);
    }

    /* Sum the per-rank histograms.  With a uniform DM mass the bin sums are exact
     * integers times m_dm, so the reduction order does not matter; for a run with
     * unequal DM masses the last bits may depend on the rank count. */
    MPI_Allreduce(MPI_IN_PLACE, dmhist, (int) (total_groups * SB_NBIN), MPI_DOUBLE, MPI_SUM, Comm);
    MPI_Allreduce(MPI_IN_PLACE, dmmom, (int) (total_groups * 4), MPI_DOUBLE, MPI_SUM, Comm);

    /* ---- Step 3: per OWNED group, flag the bound stars and (optionally) restrict. ---- */
    int64_t n_restricted = 0, n_dropped = 0;
    for(g = 0; g < local_ngroups; g++) {
        struct Group * grp = &fof->Group[g];
        /* locate this group in the replicated arrays */
        int64_t lo = 0, hi = total_groups, gi = -1;
        while(lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if(sbg[mid].GrNr < grp->base.GrNr) lo = mid + 1;
            else hi = mid;
        }
        if(lo < total_groups && sbg[lo].GrNr == grp->base.GrNr) gi = lo;

        grp->SCBoundStarMass = 0;
        grp->SCBoundStarMassUnseeded = 0;
        grp->SCBoundClusterMass = 0;
        grp->NStarBound = 0;
        grp->SCBoundRdm = 0;
        grp->SCBoundMdm = 0;

        if(gi < 0 || gend[gi] <= gstart[gi]) {
            /* no member stars: nothing can seed */
            if(apply) {
                grp->StarClusterMassUnseeded = 0;
                grp->SCcomMcut = 0;
                grp->seed_index_star = -1;
                grp->seed_task_star = -1;
            }
            continue;
        }

        const int64_t s = gstart[gi], e = gend[gi];
        const int64_t N = e - s;
        const double Rdm = sbg[gi].Rdm;
        grp->SCBoundRdm = (float) Rdm;
        grp->SCBoundMdm = (float) dmmom[gi * 4 + 3];

        /* Member stars sorted by radius; rm[].k indexes back into the block. */
        struct sb_rm * rm = (struct sb_rm *) mymalloc2("SBrm", sizeof(struct sb_rm) * N);
        for(k = 0; k < N; k++) {
            rm[k].r = sb_radius(sbs[s + k].Pos, sbg[gi].cen, BoxSize);
            rm[k].m = sbs[s + k].Mass;
            rm[k].k = k;
        }
        qsort(rm, N, sizeof(struct sb_rm), cmp_sb_rm);

        /* Star prefix (mass strictly inside r_k) and suffix (Sum m/s strictly outside),
         * both excluding the star itself, matching the exact potential's self-term. */
        double * Min = (double *) mymalloc2("SBmin", sizeof(double) * N);
        double * Tout = (double *) mymalloc2("SBtout", sizeof(double) * N);
        {
            double acc = 0;
            for(k = 0; k < N; k++) { Min[k] = acc; acc += rm[k].m; }
            double t = 0;
            for(k = N - 1; k >= 0; k--) {
                Tout[k] = t;
                t += rm[k].m / sqrt(rm[k].r * rm[k].r + eps2);
            }
        }

        /* DM shell prefix / suffix at bin resolution. */
        double dr = (Rdm > 0) ? Rdm / SB_NBIN : 0;
        double * Mdm_cum = (double *) mymalloc2("SBmdmc", sizeof(double) * (SB_NBIN + 1));
        double * Tdm     = (double *) mymalloc2("SBtdm", sizeof(double) * (SB_NBIN + 1));
        {
            int b;
            Mdm_cum[0] = 0;
            for(b = 0; b < SB_NBIN; b++)
                Mdm_cum[b + 1] = Mdm_cum[b] + dmhist[gi * SB_NBIN + b];
            Tdm[SB_NBIN] = 0;
            for(b = SB_NBIN - 1; b >= 0; b--) {
                double c = (b + 0.5) * dr;
                double sc = sqrt(c * c + eps2);
                Tdm[b] = Tdm[b + 1] + dmhist[gi * SB_NBIN + b] / sc;
            }
        }

        /* Rest frame: mass-weighted COM velocity of the member stars + the DM in the
         * sphere, i.e. the frame of the system whose binding is being tested. */
        double Vcom[3] = {0, 0, 0};
        {
            double msum = dmmom[gi * 4 + 3];
            int d;
            for(d = 0; d < 3; d++) Vcom[d] = dmmom[gi * 4 + d];
            for(k = 0; k < N; k++) {
                msum += sbs[s + k].Mass;
                for(d = 0; d < 3; d++) Vcom[d] += sbs[s + k].Mass * sbs[s + k].Vel[d];
            }
            if(msum > 0) for(d = 0; d < 3; d++) Vcom[d] /= msum;
        }

        double bound_mass = 0, bound_mass_uns = 0, bound_mgamma = 0, bound_msample = 0;
        int nbound = 0;
        double best_scm = -1.0;
        const struct sb_star * best = NULL;
        for(k = 0; k < N; k++) {
            const struct sb_star * st = &sbs[s + rm[k].k];
            const double rk_ = rm[k].r;                       /* unsoftened radius */
            const double sk = sqrt(rk_ * rk_ + eps2);         /* softened radius */

            /* DM mass inside r_k and the shell-tail beyond it (linear within the bin). */
            double Mdm_in = 0, Tdm_out = 0;
            if(dr > 0) {
                double x = rk_ / dr;
                int b = (int) x;
                if(b >= SB_NBIN) { b = SB_NBIN - 1; x = SB_NBIN; }
                if(b < 0) { b = 0; x = 0; }
                double frac = x - b;
                if(frac < 0) frac = 0;
                if(frac > 1) frac = 1;
                double mb = dmhist[gi * SB_NBIN + b];
                double cb = (b + 0.5) * dr;
                double scb = sqrt(cb * cb + eps2);
                Mdm_in = Mdm_cum[b] + mb * frac;
                Tdm_out = Tdm[b + 1] + mb * (1.0 - frac) / scb;
            }

            double potmag = (Min[k] + Mdm_in) / sk + Tout[k] + Tdm_out;
            double dvx = st->Vel[0] - Vcom[0];
            double dvy = st->Vel[1] - Vcom[1];
            double dvz = st->Vel[2] - Vcom[2];
            double ke = 0.5 * (dvx * dvx + dvy * dvy + dvz * dvz);
            double pe = atime * G * potmag;
            if(ke <= pe) {
                nbound++;
                bound_mass += st->Mass;
                if(st->is_unseeded) {
                    bound_mass_uns += st->Mass;
                    bound_mgamma += st->mGamma;
                    bound_msample += st->msample;
                    /* Same key as add_particle_to_group's MaxStarClusterMass, so the
                     * bound host is the one the unrestricted run would have chosen had
                     * the unbound stars not existed.  Ties go to the smaller ID. */
                    if(st->scm > best_scm ||
                       (best && st->scm == best_scm && st->ID < best->ID)) {
                        best_scm = st->scm;
                        best = st;
                    }
                }
            }
        }

        grp->SCBoundStarMass = bound_mass;
        grp->SCBoundStarMassUnseeded = bound_mass_uns;
        grp->SCBoundClusterMass = bound_mgamma;
        grp->NStarBound = nbound;

        if(apply) {
            /* Every quantity the seeding gates read off the UNSEEDED stars is replaced
             * by its bound-only counterpart.  Both cluster-mass variants must be done:
             * the combined sampler reads StarClusterMassUnseeded, while the plain
             * secFOF path with StarClusterSampling=1 reads StarClusterMassSampleUnseeded.
             *
             * NStarUnseeded is deliberately NOT overwritten: it is the denominator of
             * the SCMetUnseeded* metallicity histogram/sums, which were accumulated over
             * all unseeded stars in add_particle_to_group and cannot be recomputed here.
             * The multi-seed host caps derived from it therefore still count every
             * unseeded star; those modes (SeedInSecFOFMultipleSeeds, SecFOFseedsumover=0)
             * cap the seed COUNT, not the budget, so the restriction is still applied to
             * every mass that decides whether and how massive a seed is. */
            double old = fof_params.StarClusterSampling && !fof_params.SeedSecFOFcomSample ?
                grp->StarClusterMassSampleUnseeded : grp->StarClusterMassUnseeded;
            double neu = fof_params.StarClusterSampling && !fof_params.SeedSecFOFcomSample ?
                bound_msample : bound_mgamma;
            if(old > 0 && neu <= 0) n_dropped++;
            else if(neu < old) n_restricted++;

            grp->StarClusterMassUnseeded = bound_mgamma;
            grp->StarClusterMassSampleUnseeded = bound_msample;
            grp->SCcomMcut = bound_mass_uns;
            /* Repoint the seed host to the best BOUND unseeded star so a BH is never
             * seeded at an unbound star; drop the seed if none is bound (the gates drop
             * the group anyway, since the budget is now 0). */
            if(best) {
                grp->seed_index_star = best->OrigIndex;
                grp->seed_task_star  = best->OrigTask;
                grp->SeedStarID      = best->ID;
                grp->MaxStarClusterMass = best_scm;
            } else {
                grp->seed_index_star = -1;
                grp->seed_task_star  = -1;
                grp->MaxStarClusterMass = 0;
            }
        }

        /* LIFO within the iteration: Tdm, Mdm_cum, Tout, Min, rm. */
        myfree(Tdm);
        myfree(Mdm_cum);
        myfree(Tout);
        myfree(Min);
        myfree(rm);
    }

    if(apply) {
        int64_t tot[2] = {n_restricted, n_dropped}, all[2];
        MPI_Allreduce(tot, all, 2, MPI_INT64, MPI_SUM, Comm);
        message(0, "BHseedSecFOFbound=%d: restricted the seeding budget of %ld secFOF groups "
                   "(%ld lost every bound unseeded star and can no longer seed).\n",
                   mode, all[0], all[1]);
    }

    /* ---- Step 4: strict LIFO frees (reverse allocation order). ---- */
    myfree(dmmom);
    myfree(dmhist);
    myfree(gend);
    myfree(gstart);
    myfree(sbs);
    myfree(sc_displs);
    myfree(sc_counts);
    myfree(sbg);
    myfree(gc_displs);
    myfree(gc_counts);
}

int fof_get_secfof_bound_mode(void)
{
    return fof_params.BHseedSecFOFbound;
}

void fof_seed(FOFGroups * fof, ActiveParticles * act, ForceTree * tree, double atime, const RandTable * const rnd,
              int64_t ** seeded_grnr_out, int * n_seeded_out,
              double ** seeded_totmsc_out, double ** seeded_mcut_out, Cosmology * CP, MPI_Comm Comm)
{
    int i, j, n, ntot;

    int NTask;
    MPI_Comm_size(Comm, &NTask);

    /* BHseedSecFOFbound: restrict every group's seeding budget to its gravitationally
     * bound stars BEFORE any gate or sampler reads it.  This has to sit ahead of the
     * mask loop below (not inside the SeedSecFOFcomSample branch, where the older
     * SeedSeedFOFMassiveBoundStar hooks in) because it applies to every secFOF
     * seeding variant.  get_seed_in_secfof() keeps it off the primary-FOF fof_seed
     * call: the parameter requires SeedInSecFOFasStarCluster=1, and when that is on
     * run.c reaches fof_seed only through secondfof_seed. */
    if(fof_params.BHseedSecFOFbound && get_seed_in_secfof())
        fof_secfof_bound_restrict(fof, fof_params.BHseedSecFOFbound, 1, atime, CP, Comm);

    char * Marked = (char *) mymalloc2("SeedMark", fof->Ngroups);

    int Nexport = 0;
    #pragma omp parallel for reduction(+:Nexport)
    for(i = 0; i < fof->Ngroups; i++)
    {
        int SC_Mask = 0;
        int Gas_Mask = 0;
        int Halo_Mask = 0;
        if(fof_params.BlackHoleSeedStarCluster && !fof_params.SeedSecFOFcomSample){
            double sc_mass_for_seed = fof_params.StarClusterSampling ?
                fof->Group[i].StarClusterMassSampleUnseeded : fof->Group[i].StarClusterMassUnseeded;
            /* Require a star seed: SC seeding converts the largest-ClusterMass star,
             * never gas (a densest-gas seed_index, if gas is a secondary link type,
             * must not be used here). */
            SC_Mask =
                (sc_mass_for_seed >= fof_params.MinMscForBHseed)
            &&  (fof->Group[i].seed_index_star >= 0);
        }
        else{
            /* SeedSecFOFcomSample: deferred to the serial pass below
             * (the per-group stochastic draw is not OpenMP-safe). */
            SC_Mask = 0;
        }


        if (fof_params.BlackHoleSeedGasBased){
            Gas_Mask =
                (fof->Group[i].Mass >= fof_params.MinFoFMassForNewSeed)
            &&  (fof->Group[i].sfmp_mass >= fof_params.BlackHoleSeedsfmpGas)
            &&  (fof->Group[i].LenType[5] == 0)
            &&  (fof->Group[i].seed_index >= 0);
        }
        else{
            Gas_Mask = 0;
        }

        if(fof_params.BlackHoleSeedHaloBased){
            Halo_Mask =
                (fof->Group[i].Mass >= fof_params.MinFoFMassForNewSeed)
            &&  (fof->Group[i].MassType[4] >= fof_params.MinMStarForNewSeed)
            &&  (fof->Group[i].LenType[5] == 0)
            &&  (fof->Group[i].seed_index >= 0);
        }
        else{
            Halo_Mask = 0;
        }

        if(SC_Mask || Gas_Mask || Halo_Mask){
            Marked[i] = 1;
        }
        else{
            Marked[i] = 0;
        }

        /* Star-cluster seeding must seed from the star, never from gas.  Override
         * any densest-gas seed_index UNCONDITIONALLY (gas can be a secondary link
         * type in the secondary FOF, so seed_index may be >= 0 even here; the old
         * `seed_index < 0` guard left that gas index in place and converted gas).
         * Mirrors the SeedSecFOFcomSample serial pass below. */
        if(SC_Mask && fof->Group[i].seed_index_star >= 0) {
            fof->Group[i].seed_index = fof->Group[i].seed_index_star;
            fof->Group[i].seed_task = fof->Group[i].seed_task_star;
        }

        if(Marked[i]) Nexport ++;
    }

    /* SeedSecFOFcomSample: serial pass (the sampler uses the global GSL error
     * path, so the per-group draw cannot run under OpenMP).
     *   Gate 1 (cheap pre-filter): unseeded cluster-forming mass >= MinMscForBHseed.
     *   Gate 2 (decision): BHSeedMsc for the group must reach MinMscForBHseed.
     * BHSeedMsc is the single combined per-group draw, or — when
     * SeedSecFOFcomSampleParticle is on — tot_msc_fof summed from per-star draws,
     * computed collectively first by fof_secfof_particle_sample(). */
    if(fof_params.SeedSecFOFcomSample && fof_params.BlackHoleSeedStarCluster
       && fof_params.SecFOFseedsumover) {
        /* Restrict massive (> 1e8 Msun) groups to their gravitationally bound
         * unseeded stars BEFORE the sampler/gates run; overwrites the two input
         * fields (StarClusterMassUnseeded, SCcomMcut) in place. */
        if(fof_params.SeedSeedFOFMassiveBoundStar)
            fof_secfof_bound_massive_restrict(fof, atime, CP, Comm);
        if(fof_params.SeedSecFOFcomSampleParticle)
            fof_secfof_particle_sample(fof, rnd, Comm);
        for(i = 0; i < fof->Ngroups; i++) {
            if(!fof_params.SeedSecFOFcomSampleParticle) {
                fof->Group[i].BHSeedMsc = 0;
                fof->Group[i].BHSampledMscTotal = 0;
            }
            if(fof->Group[i].seed_index_star < 0)
                continue;
            if(fof->Group[i].StarClusterMassUnseeded < fof_params.MinMscForBHseed) /* Gate 1 */
                continue;
            if(!fof_params.SeedSecFOFcomSampleParticle) {
                /* Single combined per-group draw (cap applied internally). */
                double sampled_total = 0;
                double bhseed_msc = starcluster_combined_bhseed_msc(
                        fof->Group[i].SCcomMcut, fof->Group[i].StarClusterMassUnseeded,
                        (uint64_t) fof->Group[i].SeedStarID, rnd, &sampled_total, 1);
                fof->Group[i].BHSeedMsc = bhseed_msc;
                fof->Group[i].BHSampledMscTotal = sampled_total;
            }
            /* else: BHSeedMsc/BHSampledMscTotal already set by the per-particle pass. */
            if(fof->Group[i].BHSeedMsc >= fof_params.MinMscForBHseed) {     /* Gate 2 */
                if(!Marked[i]) {
                    Marked[i] = 1;
                    Nexport++;
                }
                /* Always seed at the largest-ClusterMass star.  Override any gas
                 * seed_index unconditionally: if gas is a secondary link type the
                 * group may carry a densest-gas seed_index >= 0, which must NOT be
                 * used here (that would convert gas in place instead of spawning
                 * a BH from the star). */
                fof->Group[i].seed_index = fof->Group[i].seed_index_star;
                fof->Group[i].seed_task = fof->Group[i].seed_task_star;
            }
        }
    }

    struct Group * ExportGroups = (struct Group *) mymalloc("Export", sizeof(fof->Group[0]) * Nexport);
    j = 0;
    for(i = 0; i < fof->Ngroups; i ++) {
        if(Marked[i]) {
            ExportGroups[j] = fof->Group[i];
            j++;
        }
    }
    myfree(Marked);

    qsort_openmp(ExportGroups, Nexport, sizeof(ExportGroups[0]), cmp_seed_task);

    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    memset(Send_count, 0, NTask * sizeof(int));
    for(i = 0; i < Nexport; i++) {
        Send_count[ExportGroups[i].seed_task]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int Nimport = 0;

    for(j = 0;  j < NTask; j++)
    {
        Nimport += Recv_count[j];
    }

    struct Group * ImportGroups = (struct Group *)
            mymalloc2("ImportGroups", Nimport * sizeof(struct Group));

    MPI_Alltoallv_smart(ExportGroups, Send_count, NULL, MPI_TYPE_GROUP,
                        ImportGroups, Recv_count, NULL, MPI_TYPE_GROUP,
                        Comm);

    myfree(ExportGroups);
    ta_free(Recv_count);
    ta_free(Send_count);

    MPI_Allreduce(&Nimport, &ntot, 1, MPI_INT, MPI_SUM, Comm);

    message(0, "Making %d new black hole particles.\n", ntot);

    /* Per-secFOF multi-seeding may convert additional (2nd..N_seed) stars into BHs
     * later (fof_secfof_extra_seeds).  Those also need BH slots, so include an upper
     * bound here in the single slots_reserve below (the bottom-stack relocation of
     * the force tree / ActiveParticle around slots_reserve makes the growth safe). */
    int64_t n_extra_ub = secfof_count_extra_seed_ub(fof, Comm);   /* local upper bound */
    int64_t ntot_extra = 0;
    MPI_Allreduce(&n_extra_ub, &ntot_extra, 1, MPI_INT64, MPI_SUM, Comm);

    /* Per-cluster seeding (SecFOFseedsumover=0) places all its seeds in
     * fof_secfof_random_seeds (none go through the single-seed Nimport path), so
     * reserve an upper bound here too. */
    int64_t n_random_ub = secfof_count_random_seed_ub(fof, rnd, Comm);   /* local upper bound */
    int64_t ntot_random = 0;
    MPI_Allreduce(&n_random_ub, &ntot_random, 1, MPI_INT64, MPI_SUM, Comm);

    /* Do we have enough black hole slots to create this many black holes?
     * If not, allocate more slots. */
    if(Nimport + n_extra_ub + n_random_ub + SlotsManager->info[5].size > SlotsManager->info[5].maxsize)
    {
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
        /* This is only called on a PM step, so the condition should normally be false. */
        if(act->ActiveParticle) {
            ActiveParticle_tmp = (int *) mymalloc2("ActiveParticle_tmp", act->NumActiveParticle * sizeof(int));
            memmove(ActiveParticle_tmp, act->ActiveParticle, act->NumActiveParticle * sizeof(int));
            myfree(act->ActiveParticle);
        }

        /*Now SlotsBase is the top of the bottom stack: extend the slots! */
        int64_t atleast[6];
        int64_t i;
        for(i = 0; i < 6; i++)
            atleast[i] = SlotsManager->info[i].maxsize;
        atleast[5] += (ntot + ntot_extra + ntot_random)*1.1;
        slots_reserve(1, atleast, SlotsManager);

        /*And now we need our memory back in the right place (reverse allocation order)*/
        if(ActiveParticle_tmp) {
            act->ActiveParticle = (int *) mymalloc("ActiveParticle", sizeof(int)*(act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
            memmove(act->ActiveParticle, ActiveParticle_tmp, act->NumActiveParticle * sizeof(int));
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

    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    /* Both star-cluster and gas/halo seeds are converted in-place into BHs
     * (the parent star or gas particle is consumed), so no new base particles
     * are created here — only BH slots, reserved above.  NumPart is unchanged. */
    for(n = 0; n < Nimport; n++)
    {
        fof_seed_make_one(&ImportGroups[n], ThisTask, atime, rnd);
    }

    /* Per-secFOF multi-seeding: place the extra (2nd..N_seed) BH seeds for massive
     * groups now that seed 1 exists.  Uses pre-reserved BH slots; converts the chosen
     * stars in place.  Collective (every rank participates). */
    fof_secfof_extra_seeds(fof, atime, rnd, Comm);

    /* Per-cluster seeding (SecFOFseedsumover=0): place one BH per sampled cluster
     * >= MinMscForBHseed on its chosen unseeded host star (random or largest
     * f(Z)*ClusterMass, per SeedInSecFOFRandomStarParticle).  Self-contained
     * multi-seed pass (mutually exclusive with the paths above); uses pre-reserved
     * BH slots.  Collective (every rank participates). */
    fof_secfof_random_seeds(fof, atime, rnd, CP, Comm);

    /* Optionally return the GrNr (and, for SeedSecFOFcomSampleParticle, the
     * per-group tot_msc_fof = BHSeedMsc and unseeded stellar mass = SCcomMcut) of
     * each locally-seeded group.  Use ta_malloc for the temporaries so they live
     * on the thread-local allocator, then free ImportGroups (mymalloc2), then copy
     * into mymalloc2 buffers the caller owns and frees (reverse: mcut,totmsc,grnr). */
    int64_t * seeded_grnr_tmp = NULL;
    double * seeded_totmsc_tmp = NULL;
    double * seeded_mcut_tmp = NULL;
    int n_seeded_local = 0;
    if(seeded_grnr_out && n_seeded_out && Nimport > 0) {
        seeded_grnr_tmp = ta_malloc("SeededGrNrTmp", int64_t, Nimport);
        if(seeded_totmsc_out)
            seeded_totmsc_tmp = ta_malloc("SeededTotMscTmp", double, Nimport);
        if(seeded_mcut_out)
            seeded_mcut_tmp = ta_malloc("SeededMcutTmp", double, Nimport);
        for(n = 0; n < Nimport; n++) {
            seeded_grnr_tmp[n] = ImportGroups[n].base.GrNr;
            if(seeded_totmsc_tmp) seeded_totmsc_tmp[n] = ImportGroups[n].BHSeedMsc;
            if(seeded_mcut_tmp) seeded_mcut_tmp[n] = ImportGroups[n].SCcomMcut;
        }
        n_seeded_local = Nimport;
    }

    myfree(ImportGroups);

    /* Now that ImportGroups is freed, copy the temporaries into persistent storage.
     * Allocate grnr, then totmsc, then mcut so the caller frees mcut->totmsc->grnr. */
    if(seeded_grnr_out && n_seeded_out) {
        *n_seeded_out = n_seeded_local;
        if(n_seeded_local > 0) {
            *seeded_grnr_out = (int64_t *) mymalloc2("SeededGrNr", n_seeded_local * sizeof(int64_t));
            memcpy(*seeded_grnr_out, seeded_grnr_tmp, n_seeded_local * sizeof(int64_t));
            if(seeded_totmsc_out) {
                *seeded_totmsc_out = (double *) mymalloc2("SeededTotMsc", n_seeded_local * sizeof(double));
                memcpy(*seeded_totmsc_out, seeded_totmsc_tmp, n_seeded_local * sizeof(double));
            }
            if(seeded_mcut_out) {
                *seeded_mcut_out = (double *) mymalloc2("SeededMcut", n_seeded_local * sizeof(double));
                memcpy(*seeded_mcut_out, seeded_mcut_tmp, n_seeded_local * sizeof(double));
            }
            /* ta_free in reverse allocation order */
            if(seeded_mcut_tmp) ta_free(seeded_mcut_tmp);
            if(seeded_totmsc_tmp) ta_free(seeded_totmsc_tmp);
            ta_free(seeded_grnr_tmp);
        } else {
            *seeded_grnr_out = NULL;
            if(seeded_totmsc_out) *seeded_totmsc_out = NULL;
            if(seeded_mcut_out) *seeded_mcut_out = NULL;
        }
    }

    walltime_measure("/FOF/Seeding");
}

static int fof_compare_HaloLabel_MinID(const void *a, const void *b)
{
    if(((struct fof_particle_list *) a)->MinID < ((struct fof_particle_list *) b)->MinID)
        return -1;

    if(((struct fof_particle_list *) a)->MinID > ((struct fof_particle_list *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinID(const void *a, const void *b)

{
    if(((struct BaseGroup *) a)->MinID < ((struct BaseGroup *) b)->MinID)
        return -1;

    if(((struct BaseGroup *) a)->MinID > ((struct BaseGroup *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinIDTask(const void *a, const void *b)
{
    const struct BaseGroup * p1 = (const struct BaseGroup *) a;
    const struct BaseGroup * p2 = (const struct BaseGroup *) b;
    int t1 = p1->MinIDTask;
    int t2 = p2->MinIDTask;
    if(t1 == _fof_compare_Group_MinIDTask_ThisTask) t1 = -1;
    if(t2 == _fof_compare_Group_MinIDTask_ThisTask) t2 = -1;

    if(t1 < t2) return -1;
    if(t1 > t2) return +1;
    return 0;

}

static int fof_compare_Group_OriginalIndex(const void *a, const void *b)

{
    return ((struct BaseGroup *) a)->OriginalIndex - ((struct BaseGroup *) b)->OriginalIndex;
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = labs(f->OriginalTask - f->MinIDTask);
    u[1] = f->MinID;
    u[2] = UINT64_MAX - (f->Length);
}

static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = f->MinID;
    u[1] = f->OriginalTask;
}
