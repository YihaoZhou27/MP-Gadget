#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <inttypes.h>
#include <math.h>
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

#define LARGE 1e29
#define MAXITER 400

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
    /* if 1, a secFOF group with unseeded Sum(m*Gamma) > 1e8 Msun is restricted to
     * the gravitationally bound unseeded stars before SeedSecFOFcomSample seeding */
    int SeedSeedFOFMassiveBoundStar;
    int BHseedMassScaleMsc;
    double MinMscForBHseed;
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
        fof_params.SeedSeedFOFMassiveBoundStar = param_get_int(ps, "SeedSeedFOFMassiveBoundStar");
        fof_params.BHseedMassScaleMsc = param_get_int(ps, "BHseedMassScaleMsc");
        fof_params.MinMscForBHseed = param_get_double(ps, "MinMscForBHseed");

        if(fof_params.BlackHoleSeedStarCluster && fof_params.BHseedMassScaleMsc && fof_params.MinMscForBHseed <= 0)
            endrun(1, "MinMscForBHseed must be > 0 when BlackHoleSeedStarCluster and BHseedMassScaleMsc are enabled.\n");
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
            gdst->StarClusterMassUnseeded += STARP(index).ClusterMass;
            gdst->StarClusterMassSampleUnseeded += STARP(index).StarClusterMass_sample;
            gdst->SCcomMcut += P[index].Mass;
            gdst->NStarUnseeded++;

            /* Track the unseeded star with the largest ClusterMass (or
             * StarClusterMass_sample when StarClusterSampling=1) as the seed
             * particle for star-cluster BH seeding in secondary FOF. For
             * SeedSecFOFcomSample use ClusterMass and record its ID as the RNG
             * seed for the combined draw. */
            MyFloat scm = (fof_params.StarClusterSampling && !fof_params.SeedSecFOFcomSample) ?
                STARP(index).StarClusterMass_sample : STARP(index).ClusterMass;
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
    blackhole_make_one(index, atime, rnd, seeded_by_starcluster, payload_mass, scaling_mass, init_msc, init_msc_sample, capped_star_mass, sc_metallicity, sc_metals);
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
    double   metallicity;       /* group mass-weighted metallicity */
    float    metals[NMETALS];
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
        elig[e].mGamma = STARP(i).ClusterMass;
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
                                   (MyFloat) m->capped, (MyFloat) m->metallicity, m->metals);
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
        double msc_star = starcluster_combined_bhseed_msc(
                m_cut, STARP(i).ClusterMass, (uint64_t) P[i].ID, rnd, &full, 0);
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
                m->mGamma = STARP(i).ClusterMass;
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

void fof_seed(FOFGroups * fof, ActiveParticles * act, ForceTree * tree, double atime, const RandTable * const rnd,
              int64_t ** seeded_grnr_out, int * n_seeded_out,
              double ** seeded_totmsc_out, double ** seeded_mcut_out, Cosmology * CP, MPI_Comm Comm)
{
    int i, j, n, ntot;

    int NTask;
    MPI_Comm_size(Comm, &NTask);

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
    if(fof_params.SeedSecFOFcomSample && fof_params.BlackHoleSeedStarCluster) {
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

    /* Do we have enough black hole slots to create this many black holes?
     * If not, allocate more slots. */
    if(Nimport + n_extra_ub + SlotsManager->info[5].size > SlotsManager->info[5].maxsize)
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
        atleast[5] += (ntot + ntot_extra)*1.1;
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
