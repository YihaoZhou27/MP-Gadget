#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <bigfile-mpi.h>

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/mpsort.h"
#include "utils/string.h"

#include "partmanager.h"
#include "slotsmanager.h"
#include "petaio.h"
#include "fof.h"
#include "walltime.h"
#include "cosmology.h"

#include "secondfof.h"

/*! \file secondfof.c
 *  \brief Second FOF pass for identifying star clusters.
 *
 *  Reuses the existing FOF engine by temporarily swapping parameters.
 *  Adds extra group properties: potential minimum position and group size.
 */

struct SecondFOFParams {
    int SecondFOFOn;
    int PrimaryLinkTypes;
    int SecondaryLinkTypes;
    double LinkingLength;   /* comoving, in code units (kpc/h) */
    int MinLength;
    int MinPrimaryLength;   /* drop groups with fewer primary-link particles from the catalog (0 = off) */
    int ComputeSize;        /* compute R50, R90, Rmax */
    int SecFOFonly;          /* skip primary FOF catalog, only save SecPIG */
    int SeedInSecFOFasStarCluster; /* use StarCluster BH-seeding in sec FOF catalog */
    int SeedSecFOFcomSample; /* combined per-secFOF star-cluster sampling for BH seeding */
    int SeedSecFOFcomSampleParticle; /* per-star-particle sampling variant of SeedSecFOFcomSample */
    int SecFOFStarCluster;  /* flag that sec FOF groups are star clusters (requires SecondFOFOn && StarClusterOn) */
    int SecFOFUnseededPart; /* if 1, only unseeded star particles are primary-linking particles (requires StarClusterOn) */
    int BHseedSecFOFbound;  /* 0 = off; 1 = bound to stars + DM in Rmax; 2 = ... DM in min(2*R50,Rmax) */
    char SecondFOFFileBase[256];
};

static struct SecondFOFParams sfof_params;

void set_secondfof_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        sfof_params.SecondFOFOn = param_get_int(ps, "SecondFOFOn");
        sfof_params.PrimaryLinkTypes = param_get_int(ps, "SecondFOFPrimaryLinkTypes");
        sfof_params.SecondaryLinkTypes = param_get_int(ps, "SecondFOFSecondaryLinkTypes");
        sfof_params.LinkingLength = param_get_double(ps, "SecondFOFLinkingLength");
        sfof_params.MinLength = param_get_int(ps, "SecondFOFMinLength");
        sfof_params.MinPrimaryLength = param_get_int(ps, "SecondFOFMinPrimaryLength");
        sfof_params.ComputeSize = param_get_int(ps, "SecondFOFSize");
        sfof_params.SecFOFonly = param_get_int(ps, "SecFOFonly");
        sfof_params.SeedInSecFOFasStarCluster = param_get_int(ps, "SeedInSecFOFasStarCluster");
        sfof_params.SecFOFStarCluster = param_get_int(ps, "SecFOFStarCluster");
        int StarClusterOn = param_get_int(ps, "StarClusterOn");
        /* SecFOFUnseededPart restricts the second-FOF primary-linking set to
         * unseeded star particles; it relies on the StarCluster Seeded flag. */
        sfof_params.SecFOFUnseededPart = param_get_int(ps, "SecFOFUnseededPart");
        if(sfof_params.SecFOFUnseededPart && !StarClusterOn)
            endrun(1, "SecFOFUnseededPart=1 requires StarClusterOn=1.\n");
        if(sfof_params.SeedInSecFOFasStarCluster && (!sfof_params.SecondFOFOn || !StarClusterOn || !sfof_params.SecFOFStarCluster)) {
            message(0, "SeedInSecFOFasStarCluster requires SecondFOFOn=1, StarClusterOn=1, and SecFOFStarCluster=1; disabling.\n");
            sfof_params.SeedInSecFOFasStarCluster = 0;
        }
        /* BHseedSecFOFbound only means anything for the secondary-FOF seed path. */
        sfof_params.BHseedSecFOFbound = param_get_int(ps, "BHseedSecFOFbound");
        if(sfof_params.BHseedSecFOFbound && !sfof_params.SeedInSecFOFasStarCluster)
            endrun(1, "BHseedSecFOFbound=%d requires SeedInSecFOFasStarCluster=1 "
                      "(effective: SecondFOFOn=1, StarClusterOn=1, SecFOFStarCluster=1).\n",
                      sfof_params.BHseedSecFOFbound);
        /* SeedSecFOFcomSample requires the (effective) SeedInSecFOFasStarCluster.
         * Hard error (exit) if not satisfied, per design. */
        sfof_params.SeedSecFOFcomSample = param_get_int(ps, "SeedSecFOFcomSample");
        if(sfof_params.SeedSecFOFcomSample && !sfof_params.SeedInSecFOFasStarCluster)
            endrun(1, "SeedSecFOFcomSample=1 requires SeedInSecFOFasStarCluster=1 (effective: SecondFOFOn=1, StarClusterOn=1, SecFOFStarCluster=1).\n");
        /* SeedInSecFOFMultipleSeeds requires the (effective) SeedInSecFOFasStarCluster. */
        if(param_get_int(ps, "SeedInSecFOFMultipleSeeds") && !sfof_params.SeedInSecFOFasStarCluster)
            endrun(1, "SeedInSecFOFMultipleSeeds=1 requires SeedInSecFOFasStarCluster=1 (effective: SecondFOFOn=1, StarClusterOn=1, SecFOFStarCluster=1).\n");
        if(sfof_params.SeedSecFOFcomSample && param_get_double(ps, "MinMscForBHseed") <= 0)
            endrun(1, "SeedSecFOFcomSample=1 requires MinMscForBHseed > 0.\n");
        sfof_params.SeedSecFOFcomSampleParticle = param_get_int(ps, "SeedSecFOFcomSampleParticle");
        if(sfof_params.SeedSecFOFcomSampleParticle && !sfof_params.SeedSecFOFcomSample)
            endrun(1, "SeedSecFOFcomSampleParticle=1 requires SeedSecFOFcomSample=1.\n");
        /* SeedSeedFOFMassiveBoundStar restricts massive secFOF groups to their
         * gravitationally bound unseeded stars; v1 supports the combined sampler
         * only (not the per-particle sampler). */
        if(param_get_int(ps, "SeedSeedFOFMassiveBoundStar")) {
            if(!sfof_params.SeedSecFOFcomSample)
                endrun(1, "SeedSeedFOFMassiveBoundStar=1 requires SeedSecFOFcomSample=1.\n");
            if(sfof_params.SeedSecFOFcomSampleParticle)
                endrun(1, "SeedSeedFOFMassiveBoundStar=1 is incompatible with SeedSecFOFcomSampleParticle=1 (v1 supports the combined sampler only).\n");
            if(param_get_int(ps, "SeedInSecFOFMultipleSeeds"))
                endrun(1, "SeedSeedFOFMassiveBoundStar=1 is incompatible with SeedInSecFOFMultipleSeeds=1 (only the primary seed would be bound; multi-seeding is not bound-aware).\n");
        }
        if(sfof_params.SecondFOFOn && StarClusterOn) {
            if(sfof_params.SecFOFStarCluster && sfof_params.PrimaryLinkTypes != (1 << 4) && sfof_params.PrimaryLinkTypes != ((1 << 4) | (1 << 0)))
                endrun(1, "SecFOFStarCluster requires SecondFOFPrimaryLinkTypes = 16 (star) or 17 (star+gas).\n");
        }
        /* Validate MinMscForBHseed for secondary seeding: secondfof_seed
         * forcibly enables BlackHoleSeedStarCluster at runtime, so the
         * validation in set_fof_params (which only checks the param-file
         * value of BlackHoleSeedStarCluster) can miss this. */
        if(sfof_params.SeedInSecFOFasStarCluster) {
            int BHseedMassScaleMsc = param_get_int(ps, "BHseedMassScaleMsc");
            double MinMscForBHseed = param_get_double(ps, "MinMscForBHseed");
            if(BHseedMassScaleMsc && MinMscForBHseed <= 0)
                endrun(1, "MinMscForBHseed must be > 0 when SeedInSecFOFasStarCluster and BHseedMassScaleMsc are enabled.\n");
            /* BH particles (type 5) must be a secondary link type so that newly
             * seeded BHs are included in the secondary FOF groups. */
            if(!(sfof_params.SecondaryLinkTypes & (1 << 5)))
                endrun(1, "SeedInSecFOFasStarCluster requires type 5 (BH) in SecondFOFSecondaryLinkTypes.\n");
            /* FOFPotentialMin must be on so the star at the potential minimum
             * can be identified as the seed particle location. */
            int FOFPotentialMin = param_get_int(ps, "FOFPotentialMin");
            if(!FOFPotentialMin)
                endrun(1, "SeedInSecFOFasStarCluster requires FOFPotentialMin = 1.\n");
        }
        strncpy(sfof_params.SecondFOFFileBase, param_get_string(ps, "SecondFOFFileBase"), sizeof(sfof_params.SecondFOFFileBase) - 1);
    }
    MPI_Bcast(&sfof_params, sizeof(struct SecondFOFParams), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Temporary: seeding in secondary FOF and primary FOF cannot both be
     * active in the same run. Override fof_params on all MPI ranks since
     * fof.c stores the seeding flags in a separate module-global struct. */
    if(sfof_params.SeedInSecFOFasStarCluster) {
        fof_set_seed_params(0, 0, 0);
    }
}

int get_secondfof_on(void)
{
    return sfof_params.SecondFOFOn;
}

const char * get_secondfof_filebase(void)
{
    return sfof_params.SecondFOFFileBase;
}

int get_secondfof_only(void)
{
    return sfof_params.SecFOFonly;
}

int get_seed_in_secfof(void)
{
    return sfof_params.SeedInSecFOFasStarCluster;
}

/* Extended group properties for the second FOF.
 * PotMin and PotMinPos are now computed inside fof_compile_catalogue
 * (in add_particle_to_group / fof_reduce_group) and stored in struct Group.
 * This struct only holds the size properties. */
struct SecondGroupExtra {
    float  R50;             /* Half-mass radius of primary particles */
    float  R90;             /* 90%-mass radius of primary particles */
    float  Rmax;            /* Max primary particle separation from center */
    int32_t PrimaryFOFNum;  /* Number of distinct primary FOF groups hosting particles of this sec FOF */
    int64_t PrimaryFOFID;   /* Primary FOF GrNr that hosts the largest fraction of primary-linked particles; -1 if none */
};

/* (distance, mass, GrNr) tuple for computing group sizes. */
struct dist_mass_grp {
    double dist;
    double mass;
    int64_t GrNr;
};

/* One secFOF group that just seeded a BH, broadcast to all ranks so every rank
 * can flag its local member stars and (in SeedSecFOFcomSampleParticle mode)
 * redistribute tot_msc_fof into each unseeded star's ClusterMass.
 *   totmsc = group BHSeedMsc = tot_msc_fof (per-particle summed >1e4 Msun mass);
 *   mcut   = group SCcomMcut = total unseeded stellar mass (redistribution weight
 *            denominator). Both are unused (carried as 0) outside particle mode. */
struct seeded_group {
    int64_t GrNr;
    double totmsc;
    double mcut;
};

static int cmp_seeded_group(const void * a, const void * b)
{
    int64_t va = ((const struct seeded_group *)a)->GrNr;
    int64_t vb = ((const struct seeded_group *)b)->GrNr;
    return (va > vb) - (va < vb);
}

static int cmp_dist_mass_grp_by_dist(const void * a, const void * b)
{
    const struct dist_mass_grp * da = (const struct dist_mass_grp *)a;
    const struct dist_mass_grp * db = (const struct dist_mass_grp *)b;
    return (da->dist > db->dist) - (da->dist < db->dist);
}

/* Sort by GrNr first, then by dist — so each group's particles are
 * contiguous and already distance-sorted within the block. */
static int cmp_dist_mass_grp_by_grp_dist(const void * a, const void * b)
{
    const struct dist_mass_grp * da = (const struct dist_mass_grp *)a;
    const struct dist_mass_grp * db = (const struct dist_mass_grp *)b;
    if(da->GrNr != db->GrNr)
        return (da->GrNr > db->GrNr) - (da->GrNr < db->GrNr);
    return (da->dist > db->dist) - (da->dist < db->dist);
}

/* Comparator for grp_center: sort by GrNr */
struct grp_center {
    int64_t GrNr;
    double PotMinPos[3];
};

static int cmp_grp_center_by_grnr(const void * a, const void * b)
{
    const struct grp_center * ca = (const struct grp_center *)a;
    const struct grp_center * cb = (const struct grp_center *)b;
    return (ca->GrNr > cb->GrNr) - (ca->GrNr < cb->GrNr);
}

/*
 * Compute R50, R90, Rmax for each owned group using PotMinPos as center.
 * Only considers primary-linked particles.
 *
 * Algorithm: each rank computes distances for its local primary particles,
 * then a single MPI_Allgatherv gathers ALL (dist, mass, GrNr) data to
 * every rank. Each rank then computes R50/R90/Rmax for its own groups
 * from the complete global data. This uses one collective call (no deadlock)
 * and works because the total particle count in second FOF groups is small.
 */
static void
secondfof_compute_sizes(FOFGroups * fof, struct SecondGroupExtra * extra, MPI_Comm Comm)
{
    int i, g;
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    double BoxSize = PartManager->BoxSize;

    /* Step 1: Gather all group centers so every rank can compute distances. */
    int * grp_counts = (int *) mymalloc2("SecFOF_gc", sizeof(int) * NTask);
    int local_ngroups = fof->Ngroups;
    MPI_Allgather(&local_ngroups, 1, MPI_INT, grp_counts, 1, MPI_INT, Comm);

    int64_t total_groups = 0;
    int * grp_displs = (int *) mymalloc2("SecFOF_gd", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        grp_displs[i] = total_groups;
        total_groups += grp_counts[i];
    }

    struct grp_center * all_centers = (struct grp_center *)
        mymalloc2("SecFOF_ac", sizeof(struct grp_center) * (total_groups > 0 ? total_groups : 1));
    struct grp_center * local_centers = (struct grp_center *)
        mymalloc2("SecFOF_lc", sizeof(struct grp_center) * (local_ngroups > 0 ? local_ngroups : 1));
    for(g = 0; g < local_ngroups; g++) {
        local_centers[g].GrNr = fof->Group[g].base.GrNr;
        int d;
        for(d = 0; d < 3; d++)
            local_centers[g].PotMinPos[d] = fof->Group[g].PotMinPos[d];
    }

    int * gc_bytes = (int *) mymalloc2("SecFOF_gcb", sizeof(int) * NTask);
    int * gd_bytes = (int *) mymalloc2("SecFOF_gdb", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        gc_bytes[i] = grp_counts[i] * sizeof(struct grp_center);
        gd_bytes[i] = grp_displs[i] * sizeof(struct grp_center);
    }
    MPI_Allgatherv(local_centers, local_ngroups * sizeof(struct grp_center), MPI_BYTE,
                   all_centers, gc_bytes, gd_bytes, MPI_BYTE, Comm);
    myfree(gd_bytes);
    myfree(gc_bytes);
    myfree(local_centers);

    /* Sort centers by GrNr for binary search during per-particle distance computation */
    qsort(all_centers, total_groups, sizeof(struct grp_center), cmp_grp_center_by_grnr);

    /* Step 2: Each rank computes (dist, mass, GrNr) for its local particles. */
    int64_t nlocal = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].SecGrNr < 0) continue;
        if(!((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) continue;
        nlocal++;
    }

    struct dist_mass_grp * dm_local = (struct dist_mass_grp *)
        mymalloc2("SecFOF_dml", sizeof(struct dist_mass_grp) * (nlocal > 0 ? nlocal : 1));

    int64_t n = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].SecGrNr < 0) continue;
        if(!((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) continue;

        int64_t grNr = P[i].SecGrNr;
        double center[3] = {0, 0, 0};
        /* Binary search for the group center in the sorted all_centers array */
        int64_t lo = 0, hi = total_groups;
        while(lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if(all_centers[mid].GrNr < grNr)
                lo = mid + 1;
            else
                hi = mid;
        }
        if(lo < total_groups && all_centers[lo].GrNr == grNr) {
            int d;
            for(d = 0; d < 3; d++)
                center[d] = all_centers[lo].PotMinPos[d];
        }

        double r2 = 0;
        int d;
        for(d = 0; d < 3; d++) {
            double dx = NEAREST(P[i].Pos[d] - center[d], BoxSize);
            r2 += dx * dx;
        }
        dm_local[n].dist = sqrt(r2);
        dm_local[n].mass = P[i].Mass;
        dm_local[n].GrNr = grNr;
        n++;
    }

    /* all_centers is done but dm_local sits above it on the stack,
     * so defer freeing until after dm_local is freed below. */

    /* Step 3: Allgatherv all (dist, mass, GrNr) data to every rank.
     * The total number of particles in second FOF groups is small,
     * so this is safe memory-wise. */
    int nlocal_int = (int) n;
    int * part_counts = (int *) mymalloc2("SecFOF_pc", sizeof(int) * NTask);
    MPI_Allgather(&nlocal_int, 1, MPI_INT, part_counts, 1, MPI_INT, Comm);

    int64_t total_parts = 0;
    int * part_displs = (int *) mymalloc2("SecFOF_pd", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        part_displs[i] = total_parts;
        total_parts += part_counts[i];
    }

    struct dist_mass_grp * dm_global = (struct dist_mass_grp *)
        mymalloc2("SecFOF_dmg", sizeof(struct dist_mass_grp) * (total_parts > 0 ? total_parts : 1));

    int * pc_bytes = (int *) mymalloc2("SecFOF_pcb", sizeof(int) * NTask);
    int * pd_bytes = (int *) mymalloc2("SecFOF_pdb", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        pc_bytes[i] = part_counts[i] * sizeof(struct dist_mass_grp);
        pd_bytes[i] = part_displs[i] * sizeof(struct dist_mass_grp);
    }
    MPI_Allgatherv(dm_local, nlocal_int * sizeof(struct dist_mass_grp), MPI_BYTE,
                   dm_global, pc_bytes, pd_bytes, MPI_BYTE, Comm);
    /* Only free pd_bytes and pc_bytes now (topmost on stack).
     * Everything below dm_global (part_displs, part_counts, dm_local,
     * all_centers, grp_displs, grp_counts) must wait until dm_global
     * is freed after step 4 — LIFO order. */
    myfree(pd_bytes);
    myfree(pc_bytes);

    /* Step 4: Sort dm_global by (GrNr, dist) so each group's particles
     * form a contiguous, distance-sorted block. Then binary search per group. */
    qsort(dm_global, total_parts, sizeof(struct dist_mass_grp), cmp_dist_mass_grp_by_grp_dist);

    for(g = 0; g < fof->Ngroups; g++) {
        int64_t grNr = fof->Group[g].base.GrNr;

        /* Binary search for the first particle with this GrNr */
        int64_t lo = 0, hi = total_parts;
        while(lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if(dm_global[mid].GrNr < grNr)
                lo = mid + 1;
            else
                hi = mid;
        }
        int64_t start = lo;
        while(lo < total_parts && dm_global[lo].GrNr == grNr)
            lo++;
        int64_t end = lo;
        int64_t count = end - start;

        if(count > 0) {
            /* Particles in [start, end) are already sorted by distance */
            double total_mass = 0;
            int64_t k;
            for(k = start; k < end; k++)
                total_mass += dm_global[k].mass;

            double cumul_mass = 0;
            float r50 = 0, r90 = 0, rmax = 0;
            int found50 = 0, found90 = 0;
            for(k = start; k < end; k++) {
                cumul_mass += dm_global[k].mass;
                if(!found50 && cumul_mass >= 0.5 * total_mass) {
                    r50 = dm_global[k].dist;
                    found50 = 1;
                }
                if(!found90 && cumul_mass >= 0.9 * total_mass) {
                    r90 = dm_global[k].dist;
                    found90 = 1;
                }
                rmax = dm_global[k].dist;
            }
            extra[g].R50 = r50;
            extra[g].R90 = r90;
            extra[g].Rmax = rmax;
        } else {
            extra[g].R50 = 0;
            extra[g].R90 = 0;
            extra[g].Rmax = 0;
        }
    }

    /* Free everything in strict LIFO order (reverse allocation order):
     * Stack (top→bottom): dm_global, part_displs, part_counts,
     * dm_local, all_centers, grp_displs, grp_counts */
    myfree(dm_global);
    myfree(part_displs);
    myfree(part_counts);
    myfree(dm_local);
    myfree(all_centers);
    myfree(grp_displs);
    myfree(grp_counts);
}

/* Comparator for sec_prim_pair: sort by SecGrNr first, then by GrNr */
struct sec_prim_pair {
    int64_t SecGrNr;
    int64_t GrNr;
};

static int cmp_sec_prim_pair(const void * a, const void * b)
{
    const struct sec_prim_pair * pa = (const struct sec_prim_pair *) a;
    const struct sec_prim_pair * pb = (const struct sec_prim_pair *) b;
    if(pa->SecGrNr != pb->SecGrNr)
        return (pa->SecGrNr > pb->SecGrNr) - (pa->SecGrNr < pb->SecGrNr);
    return (pa->GrNr > pb->GrNr) - (pa->GrNr < pb->GrNr);
}

/*
 * Compute PrimaryFOFNum and PrimaryFOFID for each owned second FOF group.
 * For each secondary FOF group, counts how many distinct primary FOF groups
 * its primary-linked particles belong to, and identifies which primary FOF
 * hosts the largest fraction.
 *
 * Algorithm: each rank gathers (SecGrNr, GrNr) for its local primary particles,
 * then a single MPI_Allgatherv collects all tuples globally. Each rank then
 * processes its own groups from the complete data.
 */
static void
secondfof_compute_primary_fof_info(FOFGroups * fof, struct SecondGroupExtra * extra, MPI_Comm Comm)
{
    int i, g;
    int NTask;
    MPI_Comm_size(Comm, &NTask);

    /* Step 1: Each rank builds (SecGrNr, GrNr) tuples for local primary-linked particles */
    int64_t nlocal = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].SecGrNr < 0) continue;
        if(!((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) continue;
        nlocal++;
    }

    struct sec_prim_pair * local_pairs = (struct sec_prim_pair *)
        mymalloc2("SecFOF_lp", sizeof(struct sec_prim_pair) * (nlocal > 0 ? nlocal : 1));

    int64_t n = 0;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].SecGrNr < 0) continue;
        if(!((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) continue;
        local_pairs[n].SecGrNr = P[i].SecGrNr;
        local_pairs[n].GrNr = P[i].GrNr;
        n++;
    }

    /* Step 2: Allgatherv all pairs to every rank */
    int nlocal_int = (int) n;
    int * pair_counts = (int *) mymalloc2("SecFOF_prc", sizeof(int) * NTask);
    MPI_Allgather(&nlocal_int, 1, MPI_INT, pair_counts, 1, MPI_INT, Comm);

    int64_t total_pairs = 0;
    int * pair_displs = (int *) mymalloc2("SecFOF_prd", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        pair_displs[i] = total_pairs;
        total_pairs += pair_counts[i];
    }

    struct sec_prim_pair * all_pairs = (struct sec_prim_pair *)
        mymalloc2("SecFOF_ap", sizeof(struct sec_prim_pair) * (total_pairs > 0 ? total_pairs : 1));

    int * pc_bytes = (int *) mymalloc2("SecFOF_pcb2", sizeof(int) * NTask);
    int * pd_bytes = (int *) mymalloc2("SecFOF_pdb2", sizeof(int) * NTask);
    for(i = 0; i < NTask; i++) {
        pc_bytes[i] = pair_counts[i] * sizeof(struct sec_prim_pair);
        pd_bytes[i] = pair_displs[i] * sizeof(struct sec_prim_pair);
    }
    MPI_Allgatherv(local_pairs, nlocal_int * sizeof(struct sec_prim_pair), MPI_BYTE,
                   all_pairs, pc_bytes, pd_bytes, MPI_BYTE, Comm);
    myfree(pd_bytes);
    myfree(pc_bytes);

    /* Step 3: Sort all_pairs by SecGrNr (primary key) then GrNr (secondary key)
     * so each group's particles form a contiguous, GrNr-sorted block.
     * This reduces the per-group lookup from O(Npairs) to O(log(Npairs) + group_size). */
    qsort(all_pairs, total_pairs, sizeof(struct sec_prim_pair), cmp_sec_prim_pair);

    for(g = 0; g < fof->Ngroups; g++) {
        int64_t secGrNr = fof->Group[g].base.GrNr;

        /* Binary search for the first pair with this SecGrNr */
        int64_t lo = 0, hi = total_pairs;
        while(lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if(all_pairs[mid].SecGrNr < secGrNr)
                lo = mid + 1;
            else
                hi = mid;
        }
        /* lo is now the index of the first pair with SecGrNr >= secGrNr.
         * Scan forward to find the end of the block. */
        int64_t start = lo;
        while(lo < total_pairs && all_pairs[lo].SecGrNr == secGrNr)
            lo++;
        int64_t end = lo;
        int64_t count = end - start;

        if(count == 0) {
            extra[g].PrimaryFOFNum = 0;
            extra[g].PrimaryFOFID = -1;
            continue;
        }

        /* Pairs in [start, end) are already sorted by GrNr (secondary sort key),
         * so we can count distinct primary FOF IDs in a single pass. */
        int num_distinct = 0;
        int64_t best_id = -1;
        int best_count = 0;
        int cur_count = 1;
        int64_t cur_id = all_pairs[start].GrNr;
        int64_t k;

        for(k = start + 1; k < end; k++) {
            if(all_pairs[k].GrNr == cur_id) {
                cur_count++;
            } else {
                if(cur_id >= 0) {
                    num_distinct++;
                    if(cur_count > best_count) {
                        best_count = cur_count;
                        best_id = cur_id;
                    }
                }
                cur_id = all_pairs[k].GrNr;
                cur_count = 1;
            }
        }
        /* Handle the last run */
        if(cur_id >= 0) {
            num_distinct++;
            if(cur_count > best_count) {
                best_count = cur_count;
                best_id = cur_id;
            }
        }

        extra[g].PrimaryFOFNum = num_distinct;
        extra[g].PrimaryFOFID = best_id;
    }

    /* Free in LIFO order: all_pairs, pair_displs, pair_counts, local_pairs */
    myfree(all_pairs);
    myfree(pair_displs);
    myfree(pair_counts);
    myfree(local_pairs);
}

/* ---------- IO for SecPIG catalog ---------- */

/* We define getters that read from a combined struct of Group + SecondGroupExtra.
 * Since the IO system iterates over a single base pointer, we need to pack
 * group data into a single array. We'll use a struct for this. */
struct SecondGroupOutput {
    struct Group grp;
    struct SecondGroupExtra ext;
};

#define SIMPLE_PROPERTY_SECFOF(name, field, type, items) \
    SIMPLE_GETTER(GTSec ## name, field, type, items, struct SecondGroupOutput) \
    SIMPLE_SETTER(STSec ## name, field, type, items, struct SecondGroupOutput)

SIMPLE_PROPERTY_SECFOF(GroupID, grp.base.GrNr, uint32_t, 1)
SIMPLE_PROPERTY_SECFOF(MinID, grp.base.MinID, uint64_t, 1)
SIMPLE_PROPERTY_SECFOF(Imom, grp.Imom[0][0], float, 9)
SIMPLE_PROPERTY_SECFOF(Jmom, grp.Jmom[0], float, 3)
SIMPLE_PROPERTY_SECFOF(Mass, grp.Mass, float, 1)
SIMPLE_PROPERTY_SECFOF(MassByType, grp.MassType[0], float, 6)
SIMPLE_PROPERTY_SECFOF(LengthByType, grp.LenType[0], uint32_t, 6)
SIMPLE_PROPERTY_SECFOF(StarFormationRate, grp.Sfr, float, 1)
SIMPLE_PROPERTY_SECFOF(GasMetalMass, grp.GasMetalMass, float, 1)
SIMPLE_PROPERTY_SECFOF(StellarMetalMass, grp.StellarMetalMass, float, 1)
SIMPLE_PROPERTY_SECFOF(GasMetalElemMass, grp.GasMetalElemMass[0], float, NMETALS)
SIMPLE_PROPERTY_SECFOF(StellarMetalElemMass, grp.StellarMetalElemMass[0], float, NMETALS)
SIMPLE_PROPERTY_SECFOF(MassHeIonized, grp.MassHeIonized, float, 1)
SIMPLE_PROPERTY_SECFOF(BlackholeMass, grp.BH_Mass, float, 1)
SIMPLE_PROPERTY_SECFOF(BlackholeAccretionRate, grp.BH_Mdot, float, 1)
SIMPLE_PROPERTY_SECFOF(GasSfmpMass, grp.sfmp_mass, float, 1)
SIMPLE_PROPERTY_SECFOF(StarClusterMassSample, grp.StarClusterMassSample, float, 1)
SIMPLE_PROPERTY_SECFOF(NscSample, grp.NscSample, int, 1)

SIMPLE_PROPERTY_SECFOF(PotMin, grp.PotMin, float, 1)
SIMPLE_PROPERTY_SECFOF(R50, ext.R50, float, 1)
SIMPLE_PROPERTY_SECFOF(R90, ext.R90, float, 1)
SIMPLE_PROPERTY_SECFOF(Rmax, ext.Rmax, float, 1)
SIMPLE_PROPERTY_SECFOF(PrimaryFOFNum, ext.PrimaryFOFNum, int32_t, 1)
SIMPLE_PROPERTY_SECFOF(PrimaryFOFID, ext.PrimaryFOFID, int64_t, 1)

SIMPLE_PROPERTY_SECFOF(SCMass, grp.StarClusterMass, float, 1)
SIMPLE_PROPERTY_SECFOF(SCMass_seeded, grp.SCMass_seeded, float, 1)
/* BHseedSecFOFbound diagnostics; all zero when the feature is off. */
SIMPLE_PROPERTY_SECFOF(BoundStarMass, grp.SCBoundStarMass, float, 1)
SIMPLE_PROPERTY_SECFOF(BoundStarMassUnseeded, grp.SCBoundStarMassUnseeded, float, 1)
SIMPLE_PROPERTY_SECFOF(BoundSCMass, grp.SCBoundClusterMass, float, 1)
SIMPLE_PROPERTY_SECFOF(BoundStarNum, grp.NStarBound, int, 1)
SIMPLE_PROPERTY_SECFOF(BoundRdm, grp.SCBoundRdm, float, 1)
SIMPLE_PROPERTY_SECFOF(BoundDMMass, grp.SCBoundMdm, float, 1)
SIMPLE_PROPERTY_SECFOF(SCMetallicity, grp.StarClusterMetallicity, float, 1)
SIMPLE_PROPERTY_SECFOF(SCMetalElemMass, grp.StarClusterMetalElemMass[0], float, NMETALS)

static void GTSecFirstPos(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct SecondGroupOutput * grp = (struct SecondGroupOutput *) baseptr;
    int d;
    for(d = 0; d < 3; d++) {
        out[d] = grp[i].grp.base.FirstPos[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > PartManager->BoxSize) out[d] -= PartManager->BoxSize;
        while(out[d] <= 0) out[d] += PartManager->BoxSize;
    }
}

static void STSecFirstPos(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct SecondGroupOutput * grp = (struct SecondGroupOutput *) baseptr;
    int d;
    for(d = 0; d < 3; d++) {
        grp[i].grp.base.FirstPos[d] = out[d];
    }
}

static void GTSecMassCenterPosition(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct SecondGroupOutput * grp = (struct SecondGroupOutput *) baseptr;
    int d;
    for(d = 0; d < 3; d++) {
        out[d] = grp[i].grp.CM[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > PartManager->BoxSize) out[d] -= PartManager->BoxSize;
        while(out[d] <= 0) out[d] += PartManager->BoxSize;
    }
}

static void STSecMassCenterPosition(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct SecondGroupOutput * grp = (struct SecondGroupOutput *) baseptr;
    int d;
    for(d = 0; d < 3; d++) {
        grp[i].grp.CM[d] = out[d];
    }
}

static void GTSecMassCenterVelocity(int i, float * out, void * baseptr, void * slotptr, const struct conversions * params) {
    struct SecondGroupOutput * grp = (struct SecondGroupOutput *) baseptr;
    double fac;
    if (GetUsePeculiarVelocity()) {
        fac = 1.0 / params->atime;
    } else {
        fac = 1.0;
    }
    int d;
    for(d = 0; d < 3; d++) {
        out[d] = fac * grp[i].grp.Vel[d];
    }
}

static void GTSecPotMinPos(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct SecondGroupOutput * grp = (struct SecondGroupOutput *) baseptr;
    int d;
    for(d = 0; d < 3; d++) {
        out[d] = grp[i].grp.PotMinPos[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > PartManager->BoxSize) out[d] -= PartManager->BoxSize;
        while(out[d] <= 0) out[d] += PartManager->BoxSize;
    }
}

static void STSecPotMinPos(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct SecondGroupOutput * grp = (struct SecondGroupOutput *) baseptr;
    int d;
    for(d = 0; d < 3; d++) {
        grp[i].grp.PotMinPos[d] = out[d];
    }
}

static void
secondfof_register_io_blocks(int MetalReturnOn, int ComputeSize, int SecFOFStarCluster, struct IOTable * IOTable)
{
    IOTable->used = 0;
    IOTable->allocated = 100;
    IOTable->ent = (struct IOTableEntry *) mymalloc2("SecFOFIOTable", IOTable->allocated * sizeof(IOTableEntry));

    IO_REG(SecGroupID, "u4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecMassCenterPosition, "f8", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecFirstPos, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecMinID, "u8", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecImom, "f4", 9, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecJmom, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG_WRONLY(SecMassCenterVelocity, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecLengthByType, "u4", 6, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecMassByType, "f4", 6, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecMassHeIonized, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecStarFormationRate, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    if(MetalReturnOn) {
        IO_REG(SecGasMetalMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecStellarMetalMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecGasMetalElemMass, "f4", NMETALS, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecStellarMetalElemMass, "f4", NMETALS, PTYPE_FOF_GROUP, IOTable);
    }
    IO_REG(SecBlackholeMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecBlackholeAccretionRate, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecGasSfmpMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecStarClusterMassSample, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecNscSample, "i4", 1, PTYPE_FOF_GROUP, IOTable);

    /* Second FOF specific properties */
    IO_REG(SecPotMinPos, "f8", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecPotMin, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    if(ComputeSize) {
        IO_REG(SecR50, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecR90, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecRmax, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    }
    IO_REG(SecPrimaryFOFNum, "i4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecPrimaryFOFID, "i8", 1, PTYPE_FOF_GROUP, IOTable);
    if(SecFOFStarCluster) {
        IO_REG(SecSCMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecSCMass_seeded, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecSCMetallicity, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecSCMetalElemMass, "f4", NMETALS, PTYPE_FOF_GROUP, IOTable);
        /* BHseedSecFOFbound: the gravitationally bound subset of the member stars.
         * Registered unconditionally so the catalogue schema does not depend on a
         * runtime switch; every block is identically 0 when BHseedSecFOFbound = 0.
         * These are diagnostics only -- SecSCMass, SecMassByType and SecLengthByType
         * above still describe ALL member stars in every mode. */
        IO_REG(SecBoundStarMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecBoundStarMassUnseeded, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecBoundSCMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecBoundStarNum, "i4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecBoundRdm, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecBoundDMMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    }
}

static void build_buffer_secondfof(struct SecondGroupOutput * output, int64_t Ngroups,
                                   BigArray * array, IOTableEntry * ent, struct conversions * conv)
{
    petaio_alloc_buffer(array, ent, Ngroups);
    char * p = (char *) array->data;
    int i;
    for(i = 0; i < Ngroups; i++) {
        ent->getter(i, p, output, NULL, conv);
        p += array->strides[0];
    }
}

static void secondfof_write_header(BigFile * bf, int64_t TotNgroups, const double atime,
                                   const double * MassTable, Cosmology * CP, MPI_Comm Comm)
{
    BigBlock bh;
    if(0 != big_file_mpi_create_block(bf, &bh, "Header", NULL, 0, 0, 0, Comm)) {
        endrun(0, "SecondFOF: Failed to create header\n");
    }

    int k;
    int64_t npartLocal[6] = {0};
    int64_t npartTotal[6] = {0};

    #pragma omp parallel for reduction(+: npartLocal[:6])
    for(int i = 0; i < PartManager->NumPart; i++) {
        if(P[i].SecGrNr < 0) continue;
        /* Count primary- and secondary-linked particles */
        int type_mask = sfof_params.PrimaryLinkTypes | sfof_params.SecondaryLinkTypes;
        if(!((1 << P[i].Type) & type_mask)) continue;
        npartLocal[P[i].Type]++;
    }

    MPI_Allreduce(npartLocal, npartTotal, 6, MPI_INT64, MPI_SUM, Comm);

    const double hubble = hubble_function(CP, atime);
    double RSD = 1.0 / (atime * hubble);
    int pecvel = GetUsePeculiarVelocity();
    if(!pecvel)
        RSD /= atime;

    big_block_set_attr(&bh, "NumPartInGroupTotal", npartTotal, "u8", 6);
    big_block_set_attr(&bh, "NumFOFGroupsTotal", &TotNgroups, "u8", 1);
    big_block_set_attr(&bh, "RSDFactor", &RSD, "f8", 1);
    big_block_set_attr(&bh, "MassTable", MassTable, "f8", 6);
    big_block_set_attr(&bh, "Time", &atime, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &PartManager->BoxSize, "f8", 1);
    big_block_set_attr(&bh, "OmegaLambda", &CP->OmegaLambda, "f8", 1);
    big_block_set_attr(&bh, "Omega0", &CP->Omega0, "f8", 1);
    big_block_set_attr(&bh, "HubbleParam", &CP->HubbleParam, "f8", 1);
    big_block_set_attr(&bh, "CMBTemperature", &CP->CMBTemperature, "f8", 1);
    big_block_set_attr(&bh, "OmegaBaryon", &CP->OmegaBaryon, "f8", 1);
    big_block_set_attr(&bh, "UsePeculiarVelocity", &pecvel, "i4", 1);

    /* Second FOF specific attributes */
    double ll = sfof_params.LinkingLength;
    int minlen = sfof_params.MinLength;
    int minprimlen = sfof_params.MinPrimaryLength;
    int primary = sfof_params.PrimaryLinkTypes;
    int secondary = sfof_params.SecondaryLinkTypes;
    big_block_set_attr(&bh, "SecondFOFLinkingLength", &ll, "f8", 1);
    big_block_set_attr(&bh, "SecondFOFMinLength", &minlen, "i4", 1);
    big_block_set_attr(&bh, "SecondFOFMinPrimaryLength", &minprimlen, "i4", 1);
    big_block_set_attr(&bh, "SecondFOFPrimaryLinkTypes", &primary, "i4", 1);
    big_block_set_attr(&bh, "SecondFOFSecondaryLinkTypes", &secondary, "i4", 1);

    big_block_mpi_close(&bh, Comm);
}

static void fof_radix_SecGrNr(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    const struct SecondGroupOutput * f = (const struct SecondGroupOutput *) a;
    u[0] = f->grp.base.GrNr;
}

/* Opaque result struct holding data between run and write phases */
struct SecondFOFResult {
    FOFGroups fof;
    struct SecondGroupOutput * output;
    int64_t Ngroups;
    int64_t TotNgroups;
};

void secondfof_seed(DomainDecomp * ddecomp, ActiveParticles * act, ForceTree * tree,
                    double atime, const RandTable * rnd, Cosmology * CP, MPI_Comm Comm)
{
    int i;
    message(0, "Seeding black holes using secondary FOF catalog (StarCluster criteria).\n");

    /* Save current FOF parameters */
    int save_PrimaryLT, save_SecondaryLT, save_MinLen, save_PotMin, save_MinPrimLen;
    double save_LinkLen;
    fof_get_params(&save_PrimaryLT, &save_SecondaryLT,
                   &save_LinkLen, &save_MinLen, &save_PotMin, &save_MinPrimLen);

    int save_SeedSC, save_SeedHalo, save_SeedGas;
    fof_get_seed_params(&save_SeedSC, &save_SeedHalo, &save_SeedGas);

    /* Override with secondary FOF linking parameters.
     * Enable FOFPotentialMin so that PotMin/PotMinPos are tracked for group centers.
     * MinPrimaryLength drops groups with too few primary-link particles so they
     * are not counted as a FOF at all (excluded from both seeding and the catalog). */
    fof_set_params(sfof_params.PrimaryLinkTypes, sfof_params.SecondaryLinkTypes,
                   sfof_params.LinkingLength, sfof_params.MinLength, 1,
                   sfof_params.MinPrimaryLength);

    /* Override seeding params: only StarCluster-based seeding in sec FOF */
    fof_set_seed_params(1, 0, 0);

    /* Save primary FOF GrNr into SecGrNr before the secondary FOF overwrites GrNr.
     * SecGrNr is not touched by fof_fof or fof_seed, so it is safe as temporary storage.
     * We need StoreGrNr = 1 so that particles get the secondary FOF
     * group number written into GrNr — the zeroing code below uses
     * GrNr to identify which stars belong to the seeded group. */
    int64_t NumPart_before = PartManager->NumPart;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        P[i].SecGrNr = P[i].GrNr;

    FOFGroups secfof = fof_fof(ddecomp, 1, Comm);

    /* fof_seed returns, per locally-seeded group, the GrNr and (for
     * SeedSecFOFcomSampleParticle) the per-group tot_msc_fof = BHSeedMsc and
     * unseeded stellar mass = SCcomMcut, so we don't infer them from new indices. */
    int64_t * local_seeded_grnr = NULL;
    double * local_seeded_totmsc = NULL;
    double * local_seeded_mcut = NULL;
    int n_local_seeded = 0;
    fof_seed(&secfof, act, tree, atime, rnd, &local_seeded_grnr, &n_local_seeded,
             &local_seeded_totmsc, &local_seeded_mcut, CP, Comm);

    /* Flag (Seeded=1) all type-4 stars in secondary FOF groups that just had a BH
     * seeded, so they are excluded from future seeding sums.  ClusterMass and
     * StarClusterMass_sample are kept on the star as a record (the seed payload
     * mass has already been recorded on BHP.StarClusterMass of the new BH).  In
     * SeedSecFOFcomSampleParticle mode each unseeded member star's ClusterMass is
     * additionally overwritten with its share of the group tot_msc_fof, weighted
     * by stellar mass (denominator = group unseeded stellar mass = mcut).
     * Since groups may span MPI ranks, we Allgather the seeded-group set. */
    int NTask;
    MPI_Comm_size(Comm, &NTask);

    int * recv_counts = (int *) mymalloc2("RecvCounts", NTask * sizeof(int));
    MPI_Allgather(&n_local_seeded, 1, MPI_INT, recv_counts, 1, MPI_INT, Comm);

    int n_total_seeded = 0;
    int * byte_counts = (int *) mymalloc2("ByteCounts", NTask * sizeof(int));
    int * byte_displs = (int *) mymalloc2("ByteDispls", NTask * sizeof(int));
    int boff = 0;
    for(i = 0; i < NTask; i++) {
        byte_counts[i] = recv_counts[i] * (int) sizeof(struct seeded_group);
        byte_displs[i] = boff;
        boff += byte_counts[i];
        n_total_seeded += recv_counts[i];
    }

    if(n_total_seeded > 0) {
        /* Pack local seeded groups, gather, and sort by GrNr for binary search. */
        struct seeded_group * local_sg = (struct seeded_group *) mymalloc2("LocalSeededGroups",
                (n_local_seeded > 0 ? n_local_seeded : 1) * sizeof(struct seeded_group));
        for(i = 0; i < n_local_seeded; i++) {
            local_sg[i].GrNr = local_seeded_grnr[i];
            local_sg[i].totmsc = local_seeded_totmsc ? local_seeded_totmsc[i] : 0;
            local_sg[i].mcut = local_seeded_mcut ? local_seeded_mcut[i] : 0;
        }
        struct seeded_group * all_sg = (struct seeded_group *) mymalloc2("AllSeededGroups",
                n_total_seeded * sizeof(struct seeded_group));
        MPI_Allgatherv(local_sg, n_local_seeded * (int) sizeof(struct seeded_group), MPI_BYTE,
                       all_sg, byte_counts, byte_displs, MPI_BYTE, Comm);
        qsort(all_sg, n_total_seeded, sizeof(struct seeded_group), cmp_seeded_group);

        const int particle_mode = sfof_params.SeedSecFOFcomSampleParticle;
        int64_t n_marked = 0;
        #pragma omp parallel for reduction(+:n_marked)
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].Type != 4 || P[i].GrNr < 0)
                continue;
            /* Binary search for GrNr in the seeded-group list */
            int64_t key = P[i].GrNr;
            int lo = 0, hi = n_total_seeded, found = -1;
            while(lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if(all_sg[mid].GrNr == key) { found = mid; break; }
                else if(all_sg[mid].GrNr < key) lo = mid + 1;
                else hi = mid;
            }
            if(found >= 0) {
                /* Redistribute tot_msc_fof into ClusterMass for stars still
                 * unseeded at this step (the set that fed mcut), weighted by
                 * stellar mass; the group sum is conserved (= tot_msc_fof). */
                if(particle_mode && STARP(i).Seeded == 0 && all_sg[found].mcut > 0)
                    STARP(i).ClusterMass = all_sg[found].totmsc * P[i].Mass / all_sg[found].mcut;
                /* Mark as having contributed to a BH seed; keep ClusterMass and
                 * StarClusterMass_sample as a record.  The group accumulation in
                 * fof.c keys off Seeded (not zeroed ClusterMass), so this is
                 * consistent across all seeding paths. */
                STARP(i).Seeded = 1;
                n_marked++;
            }
        }

        int64_t n_marked_total;
        MPI_Allreduce(&n_marked, &n_marked_total, 1, MPI_INT64, MPI_SUM, Comm);
        message(0, "SecondFOF seed: flagged Seeded=1 for %ld stars in %d seeded groups.%s\n",
                   n_marked_total, n_total_seeded,
                   particle_mode ? " (ClusterMass redistributed from tot_msc_fof)" : "");

        myfree(all_sg);
        myfree(local_sg);
    }

    myfree(byte_displs);
    myfree(byte_counts);
    myfree(recv_counts);
    if(local_seeded_mcut)
        myfree(local_seeded_mcut);
    if(local_seeded_totmsc)
        myfree(local_seeded_totmsc);
    if(local_seeded_grnr)
        myfree(local_seeded_grnr);

    fof_finish(&secfof);

    /* Restore primary FOF GrNr from SecGrNr, then reset SecGrNr to -1.
     * SecGrNr will be recomputed by secondfof_run if a snapshot is written. */
    #pragma omp parallel for
    for(i = 0; i < NumPart_before; i++) {
        P[i].GrNr = P[i].SecGrNr;
        P[i].SecGrNr = -1;
    }
    /* Seeds are now converted in-place (the parent star/gas becomes the BH at
     * the same index), so NumPart does not grow and this loop is normally a
     * no-op.  Kept defensively: any particle appended past NumPart_before is
     * not part of the primary FOF or any prior secondary FOF. */
    int64_t NumPart_now = PartManager->NumPart;
    for(i = NumPart_before; i < NumPart_now; i++) {
        P[i].GrNr = -1;
        P[i].SecGrNr = -1;
    }

    /* Restore original FOF parameters */
    fof_set_params(save_PrimaryLT, save_SecondaryLT,
                   save_LinkLen, save_MinLen, save_PotMin, save_MinPrimLen);
    fof_set_seed_params(save_SeedSC, save_SeedHalo, save_SeedGas);
}

SecondFOFResult * secondfof_run(DomainDecomp * ddecomp, int OutputPotential,
                                double atime, Cosmology * CP, MPI_Comm Comm)
{
    int i;

    if(!sfof_params.SecondFOFOn)
        return NULL;

    message(0, "Begin second FOF (star-primary) computation.\n");

    /* Step 1: Save current FOF parameters */
    int save_PrimaryLinkTypes, save_SecondaryLinkTypes, save_MinLength, save_PotentialMin, save_MinPrimaryLength;
    double save_ComovingLinkingLength;
    fof_get_params(&save_PrimaryLinkTypes, &save_SecondaryLinkTypes,
                   &save_ComovingLinkingLength, &save_MinLength, &save_PotentialMin,
                   &save_MinPrimaryLength);

    /* Step 2: Save current GrNr for all particles */
    int64_t * saved_GrNr = (int64_t *) mymalloc("SecFOF_SavedGrNr", sizeof(int64_t) * PartManager->NumPart);
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        saved_GrNr[i] = P[i].GrNr;

    /* Step 3: Override FOF parameters with second FOF values.
     * Enable PotMin tracking only when potential data is available.
     * MinPrimaryLength drops groups with too few primary-link particles from
     * the catalog (their particles get GrNr=-1 and are excluded automatically). */
    fof_set_params(sfof_params.PrimaryLinkTypes, sfof_params.SecondaryLinkTypes,
                   sfof_params.LinkingLength, sfof_params.MinLength, OutputPotential,
                   sfof_params.MinPrimaryLength);

    /* When enabled, restrict the primary-linking set to unseeded stars for the
     * duration of this fof_fof() call only (reset immediately afterwards). */
    fof_set_primary_unseeded_only(sfof_params.SecFOFUnseededPart);

    /* Step 4: Run the FOF algorithm */
    FOFGroups fof = fof_fof(ddecomp, 1, Comm);

    fof_set_primary_unseeded_only(0);

    /* Step 5: Copy GrNr -> SecGrNr */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        P[i].SecGrNr = P[i].GrNr;

    /* Step 5a: PotMinPos fallback.  Step 3 passed OutputPotential as the FOF
     * PotentialMin flag, and both add_particle_to_group and fof_reduce_group gate the
     * PotMin/PotMinPos update on it, so with OutputPotential = 0 PotMinPos is never
     * written and keeps its memset value (0,0,0) -- the corner of the box.  This used
     * to be repaired further down (old Step 8), but the bound pass below needs the
     * group centre, so the fallback has to happen BEFORE it: centred on (0,0,0) every
     * group would get box-scale radii, an Rmax up to half the box diagonal, and a DM
     * lookup grid that collapses to a single cell.
     * CM is a valid centre here: fof_compile_catalogue (inside fof_fof above) has
     * already divided it by the group mass and periodic-wrapped it. */
    if(!OutputPotential) {
        int g;
        for(g = 0; g < fof.Ngroups; g++) {
            fof.Group[g].PotMin = 0;
            int d;
            for(d = 0; d < 3; d++)
                fof.Group[g].PotMinPos[d] = fof.Group[g].CM[d];
        }
    }

    /* Step 5b: BHseedSecFOFbound -- fill the catalogue's SecBound* blocks.
     * Must happen HERE, while P[].GrNr still holds the secondary group number that
     * fof_secfof_bound_restrict keys on (Step 6 restores the primary one), and after
     * the Step 5a centre fallback.
     * apply = 0: the seeding decision was taken at seeding time on its own FOF pass;
     * this call only measures the bound subset for the snapshot, so the catalogue's
     * SCMass / MassByType / LengthByType keep describing ALL member stars.
     * (The seeding path is not affected by the OutputPotential issue above:
     * secondfof_seed hardcodes PotentialMin = 1 in its own fof_set_params call.) */
    if(sfof_params.BHseedSecFOFbound)
        fof_secfof_bound_restrict(&fof, sfof_params.BHseedSecFOFbound, 0,
                                  atime, CP, NULL /* no per-star mask: catalogue only */,
                                  Comm);

    /* Step 6: Restore original GrNr and free saved_GrNr
     * (must free before allocating result to respect stack allocator order) */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        P[i].GrNr = saved_GrNr[i];

    myfree(saved_GrNr);

    /* Step 6b: Allocate result struct (after saved_GrNr is freed) */
    SecondFOFResult * result = (SecondFOFResult *) mymalloc("SecFOF_Result", sizeof(SecondFOFResult));
    result->fof = fof;

    /* Step 7: Restore original FOF parameters */
    fof_set_params(save_PrimaryLinkTypes, save_SecondaryLinkTypes,
                   save_ComovingLinkingLength, save_MinLength, save_PotentialMin,
                   save_MinPrimaryLength);

    /* Step 8: Compute extra group properties.
     * PotMin/PotMinPos are already computed by fof_compile_catalogue
     * inside add_particle_to_group / fof_reduce_group; the OutputPotential == 0
     * fall-back to CM has already been applied at Step 5a, which has to run before
     * the bound pass that consumes the centre.  result->fof aliases the same Group
     * array, so nothing more is needed here.
     *
     * Everything below uses PotMinPos as the group centre:
     * secondfof_compute_sizes (R50/R90/Rmax) and the SecPotMinPos output block. */

    /* Step 9: Build output array.
     * Allocate SecFOF_Output before SecFOF_Extra so that extra
     * is on top of the stack allocator and can be freed first. */
    result->Ngroups = result->fof.Ngroups;
    result->TotNgroups = result->fof.TotNgroups;
    result->output = (struct SecondGroupOutput *)
        mymalloc("SecFOF_Output", sizeof(struct SecondGroupOutput) * (result->Ngroups > 0 ? result->Ngroups : 1));

    struct SecondGroupExtra * extra = (struct SecondGroupExtra *)
        mymalloc("SecFOF_Extra", sizeof(struct SecondGroupExtra) * (result->fof.Ngroups > 0 ? result->fof.Ngroups : 1));

    /* compute R50, R90, Rmax */
    /* This might be slow, need to optimize it. But should be fine for high-z*/
    if(sfof_params.ComputeSize) {
        secondfof_compute_sizes(&result->fof, extra, Comm);
    }

    /* Compute which primary FOF group(s) each secondary FOF belongs to */
    secondfof_compute_primary_fof_info(&result->fof, extra, Comm);

    for(i = 0; i < result->Ngroups; i++) {
        result->output[i].grp = result->fof.Group[i];
        result->output[i].ext = extra[i];
    }

    myfree(extra);

    walltime_measure("/SecondFOF/Compute");

    message(0, "SecondFOF: %ld groups found.\n", result->TotNgroups);
    return result;
}

void secondfof_write(SecondFOFResult * result, const char * OutputDir, int snapnum,
                     double atime, Cosmology * CP, const double * MassTable,
                     int MetalReturnOn, int OutputDebugFields, MPI_Comm Comm)
{
    int i;

    if(!result)
        return;

    /* Sort by GrNr */
    mpsort_mpi(result->output, result->Ngroups, sizeof(struct SecondGroupOutput),
               fof_radix_SecGrNr, 8, NULL, Comm);

    /* Write the catalog */
    char * fname = fastpm_strdup_printf("%s/%s_%03d", OutputDir, sfof_params.SecondFOFFileBase, snapnum);

    struct IOTable SecFOFIOTable = {0};
    secondfof_register_io_blocks(MetalReturnOn, sfof_params.ComputeSize, sfof_params.SecFOFStarCluster, &SecFOFIOTable);

    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, Comm)) {
        endrun(0, "SecondFOF: Failed to open file at %s\n", fname);
    }
    myfree(fname);

    struct conversions conv = {0};
    conv.atime = atime;
    conv.hubble = hubble_function(CP, atime);

    secondfof_write_header(&bf, result->TotNgroups, atime, MassTable, CP, Comm);

    for(i = 0; i < SecFOFIOTable.used; i++) {
        int ptype = SecFOFIOTable.ent[i].ptype;
        if(ptype == PTYPE_FOF_GROUP) {
            char blockname[128];
            BigArray array = {0};
            sprintf(blockname, "FOFGroups/%s", SecFOFIOTable.ent[i].name);
            build_buffer_secondfof(result->output, result->Ngroups, &array, &SecFOFIOTable.ent[i], &conv);
            message(0, "SecondFOF: Writing Block %s\n", blockname);
            petaio_save_block(&bf, blockname, &array, 1);
            petaio_destroy_buffer(&array);
        }
    }

    destroy_io_blocks(&SecFOFIOTable);

    /* Save particle catalog: temporarily swap GrNr and SecGrNr.
     * GrNr is set to SecGrNr for particle selection/sorting by secondary group,
     * and SecGrNr is set to the original GrNr so the primary FOF ID is preserved.
     * After distribution, fof_save_particles_to_bigfile swaps them back so that
     * GroupID = primary FOF ID and SecGroupID = secondary FOF ID in the output. */
    int64_t * saved_GrNr = (int64_t *) mymalloc("SecFOF_SaveGrNr",
                               sizeof(int64_t) * PartManager->NumPart);
    int64_t * saved_SecGrNr = (int64_t *) mymalloc("SecFOF_SaveSecGrNr",
                               sizeof(int64_t) * PartManager->NumPart);
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        saved_GrNr[i] = P[i].GrNr;
        saved_SecGrNr[i] = P[i].SecGrNr;
        /* Include primary- and secondary-linked particles in the second FOF catalog */
        int type_mask = sfof_params.PrimaryLinkTypes | sfof_params.SecondaryLinkTypes;
        if(P[i].SecGrNr >= 0 && ((1 << P[i].Type) & type_mask)) {
            P[i].GrNr = P[i].SecGrNr;
            P[i].SecGrNr = saved_GrNr[i];
        }
        else
            P[i].GrNr = -1;
    }

    fof_save_particles_to_bigfile(&bf, MetalReturnOn, OutputDebugFields, CP, atime, 1, Comm);

    /* Restore original GrNr and SecGrNr */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        P[i].GrNr = saved_GrNr[i];
        P[i].SecGrNr = saved_SecGrNr[i];
    }
    myfree(saved_SecGrNr);
    myfree(saved_GrNr);

    big_file_mpi_close(&bf, Comm);

    message(0, "SecondFOF: %ld groups saved.\n", result->TotNgroups);

    walltime_measure("/SecondFOF/Write");
}

void secondfof_finish(SecondFOFResult * result)
{
    if(!result)
        return;
    myfree(result->output);
    /* Must free result before Group to respect stack allocator (LIFO) order:
     * allocation order was Group -> SecFOF_Result -> SecFOF_Output */
    struct Group * grp = result->fof.Group;
    myfree(result);
    myfree(grp);

    message(0, "Finished computing second FoF groups.  (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    /* We free Group manually (not via fof_finish) because of LIFO order:
     * result must be freed before Group.  MPI_TYPE_GROUP is handled by
     * the guard in fof_fof (free-before-create) and fof_finish. */
}
