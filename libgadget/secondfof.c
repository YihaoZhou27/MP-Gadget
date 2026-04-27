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
    int ComputeSize;        /* compute R50, R90, Rmax */
    int SecFOFonly;          /* skip primary FOF catalog, only save SecPIG */
    int SeedInSecFOF;       /* seed BH using secondary FOF instead of primary */
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
        sfof_params.ComputeSize = param_get_int(ps, "SecondFOFSize");
        sfof_params.SecFOFonly = param_get_int(ps, "SecFOFonly");
        sfof_params.SeedInSecFOF = param_get_int(ps, "SeedInSecFOF");
        strncpy(sfof_params.SecondFOFFileBase, param_get_string(ps, "SecondFOFFileBase"), sizeof(sfof_params.SecondFOFFileBase) - 1);
    }
    MPI_Bcast(&sfof_params, sizeof(struct SecondFOFParams), MPI_BYTE, 0, MPI_COMM_WORLD);
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
    return sfof_params.SeedInSecFOF;
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
    int32_t PrimaryFOFID;   /* Primary FOF GrNr that hosts the largest fraction of primary-linked particles; -1 if none */
};

/* (distance, mass, GrNr) tuple for computing group sizes. */
struct dist_mass_grp {
    double dist;
    double mass;
    int64_t GrNr;
};

static int cmp_dist_mass_grp_by_dist(const void * a, const void * b)
{
    const struct dist_mass_grp * da = (const struct dist_mass_grp *)a;
    const struct dist_mass_grp * db = (const struct dist_mass_grp *)b;
    return (da->dist > db->dist) - (da->dist < db->dist);
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

    struct grp_center {
        int64_t GrNr;
        double PotMinPos[3];
    };
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
        int64_t j;
        for(j = 0; j < total_groups; j++) {
            if(all_centers[j].GrNr == grNr) {
                int d;
                for(d = 0; d < 3; d++)
                    center[d] = all_centers[j].PotMinPos[d];
                break;
            }
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

    /* Step 4: Each rank computes R50/R90/Rmax for its own groups
     * using the complete global particle data. No more MPI needed. */
    for(g = 0; g < fof->Ngroups; g++) {
        int64_t grNr = fof->Group[g].base.GrNr;

        /* Count particles belonging to this group */
        int count = 0;
        int64_t k;
        for(k = 0; k < total_parts; k++) {
            if(dm_global[k].GrNr == grNr)
                count++;
        }

        if(count > 0) {
            /* Extract and sort by distance */
            struct dist_mass_grp * grp_dm = (struct dist_mass_grp *)
                mymalloc2("SecFOF_gdm", sizeof(struct dist_mass_grp) * count);
            int idx = 0;
            for(k = 0; k < total_parts; k++) {
                if(dm_global[k].GrNr == grNr)
                    grp_dm[idx++] = dm_global[k];
            }
            qsort(grp_dm, count, sizeof(struct dist_mass_grp), cmp_dist_mass_grp_by_dist);

            double total_mass = 0;
            for(idx = 0; idx < count; idx++)
                total_mass += grp_dm[idx].mass;

            double cumul_mass = 0;
            float r50 = 0, r90 = 0, rmax = 0;
            int found50 = 0, found90 = 0;
            for(idx = 0; idx < count; idx++) {
                cumul_mass += grp_dm[idx].mass;
                if(!found50 && cumul_mass >= 0.5 * total_mass) {
                    r50 = grp_dm[idx].dist;
                    found50 = 1;
                }
                if(!found90 && cumul_mass >= 0.9 * total_mass) {
                    r90 = grp_dm[idx].dist;
                    found90 = 1;
                }
                rmax = grp_dm[idx].dist;
            }
            extra[g].R50 = r50;
            extra[g].R90 = r90;
            extra[g].Rmax = rmax;
            myfree(grp_dm);
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

static int cmp_int64(const void * a, const void * b)
{
    const int64_t va = *(const int64_t *)a;
    const int64_t vb = *(const int64_t *)b;
    return (va > vb) - (va < vb);
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
    struct sec_prim_pair {
        int64_t SecGrNr;
        int64_t GrNr;
    };

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

    /* Step 3: For each owned group, count distinct primary FOF IDs
     * and find the one with the most particles. */
    for(g = 0; g < fof->Ngroups; g++) {
        int64_t secGrNr = fof->Group[g].base.GrNr;

        /* Count particles in this sec FOF group */
        int count = 0;
        int64_t k;
        for(k = 0; k < total_pairs; k++) {
            if(all_pairs[k].SecGrNr == secGrNr)
                count++;
        }

        if(count == 0) {
            extra[g].PrimaryFOFNum = 0;
            extra[g].PrimaryFOFID = -1;
            continue;
        }

        /* Collect the primary GrNr values for this group */
        int64_t * prim_ids = (int64_t *)
            mymalloc2("SecFOF_pid", sizeof(int64_t) * count);
        int idx = 0;
        for(k = 0; k < total_pairs; k++) {
            if(all_pairs[k].SecGrNr == secGrNr)
                prim_ids[idx++] = all_pairs[k].GrNr;
        }

        /* Count distinct primary FOF IDs and find the dominant one.
         * Sort the array, then scan for unique values and track counts. */
        qsort(prim_ids, count, sizeof(int64_t), cmp_int64);

        int num_distinct = 0;
        int64_t best_id = -1;
        int best_count = 0;
        int cur_count = 1;
        int64_t cur_id = prim_ids[0];

        for(idx = 1; idx < count; idx++) {
            if(prim_ids[idx] == cur_id) {
                cur_count++;
            } else {
                /* Finished a run of cur_id */
                if(cur_id >= 0) {
                    num_distinct++;
                    if(cur_count > best_count) {
                        best_count = cur_count;
                        best_id = cur_id;
                    }
                }
                cur_id = prim_ids[idx];
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
        extra[g].PrimaryFOFID = (int32_t) best_id;

        myfree(prim_ids);
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

SIMPLE_PROPERTY_SECFOF(PotMin, grp.PotMin, float, 1)
SIMPLE_PROPERTY_SECFOF(R50, ext.R50, float, 1)
SIMPLE_PROPERTY_SECFOF(R90, ext.R90, float, 1)
SIMPLE_PROPERTY_SECFOF(Rmax, ext.Rmax, float, 1)
SIMPLE_PROPERTY_SECFOF(PrimaryFOFNum, ext.PrimaryFOFNum, int32_t, 1)
SIMPLE_PROPERTY_SECFOF(PrimaryFOFID, ext.PrimaryFOFID, int32_t, 1)

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
secondfof_register_io_blocks(int MetalReturnOn, int ComputeSize, struct IOTable * IOTable)
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

    /* Second FOF specific properties */
    IO_REG(SecPotMinPos, "f8", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecPotMin, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    if(ComputeSize) {
        IO_REG(SecR50, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecR90, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(SecRmax, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    }
    IO_REG(SecPrimaryFOFNum, "i4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(SecPrimaryFOFID, "i4", 1, PTYPE_FOF_GROUP, IOTable);
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
        /* Only count primary-linked particles */
        if(!((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) continue;
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
    int primary = sfof_params.PrimaryLinkTypes;
    int secondary = sfof_params.SecondaryLinkTypes;
    big_block_set_attr(&bh, "SecondFOFLinkingLength", &ll, "f8", 1);
    big_block_set_attr(&bh, "SecondFOFMinLength", &minlen, "i4", 1);
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

void secondfof_seed(DomainDecomp * ddecomp, ActiveParticles * act,
                    double atime, const RandTable * rnd, MPI_Comm Comm)
{
    message(0, "Seeding black holes using secondary FOF catalog.\n");

    /* Save current FOF parameters */
    int save_PrimaryLT, save_SecondaryLT, save_MinLen, save_PotMin;
    double save_LinkLen;
    fof_get_params(&save_PrimaryLT, &save_SecondaryLT,
                   &save_LinkLen, &save_MinLen, &save_PotMin);

    /* Override with secondary FOF parameters */
    fof_set_params(sfof_params.PrimaryLinkTypes, sfof_params.SecondaryLinkTypes,
                   sfof_params.LinkingLength, sfof_params.MinLength, 0);

    /* Run FOF with secondary params and seed from the result.
     * Do NOT call fof_finish() here because fof_fof() overwrites the static
     * MPI_TYPE_GROUP, and fof_finish() would free it -- the outer primary
     * fof_finish() would then double-free. Free only the Group array. */
    FOFGroups secfof = fof_fof(ddecomp, 0, Comm);
    fof_seed(&secfof, act, atime, rnd, Comm);
    myfree(secfof.Group);

    /* Restore original FOF parameters */
    fof_set_params(save_PrimaryLT, save_SecondaryLT,
                   save_LinkLen, save_MinLen, save_PotMin);
}

SecondFOFResult * secondfof_run(DomainDecomp * ddecomp, int OutputPotential, MPI_Comm Comm)
{
    int i;

    if(!sfof_params.SecondFOFOn)
        return NULL;

    message(0, "Begin second FOF (star-primary) computation.\n");

    /* Step 1: Save current FOF parameters */
    int save_PrimaryLinkTypes, save_SecondaryLinkTypes, save_MinLength, save_PotentialMin;
    double save_ComovingLinkingLength;
    fof_get_params(&save_PrimaryLinkTypes, &save_SecondaryLinkTypes,
                   &save_ComovingLinkingLength, &save_MinLength, &save_PotentialMin);

    /* Step 2: Save current GrNr for all particles */
    int64_t * saved_GrNr = (int64_t *) mymalloc("SecFOF_SavedGrNr", sizeof(int64_t) * PartManager->NumPart);
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        saved_GrNr[i] = P[i].GrNr;

    /* Step 3: Override FOF parameters with second FOF values.
     * Enable PotMin tracking only when potential data is available. */
    fof_set_params(sfof_params.PrimaryLinkTypes, sfof_params.SecondaryLinkTypes,
                   sfof_params.LinkingLength, sfof_params.MinLength, OutputPotential);

    /* Step 4: Run the FOF algorithm */
    FOFGroups fof = fof_fof(ddecomp, 1, Comm);

    /* Step 5: Copy GrNr -> SecGrNr */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
        P[i].SecGrNr = P[i].GrNr;

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
                   save_ComovingLinkingLength, save_MinLength, save_PotentialMin);

    /* Step 8: Compute extra group properties.
     * PotMin/PotMinPos are already computed by fof_compile_catalogue
     * inside add_particle_to_group / fof_reduce_group.
     * If potential was not computed (OutputPotential == 0), fall back to CM. */
    if(!OutputPotential) {
        int g;
        for(g = 0; g < result->fof.Ngroups; g++) {
            result->fof.Group[g].PotMin = 0;
            int d;
            for(d = 0; d < 3; d++)
                result->fof.Group[g].PotMinPos[d] = result->fof.Group[g].CM[d];
        }
    }

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
                     int MetalReturnOn, MPI_Comm Comm)
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
    secondfof_register_io_blocks(MetalReturnOn, sfof_params.ComputeSize, &SecFOFIOTable);

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
        /* Only include primary-linked particles in the second FOF catalog */
        if(P[i].SecGrNr >= 0 && ((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) {
            P[i].GrNr = P[i].SecGrNr;
            P[i].SecGrNr = saved_GrNr[i];
        }
        else
            P[i].GrNr = -1;
    }

    fof_save_particles_to_bigfile(&bf, MetalReturnOn, CP, atime, 1, Comm);

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

    /* Do NOT call fof_finish() here. fof_finish() frees the static
     * MPI_TYPE_GROUP datatype, but fof_fof() (called by secondfof_run)
     * overwrites that same static variable. The outer fof_finish() for
     * the primary FOF will free it; calling it here would double-free. */
}
