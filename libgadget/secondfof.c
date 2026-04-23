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

/* Extended group properties for the second FOF.
 * PotMin and PotMinPos are now computed inside fof_compile_catalogue
 * (in add_particle_to_group / fof_reduce_group) and stored in struct Group.
 * This struct only holds the size properties. */
struct SecondGroupExtra {
    float  R50;             /* Half-mass radius of primary particles */
    float  R90;             /* 90%-mass radius of primary particles */
    float  Rmax;            /* Max primary particle separation from center */
};

/* Comparison for sorting (distance, mass) pairs by distance */
struct dist_mass {
    double dist;
    double mass;
};

static int cmp_dist_mass(const void * a, const void * b)
{
    const struct dist_mass * da = (const struct dist_mass *)a;
    const struct dist_mass * db = (const struct dist_mass *)b;
    return (da->dist > db->dist) - (da->dist < db->dist);
}

/*
 * Compute R50, R90, Rmax for each owned group using PotMinPos as center.
 * Only considers primary-linked particles.
 */
static void
secondfof_compute_sizes(FOFGroups * fof, struct SecondGroupExtra * extra, MPI_Comm Comm)
{
    int i, g;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    double BoxSize = PartManager->BoxSize;

    for(g = 0; g < fof->Ngroups; g++) {
        /* Count local primary particles in this group */
        int count = 0;
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].SecGrNr != fof->Group[g].base.GrNr) continue;
            if(!((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) continue;
            count++;
        }

        /* Collect (distance, mass) pairs */
        struct dist_mass * dm_local = (struct dist_mass *) mymalloc2("SecFOF_dm", sizeof(struct dist_mass) * (count > 0 ? count : 1));
        int n = 0;
        for(i = 0; i < PartManager->NumPart; i++) {
            if(P[i].SecGrNr != fof->Group[g].base.GrNr) continue;
            if(!((1 << P[i].Type) & sfof_params.PrimaryLinkTypes)) continue;

            double dx[3];
            double r2 = 0;
            int d;
            for(d = 0; d < 3; d++) {
                dx[d] = NEAREST(P[i].Pos[d] - fof->Group[g].PotMinPos[d], BoxSize);
                r2 += dx[d] * dx[d];
            }
            dm_local[n].dist = sqrt(r2);
            dm_local[n].mass = P[i].Mass;
            n++;
        }

        /* Gather across all ranks */
        int NTask;
        MPI_Comm_size(Comm, &NTask);
        int * rcounts = (int *) mymalloc2("SecFOF_rc", sizeof(int) * NTask);
        MPI_Allgather(&n, 1, MPI_INT, rcounts, 1, MPI_INT, Comm);

        int total = 0;
        int * rdispls = (int *) mymalloc2("SecFOF_rd", sizeof(int) * NTask);
        for(i = 0; i < NTask; i++) {
            rdispls[i] = total;
            total += rcounts[i];
        }

        struct dist_mass * dm_global = (struct dist_mass *) mymalloc2("SecFOF_dmg", sizeof(struct dist_mass) * (total > 0 ? total : 1));

        int * bc = (int *) mymalloc2("SecFOF_bc", sizeof(int) * NTask);
        int * bd = (int *) mymalloc2("SecFOF_bd", sizeof(int) * NTask);
        for(i = 0; i < NTask; i++) {
            bc[i] = rcounts[i] * sizeof(struct dist_mass);
            bd[i] = rdispls[i] * sizeof(struct dist_mass);
        }
        MPI_Allgatherv(dm_local, n * sizeof(struct dist_mass), MPI_BYTE,
                       dm_global, bc, bd, MPI_BYTE, Comm);
        myfree(bd);
        myfree(bc);
        myfree(rdispls);
        myfree(rcounts);
        myfree(dm_local);

        /* Sort by distance and compute R50, R90, Rmax */
        qsort(dm_global, total, sizeof(struct dist_mass), cmp_dist_mass);

        double total_mass = 0;
        for(i = 0; i < total; i++)
            total_mass += dm_global[i].mass;

        double cumul_mass = 0;
        float r50 = 0, r90 = 0, rmax = 0;
        int found50 = 0, found90 = 0;
        for(i = 0; i < total; i++) {
            cumul_mass += dm_global[i].mass;
            if(!found50 && cumul_mass >= 0.5 * total_mass) {
                r50 = dm_global[i].dist;
                found50 = 1;
            }
            if(!found90 && cumul_mass >= 0.9 * total_mass) {
                r90 = dm_global[i].dist;
                found90 = 1;
            }
            rmax = dm_global[i].dist;
        }
        extra[g].R50 = r50;
        extra[g].R90 = r90;
        extra[g].Rmax = rmax;

        myfree(dm_global);
    }
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

    struct SecondGroupExtra * extra = (struct SecondGroupExtra *)
        mymalloc("SecFOF_Extra", sizeof(struct SecondGroupExtra) * (result->fof.Ngroups > 0 ? result->fof.Ngroups : 1));

    /* compute R50, R90, Rmax */ 
    /* This might be slow, need to optimize it. But should be fine for high-z*/
    if(sfof_params.ComputeSize) {
        secondfof_compute_sizes(&result->fof, extra, Comm);
    }

    /* Step 9: Build output array */
    result->Ngroups = result->fof.Ngroups;
    result->TotNgroups = result->fof.TotNgroups;
    result->output = (struct SecondGroupOutput *)
        mymalloc("SecFOF_Output", sizeof(struct SecondGroupOutput) * (result->Ngroups > 0 ? result->Ngroups : 1));

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
    big_file_mpi_close(&bf, Comm);

    message(0, "SecondFOF: %ld groups saved.\n", result->TotNgroups);

    walltime_measure("/SecondFOF/Write");
}

void secondfof_finish(SecondFOFResult * result)
{
    if(!result)
        return;
    myfree(result->output);
    fof_finish(&result->fof);
    myfree(result);
}
