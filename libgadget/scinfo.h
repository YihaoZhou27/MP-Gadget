#ifndef SCINFO_H
#define SCINFO_H

#include <stdio.h>
#include <stdint.h>
#include "types.h"

/* Per-seeded-star-cluster detail record.
 * One record is written every time a star-cluster-based BH seed is created
 * (BlackHoleSeedStarCluster / SeedInSecFOFasStarCluster / SeedSecFOFcomSample),
 * including seeding steps between checkpoints that are not captured by snapshots.
 * When MbhMscRelationCWmodel=1 a record is ALSO written for each sampled cluster
 * that seeds no BH because its model M_VMS < SeedBlackHoleMass: there Mbh_seed=0
 * and ID/Pos refer to the candidate host star (which stays a star).
 * Packed so the on-disk record size is identical on all architectures; size1 and
 * size2 bracket the payload (both equal sizeof(record) - 2*sizeof(int)) so a reader
 * can validate the record length, mirroring the BlackholeDetails (struct BHinfo)
 * convention. */
struct __attribute__((__packed__)) SCseedinfo {
    int      size1;
    MyIDType ID;                   /* ID of the seed BH particle (or the candidate host star for Mbh_seed=0 records). */
    int64_t  GrNr;                 /* Host (secondary) FOF group number. */
    double   a;                    /* Scale factor at seeding (formation time). */
    double   Pos[3];               /* Seed position (particle-offset corrected). */
    double   StarClusterMass;      /* Seed cluster mass: sampled >1e4 Msun sum (com mode) or the mode's seeding SC mass. */
    double   StarClusterMassTotal; /* Host group total Sum(Gamma*m_star) over ALL member stars. */
    double   SCMass_seeded;        /* Host group cluster mass already consumed by earlier seeds (SecSCMass_seeded, as-is). */
    double   Metallicity;          /* Unseeded-star metal mass ratio: Sum(BirthMet*initClusterMass)/Sum(initClusterMass). */
    double   Reff;                 /* Effective radius [pc] (per-cluster SecFOFseedsumover=0 seeding only; 0 otherwise). */
    double   Mbh_seed;             /* Subgrid mass (code units) of the BH seeded in this cluster; 0 if no BH was seeded. */
    int      NBHInGroup;           /* BH count in the host group before this seed (LenType[5]). */
    /* Distribution of the host secFOF's unseeded-star BirthMetallicity (absolute
     * Z mass fraction), per particle with equal weight.  min/max are exact; the
     * quartiles/median come from a fixed log10(Z) histogram; std is from running
     * sums.  All zero when the host group has no unseeded star. */
    double   MetUnseededMin;
    double   MetUnseededMax;
    double   MetUnseededMedian;    /* 50th percentile. */
    double   MetUnseededP25;       /* 25th percentile. */
    double   MetUnseededP75;       /* 75th percentile. */
    double   MetUnseededStd;       /* standard deviation. */
    int      size2;
};

/* Distribution stats of one host group's unseeded-star metallicity, passed to
 * scinfo_record_seed (a NULL pointer records all-zero, e.g. no unseeded star). */
struct SCmetdist {
    double min, max, median, p25, p75, std;
};

/* Register the per-rank StarClusterDetails file handle (may be NULL, which disables
 * recording).  Called once from open_outputfiles. */
void scinfo_set_file(FILE * fd);

/* Append one star-cluster seed record for the newly-made BH at base index `index`
 * (its ID and Pos are read from the particle).  Mbh_seed is the seeded BH's
 * subgrid mass (BHP.Mass, code units), or 0 for a cluster that seeded no BH
 * (MbhMscRelationCWmodel skip; `index` is then the candidate host star).
 * No-op when no file is registered.
 * Called serially from the BH-seeding paths in fof.c. */
void scinfo_record_seed(int index, double atime, double SCmass, double SCMassTotal,
                        double SCMassSeeded, double metallicity, double Reff,
                        double Mbh_seed, int NBHInGroup, int64_t GrNr,
                        const struct SCmetdist * metdist);

#endif
