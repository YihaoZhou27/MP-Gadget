#ifndef SCINFO_H
#define SCINFO_H

#include <stdio.h>
#include <stdint.h>
#include "types.h"

/* Record type, stored in the Flag field.  Mbh_seed alone cannot separate the two
 * unseeded cases, so the reason a cluster seeded no BH is recorded explicitly. */
#define SC_FLAG_SEEDED    0   /* a BH was seeded in this cluster */
#define SC_FLAG_NOVMS     1   /* cluster >= MinMscForBHseed but the CW model gave
                               * M_VMS < SeedBlackHoleMass, so no BH was seeded */
#define SC_FLAG_BELOWSEED 2   /* cluster < MinMscForBHseed: never a seed candidate
                               * (recorded only when MinMscForSCdetail is lowered) */
/* Only SC_FLAG_SEEDED records correspond to a BH particle in the simulation.  The other
 * two are model bookkeeping: their Mbh_seed is the mass the CW model predicts for that
 * cluster, but no BH was created and nothing in the run's dynamics saw it. */

/* Per-star-cluster detail record.
 * One record is written every time a star-cluster-based BH seed is created
 * (BlackHoleSeedStarCluster / SeedInSecFOFasStarCluster / SeedSecFOFcomSample),
 * including seeding steps between checkpoints that are not captured by snapshots.
 * When MbhMscRelationCWmodel=1 a record is ALSO written for each sampled cluster
 * that seeds no BH because its model M_VMS < SeedBlackHoleMass: there Mbh_seed=0
 * and ID/Pos refer to the candidate host star (which stays a star).
 * With MinMscForSCdetail below MinMscForBHseed the per-cluster secFOF mode
 * (SecFOFseedsumover=0) additionally records every sampled cluster down to that
 * mass, which can never seed a BH: there Mbh_seed=0, Flag=SC_FLAG_BELOWSEED and
 * ID/Pos refer to the host group's reference star (shared by all such clusters of
 * the same group draw, since they have no host of their own).
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
    /* Host group total STELLAR mass Sum(m_star) over ALL member stars (Group MassType[4]),
     * code units.  Distinct from StarClusterMassTotal, which is the cluster mass
     * Sum(Gamma*m_star): Gamma is the per-star birth-pressure CFE, so the two are not
     * related by any single factor.  Their ratio is the group's mass-weighted mean CFE. */
    double   StellarMassTotal;
    double   SCMass_seeded;        /* Host group cluster mass already consumed by earlier seeds (SecSCMass_seeded, as-is). */
    double   Metallicity;          /* Unseeded-star metal mass ratio: Sum(BirthMet*initClusterMass)/Sum(initClusterMass). */
    double   Reff;                 /* Effective radius [pc] (per-cluster SecFOFseedsumover=0 seeding only; 0 otherwise). */
    /* Model seed mass of this cluster, code units.  Under MbhMscRelationCWmodel this is
     * the CW-model M_VMS for EVERY record, including the clusters that seed no BH
     * (Flag != SC_FLAG_SEEDED): whether a BH particle was actually created is Flag's job,
     * not this field's.  For Flag == SC_FLAG_SEEDED it is the new BH's subgrid mass (the
     * same M_VMS when the CW model is on).  0 when the CW model is off and no BH was
     * seeded, or when the model yields no VMS at all. */
    double   Mbh_seed;
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
    int      Flag;                 /* record type: SC_FLAG_SEEDED / _NOVMS / _BELOWSEED. */
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

/* Whether records are being written on this rank.  Lets the caller skip the work of
 * generating detail records (notably the full sub-seeding-threshold population) when
 * StarClusterDetails is off. */
int scinfo_enabled(void);

/* Flush the buffered records to disk.  scinfo_record_seed flushes on its own (seed
 * events are rare); the bulk path scinfo_record_cluster does not, so its caller must
 * call this once it is done with a batch. */
void scinfo_flush(void);

/* Append one star-cluster seed record for the newly-made BH at base index `index`
 * (its ID and Pos are read from the particle).  Mbh_seed is the seeded BH's
 * subgrid mass (BHP.Mass, code units), or 0 for a cluster that seeded no BH
 * (MbhMscRelationCWmodel skip; `index` is then the candidate host star).
 * Flushes the stream.  No-op when no file is registered.
 * Called serially from the BH-seeding paths in fof.c. */
void scinfo_record_seed(int index, double atime, double SCmass, double SCMassTotal,
                        double StellarMassTotal, double SCMassSeeded, double metallicity,
                        double Reff, double Mbh_seed, int NBHInGroup, int64_t GrNr,
                        const struct SCmetdist * metdist, int flag);

/* Append one record for a sampled star cluster that has no host particle of its own
 * (MinMscForSCdetail): `id` and `pos` are the host group's reference star, `pos` in the
 * translated frame (P[].Pos), exactly as scinfo_record_seed reads them.  Mbh_seed is the
 * cluster's model seed mass, which under MbhMscRelationCWmodel is evaluated for these
 * clusters too even though no BH is created for them (see the Mbh_seed field above).
 * Does NOT flush -- these come in large batches; call scinfo_flush() after.
 * No-op when no file is registered.  Called serially from fof.c. */
void scinfo_record_cluster(MyIDType id, const double * pos, double atime, double SCmass,
                           double SCMassTotal, double StellarMassTotal, double SCMassSeeded,
                           double metallicity, double Reff, double Mbh_seed, int NBHInGroup,
                           int64_t GrNr, const struct SCmetdist * metdist, int flag);

#endif
