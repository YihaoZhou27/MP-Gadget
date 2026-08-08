#ifndef FOF_H
#define FOF_H

#include <bigfile.h>

#include "utils/paramset.h"
#include "timestep.h"
#include "slotsmanager.h"
#include "forcetree.h"
#include "utils/system.h"

void set_fof_params(ParameterSet * ps);

void fof_init(double DMMeanSeparation);

/* Allow secondfof.c to temporarily override FOF parameters */
void fof_get_params(int *PrimaryLinkTypes, int *SecondaryLinkTypes,
                    double *ComovingLinkingLength, int *MinLength,
                    int *PotentialMin, int *MinPrimaryLength);
void fof_set_params(int PrimaryLinkTypes, int SecondaryLinkTypes,
                    double ComovingLinkingLength, int MinLength,
                    int PotentialMin, int MinPrimaryLength);
/* Restrict the primary-linking set to unseeded star particles (drop seeded
 * stars). Set transiently by the second FOF when SecFOFUnseededPart=1. */
void fof_set_primary_unseeded_only(int flag);
void fof_get_seed_params(int *BlackHoleSeedStarCluster, int *BlackHoleSeedHaloBased,
                         int *BlackHoleSeedGasBased);
void fof_set_seed_params(int BlackHoleSeedStarCluster, int BlackHoleSeedHaloBased,
                         int BlackHoleSeedGasBased);
/* For the tests*/
void set_fof_testpar(int FOFSaveParticles, double FOFHaloLinkingLength, int FOFHaloMinLength);

struct BaseGroup {
    int OriginalTask;
    int OriginalIndex;
    int Length;
    /* Number of primary-linking-type particles in the group. Used by the
     * second FOF MinPrimaryLength filter; counted in fof_compile_base. */
    int LenPrimary;
    int GrNr;
    MyIDType MinID;
    int MinIDTask;
    /* Note: this is in the translated frame,
     * subtract CurrentParticleOffset to get the physical frame.*/
    float FirstPos[3];
};

/* Fixed log10(Z) histogram of the unseeded-star BirthMetallicity, per group,
 * used to derive the StarClusterDetails metallicity quartiles/median without
 * gathering per-particle values (the additive bin counts reduce like the other
 * group sums).  Bin 0 is underflow (Z <= 10^LOGMIN, including pristine Z=0),
 * bins 1..NBIN-2 are log-spaced across [LOGMIN,LOGMAX], bin NBIN-1 is overflow
 * (Z > 10^LOGMAX).  With 62 interior bins the resolution is ~0.13 dex, far
 * finer than the physical scatter.
 *
 * DO NOT raise LOGMIN for the extra resolution.  It is a real trade-off, not just a
 * plotting range, because CWmodelMetallicity 'starsample' draws per-cluster
 * metallicities off this histogram and the underflow bin returns the group's EXACT
 * zmin for every draw that lands in it -- there is no width to interpolate across.
 * Raising LOGMIN buys resolution but pushes more stars into that bin, where they all
 * collapse onto a single value that may sit far below the floor; since
 * M_VMS ~ (Z/Zsun)^-0.352, an over-weighted metal-poor value inflates seed masses.
 * Measured on paper_runs (41,701 unseeded stars; per-group |error| on <Z^-0.352> over
 * the groups that actually seed):
 *
 *   LOGMIN   dlog     underflow   median |err|   max |err|
 *    -7.0   0.129 dex    0.045%      0.0054        0.027     <-- current
 *    -6.0   0.113 dex    0.690%      0.0052        0.436
 *    -5.0   0.097 dex    4.885%      0.0898        3.344
 *
 * The median barely moves; the TAIL is what degrades, and the groups it degrades are
 * the metal-poor ones that make the massive seeds.  -6.0 was tried and reverted on that
 * basis.  Revisit only if the underflow bin is changed to interpolate between zmin and
 * 10^LOGMIN instead of returning zmin flat, which would make the floor nearly free. */
#define SC_MET_HIST_NBIN   64
#define SC_MET_HIST_LOGMIN (-7.0)
#define SC_MET_HIST_LOGMAX (1.0)

struct Group
{
    struct BaseGroup base;
    int Length;
    int LenType[6];
    double MassType[6];
    double Mass;
    double sfmp_mass;   // the mass of the star-forming, low-metallicity gas, which is used as bh-seeding criteria 
    /* Note: this is in the translated frame,
     * subtract CurrentParticleOffset to get the physical frame.*/
    double CM[3];
    double Vel[3];

    double Imom[3][3]; /* sum M r_j r_k */
    double Jmom[3]; /* sum M R_i x V_i  */

    double Sfr;
    /* Metal masses. These are the total mass in metals in the gas and stars respectively,
     * and then the species specific breakdowns for stars and gas.
     * You can obtain metallicities by dividing them by the type-specific masses.*/
    double GasMetalMass;
    double StellarMetalMass;
    float StellarMetalElemMass[NMETALS];
    float GasMetalElemMass[NMETALS];

    /* Number of gas particles in the halo which have had helium ionization happen to them*/
    float MassHeIonized;
    /*These are used for storing black hole properties*/
    double BH_Mass;
    double BH_Mdot;
    double MaxDens;

    int seed_index;
    int seed_task;

    /* Star particle with the largest ClusterMass (or StarClusterMass_sample
     * when StarClusterSampling=1), used for star-cluster BH seeding
     * in secondary FOF (where groups have no gas). */
    int seed_index_star;
    int seed_task_star;
    MyFloat MaxStarClusterMass; /*!< Largest ClusterMass (or StarClusterMass_sample) among type-4 particles */

    /***********************/
    MyFloat StarClusterMass; /*!< TOTAL ClusterMass over ALL hosted stars (seeded + unseeded). Catalogue output (SCMass). */
    MyFloat StarClusterMetallicity; /*!< Mass-weighted metallicity sum over ALL hosted stars (total). */
    float StarClusterMetalElemMass[NMETALS]; /*!< Mass-weighted species metal sums over ALL hosted stars (total). */
    /* Minimum gravitational potential among primary-linked particles.
     * Tracked during catalogue compilation, reduced across MPI ranks. */
    float PotMin;
    /* Position of the primary particle with minimum potential
     * (in the translated frame, subtract CurrentParticleOffset for physical). */
    double PotMinPos[3];
    MyFloat StarClusterMassSample; /*!< Sum of StarClusterMass_sample for all hosted stars */
    int NscSample; /*!< Sum of Nsc_sample for all hosted stars */

    /* Cluster mass split by seeding state (Seeded flag on the star slot).
     * StarClusterMassUnseeded drives ALL star-cluster BH seeding; SCMass_seeded
     * is the consumed mass kept for the catalogue. The two sum to StarClusterMass. */
    MyFloat StarClusterMassUnseeded;       /*!< Sum of ClusterMass over UNSEEDED stars (drives seeding). */
    MyFloat StarClusterMassSampleUnseeded; /*!< Sum of StarClusterMass_sample over UNSEEDED stars (sampled seeding). */
    MyFloat SCMass_seeded;                 /*!< Sum of ClusterMass over SEEDED stars (catalogue output). */
    int NStarUnseeded;                     /*!< Count of UNSEEDED type-4 stars (caps the per-secFOF multi-seed number). */
    /* Unseeded-star metallicity for the StarClusterDetails record AND for the CW
     * seed-mass model's per-cluster Z draw: the metal mass ratio
     * Sum(BirthMetallicity*initClusterMass) / Sum(initClusterMass) over UNSEEDED
     * stars. Both sums accumulated separately (over Seeded==0 stars) and reduced.
     *
     * NStarUnseeded and every SCMet* field below are OVERWRITTEN with their
     * bound-only counterparts by fof_secfof_bound_restrict(apply=1), so that under
     * BHseedSecFOFbound the metallicity the seed mass is drawn from comes from the
     * same stars that supplied the budget.  They therefore describe the bound
     * unseeded subset in the seeding path, and all unseeded stars everywhere else
     * (the catalogue call passes apply=0 and leaves them alone). */
    MyFloat SCMetalMassUnseeded;           /*!< Sum of BirthMetallicity*initClusterMass over UNSEEDED stars (metal mass). */
    MyFloat SCClusterMassUnseededInit;     /*!< Sum of initClusterMass over UNSEEDED stars (metallicity denominator). */
    /* Per-particle (equal-weight) BirthMetallicity distribution of UNSEEDED stars,
     * recorded in the StarClusterDetails file: exact min/max, running sums for the
     * standard deviation, and a fixed log10(Z) histogram for the median/quartiles.
     * The count N is NStarUnseeded.  Min is initialised to a large sentinel in
     * add_particle_to_group; all others start at 0 (memset). */
    float   SCMetUnseededMin;              /*!< min BirthMetallicity over unseeded stars. */
    float   SCMetUnseededMax;              /*!< max BirthMetallicity over unseeded stars. */
    double  SCMetUnseededSum;              /*!< Sum of BirthMetallicity (equal weight; std numerator). */
    double  SCMetUnseededSum2;             /*!< Sum of BirthMetallicity^2 (equal weight; std numerator). */
    /* Equal-weight (per star, NOT mass-weighted) log10(BirthMetallicity) sums of
     * UNSEEDED stars, floored at SC_MET_HIST_LOGMIN (covers pristine Z=0): the
     * mean/std of log10(Z) for the CW-model 'lognormal' metallicity draw. */
    double  SCMetUnseededLogSum;           /*!< Sum of log10(BirthMetallicity), floored (CWmodelMetallicity). */
    double  SCMetUnseededLogSum2;          /*!< Sum of log10(BirthMetallicity)^2, floored (CWmodelMetallicity). */
    float   SCMetUnseededHist[SC_MET_HIST_NBIN]; /*!< log10(Z) histogram (see SC_MET_HIST_* above). */

    /* BHseedSecFOFbound diagnostics: the gravitationally bound subset of the
     * group's UNSEEDED stars, as selected by fof_secfof_bound_restrict.  These
     * are pure bookkeeping -- the restriction itself acts by overwriting
     * StarClusterMassUnseeded / SCcomMcut -- and are written to the SecPIG
     * catalogue and the StarClusterDetails record.  All zero when
     * BHseedSecFOFbound = 0, i.e. when no bound selection was performed.
     * NOT accumulated in add_particle_to_group and NOT summed in
     * fof_reduce_group: they are filled after the group reduction, on the
     * owning rank only, from the globally gathered member list. */
    MyFloat SCBoundStarMass;         /*!< Sum of m_star over BOUND member stars (seeded + unseeded). */
    MyFloat SCBoundStarMassUnseeded; /*!< Sum of m_star over BOUND UNSEEDED stars (the new SCcomMcut). */
    /* Cluster mass of the bound subset, NOT f(Z)-weighted and NOT the seeding budget.
     * They use the same definition as StarClusterMass / SCMass_seeded above -- plain
     * Sum(ClusterMass) = Sum(Gamma*m_star) -- so the bound fractions are ratios of like
     * for like: SCBoundClusterMass/StarClusterMass over all stars, and
     * SCBoundClusterMassUnseeded/(StarClusterMass - SCMass_seeded) over the unseeded
     * ones.  The seeding budget is the f(Z)-weighted unseeded sum, which the apply pass
     * writes into StarClusterMassUnseeded; the two differ whenever f(Z) < 1 somewhere
     * (i.e. StarClusterSeedMetallicityMax > Min). */
    MyFloat SCBoundClusterMass;         /*!< Sum of ClusterMass over BOUND stars (seeded + unseeded). */
    MyFloat SCBoundClusterMassUnseeded; /*!< Sum of ClusterMass over BOUND UNSEEDED stars. */
    int     NStarBound;              /*!< Count of BOUND member stars (seeded + unseeded). */
    float   SCBoundRdm;              /*!< Radius of the DM sphere actually used [comoving]: Rmax
                                      *   (BHseedSecFOFbound=1) or min(2*R50, Rmax) (=2). */
    float   SCBoundMdm;              /*!< DM mass inside SCBoundRdm [code units]. */

    /* SeedSecFOFcomSample (combined per-secFOF sampling). Accumulated over
     * UNSEEDED stars (STARP.Seeded==0) only. */
    MyFloat SCcomMcut;    /*!< Sum of m_star over unseeded stars = mass-function cutoff M_cut.
                           *   OVERWRITTEN by fof_secfof_bound_restrict with the BOUND unseeded
                           *   stellar mass, so it is not the unrestricted total once
                           *   BHseedSecFOFbound is on -- use StellarMassUnseeded for that. */
    /* Unrestricted Sum(m_star) over UNSEEDED stars.  Same accumulation as SCcomMcut but
     * never overwritten by the bound restriction, so the StarClusterDetails records can
     * report the whole unseeded stellar mass alongside the bound subset. */
    MyFloat StellarMassUnseeded;
    MyFloat BHSeedMsc;    /*!< Combined-sampled seed cluster mass > 1e4 Msun (bhseed_msc); drives seed mass. Set in fof_seed */
    MyFloat BHSampledMscTotal; /*!< Combined-sampled cluster mass, full draw (no 1e4 cut); recorded as BH init_Msc_sample. Set in fof_seed */
    MyIDType SeedStarID;  /*!< ID of the max-ClusterMass unseeded star (RNG seed for the combined draw) */
};

/* Structure to hold all allocated FOF groups*/
typedef struct FOFGroups
{
    struct Group * Group;
    int64_t Ngroups;
    int64_t TotNgroups;
} FOFGroups;

/* Computes the Group structure, saved as a global array below.
 * If StoreGrNr is true, this writes to GrNr in partmanager.h.
 * Note this over-writes PeanoKey and means the tree cannot be rebuilt.*/
FOFGroups fof_fof(DomainDecomp * ddecomp, const int StoreGrNr, MPI_Comm Comm);

/*Frees the Group structure*/
void fof_finish(FOFGroups * fof);

/*Uses the Group structure to seed blackholes.
 * The active particle struct is used only because we may need to reallocate it. Random number seeds the BH mass.
 * If seeded_grnr_out != NULL, the GrNr of each locally-seeded group is written there
 * and *n_seeded_out is set to the count. The caller must myfree the returned array.
 * If seeded_totmsc_out / seeded_mcut_out != NULL, the per-group tot_msc_fof
 * (BHSeedMsc) and unseeded stellar mass (SCcomMcut) are returned in parallel
 * arrays (used by secondfof_seed for SeedSecFOFcomSampleParticle redistribution).
 * The caller must myfree any returned arrays in reverse allocation order
 * (mcut, then totmsc, then grnr). Pass NULL to skip collection (e.g. primary FOF).
 * If bound_mask_out != NULL and BHseedSecFOFbound is on, the transient per-particle
 * bound flag (see fof_secfof_bound_restrict) is returned there instead of being
 * released here, so the caller's own post-seeding passes can keep the restriction;
 * *bound_mask_out is NULL when the feature is off.  It is allocated BEFORE everything
 * else in fof_seed, so the caller frees it LAST -- after mcut/totmsc/grnr and before
 * anything that predates the fof_seed call (notably fof_finish, whose Group array is
 * an older mymalloc2 block). */
/* tree (the live gas/BH force tree from run.c, or NULL) is relocated off the
 * MAIN bottom stack around slots_reserve so SlotsBase can be grown (LIFO). */
void fof_seed(FOFGroups * fof, ActiveParticles * act, ForceTree * tree, double atime, const RandTable * const rnd,
              int64_t ** seeded_grnr_out, int * n_seeded_out,
              double ** seeded_totmsc_out, double ** seeded_mcut_out,
              char ** bound_mask_out, Cosmology * CP, MPI_Comm Comm);

/* BHseedSecFOFbound (see gadget/params.c and the implementation in fof.c).
 * Fills the SCBound* fields of every owned group with the gravitationally bound
 * subset of its member stars.  `mode` is 1 (DM inside Rmax) or 2 (DM inside
 * min(2*R50, Rmax)); any other value is a no-op.  With `apply` set it additionally
 * overwrites StarClusterMassUnseeded / SCcomMcut / the seed-star pointer so only the
 * bound stars drive BH seeding; with `apply` clear it only reports (used on the
 * catalogue path, where the seeding decision has already been taken).
 *
 * `bound_mask`, when not NULL, is a caller-owned char array of PartManager->NumPart
 * entries which the call fills with the per-LOCAL-PARTICLE bound flag (1 = a member
 * star of some secFOF group that passed the bound test, 0 = everything else).  It is
 * the transient per-star flag the group-level fields cannot carry, and it lets the
 * seeding paths that re-scan the unseeded stars individually keep the restriction.
 * Producing it makes every rank evaluate EVERY group rather than only its own (a
 * group's member stars are spread over all ranks), which costs a few percent of wall
 * time and no communication; pass NULL when the mask is not needed.  The mask indexes
 * P[] directly, so it is only valid while NumPart and the particle order are
 * unchanged -- seeding converts particles in place, so it survives fof_seed.
 *
 * Collective: must be called by every rank of Comm. */
void fof_secfof_bound_restrict(FOFGroups * fof, int mode, int apply,
                               double atime, Cosmology * CP, char * bound_mask,
                               MPI_Comm Comm);
/* Whether BHseedSecFOFbound is active, and in which mode (0 = off). */
int fof_get_secfof_bound_mode(void);

/* Saves the Group structure to disc.
 Returns 1 if a domain_exchange is needed afterwards.*/
int fof_save_groups(FOFGroups * fof, const char * OutputDir, const char * FOFFileBase, int num, Cosmology * CP, double atime, const double * MassTable, int MetalReturnOn, const int OutputDebugFields, MPI_Comm Comm);

/* Does the actual saving of the particles
 Returns 1 if a domain_exchange is needed afterwards.*/
int fof_save_particles(FOFGroups * fof, char * fname, int SaveParticles, Cosmology * CP, double atime, const double * MassTable, int MetalReturnOn, const int OutputDebugFields, MPI_Comm Comm);

/* Save particle catalog (type subdirectories 0/, 1/, ..., 5/) into an already-open BigFile.
 * Particles with GrNr >= 0 are selected and sorted by (Type, GrNr).
 * If swap_group_ids is set, GrNr and SecGrNr are swapped on the distributed
 * particles before writing IO blocks, so that GroupID gets the original primary
 * FOF value and SecGroupID gets the secondary FOF value.
 * Returns 1 if a domain_maintain is needed afterwards (when PartManager was reused). */
int fof_save_particles_to_bigfile(BigFile * bf, int MetalReturnOn, int OutputDebugFields, Cosmology * CP, double atime, int swap_group_ids, MPI_Comm Comm);

/* Selection function: returns true for particles that belong to a FOF group. */
int fof_select_func(int i, const struct particle_data * Parts);

#endif
