#ifndef __BLACKHOLE_H
#define __BLACKHOLE_H
#include "utils/paramset.h"
#include "forcetree.h"
#include "density.h"
#include "utils/system.h"
#include "slotsmanager.h"

/* Full state of the star cluster carried by one BH, passed through the BH merger treewalk:
 * at a merger the remnant keeps the heaviest of the merging clusters (compared by cluster
 * mass, ties broken by the larger BH ID) and the lighter ones are removed. */
struct bh_sc_state {
    MyIDType ID;                  /* ID of the BH carrying the cluster (tie-break only) */
    MyFloat Mass;                 /* StarClusterMass */
    MyFloat FormationTime;        /* StarClusterFormationTime */
    MyFloat Metallicity;          /* StarClusterMetallicity */
    MyFloat TotalMassReturned;    /* StarClusterTotalMassReturned */
    MyFloat InitReff;             /* SC_initReff */
    MyFloat Reff;                 /* SC_Reff */
    MyFloat RlxPendingMyr;        /* SC_RlxPendingMyr */
    float Metals[NMETALS];        /* StarClusterMetals */
    float LastEnrichmentMyr;      /* StarClusterLastEnrichmentMyr */
};

struct BHPriv {
    /* Temporary array to store the IDs of the swallowing black hole for gas.
     * We store ID + 1 so that SwallowID == 0 can correspond to the unswallowed case. */
    MyIDType * SPH_SwallowID;
    /* Similar for IDs of BH mergers*/
    MyIDType * BH_SwallowID;
    /* These are temporaries used in the accretion treewalk*/
    MyFloat * BH_Entropy;
    MyFloat (*BH_SurroundingGasVel)[3];

    /* These are temporaries used in the feedback treewalk.*/
    MyFloat * BH_accreted_Mass;
    MyFloat * BH_accreted_BHMass;
    MyFloat (*BH_accreted_momentum)[3];
    MyFloat * BH_accreted_StarClusterMass; /* summed cluster mass of the swallowed BHs (Mtrack bookkeeping) */
    struct bh_sc_state * BH_accreted_SCmax; /* heaviest cluster among the swallowed BHs */
    MyFloat (*BH_GWRecoilKick)[3]; /* Accumulated GW recoil kick velocity from BH mergers */

    /* This is a temporary computed in the accretion treewalk and used
     * in the feedback treewalk*/
    MyFloat * BH_FeedbackWeightSum;

    /* temporary computed for kinetic feedback energy threshold*/
    MyFloat * NumDM;
    MyFloat * MgasEnc;
    /* mark the state of AGN kinetic feedback, 1 accumulate, 2 release */
    int * KEflag;

    /* Accumulated SPH vorticity vector (3-component, before density normalization) */
    MyFloat (*BH_VorticityVec)[3];
    /* Dimensionless vorticity magnitude: omega_star = omega * G * M_BH / c_s^3 */
    MyFloat * BH_Vorticity;
    /* Sound speed of surrounding gas (used in dimensionless vorticity) */
    MyFloat * BH_SoundSpeed;

    /* Time factors*/
    double atime;
    double a3inv;
    double hubble;
    struct UnitSystem units;
    Cosmology * CP;
    /* Counters*/
    int64_t * N_sph_swallowed;
    int64_t * N_BH_swallowed;
    struct kick_factor_data * kf;
    RandTable * rnd;
    int is_PM; /* Whether this timestep is a PM step */
};
#define BH_GET_PRIV(tw) ((struct BHPriv *) (tw->priv))

enum BlackHoleFeedbackMethod {
     BH_FEEDBACK_TOPHAT   = 0x2,
     BH_FEEDBACK_SPLINE   = 0x4,
     BH_FEEDBACK_MASS     = 0x8,
     BH_FEEDBACK_VOLUME   = 0x10,
     BH_FEEDBACK_OPTTHIN  = 0x20,
};

/*Set the parameters of the star formation module*/
void set_blackhole_params(ParameterSet * ps);

/* Returns 1 if tidal field computation is enabled for BH particles */
int get_bh_tidalfield_on(void);

/* Returns the SeedBHDynMass parameter value */
double get_bh_seed_dyn_mass(void);

/* Returns the SeedBlackHoleMass parameter value (code mass units) */
double get_bh_seed_mass(void);

/* Returns 1 only for StarClusterBHDyn=1 (evolving SC mass attached to the BH and
 * included in P.Mass). Mode 2 returns 0: no SC payload is attached; the seed
 * cluster mass (init_Msc) only sets a frozen per-BH dynamical-mass floor. */
int get_starcluster_bhdyn_on(void);

/* Warn (not abort) when StarClusterBHDyn=2 and MinMscForBHseed is below the dark
 * matter particle mass (dm_particle_mass = header MassTable[1], code units). */
void blackhole_check_seed_dm_resolution(double dm_particle_mass);

/* Does the black hole feedback and accretion.
 * TimeNextSeedingCheck is the time of the BH next seeding check.
 * It will be compared to the current time and updated after seeding takes place.
 * tree is a valid ForceTree.
 */
void blackhole(const ActiveParticles * act, double atime, Cosmology * CP, ForceTree * tree, DomainDecomp * ddecomp, DriftKickTimes * times, RandTable * rnd, const struct UnitSystem units, FILE * FdBlackHoles, FILE * FdBlackholeDetails, size_t *bhdetailswritten, int is_PM);

/* Make a black hole from the particle at index. Random number generator used
 * for the initial mass drawn from a power law.
 * seeded_by_starcluster: 1 if seeded by star-cluster criteria, 0 otherwise.
 * The parent particle is converted IN PLACE into the black hole (keeping its ID
 * and full mass): a gas particle (type 0) for gas/halo-based seeding, or a star
 * particle (type 4) for star-cluster seeding. The parent is consumed, and its
 * mass is carried over as the BH's initial Mtrack.
 * SeedMassOverride > 0 sets BHP.Mass directly (MbhMscRelationCWmodel: the
 * Williams et al. 2026 VMS mass), bypassing the SeedBlackHoleMass /
 * BHseedMassScaleMsc prescription; pass 0 for the normal seed-mass logic.
 * SC_initReff: effective radius (physical pc) drawn for the seed cluster; pass 0 when the
 * seeding path draws no per-cluster radius (a BH carrying a cluster then gets the
 * StarClusterReffRelation median at its cluster mass). Also the starting SC_Reff. */
void blackhole_make_one(int index, const double atime, const RandTable * const rnd, int seeded_by_starcluster, MyFloat StarClusterMass, MyFloat ScalingMass, MyFloat init_Msc, MyFloat init_Msc_sample, MyFloat CappedStarMass, int BHNgbAtSeeding, MyFloat StarClusterMetallicity, const float * StarClusterMetals, MyFloat SeedMassOverride, MyFloat SC_initReff);

/* Seed black holes from individual star particles whose star cluster mass
 * exceeds MinMscForBHseed.  Called every PM step when BlackholeSeedSCparticle=1,
 * or every timestep (after star formation) when BHseedEveryTimestep=1.
 * If NewStars/NumNewStar are provided (non-NULL, > 0), only those particle
 * indices are checked (newly formed stars).  Otherwise falls back to a full
 * scan of all type-4 particles. */
void blackhole_seed_sc_particle(ActiveParticles * act, ForceTree * tree, double atime,
                                const RandTable * const rnd, MPI_Comm Comm,
                                int * NewStars, int64_t NumNewStar);

#endif
