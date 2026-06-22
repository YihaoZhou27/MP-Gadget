#ifndef __BLACKHOLE_H
#define __BLACKHOLE_H
#include "utils/paramset.h"
#include "forcetree.h"
#include "density.h"
#include "utils/system.h"
#include "slotsmanager.h"

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
    MyFloat * BH_accreted_StarClusterMass;
    MyFloat * BH_accreted_SCMetallicityWeighted;
    MyFloat (* BH_accreted_SCMetalsWeighted)[NMETALS];
    MyFloat * BH_accreted_SCTotalMassReturned;
    MyFloat (*BH_GWRecoilKick)[3]; /* Accumulated GW recoil kick velocity from BH mergers */
    MyFloat * BH_accreted_SCFormTimeMin; /* min StarClusterFormationTime across swallowed BHs */
    float * BH_accreted_SCLastEnrichMax; /* max StarClusterLastEnrichmentMyr across swallowed BHs */

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

/* Returns 1 if StarClusterBHDyn is enabled (SC mass included in P.Mass) */
int get_starcluster_bhdyn_on(void);

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
 * mass is carried over as the BH's initial Mtrack. */
void blackhole_make_one(int index, const double atime, const RandTable * const rnd, int seeded_by_starcluster, MyFloat StarClusterMass, MyFloat ScalingMass, MyFloat init_Msc, MyFloat init_Msc_sample, MyFloat CappedStarMass, int BHNgbAtSeeding, MyFloat StarClusterMetallicity, const float * StarClusterMetals);

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
