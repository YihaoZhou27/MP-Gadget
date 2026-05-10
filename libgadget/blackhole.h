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

    /* This is a temporary computed in the accretion treewalk and used
     * in the feedback treewalk*/
    MyFloat * BH_FeedbackWeightSum;

    /* temporary computed for kinetic feedback energy threshold*/
    MyFloat * NumDM;
    MyFloat * MgasEnc;
    /* mark the state of AGN kinetic feedback, 1 accumulate, 2 release */
    int * KEflag;

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

/* Does the black hole feedback and accretion.
 * TimeNextSeedingCheck is the time of the BH next seeding check.
 * It will be compared to the current time and updated after seeding takes place.
 * tree is a valid ForceTree.
 */
void blackhole(const ActiveParticles * act, double atime, Cosmology * CP, ForceTree * tree, DomainDecomp * ddecomp, DriftKickTimes * times, RandTable * rnd, const struct UnitSystem units, FILE * FdBlackHoles, FILE * FdBlackholeDetails, size_t *bhdetailswritten, int is_PM);

/* Make a black hole from the particle at index. Random number generator used
 * for the initial mass drawn from a power law.
 * seeded_by_starcluster: 1 if seeded by star-cluster criteria, 0 otherwise.
 * When seeded_by_starcluster && P[index].Type==4, a new particle is spawned
 * next to the star (secondary FOF path); otherwise the gas particle at
 * index is converted in-place (primary FOF / halo / gas-based path). */
void blackhole_make_one(int index, const double atime, const RandTable * const rnd, int seeded_by_starcluster, MyFloat StarClusterMass, MyFloat StarClusterMetallicity, const float * StarClusterMetals);

/* Seed black holes from individual star particles whose star cluster mass
 * exceeds MinMscForBHseed.  Called every PM step when BlackholeSeedSCparticle=1,
 * or every timestep (after star formation) when BHseedEveryTimestep=1. */
void blackhole_seed_sc_particle(ActiveParticles * act, double atime,
                                const RandTable * const rnd, MPI_Comm Comm);

#endif
