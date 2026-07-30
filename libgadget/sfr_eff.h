#ifndef __SFR_H
#define __SFR_H

#include "forcetree.h"
#include "utils/paramset.h"
#include "utils/system.h"
#include "timestep.h"
#include "partmanager.h"
#include "slotsmanager.h"

#define  METAL_YIELD       0.02	/*!< effective metal yield for star formation */

/*
 * additional sfr criteria
 */
enum StarformationCriterion {
    SFR_CRITERION_DENSITY = 1,
    SFR_CRITERION_MOLECULAR_H2 = 3, /* 2 + 1 */
    SFR_CRITERION_SELFGRAVITY = 5,  /* 4 + 1 */
    /* below are additional flags in SELFGRAVITY */
    SFR_CRITERION_CONVERGENT_FLOW = 13, /* 8 + 4 + 1 */
    SFR_CRITERION_CONTINUOUS_CUTOFF= 21, /* 16 + 4 + 1 */
};

/*Set the parameters of the star formation module*/
void set_sfr_params(ParameterSet * ps);

void init_cooling_and_star_formation(int CoolingOn, int StarformationOn, Cosmology * CP, const double avg_baryon_mass, const double BoxSize, const struct UnitSystem units);
/*Do the cooling and the star formation. The tree is required for the winds only.
 * If NewStars_out and NumNewStar_out are non-NULL, the NewStars array is returned
 * to the caller (who must myfree it) instead of being freed internally.*/
void cooling_and_starformation(ActiveParticles * act, double Time, double dloga, ForceTree * tree, struct grav_accel_store GravAccel, DomainDecomp * ddecomp, Cosmology *CP, MyFloat * GradRho, RandTable * rnd, FILE * FdSfr, int **NewStars_out, int64_t *NumNewStar_out);

/*Get the neutral fraction of a particle correctly, even when on the star-forming equation of state.
 * This calls the cooling routines for the current internal energy when off the equation of state, but
 * when on the equation of state calls them separately for the cold and hot gas.*/
double get_neutral_fraction_sfreff(double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata);

/*Get the helium ionic fraction of a particle correctly, even when on the star-forming equation of state.
 * This calls the cooling routines for the current internal energy when off the equation of state, but
 * when on the equation of state calls them separately for the cold and hot gas.*/
double get_helium_neutral_fraction_sfreff(int ion, double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata);

/* Return whether we are using a star formation model that needs grad rho computed for the gas particles*/
int sfr_need_to_compute_sph_grad_rho(void);

/* Get the number of generations of stars that may form*/
int get_generations(void);

/* Returns 1 if particle is on effective EOS, 0 otherwise*/
int sfreff_on_eeqos(const struct sph_particle_data * sph, const double a3inv);

/* Get the Minimum temperature in internal energy*/
double get_MinEgySpec(void);

/* Returns the density threshold for star formation in comoving units*/
double sfr_density_threshold(const double atime);

/* Combined per-secFOF star-cluster sampling for BH seeding (SeedSecFOFcomSample).
 * Also used per-star-particle for SeedSecFOFcomSampleParticle.
 * Mcut       : mass-function cutoff (code units). Group total unseeded stellar
 *              mass in the combined mode; min(M_cstar, group stellar mass) per
 *              star in the per-particle mode.
 * sum_mGamma : sum of m_star*Gamma over the sampled stars (code units): the group
 *              total in combined mode, or a single star's m_star*Gamma per-particle.
 * rand_id    : RNG seed (reproducible)
 * allow_cap  : if 1 and SCmasscapSecFOFstarmass is set, cap the result at Mcut;
 *              pass 0 for the per-particle mode (the cap is applied on the group sum).
 * Returns bhseed_msc = sum of sampled cluster masses > 1e4 Msun (code units).
 * If total_sampled_out != NULL, also returns the summed mass of ALL sampled
 * clusters there (full draw, no threshold).
 * Serial only (uses the global GSL error handler). */
double starcluster_combined_bhseed_msc(double Mcut, double sum_mGamma,
                                       uint64_t rand_id, const RandTable * const rnd,
                                       double * total_sampled_out, int allow_cap);

/* Per-cluster secFOF seeding (SecFOFseedsumover=0): draw the combined per-secFOF cluster population
 * (same Poisson + inverse-CDF draw and RNG offsets as starcluster_combined_bhseed_msc)
 * and return the INDIVIDUAL cluster masses >= min_seed_mass. Fills out_masses with the
 * largest min(n_qualify, cap) such masses, sorted descending, and returns n_qualify =
 * the TOTAL number of clusters >= min_seed_mass (which may exceed cap, so the caller can
 * detect a shortage of unseeded stars). out_masses may be NULL when cap == 0. All masses
 * in code units. Serial only (uses the global GSL error handler). */
int starcluster_combined_seed_masslist(double Mcut, double sum_mGamma,
                                       uint64_t rand_id, const RandTable * const rnd,
                                       double min_seed_mass, double * out_masses, int cap);

/* Callback receiving one sampled star cluster: its mass (code units) and its index
 * s in the group's draw sequence (used to key per-cluster random numbers). */
typedef void (*sc_detail_cb)(double mass, int draw_index, void * data);

/* StarClusterDetails full-population pass (MinMscForSCdetail). Redraws the SAME
 * cluster population as starcluster_combined_seed_masslist for this group (identical
 * Poisson count, identical per-cluster inverse-CDF and RNG offsets from rand_id) and
 * hands every cluster with mass_lo <= m < mass_hi to cb, in draw order. Used to record
 * the clusters that are too light to ever seed a BH, which are never buffered or
 * communicated. No-op if cb is NULL or the mass window is empty.
 * Serial only (uses the global GSL error handler). */
void starcluster_seed_masslist_detail(double Mcut, double sum_mGamma,
                                      uint64_t rand_id, const RandTable * const rnd,
                                      double mass_lo, double mass_hi,
                                      sc_detail_cb cb, void * cbdata);

/* Metallicity-dependent BH-seeding factor f(Z) in [0,1], applied to the per-star
 * cluster mass (Gamma*m_star) used for star-cluster BH seeding. Z is the absolute
 * star metallicity (BirthMetallicity). Returns 1 when the feature is disabled
 * (StarClusterSeedMetallicityMax <= Min). See definition in sfr_eff.c. */
double get_seed_metallicity_factor(double Z);

/* Whether SCmasscapSecFOFstarmass is enabled (cap SC mass at the group's unseeded
 * stellar mass). Used by the per-particle seeder to cap the group-summed mass. */
int get_scmasscap_secfof_starmass(void);

/* Per-secFOF multi-seed mass threshold (1e8 Msun) in code mass units. */
double get_msc_multiseed_thresh_code(void);

/* Sample an effective radius (in pc) for a seeded star cluster of code-unit mass
 * mcl_code, from the size-mass relation R_eff = 1.4 pc * (M_cl/1e4 Msun)^0.25 with a
 * 0.5 dex lognormal scatter (reproducibly keyed on the host star ID rand_id), and
 * log10(R/pc) clipped to [-1, 2]. Used only by the StarClusterDetails record in the
 * per-cluster (SecFOFseedsumover=0) seeding path. Returns 0 for a non-positive mass. */
double starcluster_sample_reff_pc(double mcl_code, uint64_t rand_id, const RandTable * const rnd);

#endif
