#ifndef STARCLUSTER_EVOLUTION_H
#define STARCLUSTER_EVOLUTION_H

#include "forcetree.h"
#include "timestep.h"
#include "cosmology.h"

/* Each of the three routines below records its own contribution to the cluster's evolution in
 * the BH fields SC_MlossStellar / SC_MlossRelax / SC_MlossTDE / SC_MlossTDEMerger (cumulative
 * mass lost, code units; the last two are the two channels of starcluster_tde_growth) and
 * SC_dlnReffStellar / SC_dlnReffRelax (cumulative ln R_eff change), see slotsmanager.h; a
 * dissolved cluster resets them with the rest of its state. */

/* Stellar-evolution mass loss (SCEvolutionStellar) of the star clusters attached to the
 * active BH particles: StarClusterMass becomes the cluster's birth mass times one minus the
 * cumulative SSP return fraction at its age (same yield tables and Chabrier IMF as
 * metal_return), and StarClusterTotalMassReturned the mass lost.  Nothing is deposited into
 * the gas: the cluster mass is part of the star particles' mass, whose return metal_return
 * already does.  Independent of MetalReturnOn.  size_evolution: SC_Reff expands by the inverse
 * of the mass ratio (Guerra et al. 2026 sect. 2.2.3). */
void starcluster_stellar_evolution(const ActiveParticles * act, Cosmology * CP, const double atime, const int size_evolution);

/* Two-body relaxation mass loss (SCEvolutionRelaxation) of the star clusters attached to
 * the active BH particles, over each BH's own step.  mode 1: E-MOSAICS law; mode 2: GB08
 * law of EMP-Pathfinder, with r_h = (4/3) SC_Reff.  Both are driven by the BH tidal field
 * (BlackholeTidalField=1).  Clusters below 100 Msun are dissolved; the stripped mass leaves
 * StarClusterMass (and P.Mass when StarClusterBHDyn=1) and StarClusterTotalMassReturned is
 * rescaled with it.  size_evolution (mode 2 only): SC_Reff follows Guerra et al. 2026 eq. 22.
 * A BH with no tidal field yet (all eigenvalues zero, e.g. a new seed before its first PM
 * step under hierarchical gravity) is not relaxed: its time accumulates in SC_RlxPendingMyr
 * and is relaxed with its first field. */
void starcluster_relaxation(const ActiveParticles * act, const Cosmology * CP, const double atime, const int mode, const int size_evolution, const struct UnitSystem units);

/* Start-up description of the active relaxation model and of the tidal-field cadence
 * (hierarchical = SplitGravityTimestepsOn). */
void starcluster_relaxation_message(const int mode, const int hierarchical);

/* Start-up description of the active size-evolution terms (StarClusterSizeEvolution). */
void starcluster_size_evolution_message(const int size_stellar, const int size_relaxation);

/* Growth of the active BH particles by tidal disruptions, both channels in one pass over the
 * active BHs (each switched by its flag):
 *  - single_on (StarClusterTDEtoBH): disruption of their star cluster's stars at the Rizzuto et al.
 *    (2023) eq. 9 rate on the Williams et al. (2026) cluster-with-BH structure, from the BH mass,
 *    StarClusterMass and SC_Reff.  Sets NdotTDE and MdotTDE, adds MdotTDE x dt to the BH mass and
 *    removes the disrupted stars from the cluster.  BHs without a cluster get 0.
 *  - merger_on (StarClusterEnhancedTDE4Merger): the burst reservoir MergerTDEMassLeft filled at the
 *    BH's mergers (blackhole_feedback_postprocess) is added at a constant rate over the window
 *    MergerTDETimeLeftMyr; sets MdotTDEMerger.  The stars of each step's share leave the cluster
 *    the remnant carries (1/f_acc x the BH gain, recorded in SC_MlossTDEMerger); the BH gain
 *    itself does not depend on a cluster still being attached.
 * Both additions are on top of, and not limited by, the Eddington-capped gas accretion of
 * blackhole(), and are credited to Mtrack as well (stellar debris, not gas). */
void starcluster_tde_growth(const ActiveParticles * act, const Cosmology * CP, const double atime, const struct UnitSystem units, const int single_on, const int merger_on);

/* Start-up description of the TDE growth model. */
void starcluster_tde_message(void);

/* StarClusterEnhancedTDE4Merger: number of stars disrupted in the eccentric Kozai-Lidov burst of
 * Mockler et al. (2023) around the LIGHTER BH of a merging pair (the secondary, mass m1_msun, in its
 * own cluster of stellar mass msc_msun and effective radius reff_pc): N_TDE = eps_dis x the stars
 * inside the hierarchical radius a_hier = 0.1 a_bin (1 - e^2)/e of a binary at a_bin = 0.5 r_h,1
 * with e = 0.5 (their eq. 2 and fiducial setup), r_h,1 the sphere holding 2 m1 of cluster stars
 * (at most the cluster's edge r_max = 1.4 R_eff), on the same rho ~ r^-7/4 cusp as the single-BH
 * rate; eps_dis = 0.25.  N_TDE = 2 (0.075)^(5/4) eps_dis m1/m_* = 0.0196 m1/m_* whenever
 * 2 m1 <= M_SC, and 0.075^(5/4) eps_dis M_SC/m_* = 0.0098 M_SC/m_* otherwise.  0 without a cluster.
 * sc_merger_tde_mass_msun is the mass the remnant gains from it, f_acc m_* N_TDE.  Exposed for
 * testing. */
double sc_merger_tde_ntde(double m1_msun, double msc_msun, double reff_pc);
double sc_merger_tde_mass_msun(double m1_msun, double msc_msun, double reff_pc);

/* Duration (Myr) over which a merger burst is added to the remnant (every merger restarts it) */
#define SC_MTDE_TBIN_MYR 10.0

/* Start-up description of the merger-burst model. */
void starcluster_merger_tde_message(void);

/* The TDE rate [Myr^-1] of a BH of mass mbh_msun inside a cluster of stellar mass msc_msun and
 * effective radius reff_pc; exposed for testing.  0 without a BH or a cluster. */
double sc_tde_rate_per_myr(double mbh_msun, double msc_msun, double reff_pc);

/* Relaxation-driven size change over a step, r_new/r_old = (m_new/m_old)^(2 - zeta/xi)
 * (Guerra et al. 2026 eq. 22 without tidal shocks); exposed for testing. */
double sc_size_factor_rlx(double m_old, double m_new, double xi);

/* GB08 mass (returned) and, with size_evolution, R_eff (*reff_out) after dt_myr, sub-cycled so
 * no sub-step removes more than 0.2% of the mass; what starcluster_relaxation applies per step. */
double sc_relax_gb08_step(double m_msun, double reff_pc, double T_myr2, double dt_myr, int size_evolution, double * reff_out);

/* The single-cluster relaxation laws, exposed for testing.  Masses in Msun, radii in pc,
 * times in Myr, T the E-MOSAICS tidal strength max(lambda)+Omega^2 (physical).
 * Each returns the cluster mass after dt_myr (0 once fully disrupted). */
double sc_relax_mass_emosaics(double m_msun, double T_gyr2, double dt_myr);
double sc_relax_mass_gb08(double m_msun, double rh_pc, double T_myr2, double dt_myr);
/* GB08 ingredients: half-mass relaxation time (Myr) and escape fraction per t_rh. */
double sc_relax_trh_myr(double m_msun, double rh_pc);
double sc_relax_xi(double m_msun, double rh_pc, double T_myr2);

#endif
