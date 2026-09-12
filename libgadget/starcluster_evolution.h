#ifndef STARCLUSTER_EVOLUTION_H
#define STARCLUSTER_EVOLUTION_H

#include "forcetree.h"
#include "timestep.h"
#include "cosmology.h"

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
