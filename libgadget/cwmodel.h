#ifndef CWMODEL_H
#define CWMODEL_H

/* Williams et al. 2026 (arXiv:2603.26872) VMS-in-a-dense-star-cluster model
 * (no black hole track): given a star cluster, compute the final very-massive-
 * star mass M_VMS from the collision-product inflow / Vink-2018-wind
 * equilibrium.  Used as the BH seed mass when MbhMscRelationCWmodel=1
 * (SeedInSecFOFRandomStarParticle seeding path).
 *
 * C port of code_v2/WilliamModel/scmodel.py final_vms_mass() with all Params
 * defaults except inflow_rmin_factor=5 (the kappa=5 inflow normalization that
 * reproduces the published Fig. 4; see the WilliamModel NOTES.md diagnosis).
 *
 * Inputs:
 *   M_msun         cluster mass [Msun]
 *   r_max_pc       cluster (virial) radius [pc]; = 1.4 * effective radius
 *   Z_massfrac     cluster metal mass fraction (absolute Z); converted to
 *                  Z/Zsun with Zsun=0.0134 and floored at Z/Zsun=1e-4
 *   t_universe_sec age of the universe at the seeding redshift [s]
 *                  (simulation cosmology; enters t_d = min(t_ms,t_uni,t_cc))
 *   alpha          density power-law index rho ~ r^-alpha (CWmodelAlpha input,
 *                  default 1.2, 0<alpha<3); the Rose et al. 2020 eccentricity
 *                  functions are recomputed for this alpha
 * Returns M_VMS in Msun; 0 when there is no net inflow (no VMS forms).
 * Clusters at or above the mean-density cap rho_mean = M/(4/3 pi r_max^3)
 * >= 6e7 Msun/pc^3 (scmodel.py rho_mean_cap system exclusion) skip the
 * collision model and return 0.01 * M_msun directly. */
double cw_final_vms_mass_msun(double M_msun, double r_max_pc, double Z_massfrac,
                              double t_universe_sec, double alpha);

#endif
