#ifndef CWMODEL_H
#define CWMODEL_H

/* Williams et al. 2026 (arXiv:2603.26872) VMS-in-a-dense-star-cluster model
 * (no black hole track): given a star cluster, compute the final very-massive-
 * star mass M_VMS from the collision-product inflow / Vink-2018-wind
 * equilibrium.  Used as the BH seed mass when MbhMscRelationCWmodel=1
 * (SeedInSecFOFRandomStarParticle seeding path).
 *
 * C port of the AUTHOR'S ORIGINAL code (code/CWmodel_oricode,
 * src/timescales/analysis/modelv2.py STAR-ONLY branch) at upstream commit
 * 0f2d04a (2026-08-06), which POSTDATES the published Fig. 4 and yields ~+0.31
 * dex higher M_VMS than that figure.  Closed-form throughout: r_min = relaxation
 * radius (t_relax=P_orb), r_df from stellar_df_radius (q=Mc/Mstar=1), Mdot_in =
 * Mdot_df - Mdot_dep - Mdot_bin all *(1-f_vms), binary
 * heating ON (fixed sigma=20 km/s), and M_VMS from the direct wind equilibrium
 * (no fml_vms iteration).  Simulation adaptations: t_merger omitted, the
 * mean-density cap returns 0.01*M_cl, Z is the per-cluster simulation value with
 * no lower clamp, and f_IMF is MP-Gadget's Chabrier 0.0969 (not Salpeter 0.0649).
 *
 * Inputs:
 *   M_msun         cluster mass [Msun]
 *   r_max_pc       cluster (virial) radius [pc]; = 1.4 * effective radius
 *   Z_massfrac     cluster metal mass fraction (absolute Z); converted to
 *                  Z/Zsun with Zsun=0.0134 and used as given (no lower clamp).
 *                  Z <= 0 means no wind, so the equilibrium VMS mass is
 *                  unbounded and M_msun (the whole cluster) is returned
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
