#ifndef STARCLUSTER_H
#define STARCLUSTER_H

/* Given P/k_B in [K/cm^3], return the Cluster Formation Efficiency
 * via log-linear interpolation of the Kruijssen (2012) table. */
double get_cluster_formation_efficiency(double Pressure_over_kB);

/* Kruijssen (2012) local cluster formation efficiency, eq. 26 without the cruel-cradle effect, evaluated directly
 * for a local density rho, 1-D turbulent velocity dispersion sigma and cold-gas sound speed cs (the table above is
 * this model on EAGLE's equation of state with sigma = sqrt(P/rho), cs = 0.3 km/s). G, t_sn (supernova timescale),
 * phi_fb (feedback efficiency) and t_inc (incomplete-star-formation time) are passed in, so any consistent unit
 * system works (sfr_eff.c uses code units: G = GravInternal, t_sn_code, phi_fb_code, t_inc_code).
 * sigma = 0 is the delta-function limit of the lognormal density PDF (no density contrast):
 * Gamma = min(eps_core, sSFR_ff t_sn / t_ff, sSFR_ff t_inc / t_ff) / eps_core. Capped at 1. */
double get_cluster_formation_efficiency_k12(double rho, double sigma, double cs, double G, double t_sn, double phi_fb, double t_inc);

#endif
