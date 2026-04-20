#ifndef STARCLUSTER_H
#define STARCLUSTER_H

/* Given P/k_B in [K/cm^3], return the Cluster Formation Efficiency
 * via log-linear interpolation of the Kruijssen (2012) table. */
double get_cluster_formation_efficiency(double Pressure_over_kB);

#endif
