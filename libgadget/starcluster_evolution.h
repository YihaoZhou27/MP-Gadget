#ifndef STARCLUSTER_EVOLUTION_H
#define STARCLUSTER_EVOLUTION_H

#include "forcetree.h"
#include "timestep.h"
#include "cosmology.h"

/* Perform stellar evolution (mass and metal return) for star clusters
 * attached to black hole particles. Uses the same yield tables and IMF
 * as the regular metal_return module. */
void starcluster_metal_return(const ActiveParticles * act, ForceTree * gasTree, Cosmology * CP, const double atime, const double AvgGasMass);

#endif
