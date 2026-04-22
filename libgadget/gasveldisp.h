#ifndef GASVELDISP_H
#define GASVELDISP_H

#include "forcetree.h"
#include "timestep.h"
#include "cosmology.h"
#include "domain.h"

/* Compute the gas and stellar velocity dispersion, total mass, and neighbor count
 * within Hsml for each gas particle. Should be called at every PM step.
 * Reuses the existing gasTree for gas neighbors; builds a small star-only tree internally. */
void gas_star_veldisp(const ActiveParticles * act, Cosmology * CP,
                      const DriftKickTimes * times,
                      const ForceTree * gasTree,
                      DomainDecomp * ddecomp, const char * OutputDir);

#endif
