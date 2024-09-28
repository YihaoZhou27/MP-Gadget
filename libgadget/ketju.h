// Ketju module for GADGET-4.
// Copyright (C) 2022 Matias Mannerkoski and contributors.
// Licensed under the GPLv3 license. See LICENSE for details.


//#include "gadgetconfig.h"


#ifndef KETJU_INTERFACE_H
#define KETJU_INTERFACE_H

#include <stdbool.h>
#include "timestep.h"
#include "utils/paramset.h"
#include "utils/unitsystem.h"

// Default options
#ifndef KETJU_STAR_PARTICLE_TYPE_FLAGS
#define KETJU_STAR_PARTICLE_TYPE_FLAGS (1 << 4)
#endif

#ifndef KETJU_DM_PARTICLE_TYPE_FLAGS
#define KETJU_DM_PARTICLE_TYPE_FLAGS (1 << 1)
#endif

#ifndef KETJU_BH_PARTICLE_TYPE_FLAGS
#define KETJU_BH_PARTICLE_TYPE_FLAGS (1 << 5)
#endif

#ifndef KETJU_INTEGRATOR_MAX_STEP_COUNT
#define KETJU_INTEGRATOR_MAX_STEP_COUNT 5000000
#endif

#ifndef KETJU_TIMESTEP_LIMITING_RADIUS_FACTOR
#define KETJU_TIMESTEP_LIMITING_RADIUS_FACTOR 100.
#endif

#define SOFTFAC1 (32.0 / 3) /**< Coefficients for gravitational softening */
#define SOFTFAC2 32.0
#define SOFTFAC3 (-38.4)
#define SOFTFAC4 (-2.8)
#define SOFTFAC5 (16.0 / 3)
#define SOFTFAC6 6.4
#define SOFTFAC7 (-9.6)
#define SOFTFAC8 (64.0 / 3)
#define SOFTFAC9 (-48.0)
#define SOFTFAC10 38.4
#define SOFTFAC11 (-32.0 / 3)
#define SOFTFAC12 (-1.0 / 15)
#define SOFTFAC13 (-3.2)
#define SOFTFAC14 (1.0 / 15)
#define SOFTFAC15 (-16.0)
#define SOFTFAC16 9.6
#define SOFTFAC17 (-64.0 / 30)
#define SOFTFAC18 128.0
#define SOFTFAC19 (-115.2)
#define SOFTFAC20 (64.0 / 3)
#define SOFTFAC21 (-96.0)
#define SOFTFAC22 115.2
#define SOFTFAC23 (-128.0 / 3)
#define SOFTFAC24 (4.0 / 30)

#ifdef __cplusplus
extern "C" {
#endif

//Forward declare some types needed for arguments
// class simparticles;
// class IO_Def;
// class restart;
 
// template <typename T>
// class domain;


// configurable options, set from the config file
struct Ketju_Options
{
  // Main behavior options

  double minimum_bh_mass;
  // Minimum mass of a BH particle that gets a ketju region, smaller ones
  // are treated like stellar particles by the subsystem integrator.

  double region_physical_radius;
  double output_time_interval;

  // changed: added para
  double star_smoothing;
  double MaxSizeTimestep;
  double ErrTolIntAccuracy;
  int ComovingIntegrationOn;
  int IntegrateBinaryRegion;
  unsigned int region_member_flags; 


  char PN_terms[20];
  int enable_bh_merger_kicks;
  int use_star_star_softening;
  double expand_tight_binaries_period_factor;

  // accuracy parameters

  double integration_relative_tolerance;
  double output_time_relative_tolerance;

  // performance tuning

  double minimum_particles_per_task;
  // A tuning parameter for parallelization of the integrator to avoid using
  // too many tasks.

  int use_divide_and_conquer_mst;
  int max_tree_distance;
  int steps_between_mst_reconstruction;
};



// Interface to the other code through free functions,
// so any data storage classes can be hidden in the implementation file.
// These modify the particle data through the passes pointer.
void Ketju_init_ketjuRM(Cosmology * CP, const double TimeBegin, const struct UnitSystem units);
void Ketju_find_regions(const ActiveParticles * act, DriftKickTimes * times, MPI_Comm Comm, int FastParticleType, double asmth, const struct UnitSystem units);
//void Ketju_run_integration(DriftKickTimes * times, const struct UnitSystem units, MPI_Comm Comm);
void Ketju_run_integration(DriftKickTimes * times, const struct UnitSystem units, MPI_Comm Comm, FILE * FdSingleBH, FILE * FdKetjuRegion, int trace_bhid, Cosmology * CP, int RestartSnapNum, const char * OutputDir);
//void Ketju_set_final_velocities();
void Ketju_set_final_velocities(const ActiveParticles * act);
void Ketju_finish_step(const ActiveParticles * act, DriftKickTimes * times, const struct UnitSystem units, int NumCurrentTiStep, MPI_Comm Comm);
int Ketju_is_particle_timestep_limited(int index);
int Ketju_set_limited_timebins(int min_timebin, int push_down_flag);
void Ketju_set_ketju_params(ParameterSet * ps);


#ifdef __cplusplus
}
#endif

#endif  // ifndef KETJU_INTERFACE_H
