// Ketju module for GADGET-4.
// Copyright (C) 2022 Matias Mannerkoski and contributors.
// Licensed under the GPLv3 license. See LICENSE for details.

// #include "gadgetconfig.h"

#include <hdf5.h>


// Unlike the rest of the code, we're using STL containers here.
// This bypasses the global memory allocation pool, but the only real problem
// that could cause is crashing with a not-so-nice error message when the memory
// is full, and some undercounting of total memory usage.
// It might be possible to implement a custom allocator that uses the global
// allocation pool if this choice causes issues, but so far such issues have not appeared.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cfloat>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <math.h>
#include <mpi.h>
// #include "../data/allvars.h"
// #include "../data/simparticles.h"
// #include "../domain/domain.h"
// #include "../io/restart.h"
// #include "../logs/logs.h"
// #include "../mpi_utils/mpi_utils.h"
// #include "../mpi_utils/shared_mem_handler.h"
// #include "../time_integration/driftfac.h"
#include "ketju_integrator/ketju_integrator.h"

extern "C"
{
#include "slotsmanager.h"
#include "gravity.h"
#include "physconst.h"
#include "cosmology.h"
#include "timefac.h"
#include "timebinmgr.h"
#include "physconst.h"
#include "walltime.h"
#include "ketju.h"
#include "partmanager.h"
#include "stats.h"
}
// MP-Gadget File

// This file consists mostly of class definitions and their methods that are only
// called through the few interface functions declared in the ketju.h header.
// There's quite a bit of code, and consequently this file is somewhat long,
// but breaking it into separate files is somewhat inconvenient due to the way
// the build system is implemented.
// The classes and their methods are organized into clear blocks,
// containing also any helper functions needed by the methods,
// with the interface functions at the end,
// so this file should be clear enough as is.

// Anonymous namespace for implementation details (internal linkage)

// Now only availble for non-comoving simulation


// Data that persists after restarts in All and is the same for all tasks
struct Ketju_Data
{
  struct Ketju_Options options;
  long long final_written_Ti;
  int output_timebin;
  int next_tstep_index;
} KetjuData;


void Ketju_set_ketju_params(ParameterSet * ps)
{
  int ThisTask;
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  if(ThisTask==0){

    //auto &ketju_opt = KetjuData.options;

    int KetjuOn = param_get_int(ps, "KetjuOn");
    KetjuData.options.enable_bh_merger_kicks = param_get_int(ps, "KetjuBHMergerKicks");
    KetjuData.options.minimum_bh_mass = param_get_double(ps, "KetjuMinimumBHMass");
    KetjuData.options.region_physical_radius = param_get_double(ps, "KetjuRegionPhysicalRadius");
    KetjuData.options.output_time_interval = param_get_double(ps, "KetjuOutputTimeInterval");
    //KetjuData.options.PN_terms = param_get_string(ps, "KetjuPNTerms");
    param_get_string2(ps, "KetjuPNTerms", KetjuData.options.PN_terms, sizeof(KetjuData.options.PN_terms));
    KetjuData.options.use_star_star_softening = param_get_int(ps, "KetjuUseStarStarSoftening");
    KetjuData.options.expand_tight_binaries_period_factor = param_get_double(ps, "KetjuExpandTightBinaries");
    KetjuData.options.integration_relative_tolerance = param_get_double(ps, "KetjuIntegrationRelativeTolerance");
    KetjuData.options.output_time_relative_tolerance = param_get_double(ps, "KetjuOutputTimeRelativeTolerance");
    KetjuData.options.minimum_particles_per_task = param_get_double(ps, "KetjuMinimumParticlesPerTask");
    KetjuData.options.use_divide_and_conquer_mst = param_get_int(ps, "KetjuUseDnC");
    KetjuData.options.max_tree_distance = param_get_int(ps, "KetjuMaxTreeDistance");
    KetjuData.options.steps_between_mst_reconstruction = param_get_int(ps, "KetjuStepsBetweenMSTReconstruction");

    // added param
    KetjuData.options.star_smoothing = 2.8 * param_get_double(ps, "SofteningType4");
    KetjuData.options.MaxSizeTimestep = param_get_double(ps, "MaxSizeTimestep");
    KetjuData.options.ErrTolIntAccuracy = param_get_double(ps, "ErrTolIntAccuracy");
    KetjuData.options.ComovingIntegrationOn = param_get_int(ps, "ComovingIntegrationOn");
    KetjuData.options.IntegrateBinaryRegion = param_get_int(ps, "KetjuIntegrateBinaryRegion");

    KetjuData.options.region_member_flags = (KETJU_STAR_PARTICLE_TYPE_FLAGS) | (KETJU_BH_PARTICLE_TYPE_FLAGS);
    if(param_get_int(ps, "KetjuUseDMNonsoftening")){
      KetjuData.options.region_member_flags |= KETJU_DM_PARTICLE_TYPE_FLAGS;


    message(0, "KETJU region member: %d \n", KetjuData.options.region_member_flags);


    }

    // parameter_sanity_check

    
    if(KetjuOn == 1){
      if(KetjuData.options.ComovingIntegrationOn == 1)
        endrun(1, "Now the Ketju is only availble for ComovingIntegrationOn=1!");

      if(KetjuData.options.region_physical_radius <= 0)
        endrun(1, "Value of KetjuRegionPhysicalRadius must be positive");


      if(KetjuData.options.expand_tight_binaries_period_factor < 0)
        endrun(1, "Value of KetjuExpandTightBinaries must be non-negative");

      if(KetjuData.options.integration_relative_tolerance <= 0)
        endrun(1, "Value of KetjuIntegrationRelativeTolerance must be positive");

      if(param_get_int(ps, "ForceEqualTimesteps") == 1)
        endrun(1, "KETJU is not compatible with OUTPUT_NON_SYNCHRONIZED_ALLOWED");

      if(param_get_int(ps, "LightconeOn") == 1)
        endrun(1, "KETJU is not compatible with LIGHTCONE");

      if(param_get_int(ps, "BlackHoleRepositionEnabled") == 1)
        endrun(1, "KETJU is not compatible with BlackHoleRepositionEnabled");

      

      double star_sml = KetjuData.options.star_smoothing;
      double bh_sml = 2.8 * param_get_double(ps, "SofteningType5");
      if(star_sml != bh_sml)
      {
        endrun(1, 
        "Ketju: Star particles and blackhole must use the same Softening Length.");
      }

      if(param_get_int(ps, "KetjuUseDMNonsoftening")){
        double dm_sml = 2.8 * param_get_double(ps, "SofteningType1");

        if(dm_sml != star_sml)
            endrun(1, "Ketju: Dark matter must use the same softening length with the black holes if DM_nonsofteing is implemented.");

      }


      double max_soft = param_get_double(ps, "SofteningType4");

      const double min_radius = 2.8 * max_soft;

      const double region_radius = KetjuData.options.region_physical_radius;
      if(region_radius < min_radius)
        {
          char buf[300];
          std::snprintf(buf, sizeof buf,
                        "Ketju: KetjuRegionPhysicalRadius = %3.3g is too "
                        "small! "
                        "The region radius needs to be at least 2.8 times larger than "
                        "the star and BH (Bndry) physical softening lengths. Current "
                        "softening lengths require KetjuRegionPhysicalRadius > "
                        "%3.3g.",
                        region_radius, min_radius);
          endrun(1, buf);
        }

    }
    //MPI_Bcast(&blackhole_params, sizeof(struct BlackholeParams), MPI_BYTE, 0, MPI_COMM_WORLD);

  }
    MPI_Bcast(&KetjuData, sizeof(struct Ketju_Data), MPI_BYTE, 0, MPI_COMM_WORLD);
}


// namespace
// {
// using namespace ketju;

// unsigned int region_member_flags = (KETJU_STAR_PARTICLE_TYPE_FLAGS) | (KETJU_BH_PARTICLE_TYPE_FLAGS);
constexpr unsigned int star_particle_flags = (KETJU_STAR_PARTICLE_TYPE_FLAGS);
constexpr unsigned int bh_particle_flags   = (KETJU_BH_PARTICLE_TYPE_FLAGS);
constexpr unsigned int all_particles_flags = ~0;

static_assert((star_particle_flags & bh_particle_flags) == 0, "Cannot have overlapping star and BH particle types");
// static_assert((region_member_flags & 1) == 0, "Cannot have gas as a KETJU region member type (star or BH)");
static_assert(star_particle_flags, "Cannot have empty KETJU_STAR_PARTICLE_TYPE_FLAGS");
static_assert(bh_particle_flags, "Cannot have empty KETJU_BH_PARTICLE_TYPE_FLAGS");

bool check_type_flag(int type, unsigned int flags) { return (1 << type) & flags; }
bool is_bh_type(int type) { return check_type_flag(type, bh_particle_flags); }



//changed: add a new function to set parameteres related to ketju
//using this to replace all the "All.KetjuData." 

//TODO: these functions are required for the comoving case (referring to time_integration/driftfac.cc of Gadget-4)
// double get_cosmic_time(inttime_t t0);
// double get_cosmic_timestep(inttime_t t0, inttime_t t1);
// double get_gravkick_factor(inttime_t t0, inttime_t t1);

///// rewrite some functions from Gadget-4
void timebins_get_bin_and_do_validity_checks(inttime_t ti_step, inttime_t Ti_Current, int *bin_new, int bin_old)
{
  /* make it a power 2 subdivision */
  
  inttime_t ti_min = TIMEBASE;
  while(ti_min > ti_step)
    ti_min >>= 1;
  ti_step = ti_min;
  
  /* get timestep bin */
  int bin = -1;

  if(ti_step == 0)
    bin = 0;

  if(ti_step == 1)
    endrun(1, "time-step of integer size 1 not allowed\n");

  while(ti_step)
    {
      bin++;
      ti_step >>= 1;
    }
  
  if(bin > bin_old) /* timestep wants to increase */
    {

      while(is_timebin_active(bin, Ti_Current) == 0 && bin > bin_old) /* make sure the new step is synchronized */
        bin--;

      ti_step = bin ? (((inttime_t)1) << bin) : 0;
    }
  
  // if(Ti_Current >= TIMEBASE) /* we here finish the last timestep. */
  //   {
  //     ti_step = 0;
  //     bin     = 0;
  //   }

  // if((TIMEBASE - Ti_Current) < ti_step) /* check that we don't run beyond the end */
  //   {
  //     endrun(1, "we are beyond the end of the timeline"); /* should not happen */
  //   }
  

  *bin_new = bin;
}

// changed: add the numcurrenttistep
template <typename... Ts>
void ketju_printf(const char *msg_template, const Ts &...args)
{
  // We can't always print from task 0, so the messages might get out of order.
  // Print the sync-point to make tracking this easier.
  //std::printf("Ketju (Sync-Point %d): ", NumCurrentTiStep);
  std::printf("Ketju: ");
  std::printf(msg_template, args...);
  std::fflush(stdout);
}

// changed: add the numcurrenttistep
template <typename... Ts>
void ketju_debug_printf(const char *msg_template, const int NumCurrentTiStep, const Ts &...args)
{
#ifdef KETJU_DEBUG
  std::printf("Ketju DEBUG (Sync-Point %d): ", const int NumCurrentTiStep,);
  std::printf(msg_template, args...);
  std::fflush(stdout);
#endif
}


double vector_norm(const double vec[3]) { return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]); }

/////////////////

// Particle data communicated between tasks, values are in the global gadget frame.
// changed: MyIntPosType IntPos[3] --> double Pos[3]
struct mpi_particle
{
  MyIDType ID;
  int Type;
  // where in the particle data array (simparticles::P) and on which task the particle is located
  int Task;
  int Index;

  double Mass;
  double Pos[3];
  double Vel[3];
  double Spin[3];
  double potential_energy_correction;  // only needed with EVALPOTENTIAL, but simplest to include always

  static MPI_Datatype get_mpi_datatype();
  static mpi_particle from_gadget_particle_index(int task, int index); // changed: delete the Sp para for from_gadget_partcile_index
};


//changed: since we change the mpi struct mpi_particle.
MPI_Datatype mpi_particle::get_mpi_datatype()
{
  // construct the datatype on the first call
  static MPI_Datatype dtype = []() {
    MPI_Datatype dtype;
    const int n_items         = 9;
    const int block_lengths[] = {1, 1, 1, 1, 1, 3, 3, 3, 1};

    static_assert(std::is_same<MyIDType, unsigned long int>::value || std::is_same<MyIDType, unsigned long long>::value);
    MPI_Datatype ID_type = std::is_same<MyIDType, unsigned long int>::value ? MPI_UNSIGNED_LONG : MPI_UNSIGNED_LONG_LONG;

    MPI_Datatype types[] = {ID_type, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[]   = {
        offsetof(mpi_particle, ID),    offsetof(mpi_particle, Type), offsetof(mpi_particle, Task),
        offsetof(mpi_particle, Index), offsetof(mpi_particle, Mass), offsetof(mpi_particle, Pos),
        offsetof(mpi_particle, Vel),   offsetof(mpi_particle, Spin), offsetof(mpi_particle, potential_energy_correction)};

    MPI_Type_create_struct(n_items, block_lengths, offsets, types, &dtype);
    MPI_Type_commit(&dtype);

    return dtype;
  }();

  return dtype;
}

// changed //remove spin property since BH particle doesn't have spin in MP-Gadget
mpi_particle mpi_particle::from_gadget_particle_index(int task, int index)
{
  //const auto &p     = P[index];
  const auto type   = P[index].Type;
  double zero_spin[3]     = {0, 0, 0};
  const double *spin_data = zero_spin;
  float mass; 
  if(type==5){
    mass = BHP(index).Mass;
    spin_data = P[index].Spin;
  }
  else{
    mass = P[index].Mass;
  }
  


  return {P[index].ID,
        type,
        task,
        index,
        mass,
        {P[index].Pos[0], P[index].Pos[1], P[index].Pos[2]},
        {P[index].Vel[0], P[index].Vel[1], P[index].Vel[2]},
        {spin_data[0], spin_data[1], spin_data[2]}};
}

/////////////////
// Data stored for BH mergers into the output file
struct bh_merger_data
{
  MyIDType ID1, ID2, ID_remnant;
  double m1, m2, m_remnant;
  double chi1, chi2, chi_remnant;
  double v_kick;    // kick velocity that is applied if kicks are enabled, stored even if it is not applied
  double t_merger;  // physical time of the merger
  double z_merger;  // redshift of the merger

  static MPI_Datatype get_mpi_datatype();
};

MPI_Datatype bh_merger_data::get_mpi_datatype()
{
  // construct the datatype on the first call
  static MPI_Datatype dtype = []() {
    MPI_Datatype dtype;
    const int n_items = 12;
    std::vector<int> block_lengths(n_items, 1);

    static_assert(std::is_same<MyIDType, unsigned long int>::value || std::is_same<MyIDType, unsigned long long>::value);
    MPI_Datatype ID_type = std::is_same<MyIDType, unsigned long int>::value ? MPI_UNSIGNED_LONG : MPI_UNSIGNED_LONG_LONG;

    MPI_Datatype types[] = {
        ID_type,    ID_type,    ID_type,    MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
    };
    MPI_Aint offsets[] = {offsetof(bh_merger_data, ID1),    offsetof(bh_merger_data, ID2),      offsetof(bh_merger_data, ID_remnant),
                          offsetof(bh_merger_data, m1),     offsetof(bh_merger_data, m2),       offsetof(bh_merger_data, m_remnant),
                          offsetof(bh_merger_data, chi1),   offsetof(bh_merger_data, chi2),     offsetof(bh_merger_data, chi_remnant),
                          offsetof(bh_merger_data, v_kick), offsetof(bh_merger_data, t_merger), offsetof(bh_merger_data, z_merger)};

    MPI_Type_create_struct(n_items, block_lengths.data(), offsets, types, &dtype);
    MPI_Type_commit(&dtype);

    return dtype;
  }();

  return dtype;
}


int parse_PN_terms()
{
  auto &terms = KetjuData.options.PN_terms;
  for(size_t i = 0; terms[i] != '\0'; ++i)
    {
      terms[i] = std::tolower(terms[i]);
      if(i >= sizeof terms)
        endrun(1, "Error: KetjuPNTerms too long for buffer!");
    }

  // special flags
  if(std::strcmp(terms, "all") == 0)
    return KETJU_PN_ALL;

  if(std::strcmp(terms, "no_spin") == 0)
    return KETJU_PN_DYNAMIC_ALL;

  if(std::strcmp(terms, "none") == 0)
    return KETJU_PN_NONE;

  int flags = KETJU_PN_NONE;
  for(int i = 0; terms[i] != '\0'; ++i)
    {
      switch(terms[i])
        {
          case '2':
            flags = flags | KETJU_PN_1_0_ACC;
            break;
          case '4':
            flags = flags | KETJU_PN_2_0_ACC;
            break;
          case '5':
            flags = flags | KETJU_PN_2_5_ACC;
            break;
          case '6':
            flags = flags | KETJU_PN_3_0_ACC;
            break;
          case '7':
            flags = flags | KETJU_PN_3_5_ACC;
            break;
          case 's':
            flags = flags | KETJU_PN_SPIN_ALL;
            break;
          case 'c':
            flags = flags | KETJU_PN_THREEBODY;  // cross terms
            break;
          default:
            char buf[50];
            std::snprintf(buf, sizeof buf, "Invalid character '%c' in KetjuPNTerms string!", terms[i]);
            endrun(1, buf);
        }
    }
  return flags;
}

int get_PN_flags()
{
  static int res = parse_PN_terms();
  return res;
}





class IntegratorSystem : ketju_system
{
  struct extra_data
  {
    MyIDType ID;
    int Task;
    int Index;
    int Type;
  };
  int initial_num_particles;
  std::vector<extra_data> extra_data_storage;
 public:
  IntegratorSystem(int num_bhs, int num_other_particles, const double cf_atime, Cosmology * CP, const struct UnitSystem units, MPI_Comm Comm)
      : initial_num_particles(num_bhs + num_other_particles), extra_data_storage(initial_num_particles)
  {
    ketju_create_system(this, num_bhs, num_other_particles, Comm);
    particle_extra_data           = extra_data_storage.data();
    particle_extra_data_elem_size = sizeof(extra_data);
    constants->G                  = CP->GravInternal;
    constants->c                  = LIGHTCGS / units.UnitVelocity_in_cm_per_s;

    options->PN_flags = get_PN_flags();

    const auto &opt                 = KetjuData.options;
    options->enable_bh_merger_kicks = opt.enable_bh_merger_kicks;
    options->star_star_softening    = opt.use_star_star_softening ? KetjuData.options.star_smoothing * cf_atime : 0;

    options->gbs_relative_tolerance         = opt.integration_relative_tolerance;
    options->output_time_relative_tolerance = opt.output_time_relative_tolerance;

    options->mst_algorithm                    = opt.use_divide_and_conquer_mst ? KETJU_MST_DIVIDE_AND_CONQUER_PRIM : KETJU_MST_PRIM;
    options->steps_between_mst_reconstruction = opt.steps_between_mst_reconstruction;
    options->max_tree_distance                = opt.max_tree_distance;

    options->max_step_count = (KETJU_INTEGRATOR_MAX_STEP_COUNT);
  }
  // We won't need these, even though they could be defined.
  // Make sure they're not available to avoid errors.
  IntegratorSystem(const IntegratorSystem &)            = delete;
  IntegratorSystem &operator=(const IntegratorSystem &) = delete;
  IntegratorSystem(IntegratorSystem &&)                 = delete;
  IntegratorSystem &operator=(IntegratorSystem &&)      = delete;

  ~IntegratorSystem() { ketju_free_system(this); }

  ketju_integration_options &get_options() { return *options; }
  const ketju_integration_options &get_options() const { return *options; }
  const ketju_performance_counters &get_perf() const { return *perf; }
  double get_time() const { return physical_state->time; }
  int get_num_pn_particles() const { return num_pn_particles; }
  int get_num_particles() const { return num_particles; }
  int get_num_mergers() const { return num_mergers; }
  const ketju_bh_merger_info *get_merger_info() const { return merger_infos; }
  void run(double integration_timespan) { ketju_run_integrator(this, integration_timespan); }

  // A simple class to allow accessing individual particles conveniently like single objects/structs,
  // even though the data is stored as a struct of arrays.
  template <typename IS>
  class ParticleProxyT
  {
    static_assert(std::is_same<typename std::remove_const<IS>::type, IntegratorSystem>::value);
    IS *const system;
    const int index;
    // Only allow construction through IntegratorSystem methods
    friend class IntegratorSystem;
    ParticleProxyT(IS *system, int index) : system(system), index(index) {}

   public:
    auto pos() -> double (&)[3] { return system->physical_state->pos[index]; }
    auto vel() -> double (&)[3] { return system->physical_state->vel[index]; }
    auto spin() -> double (&)[3]
    {
      if(index >= system->get_num_pn_particles())
        {
          endrun(1, "Out of bounds particle spin access!");
        }
      return system->physical_state->spin[index];
    }
    double &mass() { return system->physical_state->mass[index]; }
    MyIDType &ID() { return system->extra_data_storage[index].ID; }
    int &Task() { return system->extra_data_storage[index].Task; }
    int &Index() { return system->extra_data_storage[index].Index; }
    int &Type() { return system->extra_data_storage[index].Type; }

    auto pos() const -> const double (&)[3] { return system->physical_state->pos[index]; }
    auto vel() const -> const double (&)[3] { return system->physical_state->vel[index]; }
    auto spin() const -> const double (&)[3]
    {
      if(index >= system->get_num_pn_particles())
        {
          endrun(1, "Out of bounds particle spin access!");
        }
      return system->physical_state->spin[index];
    }
    const double &mass() const { return system->physical_state->mass[index]; }
    const MyIDType &ID() const { return system->extra_data_storage[index].ID; }
    const int &Task() const { return system->extra_data_storage[index].Task; }
    const int &Index() const { return system->extra_data_storage[index].Index; }
    const int &Type() const { return system->extra_data_storage[index].Type; }
  };

  using ParticleProxy      = ParticleProxyT<IntegratorSystem>;
  using ConstParticleProxy = ParticleProxyT<const IntegratorSystem>;

  ParticleProxy operator[](int index)
  {
    // Allow accessing merged particle data at the end of the array as well.
    if(index >= initial_num_particles)
      {
        endrun(1, "Out of bounds particle access!");
      }
    return ParticleProxy(this, index);
  }

  const ConstParticleProxy operator[](int index) const
  {
    if(index >= initial_num_particles)
      {
        endrun(1, "Out of bounds particle access!");
      }
    return ConstParticleProxy(this, index);
  }
};

//////////////////

class mpi_task_group
{
  // Default state is similar to what we get when current task is not a part of the group
  MPI_Group group = MPI_GROUP_NULL;
  MPI_Comm comm   = MPI_COMM_NULL;
  int rank        = MPI_UNDEFINED;
  int size        = 0;
  int root        = MPI_UNDEFINED;
  int root_sim    = MPI_UNDEFINED;

 public:
  mpi_task_group() = default;
  mpi_task_group(const std::vector<int> &task_sim_indices, MPI_Comm Comm, int tag = 0)
  {
    MPI_Group sim_group;
    MPI_Comm_group(Comm, &sim_group);
    MPI_Group_incl(sim_group, task_sim_indices.size(), task_sim_indices.data(), &group);
    MPI_Group_size(group, &size);
    MPI_Group_rank(group, &rank);
    MPI_Comm_create_group(Comm, group, tag, &comm);
    MPI_Group_free(&sim_group);

    root     = 0;
    root_sim = task_sim_indices[root];
  }

  mpi_task_group(const mpi_task_group &)            = delete;
  mpi_task_group &operator=(const mpi_task_group &) = delete;

  mpi_task_group &operator=(mpi_task_group &&other) noexcept
  {
    std::swap(group, other.group);
    std::swap(comm, other.comm);

    rank     = other.rank;
    size     = other.size;
    root     = other.root;
    root_sim = other.root_sim;
    return *this;
  }
  mpi_task_group(mpi_task_group &&other) noexcept { *this = std::move(other); }

  ~mpi_task_group()
  {
    if(comm != MPI_COMM_NULL)
      MPI_Comm_free(&comm);
    if(group != MPI_GROUP_NULL)
      MPI_Group_free(&group);
  }

  int get_rank() const { return rank; }
  int get_size() const { return size; }
  int get_root() const { return root; }
  int get_root_sim() const { return root_sim; }
  MPI_Group get_group() const { return group; }
  MPI_Comm get_comm() const { return comm; }

  // Set both groups to have the same root task if possible
  void set_common_root(mpi_task_group &other, MPI_Comm Comm)
  {
    if(group == MPI_GROUP_NULL || other.group == MPI_GROUP_NULL)
      return;

    MPI_Group sim_group;
    MPI_Comm_group(Comm, &sim_group);

    MPI_Group intersection;
    MPI_Group_intersection(group, other.group, &intersection);
    if(intersection != MPI_GROUP_EMPTY)
      {
        // found overlapping tasks, lets just set the first of them as the
        // root for both groups
        const int root_intersection_rank = 0;
        MPI_Group_translate_ranks(intersection, 1, &root_intersection_rank, group, &root);
        MPI_Group_translate_ranks(intersection, 1, &root_intersection_rank, other.group, &other.root);

        // Find the sim ranks of the root tasks
        MPI_Group_translate_ranks(group, 1, &root, sim_group, &root_sim);
        MPI_Group_translate_ranks(other.group, 1, &other.root, sim_group, &other.root_sim);
      }
    // nothing to do if no overlap between the groups

    MPI_Group_free(&intersection);
    MPI_Group_free(&sim_group);
  }
};



////////////////////

struct region_compute_info
{
  int compute_sequence_position;
  int first_task_index;
  int final_task_index;
};

// A single Ketju region.
class ketju_Region
{
  int region_index;
  // Set of particle indices on the local task within this region
  std::set<int> local_member_indices;

  // Data of the contained BHs before integration
  std::vector<mpi_particle> bhs;

  int timestep_region_index = -1;

  int total_particle_count;
  int local_star_count;
  int star_count;

  // CoM position, velocity in Gadget coordinates
  //changed save the CoM with float pos rather than int pos
  //MyIntPosType CoM_IntPos[3];
  double CoM_Pos[3];
  double CoM_Vel[3];

  double system_scale_fac;  // scale factor at the system's integrated time
  double system_hubble;     // hubble parameter likewise

  mpi_task_group affected_tasks, compute_tasks;

  std::vector<int> particle_counts_on_affected_tasks;
  std::vector<int> affected_task_sim_indices;

  region_compute_info compute_info;

  // Integrator data is only allocated on the compute tasks
  std::unique_ptr<IntegratorSystem> integrator_system;

  std::vector<mpi_particle> output_bh_data;
  std::vector<bh_merger_data> output_merger_data;
  Cosmology * CP;

  void gadget_to_integrator_pos(const double (&gadget_pos)[3], double (&integrator_pos)[3]) const;
  void integrator_to_gadget_pos(const double (&integrator_pos)[3], double (&gadget_pos)[3]) const;
  void gadget_to_integrator_vel(const double (&gadget_vel)[3], const double (&integrator_pos)[3], double (&integrator_vel)[3]) const;
  void integrator_to_gadget_vel(const double (&integrator_vel)[3], const double (&integrator_pos)[3], double (&gadget_vel)[3]) const;

  std::vector<mpi_particle> get_particle_data_on_compute_tasks(MPI_Comm Comm) const;
  void set_CoM_from_particles(const std::vector<mpi_particle> &particles);

  void do_negative_halfstep_kick(double halfstep_kick_factor, inttime_t Ti_Current);
  void expand_tight_binaries(double timestep);
  int setup_output_storage(const inttime_t region_Ti_step, const inttime_t output_Ti_step, const inttime_t Ti_Current);
  void store_output_data_on_compute_root(int &output_index);
  void store_merger_data_on_compute_root(inttime_t t0, inttime_t Ti_Current);
  void run_integration(int timebin, inttime_t Ti_Current);

  std::vector<double> get_potential_energy_corrections_on_compute_root(inttime_t Ti_Current) const;
  std::vector<mpi_particle> get_particle_data_on_affected_root(inttime_t Ti_Current, MPI_Comm Comm) const;

 public:
  // changed: add para to the constructor: cf_atime (to replace the original All.cf_atime), a_hubble (to get the hubble)
  ketju_Region(std::vector<mpi_particle> &&bhs, std::set<int> &&local_member_indices, Cosmology * CP, double cf_hubble_a, int region_index, double cf_atime, int local_star_count)
      : region_index(region_index),
        local_member_indices{std::move(local_member_indices)},
        bhs{std::move(bhs)},
        system_scale_fac(cf_atime),
        system_hubble(cf_hubble_a),
        CP(CP),
        local_star_count(local_star_count)
  {
  }

  const std::vector<mpi_particle> &get_bhs() const { return bhs; }
  int get_bh_count() const { return bhs.size(); }
  const std::set<int> &get_local_member_indices() const { return local_member_indices; }

  void set_timestep_region_index(int i) { timestep_region_index = i; }
  int get_timestep_region_index() const { return timestep_region_index; }
  int get_region_index() const { return region_index; }
  int get_total_particle_count() const { return total_particle_count; }
  int get_star_count() const {return star_count; }
  void set_compute_info(const region_compute_info &info) { compute_info = info; }
  const region_compute_info &get_compute_info() const { return compute_info; }
  int get_compute_sequence_position() const { return compute_info.compute_sequence_position; }
  int get_compute_root_sim() const { return compute_tasks.get_root_sim(); }
  bool this_task_is_in_compute_tasks() const { return compute_tasks.get_rank() != MPI_UNDEFINED; }
  bool this_task_is_compute_root() const { return compute_tasks.get_rank() == compute_tasks.get_root(); }
  bool this_task_is_affected_root() const { return affected_tasks.get_rank() == affected_tasks.get_root(); }
  bool this_task_is_in_affected_tasks() const { return affected_tasks.get_rank() != MPI_UNDEFINED; }
  const std::vector<mpi_particle> &get_output_bh_data() const { return output_bh_data; }
  const std::vector<bh_merger_data> &get_output_merger_data() const { return output_merger_data; }
  double get_normalized_integration_cost_from_compute_root(int timebin) const
  {
    if(!this_task_is_compute_root())
      return 0;

    return integrator_system->get_perf().total_work / (inttime_t(1) << timebin);
  }

  //double get_cf_hubble_a(inttime_t Ti_Current) const;
  int get_max_timebin(inttime_t Ti_Current, MPI_Comm Comm) const;
  void find_affected_tasks_and_particle_counts(MPI_Comm Comm);
  void set_up_compute_comms(MPI_Comm Comm);
  void set_up_integrator(MPI_Comm Comm, inttime_t Ti_Current, const struct UnitSystem units);
  void do_integration_step(int timebin, inttime_t Ti_Current, FILE * FdSingleBH, FILE * FdKetjuRegion, int trace_bhid);
  void update_sim_data(inttime_t Ti_Current, MPI_Comm Comm);
  void Region_info(FILE * FdKetjuRegion, const double Time, double timestep, int timebin);
};

double get_cf_hubble_a(inttime_t Ti_Current, Cosmology * CP)
{
  double cf_atime;

  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
  }

  //return hubble_function(CP, cf_atime);
  return 1.0;
}


int ketju_Region::get_max_timebin(inttime_t Ti_Current, MPI_Comm Comm) const
{
  // get cf_atime and cf_a2inv from ti_current
  double cf_atime, cf_a2inv;

  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
    cf_a2inv = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
    cf_a2inv = 1 / (cf_atime * cf_atime);
  }
  //double cf_hubble_a = get_cf_hubble_a(Ti_Current, CP);
  const double cf_hubble_a = hubble_function(CP, cf_atime);
  double timebase_interval = Dloga_interval_ti(Ti_Current);
  double max_dt = std::min(TIMEBASE * timebase_interval / cf_hubble_a, KetjuData.options.MaxSizeTimestep / cf_hubble_a);
  //double max_dt = std::min(TIMEBASE * All.Timebase_interval / cf_hubble_a, KetjuData.options.MaxSizeTimestep / cf_hubble_a);
  double stellar_com_vel[3] = {};
  double stellar_sigma      = 0;

  int NTask, ThisTask;
  MPI_Comm_size(Comm, &NTask);
  MPI_Comm_rank(Comm, &ThisTask);
    
  
  // Consider the motion of the CoM for limiting the timestep.
  // Generally the surrounding stellar system will place a stricter limit, but this guards against cases where it doesn't.
  // Also calculate the CoM vel and sigma of stellar particles for use further down.
  if(this_task_is_in_affected_tasks())
    {
  
      double com_data[7]           = {};  // vel + acc + mass
      double *vel_data             = com_data;
      double *acc_data             = com_data + 3;
      double stellar_vel_data[7]   = {};  // stellar com vel, vel^2, mass
      double *stellar_com_vel_data = stellar_vel_data;
      double *stellar_vel2_data    = stellar_vel_data + 3;
      for(int p : local_member_indices)
        {
          if(P[p].Type == 5 ){
            for(int k = 0; k < 3; ++k)
              {
                vel_data[k] += BHP(p).Mass * P[p].Vel[k] / cf_atime;
                acc_data[k] += BHP(p).Mass * P[p].FullTreeGravAccel[k] * cf_a2inv;
                //#if defined(PMGRID) && !defined(TREEPM_NOTIMESPLIT)
                acc_data[k] += BHP(p).Mass * P[p].GravPM[k] * cf_a2inv;
                //#endif
              }
            com_data[6] += BHP(p).Mass;            
          }
          else{
            for(int k = 0; k < 3; ++k)
              {
          
                vel_data[k] += P[p].Mass * P[p].Vel[k] / cf_atime;
                acc_data[k] += P[p].Mass * P[p].FullTreeGravAccel[k] * cf_a2inv;
                //#if defined(PMGRID) && !defined(TREEPM_NOTIMESPLIT)
                acc_data[k] += P[p].Mass * P[p].GravPM[k] * cf_a2inv;
                //#endif
              }
            com_data[6] += P[p].Mass;
          }

          if(P[p].Type == 4 )
            {
              for(int k = 0; k < 3; ++k)
                {
                  stellar_com_vel_data[k] += P[p].Mass * P[p].Vel[k] / cf_atime;
                  stellar_vel2_data[k] += P[p].Mass * std::pow(P[p].Vel[k] / cf_atime, 2);
                }
              stellar_vel_data[6] += P[p].Mass;
            }
        }

      double com_res[7];
      MPI_Reduce(com_data, com_res, 7, MPI_DOUBLE, MPI_SUM, affected_tasks.get_root(), affected_tasks.get_comm());
      double stellar_vel_res[7];
      MPI_Reduce(stellar_vel_data, stellar_vel_res, 7, MPI_DOUBLE, MPI_SUM, affected_tasks.get_root(), affected_tasks.get_comm());
      if(this_task_is_affected_root())
        {
          const double com_mass = com_res[6];
          const double com_vel  = vector_norm(com_res) / com_mass;
          const double com_acc  = vector_norm(com_res + 3) / com_mass;

          // Check the acceleration time step criterion.
          // Use the same criterion as for softened particles,
          // but with a tenth of the ketju region radius as the softening
          // for a more conservative timestep.
          const double acc_dt = std::sqrt(0.2 * KetjuData.options.ErrTolIntAccuracy * KetjuData.options.region_physical_radius / com_acc);

          // Limit also so that the CoM of the system doesn't move too much relative to the region size
          const double com_vel_dt = 0.1 * KetjuData.options.region_physical_radius / com_vel;

          ketju_debug_printf("Region %d: max_dt=%g, acc_dt=%g, com_vel_dt=%g, ", region_index, max_dt, acc_dt, com_vel_dt);

          max_dt = std::min(max_dt, acc_dt);
          max_dt = std::min(max_dt, com_vel_dt);

          const double stellar_com_mass = stellar_vel_res[6];
          if(stellar_com_mass > 0)
            {  // Leave the stellar com vel and sigma as 0 if no stars in the region
              stellar_sigma = 0;
              for(int k = 0; k < 3; ++k)
                {
                  stellar_com_vel[k] = stellar_vel_res[k] / stellar_com_mass;
                  stellar_sigma += stellar_vel_res[k + 3] / stellar_com_mass - std::pow(stellar_com_vel[k], 2);
                }
              stellar_sigma /= 3;
              stellar_sigma = std::sqrt(stellar_sigma);
            }
        }
    }
  
  if(this_task_is_affected_root())
    {
      // A fairly conservative estimate based on BH velocities:
      // limit so that any BH only travels at most 30% of the region radius
      // relative to the stellar CoM vel during a single step assuming constant velocity.
      // This avoids passing through regions of stars too quickly for the stars to enter
      // the region, either due to the BH or stellar system motion, while avoiding
      // unnecessary limiting due to e.g. the whole galaxy moving.
      // In the outer 30% of the region the softened gravity is still nearly Newtonian
      // even for the smallest possible region.
      // We assume suitable continuity, so that the values calculated from the stars
      // in the region currently are representative of the surroundings.
      // For tight binaries this condition might be too strict, but with typical parameters should
      // still allow for several orbits in a gadget timestep,
      // and automatically deals with possible large GW recoil kicks as well.
      double max_vel = 0;
      for(const auto &bh : bhs)
        {
          double vel_diff[3];
          for(int k = 0; k < 3; ++k)
            {
              vel_diff[k] = bh.Vel[k] / cf_atime - stellar_com_vel[k];
            }

          max_vel = std::max(max_vel, vector_norm(vel_diff));
        }
      const double max_bh_vel_dt = 0.3 * KetjuData.options.region_physical_radius / max_vel;
      max_dt                     = std::min(max_dt, max_bh_vel_dt);

      // Limit also so that 3 * (stellar sigma) only covers at most 50% of region radius.
      // This should ensure that essentially all stars become active in the buffer zone,
      // and also that most stars enter the region within the outer 30% for the same reasons as above.
      const double stellar_sigma_dt = .5 * KetjuData.options.region_physical_radius / (3 * stellar_sigma);
      max_dt                        = std::min(max_dt, stellar_sigma_dt);

      ketju_debug_printf("max_bh_vel_dt=%g, stellar_sigma_dt=%g\n", max_bh_vel_dt, stellar_sigma_dt);
    }
 
  MPI_Barrier(Comm);
  MPI_Bcast(&max_dt, 1, MPI_DOUBLE, affected_tasks.get_root_sim(), Comm);



  inttime_t ti_step = max_dt / timebase_interval;

  // copied from timestep.cc simparticles::get_timestep_bin since getting access to that here is inconvenient
  int bin = -1;
  if(ti_step == 0)
    return 0;

  if(ti_step == 1)
    endrun(1, "time-step of integer size 1 not allowed \n");

  while(ti_step)
    {
      bin++;
      ti_step >>= 1;
      // message(0, "DEBUG.. ketju 968 bin: %d \n", bin);
    }
  // message(0, "DEBUG.. ketju 970 bin: %d \n", bin);
  return bin;
}


void ketju_Region::find_affected_tasks_and_particle_counts(MPI_Comm Comm)
{
  int NTask, ThisTask;
  MPI_Comm_size(Comm, &NTask);
  MPI_Comm_rank(Comm, &ThisTask);


  std::vector<int> particle_counts(NTask);
  std::vector<int> star_particles_counts(NTask);
  star_particles_counts[ThisTask] = local_star_count;
  particle_counts[ThisTask] = local_member_indices.size();
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, particle_counts.data(), 1, MPI_INT, Comm);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, star_particles_counts.data(), 1, MPI_INT, Comm);

  total_particle_count = 0;
  star_count = 0;

  for(int i = 0; i < NTask; ++i)
    {
      if(particle_counts[i] > 0)
        {
          total_particle_count += particle_counts[i];
          star_count += star_particles_counts[i];
          //message(0, "DEBUG.. ketju 1035 taskidx %d update star_count: %d from adding task_star_num %d, update total part %d from adding %d \n",
           //i, star_count, star_particles_counts[i], total_particle_count, particle_counts[i]);
          particle_counts_on_affected_tasks.push_back(particle_counts[i]);
          affected_task_sim_indices.push_back(i);
        }
    }
  // set up communicator
  affected_tasks = mpi_task_group(affected_task_sim_indices, Comm);
  //message(0, "DEBUG.. ketju 1043 star_count %d (task idx %d) \n", star_count, ThisTask);
}

void ketju_Region::set_up_compute_comms(MPI_Comm Comm)
{
  affected_tasks = mpi_task_group(affected_task_sim_indices, Comm);
  std::vector<int> compute_task_indices(compute_info.final_task_index - compute_info.first_task_index + 1);
  std::iota(compute_task_indices.begin(), compute_task_indices.end(), compute_info.first_task_index);
  compute_tasks = mpi_task_group(compute_task_indices, Comm);

  compute_tasks.set_common_root(affected_tasks, Comm);
}

void ketju_Region::set_CoM_from_particles(const std::vector<mpi_particle> &particles)
{
  // To maintain the position resolution, calculate using position
  // offsets to the first particle.
  double M = 0;
  std::fill(std::begin(CoM_Vel), std::end(CoM_Vel), 0);
  double CoM_shift[3] = {0, 0, 0};
  for(const auto &p : particles)
    {
      M += p.Mass;
      double p_shift[3];
      for(int i = 0; i< 3; ++i){
        p_shift[i] = p.Pos[i] - particles[0].Pos[i];
      }

      //conv.nearest_image_intpos_to_pos(p.IntPos, particles[0].IntPos, p_shift);
      for(int i = 0; i < 3; ++i)
        {
          CoM_Vel[i] += p.Mass * p.Vel[i];
          CoM_shift[i] += p.Mass * p_shift[i];
        }
    }
  for(int i = 0; i < 3; ++i)
    {
      CoM_Vel[i] /= M;
      CoM_shift[i] /= M;
    }
  // MySignedIntPosType CoM_Int_shift[3];
  // conv.pos_to_signedintpos(CoM_shift, CoM_Int_shift);
  for(int i = 0; i < 3; ++i)
    {
      CoM_Pos[i] = particles[0].Pos[i] + CoM_shift[i];
      //CoM_IntPos[i] = particles[0].IntPos[i] + CoM_Int_shift[i];
    }
}

void ketju_Region::gadget_to_integrator_pos(const double (&gadget_pos)[3],
                                      double (&integrator_pos)[3]) const
{
  for(int i = 0; i< 3; ++i){
    integrator_pos[i] = gadget_pos[i] - CoM_Pos[i];
  }
  //conv.nearest_image_intpos_to_pos(gadget_pos, CoM_IntPos, integrator_pos);
  if(KetjuData.options.ComovingIntegrationOn)
    {
      for(auto &c : integrator_pos)
        {
          c *= system_scale_fac;
        }
    }
}


void ketju_Region::integrator_to_gadget_pos(const double (&integrator_pos)[3],
                                      double (&gadget_pos)[3]) const
{
  double pos[3];
  for(int i = 0; i < 3; ++i)
    {
      pos[i] = integrator_pos[i] / system_scale_fac;
    }
  //MySignedIntPosType intpos_shift[3];
  //conv.pos_to_signedintpos(pos, intpos_shift);

  for(int i = 0; i < 3; ++i)
    {
      // This may wrap as intended in a periodic box, while the safety buffer regions around the particles should keep it from
      // happening in non-periodic runs.
      // TODO: Some toy examples like a very tight, eccentric binary may end up breaking this if the initial RegionLen value is very
      // small. Is there anything that can be done to prevent that?
      gadget_pos[i] = CoM_Pos[i] + pos[i];
    }
  //conv.constrain_intpos(gadget_pos);
}

void ketju_Region::gadget_to_integrator_vel(const double (&gadget_vel)[3], const double (&integrator_pos)[3],
                                      double (&integrator_vel)[3]) const
{
  if(KetjuData.options.ComovingIntegrationOn)
    {
      for(int i = 0; i < 3; ++i)
        {
          // Include the small Hubble flow contribution to the internal relative velocities.
          // Note that gadget_vel is not the comoving velocity, but the canonical momentum.
          integrator_vel[i] = (gadget_vel[i] - CoM_Vel[i]) / system_scale_fac + system_hubble * integrator_pos[i];
        }
    }
  else
    {
      for(int i = 0; i < 3; ++i)
        {
          integrator_vel[i] = gadget_vel[i] - CoM_Vel[i];
        }
    }
}

void ketju_Region::integrator_to_gadget_vel(const double (&integrator_vel)[3], const double (&integrator_pos)[3],
                                      double (&gadget_vel)[3]) const
{
  if(KetjuData.options.ComovingIntegrationOn)
    {
      for(int i = 0; i < 3; ++i)
        {
          gadget_vel[i] = (integrator_vel[i] - system_hubble * integrator_pos[i]) * system_scale_fac + CoM_Vel[i];
        }
    }
  else
    {
      for(int i = 0; i < 3; ++i)
        {
          gadget_vel[i] = integrator_vel[i] + CoM_Vel[i];
        }
    }
}

std::vector<mpi_particle> ketju_Region::get_particle_data_on_compute_tasks(MPI_Comm Comm) const
{
  // Now the implementation doesn't make use of the shared memory between tasks.
  // TODO: This could be improved if performance requires.

  std::vector<mpi_particle> recv_particle_data;

  if(this_task_is_in_compute_tasks() || this_task_is_affected_root())
    {
      recv_particle_data.resize(total_particle_count);
    }

  if(this_task_is_in_affected_tasks())
    {
      // Gather particle data on affected group root
      std::vector<mpi_particle> send_particle_data;
      send_particle_data.reserve(local_member_indices.size());

      const int thistask = affected_tasks.get_rank();
      for(int i : local_member_indices)
        {
          send_particle_data.emplace_back(mpi_particle::from_gadget_particle_index(thistask, i));
        }
      std::vector<int> displs(particle_counts_on_affected_tasks.size());
      std::partial_sum(particle_counts_on_affected_tasks.begin(), particle_counts_on_affected_tasks.end() - 1, displs.begin() + 1);
      MPI_Gatherv(send_particle_data.data(), send_particle_data.size(), mpi_particle::get_mpi_datatype(), recv_particle_data.data(),
                  particle_counts_on_affected_tasks.data(), displs.data(), mpi_particle::get_mpi_datatype(), affected_tasks.get_root(),
                  affected_tasks.get_comm());
    }

  // Send data from affected root to compute root if needed
  if(!(this_task_is_affected_root() && this_task_is_compute_root()))
    {
      if(this_task_is_affected_root())
        {
          MPI_Send(recv_particle_data.data(), recv_particle_data.size(), mpi_particle::get_mpi_datatype(),
                   compute_tasks.get_root_sim(), region_index, Comm);
        }
      if(this_task_is_compute_root())
        {
          MPI_Recv(recv_particle_data.data(), recv_particle_data.size(), mpi_particle::get_mpi_datatype(),
                   affected_tasks.get_root_sim(), region_index, Comm, MPI_STATUS_IGNORE);
        }
    }

  if(this_task_is_compute_root())
    {
      // Sort particle data to have BHs at the start, in
      // descending order by their masses.
      // The integrator requires BHs to be at the start of the data,
      // and ordering by masses is needed to support the minimum mass cutoff for
      // PN enabled BHs and also gives more logical merger remnant IDs.
      std::sort(recv_particle_data.begin(), recv_particle_data.end(), [](const mpi_particle &a, const mpi_particle &b) {
        if((a.Type == 5) && (b.Type == 5))
          {
            return a.Mass > b.Mass;
          }

        return a.Type == 5;
      });
    }

  if(this_task_is_in_compute_tasks())
    {
      MPI_Bcast(recv_particle_data.data(), recv_particle_data.size(), mpi_particle::get_mpi_datatype(), compute_tasks.get_root(),
                compute_tasks.get_comm());
    }

  return recv_particle_data;
}


void ketju_Region::set_up_integrator(MPI_Comm Comm, inttime_t Ti_Current, const struct UnitSystem units)
{
  std::vector<mpi_particle> particle_data = get_particle_data_on_compute_tasks(Comm);

  if(!this_task_is_in_compute_tasks())
    return;

  set_CoM_from_particles(particle_data);

  double cf_atime;
  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
  }


  const int num_bhs = get_bh_count();
  integrator_system =
      std::unique_ptr<IntegratorSystem>(new IntegratorSystem(num_bhs, total_particle_count - num_bhs, cf_atime,  CP, units, compute_tasks.get_comm()));

  // message(1, "DEBUG...ketju.c 1225: region (%d) num_bhs(%d) num_part(%d): CoMPos xyz=<%g|%g|%g>, vel=<%g|%g|%g> \n",
  //   region_index, num_bhs, total_particle_count, 
  //   CoM_Pos[0], CoM_Pos[1], CoM_Pos[2], CoM_Vel[0], CoM_Vel[1], CoM_Vel[2]);

  for(int i = 0; i < total_particle_count; ++i)
    {
      auto &&p_int = (*integrator_system)[i];
      auto &p_gad  = particle_data[i];

      p_int.mass()  = p_gad.Mass;
      p_int.ID()    = p_gad.ID;
      p_int.Task()  = p_gad.Task;
      p_int.Index() = p_gad.Index;
      p_int.Type()  = p_gad.Type;

      gadget_to_integrator_pos(p_gad.Pos, p_int.pos());
      gadget_to_integrator_vel(p_gad.Vel, p_int.pos(), p_int.vel());

      if(i < num_bhs)
        {
          std::copy(std::begin(p_gad.Spin), std::end(p_gad.Spin), std::begin(p_int.spin()));
        }
    }
}

// Schedule a block of work consisting of two loops of 0 <= i < Nloop-1,
// i < j < Nloop and divide it approximately evenly among num_proc tasks.
// edge_index == e gives the starting point for
// e'th process and edge_index == e+1 gives the ending point (non-inclusive).
int loop_scheduling_block_edge(int Nloop, int num_proc, int edge_index)
{
  if(Nloop <= 0 || num_proc <= 0)
    endrun(1, "Invalid call to ketju loop_scheduling_block_edge");

  if(num_proc >= Nloop)  // more tasks than available work
    num_proc = Nloop;

  // Handle edges explicitly to avoid any possible issues with rounding,
  // otherwise possibly being off by one from the optimal value isn't that important.
  if(edge_index <= 0)
    return 0;

  if(edge_index >= num_proc)
    return Nloop - 1;

  // Use floating point math to avoid intermediate overflows
  const double Proc = num_proc;
  const double N = Nloop;
  // Analytic solution for the edge position below which the total work is nearest to edge_index * N * (N - 1) / (2 * P)
  return std::ceil(N - 0.5 - std::sqrt(N * (N - 1) * (Proc - edge_index) / Proc + 0.25));
}

double softened_inverse_r3(double h, double r)
{
  const double rinv = 1. / r;

  // Copied from gravtree::get_gfactors_monopole since getting
  // access to the Sim.Gravtree object here seems too annoying.
  if(r > h)
    {
      return rinv * rinv * rinv;
    }
  else
    {
      double h_inv  = 1 / h;
      double h2_inv = h_inv * h_inv;
      double u      = r * h_inv;

      if(u < 0.5)
        {
          double u2 = u * u;
          return rinv * h2_inv * u * ((SOFTFAC1) + u2 * ((SOFTFAC2)*u + (SOFTFAC3)));
        }
      else
        {
          double u2 = u * u;
          double u3 = u2 * u;
          return rinv * h2_inv * u * ((SOFTFAC8) + (SOFTFAC9)*u + (SOFTFAC10)*u2 + (SOFTFAC11)*u3 + (SOFTFAC12) / u3);
        }
    }
}

void ketju_Region::do_negative_halfstep_kick(double halfstep_kick_factor, inttime_t Ti_Current)
{
  if(!this_task_is_in_compute_tasks())
    return;

  auto &syst           = *integrator_system;
  const int num_part   = syst.get_num_particles();
  const int loop_start = loop_scheduling_block_edge(num_part, compute_tasks.get_size(), compute_tasks.get_rank());
  const int loop_end   = loop_scheduling_block_edge(num_part, compute_tasks.get_size(), compute_tasks.get_rank() + 1);

  // seems there isn't an easy way to get an automatically managed 2d array, so need to use new/delete.
  auto dv = new double[num_part][3]();

  // Assumes that the softenings are constant in physical coordinates also in comoving integration,
  // the scale factor is only due to how the values are stored.
  
  double cf_atime;
  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
  }

  const double h = KetjuData.options.star_smoothing * cf_atime;
  //const double h = All.ForceSoftening[get_star_softening_class()] * All.cf_atime;

  // Compute the kicks for this task
  for(int i = loop_start; i < loop_end; ++i)
    {
      for(int j = i + 1; j < num_part; ++j)
        {
          auto &&pi = syst[i];
          auto &&pj = syst[j];
          double dr[3];
          double r2 = 0;
          // Operating in CoM coordinates, so always
          // non-periodic.
          for(int k = 0; k < 3; ++k)
            {
              dr[k] = pi.pos()[k] - pj.pos()[k];
              r2 += dr[k] * dr[k];
            }
          double dt_G_per_soft_r3 = halfstep_kick_factor * CP->GravInternal * softened_inverse_r3(h, std::sqrt(r2));
          for(int k = 0; k < 3; ++k)
            {
              dv[i][k] += pj.mass() * dt_G_per_soft_r3 * dr[k];
              dv[j][k] -= pi.mass() * dt_G_per_soft_r3 * dr[k];
            }
        }
    }

  // Collect results and write to the ketju_system state
  MPI_Allreduce(MPI_IN_PLACE, dv, 3 * num_part, MPI_DOUBLE, MPI_SUM, compute_tasks.get_comm());

  for(int i = 0; i < num_part; ++i)
    {
      auto &&p = syst[i];

      for(int k = 0; k < 3; ++k)
        {
          p.vel()[k] += dv[i][k];
        }

    }

  delete[] dv;
}

// some helpers for Region::expand_tight_binaries
struct region_binaries
{
  struct binary_particles
  {
    int i, j;
    double period;
  };

  std::vector<binary_particles> binaries;
  double min_BH_binary_period;
};

// Find the bound binaries in the system
region_binaries find_binaries(int loop_start, int loop_end, const IntegratorSystem &integrator_system, Cosmology * CP)
{
  double min_BH_binary_period = DBL_MAX;
  std::vector<region_binaries::binary_particles> binaries;
  const int num_part = integrator_system.get_num_particles();

  for(int i = loop_start; i < loop_end; ++i)
    {
      for(int j = i + 1; j < num_part; ++j)
        {
          double dr[3];
          double r2 = 0;
          double dv[3];
          double v2       = 0;
          const auto &&pi = integrator_system[i];
          const auto &&pj = integrator_system[j];
          // Operating in CoM coordinates, so always
          // non-periodic.
          for(int k = 0; k < 3; ++k)
            {
              dr[k] = pi.pos()[k] - pj.pos()[k];
              r2 += dr[k] * dr[k];
              dv[k] = pi.vel()[k] - pj.vel()[k];
              v2 += dv[k] * dv[k];
            }

          double GM = CP->GravInternal * (pi.mass() + pj.mass());
          // energy per reduced mass of the potential binary
          double E = .5 * v2 - GM / std::sqrt(r2);

          if(E > 0)
            continue;  // not bound

          double a      = -GM / (2 * E);
          double period = 2 * M_PI * a * std::sqrt(a / GM);

          if(is_bh_type(pi.Type()) && is_bh_type(pj.Type()))
            {
              // For BH binaries the only interesting thing is the minimum
              // period, we'll never touch them otherwise.
              if(period < min_BH_binary_period)
                {
                  min_BH_binary_period = period;
                }
              continue;
            }
          // Otherwise store the binary for further processing
          binaries.push_back({i, j, period});
        }
    }
  return {std::move(binaries), min_BH_binary_period};
}

struct binary_expansion_info
{
  double old_a, new_a, delta_v;
};

// Expand the binary by kicking the particles to double the period.
// This breaks energy conservation.
binary_expansion_info double_binary_period(IntegratorSystem::ParticleProxy &pi, IntegratorSystem::ParticleProxy &pj, Cosmology * CP)
{
  double r2 = 0;
  double dv[3];
  double v2 = 0;
  for(int k = 0; k < 3; ++k)
    {
      double dr = pi.pos()[k] - pj.pos()[k];
      r2 += dr * dr;
      dv[k] = pi.vel()[k] - pj.vel()[k];
      v2 += dv[k] * dv[k];
    }

  const double M     = (pi.mass() + pj.mass());
  const double potE  = CP->GravInternal * M / std::sqrt(r2);
  const double E     = .5 * v2 - potE;
  const double newE  = 0.63 * E;  // approx. the factor required for doubling the period (2^-2/3)
  const double new_v = std::sqrt(2 * (potE + newE));
  const double old_v = std::sqrt(v2);

  for(int k = 0; k < 3; ++k)
    {
      dv[k] *= (new_v - old_v) / old_v;
      pi.vel()[k] += dv[k] * pj.mass() / M;
      pj.vel()[k] -= dv[k] * pi.mass() / M;
    }

  const double a     = CP->GravInternal * M / (-2 * E);
  const double new_a = CP->GravInternal * M / (-2 * newE);
  return {a, new_a, new_v - old_v};
}

void ketju_Region::expand_tight_binaries(double timestep)
{
  const int num_part = integrator_system->get_num_particles();
  const int num_bh   = integrator_system->get_num_pn_particles();

  // Parallelize the calculation
  int loop_start = loop_scheduling_block_edge(num_part, compute_tasks.get_size(), compute_tasks.get_rank());
  int loop_end   = loop_scheduling_block_edge(num_part, compute_tasks.get_size(), compute_tasks.get_rank() + 1);

  if(KetjuData.options.use_star_star_softening)
    {
      // Only check binaries containing a BH.
      // This loop parallelization wastes some computing power, but this is very cheap in this case anyway
      loop_start = std::min(loop_start, num_bh);
      loop_end   = std::min(loop_end, num_bh);
    }

  auto found_binaries         = find_binaries(loop_start, loop_end, *integrator_system, CP);
  auto &binaries              = found_binaries.binaries;
  double min_BH_binary_period = found_binaries.min_BH_binary_period;
  MPI_Allreduce(MPI_IN_PLACE, &min_BH_binary_period, 1, MPI_DOUBLE, MPI_MIN, compute_tasks.get_comm());

  // sort the binaries to find the shortest periods
  std::sort(binaries.begin(), binaries.end(),
            [](const decltype(binaries[0]) &a, const decltype(binaries[0]) &b) { return a.period < b.period; });
  struct
  {
    double period;
    int rank;
  } min_period_rank = {binaries.empty() ? DBL_MAX : binaries[0].period, compute_tasks.get_rank()};

  MPI_Allreduce(MPI_IN_PLACE, &min_period_rank, 1, MPI_DOUBLE_INT, MPI_MINLOC, compute_tasks.get_comm());
  const double min_period = min_period_rank.period;

  if(this_task_is_compute_root())
    {
      ketju_debug_printf(
          "expand_tight_binaries: region %d min_period = %g, "
          "timestep = %g\n",
          region_index, min_period, timestep);
    }

  // No need to expand the binary if the period is longer than the timestep or
  // the minimum BH binary period times the period factor
  const double fac = KetjuData.options.expand_tight_binaries_period_factor;
  if(min_period > fac * timestep || min_period > fac * min_BH_binary_period)
    {
      return;
    }

  // Decide if the binary with the smallest period needs to be expanded.
  // If there are less than a set limit of  particles with periods < 2*min_period,
  // the tightest binary should be an outlier that can be expanded to improve performance.
  // Only one binary per region is expanded at once.
  // They should be rare enough for there to be no need to, and it's safer to
  // limit the artificial fudging of the physics.

  int count = 0;
  for(const auto &binary : binaries)
    {
      if(binary.period > min_period * 2)
        {
          break;  // the binaries are sorted in ascending order
        }
      ++count;
    }
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, compute_tasks.get_comm());

  if(count > std::max(num_part / 100., 5.))
    {
      return;
    }

  // get the particle indices of the binary to expand
  int ij[2];
  if(min_period_rank.rank == compute_tasks.get_rank())
    {
      ij[0] = binaries[0].i;
      ij[1] = binaries[0].j;
    }
  MPI_Bcast(ij, 2, MPI_INT, min_period_rank.rank, compute_tasks.get_comm());

  auto &&pi = (*integrator_system)[ij[0]];
  auto &&pj = (*integrator_system)[ij[1]];

  auto info = double_binary_period(pi, pj, CP);

  if(this_task_is_compute_root())
    {
      ketju_printf(
          "expanding excessively tight binary in region %d\n"
          "    Particles (ID %llu type %d) and (ID %llu type %d)\n"
          "    Old period %g, semimajor axis %g, new period %g,"
          " new semimajor axis %g, delta_v %g\n",
          region_index, static_cast<unsigned long long>(pi.ID()), pi.Type(), static_cast<unsigned long long>(pj.ID()), pj.Type(),
          min_period, info.old_a, 2 * min_period, info.new_a, info.delta_v);
    }
}

void ketju_Region::store_merger_data_on_compute_root(inttime_t t0, inttime_t Ti_Current)
{
  if(!this_task_is_compute_root())
    return;

  const int num_mergers = integrator_system->get_num_mergers();
  if(num_mergers == 0)
    return;

  double cf_atime;
  double allTime = get_atime(Ti_Current);

  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
    allTime = log(allTime);
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
  }


  //const double phys_t0 = KetjuData.options.ComovingIntegrationOn ? get_cosmic_time(t0) : allTime;
  const double phys_t0 = allTime;
  const double z0      = 1. / cf_atime - 1;
  const double z1      = 1. / system_scale_fac - 1;

  const ketju_bh_merger_info *infos = integrator_system->get_merger_info();

  for(int m = 0; m < num_mergers; ++m)
    {
      auto &info = infos[m];

      MyIDType ID1 = (*integrator_system)[info.extra_index1].ID();
      MyIDType ID2 = (*integrator_system)[info.extra_index2].ID();

      output_merger_data.emplace_back();
      bh_merger_data &data = output_merger_data.back();

      data.ID1         = ID1;
      data.ID2         = ID2;
      data.ID_remnant  = ID1;
      data.m1          = info.m1;
      data.m2          = info.m2;
      data.m_remnant   = info.m_remnant;
      data.chi1        = info.chi1;
      data.chi2        = info.chi2;
      data.chi_remnant = info.chi_remnant;
      data.v_kick      = info.v_kick;
      // The time fields start from zero at the beginning of the step.
      data.t_merger = info.t_merger + phys_t0;
      // Approximate with a linear interpolation, should be accurate enough.
      data.z_merger = z0 + (z1 - z0) * info.t_merger / integrator_system->get_time();

      // Output data to stdout in addition to the main output file.
      ketju_printf(
          "BH merger in region %d:\n"
          "  Phys. time: %.5g (redshift %.5f) + %.5g \n"
          "  ID %llu + %llu -> %llu\n"
          "  mass %.4g + %.4g -> %.4g\n"
          "  chi %3.2f + %3.2f -> %3.2f\n"
          "  Kick velocity %.4g (kicks are %s)\n",
          region_index, phys_t0, z0, info.t_merger, static_cast<unsigned long long>(ID1), static_cast<unsigned long long>(ID2),
          static_cast<unsigned long long>(ID1), info.m1, info.m2, info.m_remnant, info.chi1, info.chi2, info.chi_remnant, info.v_kick,
          (KetjuData.options.enable_bh_merger_kicks ? "enabled" : "disabled"));
    }
}

int ketju_Region::setup_output_storage(const inttime_t region_Ti_step, const inttime_t output_Ti_step, const inttime_t Ti_Current)
{
  // Output is stored at the points where output_timebin is synchronized.
  // Initial state only stored on the first step, otherwise it has been stored
  // at the end of the  the previous integration if needed.
  const bool output_step_synced_at_end_of_current_step =
      Ti_Current + region_Ti_step == (1 + Ti_Current / output_Ti_step) * output_Ti_step;

  int num_output_points = region_Ti_step / output_Ti_step;

  if(num_output_points == 0 && output_step_synced_at_end_of_current_step)
    {
      num_output_points = 1;
    }

  if(Ti_Current == 0)
    {
      num_output_points += 1;
    }

  if(this_task_is_compute_root())
    {
      output_bh_data.resize(num_output_points * get_bh_count());
    }

  return num_output_points;
}

// Stores the BH data to positions given by output_index and advances output_index
void ketju_Region::store_output_data_on_compute_root(int &output_index)
{
  if(!this_task_is_compute_root())
    return;

  // The size of the output data storage is based on the initial number of BHs
  const int num_output_points = output_bh_data.size() / get_bh_count();

  if(num_output_points <= output_index)
    endrun(1, "Ketju: incorrect store_output_data calls!");

  // but data is only stored for the currently existing BHs, obviously
  const int num_bhs = integrator_system->get_num_pn_particles();
  for(int i = 0; i < num_bhs; ++i)
    {
      auto &p_out        = output_bh_data[i * num_output_points + output_index];
      const auto &&p_int = (*integrator_system)[i];

      p_out.Mass  = p_int.mass();
      p_out.ID    = p_int.ID();
      p_out.Task  = p_int.Task();
      p_out.Index = p_int.Index();
      p_out.Type  = p_int.Type();

      integrator_to_gadget_pos(p_int.pos(), p_out.Pos);
      integrator_to_gadget_vel(p_int.vel(), p_int.pos(), p_out.Vel);
      auto &spin = p_int.spin();
      std::copy(std::begin(spin), std::end(spin), std::begin(p_out.Spin));
    }
  ++output_index;
}


void ketju_Region::run_integration(int timebin, inttime_t Ti_Current)
{
  if(!this_task_is_in_compute_tasks())
    return;

  double cf_atime;
  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
  }

  const inttime_t t0             = Ti_Current;
  const inttime_t region_Ti_step = inttime_t(1) << timebin;
  const inttime_t t1             = t0 + region_Ti_step;

  const int output_timebin         = KetjuData.output_timebin;
  const inttime_t output_Ti_step = ((inttime_t)1 << output_timebin);

  const int num_output_points = setup_output_storage(region_Ti_step, output_Ti_step, Ti_Current);

  int output_index = 0;
  double timebase_interval = Dloga_interval_ti(Ti_Current);

  // output initial state
  if(Ti_Current == 0)
    {
      store_output_data_on_compute_root(output_index);
    }

  // Ti_step is set so that output is written at the end of each iteration,
  // unless we don't need to do further output on this integration step.
  const inttime_t Ti_step       = std::min(region_Ti_step, output_Ti_step);
  //const inttime_t Ti_step = region_Ti_step;
  const bool need_to_store_output = num_output_points > output_index;

  auto &syst           = *integrator_system;
  const int num_part   = syst.get_num_particles();

  for(inttime_t t_cur = t0; t_cur < t1; t_cur += Ti_step)
    {
      double scale_fac = 1., hubble = 0.;
      double dt, CoM_dt;
      // if(KetjuData.options.ComovingIntegrationOn)
      //   {
      //     // The CoM is propagated in gadget coordinates
      //     CoM_dt = get_exact_drift_factor(CP, t_cur, t_cur + Ti_step);
          

      //     // The subsystems are propagated in physical time
          
      //     dt = get_cosmic_timestep(t_cur, t_cur + Ti_step);
      //     // factors at the end of the step
      //     scale_fac = TimeBegin * std::exp((t_cur + Ti_step) * timebase_interval);

      //     hubble = hubble_function(CP, scale_fac);
          
      //   }
      // else
      //   {
      dt = CoM_dt = Ti_step * timebase_interval / hubble_function(CP, cf_atime);
      //  }

      integrator_system->run(dt);


      system_scale_fac = scale_fac;
      system_hubble    = hubble;
      
      // message(1, "DEBUG... ketju.c 1810: region %d, update CoM pos: dt %g CoM_Vel (%g|%g|%g) CoM_Pos (%g|%g|%g) \n",
      // region_index, CoM_dt, CoM_Vel[0], CoM_Vel[1], CoM_Vel[2], 
      // CoM_Pos[0], CoM_Pos[1], CoM_Pos[2]);

      for(int k = 0; k < 3; ++k)
        {
          //CoM_IntPos[k] += conv.pos_to_signedintpos(CoM_dt * CoM_Vel[k]);
          CoM_Pos[k] += CoM_dt * CoM_Vel[k];
         }
      
      if(need_to_store_output)
        {
          store_output_data_on_compute_root(output_index);
        }
    }

  store_merger_data_on_compute_root(t0, Ti_Current);

  // Some manual timer log manipulation to log the integrator internal timers as well.
  auto &perf          = integrator_system->get_perf();
  // double compute_time = perf.time_force + perf.time_gbsextr;
  // double comm_time    = perf.time_force_comm + perf.time_gbscomm;
  //TIMER_ADD(CPU_KETJU_INTEGRATOR_COMPUTE, compute_time);
  //TIMER_ADD(CPU_KETJU_INTEGRATOR_COMM, comm_time);
  //TIMER_ADD(CPU_KETJU_INTEGRATOR, perf.time_total - (compute_time + comm_time));
  //TIMER_ADD(CPU_KETJU_INTEGRATION, -perf.time_total);

  if(this_task_is_compute_root())
    {
      ketju_printf("integrated region %d with %d successful + %d rejected steps\n", region_index, perf.successful_steps,
                   perf.failed_steps);
    }
}



void ketju_Region::do_integration_step(int timebin, inttime_t Ti_Current, FILE * FdSingleBH,  FILE * FdKetjuRegion, int trace_bhid)
{

  double newatime = get_atime(Ti_Current);  
  double floattime = log(newatime);
  double timebase_interval = Dloga_interval_ti(Ti_Current);

  double timestep                    = timebase_interval * (inttime_t(1) << timebin) / hubble_function(CP, 1);
  double first_halfstep_kick_factor  = .5 * timestep;
  double second_halfstep_kick_factor = .5 * timestep;

  if(this_task_is_compute_root())
    {
      ketju_printf(
          "integrating region %d around BH ID %llu with %lu BH(s), %d particle(s) "
          "on tasks %d-%d on timebin %d (physical timestep %g)\n",
          get_region_index(), static_cast<unsigned long long>(get_bhs()[0].ID), get_bh_count(), get_total_particle_count(),
          get_compute_info().first_task_index, get_compute_info().final_task_index, timebin, timestep);


      for(int p : local_member_indices)
        {
          // ketju_printf("DEBUG...ketju 1893: region %d %d members: type %d ID %ld \n", get_region_index(), p, P[p].Type, P[p].ID);
          // message(0, "DEBUG...ketju 1894: %d region members: type %d ID %ld, \n", get_region_index(), P[p].Type, P[p].ID);
          // ketju_printf("DEBUG...ketju 1894: %d region members: type %d ID %ld, %d, \n", p, P[p].Type, P[p].ID, P[p].Type);
        }
      Region_info(FdKetjuRegion, floattime, timestep, timebin);

      
    }


  do_negative_halfstep_kick(first_halfstep_kick_factor, Ti_Current);
  //trace_singleblackhole(FdSingleBH, floattime, "KETJU_first_negative_kick", trace_bhid, PartManager);
  if(KetjuData.options.expand_tight_binaries_period_factor > 0)
    {
      expand_tight_binaries(timestep);
    }
  run_integration(timebin, Ti_Current);
  //trace_singleblackhole(FdSingleBH, floattime, "KETJU_run_integration", trace_bhid, PartManager);
  //message(0, "DEBUG...ketju 1905: %d region run_integration done. \n", get_region_index());
  do_negative_halfstep_kick(second_halfstep_kick_factor, Ti_Current);
  //trace_singleblackhole(FdSingleBH, floattime, "KETJU_second_negative_kick", trace_bhid, PartManager);
}


// difference between -1/r and the softened potential
double softened_inverse_r_correction(double h, double r)
{
  if(r > h)
    {
      return 0.;
    }
  else
    {
      const double h_inv = 1 / h;
      const double u     = r * h_inv;
      const double u2    = u * u;

      if(u < 0.5)
        {
          return -1. / r - h_inv * ((SOFTFAC4) + u2 * ((SOFTFAC5) + u2 * ((SOFTFAC6)*u + (SOFTFAC7))));
        }
      else
        {
          return -1. / r -
                 h_inv * ((SOFTFAC13) + (SOFTFAC14) / u + u2 * ((SOFTFAC1) + u * ((SOFTFAC15) + u * ((SOFTFAC16) + (SOFTFAC17)*u))));
        }
    }
}

std::vector<double> ketju_Region::get_potential_energy_corrections_on_compute_root(inttime_t Ti_Current) const
{
  if(!this_task_is_in_compute_tasks())
    return {};

  // Return values for the particle count before integration, so any merged particles just get zeros
  std::vector<double> potcorr(total_particle_count, 0);
  // Return just zeros if values not needed, easier than guarding every access
  // to the array.
  auto &syst         = *integrator_system;
  const int num_part = syst.get_num_particles();
  double cf_atime;

  int loop_start, loop_end, inner_loop_end;
  if(KetjuData.options.use_star_star_softening)
    {
      loop_start     = loop_scheduling_block_edge(num_part, compute_tasks.get_size(), compute_tasks.get_rank());
      loop_end       = loop_scheduling_block_edge(num_part, compute_tasks.get_size(), compute_tasks.get_rank() + 1);
      inner_loop_end = num_part;
    }
  else
    {
      // Need to only compute interactions involving BHs, star-star potential is correct already
      const int part_per_task = num_part / compute_tasks.get_size();
      const int remainder     = num_part % compute_tasks.get_size();
      loop_start              = part_per_task * compute_tasks.get_rank();
      loop_start += std::min(remainder, compute_tasks.get_rank());
      loop_end       = loop_start + part_per_task + (compute_tasks.get_rank() < remainder ? 1 : 0);
      inner_loop_end = syst.get_num_pn_particles();
    }

  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
  }


  const double h = KetjuData.options.star_smoothing * cf_atime;
  //const double h = All.ForceSoftening[get_star_softening_class()] * All.cf_atime;
  for(int i = loop_start; i < loop_end; ++i)
    {
      const int inner_loop_start = KetjuData.options.use_star_star_softening ? 0 : i + 1;
      for(int j = inner_loop_start; j < inner_loop_end; ++j)
        {
          if(i == j)  // only possible in the use_star_star_softening case
            continue;

          auto &&pi = syst[i];
          auto &&pj = syst[j];

          double r2 = 0;
          for(int k = 0; k < 3; ++k)
            {
              const double dr = pi.pos()[k] - pj.pos()[k];
              r2 += dr * dr;
            }
          const double corr = CP->GravInternal * softened_inverse_r_correction(h, std::sqrt(r2));
          potcorr[i] += pj.mass() * corr;
          potcorr[j] += pi.mass() * corr;
        }
    }

  if(this_task_is_compute_root())
    {
      MPI_Reduce(MPI_IN_PLACE, potcorr.data(), potcorr.size(), MPI_DOUBLE, MPI_SUM, compute_tasks.get_root(),
                 compute_tasks.get_comm());
    }
  else
    {
      MPI_Reduce(potcorr.data(), nullptr, potcorr.size(), MPI_DOUBLE, MPI_SUM, compute_tasks.get_root(), compute_tasks.get_comm());
      potcorr.clear();
    }


  //(void)softened_inverse_r_correction;  // silence unused warning


  return potcorr;
}

template <typename part_data>
void update_sim_particle_(part_data &p_gad, const mpi_particle &p, Cosmology * CP, inttime_t Ti_Current)
{
  
  //message(1, "DEBUG...ketju 1942: insdie the update_sim_particle_ \n");
  // if(P[p.Index].Type == 5){
  //     message(1, "DEBUG...ketju 1945 update for BH %ld vel(%g|%g|%g) xyz(%g|%g|%g) \n", P[p.Index].ID, 
  //     P[p.Index].Vel[0], P[p.Index].Vel[1], P[p.Index].Vel[2], P[p.Index].Pos[0], P[p.Index].Pos[1], P[p.Index].Pos[2]);
  // }
  p_gad.KetjuIntegrated = 1;

  if(P[p.Index].ID != p.ID)
    {
      endrun(1, "Ketju: particle indexing error!");
    }

  if(p.Mass == 0)  // merged away particle, no need for other fields
    return;

  inttime_t t0        = Ti_Current;
  inttime_t t1        = t0 + (inttime_t(1) << P[p.Index].TimeBinGravity);
  
  // double timebase_interval = Dloga_interval_ti(Ti_Current);
  // double test1 = timebase_interval * (t1 - t0);
  // message(0, "DEBUG...ketju.c 1954: ");

  //const double driftfac = KetjuData.options.ComovingIntegrationOn ? get_exact_drift_factor(CP, t0, t1) : (timebase_interval * (t1 - t0)/hubble_function(CP, 1));
  //const double driftfac = All.ComovingIntegrationOn ? Driftfac.get_drift_factor(t0, t1) : All.Timebase_interval * (t1 - t0);
  const double driftfac = get_exact_drift_factor(CP, t0, t1);
  double diff_pos[3];
  for(int i = 0; i< 3; ++i){
    diff_pos[i] = p.Pos[i] - P[p.Index].Pos[i];
  }
    
  // if(P[p.Index].Type == 5){
  //     message(1, "DEBUG...ketju 1972 update for part %ld driftfac %g, double timebase_interval %g, int t0: %d, int t1: %d \n", P[p.Index].ID, driftfac, timebase_interval, t0, t1);
  //     message(1, "DEBUG...ketju 1973 update for BH %ld ori pos xyz(%g|%g|%g) \n", P[p.Index].ID, P[p.Index].Pos[0], P[p.Index].Pos[1], P[p.Index].Pos[2]);
  //     message(1, "DEBUG...ketju 1974 update for BH %ld updated pos xyz(%g|%g|%g) \n", P[p.Index].ID, p.Pos[0], p.Pos[1], p.Pos[2]);
  //     message(1, "DEBUG...ketju 1975 update for BH %ld diff_pos xyz(%g|%g|%g) \n", P[p.Index].ID, diff_pos[0], diff_pos[1], diff_pos[2]);
  // }
  //Sp.nearest_image_intpos_to_pos(p.Pos, p_gad.Pos, diff_pos);

  for(int k = 0; k < 3; ++k)
    {
      // The velocity is set to a mean velocity so that after the drift step finishes the particle position is the one given by the
      // integrator within double precision rounding error. The positions aren't accurate to machine precision anyway, so such
      // small errors are no problem, and we avoid having to store the integrated position explicitly.
      P[p.Index].Vel[k] = diff_pos[k] / driftfac;
      // The final velocity is moved into the main Vel field after the leapfrog step is complete.

      p_gad.KetjuFinalVel[k] = p.Vel[k];
      
    }


    
    

  if(is_bh_type(p.Type) && p.Mass >= KetjuData.options.minimum_bh_mass)
    {
      std::copy(std::begin(p.Spin), std::end(p.Spin), std::begin(p_gad.Spin));
    }
  p_gad.KetjuPotentialEnergyCorrection = p.potential_energy_correction;
}


void update_sim_particle(const mpi_particle &p, Cosmology * CP, inttime_t Ti_Current)
{

  auto &p_gad = P[p.Index];

  if(P[p.Index].Type == 5){
    BHP(p.Index).Mass = p.Mass;
    }
  else{
    P[p.Index].Mass = p.Mass;
  }

  update_sim_particle_(p_gad, p, CP, Ti_Current);}

std::vector<mpi_particle> ketju_Region::get_particle_data_on_affected_root(inttime_t Ti_Current, MPI_Comm Comm) const
{
  std::vector<mpi_particle> full_particle_data;

  // Needs to be called here, since does collective comms
  std::vector<double> potcorrs = get_potential_energy_corrections_on_compute_root(Ti_Current);

  if(this_task_is_compute_root())
    {
      const int num_bhs       = integrator_system->get_num_pn_particles();
      const int Npart_current = integrator_system->get_num_particles();

      full_particle_data.resize(total_particle_count);
      for(int i = 0; i < total_particle_count; ++i)
        {
          auto &p_send = full_particle_data[i];
          auto &&p_int = (*integrator_system)[i];

          p_send.ID    = p_int.ID();
          p_send.Task  = p_int.Task();
          p_send.Index = p_int.Index();
          p_send.Type  = p_int.Type();

          if(i >= Npart_current)
            {
              // We're looking at a merged away particle, so set the particle mass to zero.
              // No need for other fields beyond this.
              // These particles get deleted during domain decomposition in the domain_rearrange_particle_sequence function.
              p_send.Mass = 0;
            }
          else
            {
              p_send.Mass = p_int.mass();
            }

          integrator_to_gadget_pos(p_int.pos(), p_send.Pos);
          
          integrator_to_gadget_vel(p_int.vel(), p_int.pos(), p_send.Vel);
          
          if(i < num_bhs)
            {
              auto &spin = p_int.spin();
              std::copy(std::begin(spin), std::end(spin), std::begin(p_send.Spin));
            }
          p_send.potential_energy_correction = potcorrs[i];
        }

      // Send data from compute root to affected root if needed
      if(!this_task_is_affected_root())
        {
          MPI_Send(full_particle_data.data(), full_particle_data.size(), mpi_particle::get_mpi_datatype(),
                   affected_tasks.get_root_sim(), region_index, Comm);
          full_particle_data.clear();
        }
    }
  else if(this_task_is_affected_root())
    {
      full_particle_data.resize(total_particle_count);
      MPI_Recv(full_particle_data.data(), full_particle_data.size(), mpi_particle::get_mpi_datatype(), compute_tasks.get_root_sim(),
               region_index, Comm, MPI_STATUS_IGNORE);
    }

  return full_particle_data;
}

void ketju_Region::update_sim_data(inttime_t Ti_Current, MPI_Comm Comm)
{
  std::vector<mpi_particle> full_particle_data = get_particle_data_on_affected_root(Ti_Current, Comm);

  if(!this_task_is_in_affected_tasks())
    return;

  std::vector<int> displs(affected_tasks.get_size());
  if(this_task_is_affected_root())
    {
      std::sort(full_particle_data.begin(), full_particle_data.end(),
                [](const mpi_particle &a, const mpi_particle &b) { return a.Task < b.Task; });

      std::partial_sum(particle_counts_on_affected_tasks.begin(), particle_counts_on_affected_tasks.end() - 1, displs.begin() + 1);
    }

  std::vector<mpi_particle> local_particle_data(particle_counts_on_affected_tasks[affected_tasks.get_rank()]);
  MPI_Scatterv(full_particle_data.data(), particle_counts_on_affected_tasks.data(), displs.data(), mpi_particle::get_mpi_datatype(),
               local_particle_data.data(), local_particle_data.size(), mpi_particle::get_mpi_datatype(), affected_tasks.get_root(),
               affected_tasks.get_comm());

  for(auto &p : local_particle_data)
    {
      update_sim_particle(p, CP, Ti_Current);
    }
}

//////////////////

//////////////////

class TimestepRegion
{
  int timebin = -1;
  std::set<int> limiting_particle_indices;
  std::set<int> limited_particle_indices;

 public:
  TimestepRegion(std::set<int> &&limited_particle_indices, std::set<int> &&limiting_particle_indices)
      : limiting_particle_indices{std::move(limiting_particle_indices)},
       limited_particle_indices{std::move(limited_particle_indices)}
  {
  }

  const std::set<int> &get_limited_particle_indices() const { return limited_particle_indices; }
  const std::set<int> &get_limiting_particle_indices() const { return limiting_particle_indices; }

  void decrease_timebin(int new_timebin)
  {
    if(new_timebin < timebin || timebin == -1)
      {
        timebin = new_timebin;
      }
  }

  int get_timebin() const { return timebin; }
};

///////////////////////////

// Provides storage for the Regions and methods that the interface
// functions call into
class RegionManager
{
  std::vector<ketju_Region> regions;
  std::vector<TimestepRegion> timestep_regions;
  std::unordered_set<int> all_limited_particle_indices;

  std::unordered_map<MyIDType, double> region_previous_cost;

  int num_sequential_computations;
  int GasEnabled;
  Cosmology CP;
  double region_cost_estimate(int region_index) const;
  using parallel_task_allocation = std::unordered_map<int, region_compute_info>;
  //< maps region index to assigned task data
  parallel_task_allocation allocate_parallel_run(const std::vector<int> &region_indices, int run_index) const;
  std::vector<parallel_task_allocation> allocate_sequential_runs(int num_runs) const;
  double estimate_total_time(const std::vector<parallel_task_allocation> &alloc) const;

 public:

  void RM_init(Cosmology * CP_all, const double TimeBegin, const struct UnitSystem units)
  {
    CP.CMBTemperature = CP_all->CMBTemperature;
    CP.RadiationOn = CP_all->RadiationOn;
    CP.Omega0 = CP_all->Omega0;
    CP.OmegaBaryon = CP_all->OmegaBaryon;
    CP.OmegaLambda = CP_all->OmegaLambda;
    CP.Omega_fld = CP_all->Omega_fld;
    CP.w0_fld = CP_all->w0_fld;
    CP.wa_fld = CP_all->wa_fld;
    CP.Omega_ur = CP_all->Omega_ur;
    CP.HubbleParam = CP_all->HubbleParam;
    CP.NonPeriodic = CP_all->NonPeriodic;
    CP.ComovingIntegrationOn = CP_all->ComovingIntegrationOn;
    CP.Redshift = CP_all->Redshift;
    CP.MassiveNuLinRespOn = CP_all->MassiveNuLinRespOn;
    CP.HybridNeutrinosOn = CP_all->HybridNeutrinosOn;
    CP.MNu[0] = CP_all->MNu[0];
    CP.MNu[1] = CP_all->MNu[1];
    CP.MNu[2] = CP_all->MNu[2];
    CP.HybridVcrit = CP_all->HybridVcrit;
    CP.HybridNuPartTime = CP_all->HybridNuPartTime;


    init_cosmology(&CP, TimeBegin, units);
    GasEnabled = SlotsManager->info[0].enabled;
  }
  const std::vector<ketju_Region> &get_regions() const { return regions; }
  const std::vector<TimestepRegion> &get_timestep_regions() const { return timestep_regions; }
  
  void find_regions(const ActiveParticles * act, inttime_t Ti_Current, const struct UnitSystem units);
  void find_timestep_regions(const ActiveParticles * act, inttime_t Ti_Current);
  void set_region_timebins(int FastParticleType, double asmth, DriftKickTimes * times, MPI_Comm Comm);
  int is_particle_timestep_limited(int target) const;
  int set_limited_timebins(int min_timebin, int push_down_flag);
  void set_up_regions_for_integration(inttime_t Ti_Current, const struct UnitSystem units);
  void allocate_compute_tasks();
  void integrate_regions(inttime_t Ti_Current, FILE * FdSingleBH, FILE * FdKetjuRegion, int trace_bhid);
  std::pair<std::vector<std::vector<mpi_particle>>, std::vector<int>> comm_and_get_output_data_on_root(MPI_Comm Comm) const;
  std::vector<bh_merger_data> comm_and_get_merger_data_on_root(MPI_Comm Comm) const;
  void update_sim_data(inttime_t Ti_Current, MPI_Comm Comm);
  void set_output_timebin(DriftKickTimes * times, MPI_Comm Comm) const;

  void clear_data()
  {
    regions.clear();
    timestep_regions.clear();
    all_limited_particle_indices.clear();
  }
};


void bh_spin_sanity_check(int target, UnitSystem units, Cosmology * CP)
{
  const double chi =
      vector_norm(P[target].Spin) / (CP->GravInternal * std::pow(BHP(target).Mass, 2)) * (LIGHTCGS / units.UnitVelocity_in_cm_per_s);

  if(!(chi <= 1))
    {
      char msg[200];
      std::snprintf(msg, sizeof msg, "Ketju: BH ID %llu has unphysical spin |chi| = %g, should have |chi| < 1.\n",
                    static_cast<unsigned long long>(P[target].ID), chi);
      endrun(1, msg);
    }
}


std::vector<mpi_particle> find_bhs(const ActiveParticles * act, MPI_Comm Comm, UnitSystem units, Cosmology * CP)
{
  std::vector<mpi_particle> local_bhs;
  int ThisTask;

  //#pragma omp parallel for
  for(int i = 0; i < act->NumActiveParticle; ++i)
    {
      const int target = act->ActiveParticle ? act->ActiveParticle[i] : i;;

      if((P[target].Type==5))
        {
          const double mass = BHP(target).Mass;
          if(mass == 0 || mass < KetjuData.options.minimum_bh_mass)
            continue;

          
          bh_spin_sanity_check(target, units, CP);

          MPI_Comm_rank(Comm, &ThisTask);

          local_bhs.push_back(mpi_particle::from_gadget_particle_index(ThisTask, target));
          // message(1, "DEBUG...ketju 2261: find a new blackhole: %d (task %d)\n", P[target].ID, ThisTask);
        }
    }

  int NTask;
  MPI_Comm_size(Comm, &NTask);

  std::vector<int> bh_counts(NTask);
  const int bh_count = local_bhs.size();
  MPI_Allgather(&bh_count, 1, MPI_INT, bh_counts.data(), 1, MPI_INT, Comm);

  std::vector<int> displs(NTask);
  std::vector<int> recvcounts(NTask);
  int num_bhs = 0;
  for(int i = 0; i < NTask; ++i)
    {
      displs[i]     = num_bhs;
      recvcounts[i] = bh_counts[i];
      num_bhs += bh_counts[i];
      
    }

  std::vector<mpi_particle> bhs(num_bhs);
  MPI_Allgatherv(local_bhs.data(), bh_count, mpi_particle::get_mpi_datatype(), bhs.data(), recvcounts.data(), displs.data(),
                 mpi_particle::get_mpi_datatype(), Comm);
  
  return bhs;
}

// Center is a type that has an Pos field
template <typename Center>
std::set<int> find_local_particles(const ActiveParticles * act, const std::vector<Center> &centers, const double radius,
                                   const unsigned int type_flags, int * star_num)
{
  // Do a simple linear search through the particles, since the particle tree structures
  // are constructed and freed in the functions where they are used, so that we don't have
  // access to one for free here.
  // Constructing one would likely be at least as expensive as doing this simple search,
  // since we only deal with a few center points at any time.
  std::set<int> particle_indices;
  int part_star_count = 0;
  //#pragma omp parallel for
  for(int i = 0; i < act->NumActiveParticle; ++i) 
    {
      const int target = act->ActiveParticle ? act->ActiveParticle[i] : i;
      const auto type  = P[target].Type;
      if(!check_type_flag(type, type_flags) || P[target].Mass <= 0)
        continue;
      


      for(const auto &c : centers)
        {
          double offset[3];
      
          for(int i = 0; i< 3; ++i){
            offset[i] = P[target].Pos[i] - c.Pos[i];
          }
          

          if(vector_norm(offset) <= radius)
            {
              particle_indices.insert(target);

              if(check_type_flag(type, star_particle_flags)){
                part_star_count += 1;
                //message(1, "DEBUG...ketju.c 2318 update part_star_count %d (part type %d, id star_particle_flags %d)\n", part_star_count, type, star_particle_flags);
              }
              //message(1, "DEBUG...ketju.c 2293 region with offset pos (xyz=<%g|%g|%g>) get one particle: type %d, ID %ld, radius %g\n",
              //offset[0], offset[1], offset[2], P[target].Type, P[target].ID, radius);
              break;   // each active particle assgined at most once. 
            }
        }
    }
  //message(1, "DEBUG...ketju.c 2339 part_star_count %d \n", part_star_count);
  *star_num = part_star_count;
  //message(1, "DEBUG...ketju.c 2341 star_num %d \n", *star_num);
  return particle_indices;
}


// The regions and timestep regions consist of unions of spheres with BHs at the
// centers, and finding the regions consists mainly of combining overlapping
// regions.

// Check for overlap of two regions defined by center points and distance
// required for overlap. For regions consisting of spheres
// overlap_distance=2*sphere_radius, but for handling timestep regions we need a
// slightly more general definition.
// Center is a type with an IntPos field, e.g. mpi_particle.
template <typename Center>
int regions_overlap(const std::vector<Center> &reg1_centers, const std::vector<Center> &reg2_centers, double overlap_distance)
{
  for(const Center &c1 : reg1_centers)
    {
      for(const Center &c2 : reg2_centers)
        {
          double offset[3];
  
          for(int i = 0; i< 3; ++i){
            offset[i] = c1.Pos[i] - c2.Pos[i];
          }
          //conv.nearest_image_intpos_to_pos(c1.Pos, c2.Pos, offset);

          if(vector_norm(offset) < overlap_distance)
            return 1;
        }
    }
  return 0;
}

// Do the actual combining.
// Needs to handle input regions with multiple centers for efficiently combining timestep regions where the regions we start with may
// already contain multiple BHs.
template <typename Center>
std::vector<std::vector<Center>> combine_overlapping_sphere_union_regions(const std::vector<std::vector<Center>> &region_centers,
                                                                          double overlap_distance)
{
  std::vector<std::vector<Center>> combined_region_centers;
  for(auto &this_region_centers : region_centers)
    {
      std::vector<decltype(combined_region_centers.begin())> regions_this_region_overlaps_iterators;
      for(auto it = combined_region_centers.begin(); it != combined_region_centers.end(); ++it)
        {
          if(regions_overlap(this_region_centers, *it, overlap_distance))
            {
              regions_this_region_overlaps_iterators.push_back(it);
            }
        }

      if(regions_this_region_overlaps_iterators.empty())
        {
          combined_region_centers.emplace_back(this_region_centers);
        }
      else
        {
          // Merge all the regions this region overlaps.
          // Iterate in reverse to avoid iterator invalidation with erase
          // message(1, "DEBUG...ketju 2360 merging some regions!! \n");
          auto &main_region       = *regions_this_region_overlaps_iterators.front();
          const auto rit_to_first = regions_this_region_overlaps_iterators.rend() - 1;
          for(auto rit = regions_this_region_overlaps_iterators.rbegin(); rit != rit_to_first; ++rit)
            {
              auto &reg_bhs = **rit;  // rit is an iterator to a vector of iterators
              main_region.insert(main_region.end(), reg_bhs.begin(), reg_bhs.end());
              combined_region_centers.erase(*rit);
            }
          main_region.insert(main_region.end(), this_region_centers.begin(), this_region_centers.end());
        }
    }
  return combined_region_centers;
}

// Overload for when there's only one center per initial region
template <typename Center>
std::vector<std::vector<Center>> combine_overlapping_sphere_union_regions(const std::vector<Center> &region_centers,
                                                                          double overlap_distance)
{
  std::vector<std::vector<Center>> wrapped_centers;
  std::transform(region_centers.begin(), region_centers.end(), std::back_inserter(wrapped_centers),
                 [](const Center &c) { return std::vector<Center>{c}; });
  return combine_overlapping_sphere_union_regions(wrapped_centers, overlap_distance);
}

void RegionManager::find_regions(const ActiveParticles * act, inttime_t Ti_Current, UnitSystem units)
{
  
  double cf_atime, cf_ainv;
  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
    cf_ainv = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
    cf_ainv = 1 / cf_atime;
  }
  //double cf_hubble_a = get_cf_hubble_a(Ti_Current, &CP);
  double cf_hubble_a = hubble_function(&CP, cf_atime);
  const double region_radius = KetjuData.options.region_physical_radius * cf_ainv;
  auto region_bh_lists = combine_overlapping_sphere_union_regions(find_bhs(act, MPI_COMM_WORLD, units, &CP), 2 * region_radius);
  int region_index = 0;
  int region_star_num = 0;
  for(auto &reg_bhs : region_bh_lists)
    {
      // Sort BHs by mass (and ID) to make the most massive one the main BH used for identifying the region
      std::sort(reg_bhs.begin(), reg_bhs.end(), [](const mpi_particle &bh1, const mpi_particle &bh2) {
        if(bh1.Mass == bh2.Mass)
          return bh1.ID < bh2.ID;  // ascending ID for equal masses

        return bh1.Mass > bh2.Mass;  // descending by mass
      });


      if(KetjuData.options.IntegrateBinaryRegion){
        if(reg_bhs.size() < 2){ 
          message(1, "KETJU ignore the region around %llu since it only has %d bh\n", static_cast<unsigned long long>(reg_bhs[0].ID), reg_bhs.size());
          continue;
          }
        }

      auto local_particles = find_local_particles(act, reg_bhs, region_radius, KetjuData.options.region_member_flags, &region_star_num);
      //message(0, "DEBUG... KETJU 2463: find %d regions: region_star_num %d \n", region_index, region_star_num);
      regions.emplace_back(std::move(reg_bhs), std::move(local_particles), &CP,  cf_hubble_a, region_index++, cf_atime, region_star_num);
      //message(0, "DEBIG... KETJU 2465: find %d regions has %d bh and %d part region_star_num %d \n", region_index, reg_bhs.size(), local_particles.size(), region_star_num);
    }


  for(auto &reg : regions)
    {
      reg.find_affected_tasks_and_particle_counts(MPI_COMM_WORLD);
      //message(0, "DEBUG...ketju 2470 , regions has id: %d \n",reg.get_region_index());
    }
  // message(0, "KETJU....2471: find %d regions ... \n", regions.size());
}


void RegionManager::find_timestep_regions(const ActiveParticles * act, inttime_t Ti_Current)
{

  double cf_atime, cf_ainv;
  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
    cf_ainv = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(Ti_Current));
    cf_ainv = 1 / cf_atime;
  }

  int region_star_count; 
  const double region_radius = KetjuData.options.region_physical_radius * cf_ainv;

  // All particles that are not in the regularized regions but within timestep_limiting_radius are used for setting the limited
  // timesteps. The limited timestep is the minimum used within this region.
  const double timestep_limiting_radius = region_radius * (KETJU_TIMESTEP_LIMITING_RADIUS_FACTOR);
  // Limited timesteps are set for particles that can go into the regions but within
  // a larger radius, to ensure that they are active when they enter within region_radius.
  const double timestep_limited_radius = 2 * region_radius;

  struct region_index_bh_pos
  {
    int region_index;
    double Pos[3];
  };
  std::vector<std::vector<region_index_bh_pos>> region_indices_bh_pos;
  region_indices_bh_pos.reserve(regions.size());
  for(const auto &reg : regions)
    {
      region_indices_bh_pos.emplace_back();
      auto &vec = region_indices_bh_pos.back();

      for(const auto &bh : reg.get_bhs())
        {
          vec.push_back({reg.get_region_index(), {bh.Pos[0], bh.Pos[1], bh.Pos[2]}});
        }
    }

  // Timestep regions merged if the limited radius of one region falls inside
  // the limiting radius of another, since then the timesteps must be the same.
  auto combined_region_indices_bh_pos =
      combine_overlapping_sphere_union_regions(region_indices_bh_pos, timestep_limiting_radius + timestep_limited_radius);

  for(const auto &tsr : combined_region_indices_bh_pos)
    {
      // Limiting particles are the particles within the
      // timestep_limiting_radius but outside the actual regularized region.
      auto limiting_particles = find_local_particles(act, tsr, timestep_limiting_radius, all_particles_flags, &region_star_count);

      for(const auto &ind_pos : tsr)
        {
          for(auto j : regions[ind_pos.region_index].get_local_member_indices())
            {
              limiting_particles.erase(j);  // remove bh and stellar particles in the limiting particles group 
            }
        }

      timestep_regions.emplace_back(find_local_particles(act, tsr, timestep_limited_radius, KetjuData.options.region_member_flags, &region_star_count),
                                    std::move(limiting_particles));

      for(const auto &ind_pos : tsr)
        {
          regions[ind_pos.region_index].set_timestep_region_index(timestep_regions.size() - 1);
        }

      const auto &limited_inds = timestep_regions.back().get_limited_particle_indices();
      all_limited_particle_indices.insert(limited_inds.begin(), limited_inds.end());
    }
}

void RegionManager::set_region_timebins(int FastParticleType, double asmth, DriftKickTimes * times, MPI_Comm Comm)
{
  double cf_atime;
  if(!KetjuData.options.ComovingIntegrationOn){
    cf_atime = 1;
  }
  else{
    cf_atime = exp(loga_from_ti(times->Ti_Current));
  }
  //const double cf_hubble_a = hubble_function(&CP, cf_atime);
  for(auto &reg : regions)
    {
      //cf_hubble_a = get_cf_hubble_a(times->Ti_Current, &CP);

      int timemax = reg.get_max_timebin(times->Ti_Current, Comm);
      //message(0, "DEBUG... ketju 2638 set_region_timebin, region_idx:%d region_timemax %d \n", reg.get_region_index(), timemax);
      timestep_regions[reg.get_timestep_region_index()].decrease_timebin(reg.get_max_timebin(times->Ti_Current, Comm));

    }


  int timestep_reg_idx=0;
  for(auto &treg : timestep_regions)
    {
      
      inttime_t ti_step = inttime_t(1) << treg.get_timebin();
      for(int i : treg.get_limiting_particle_indices())
        {
          const int isPM = is_PM_timestep(times);
          inttime_t dti_max = times->PM_length;
          if(isPM)
              dti_max = get_PM_timestep_ti(times, cf_atime, &CP, FastParticleType, asmth);
        
          // cf_hubble_a = get_cf_hubble_a(times->Ti_Current, &CP);
          const double cf_hubble_a = hubble_function(&CP, cf_atime);
          enum TimeStepType titype = TI_ACCEL;
          double dloga = get_timestep_gravity_dloga(i, P[i].FullTreeGravAccel, cf_atime, cf_hubble_a);
          if(GasEnabled){
            double dloga_hydro = get_timestep_hydro_dloga(i, times->Ti_Current, cf_atime, cf_hubble_a, &titype);
            if(dloga_hydro < dloga) {
                dloga = dloga_hydro;
            }
          }
          inttime_t ts =  convert_timestep_to_ti(dloga, i, dti_max, times->Ti_Current, titype);

          if(ts < ti_step && ts > 0)
            {
              ti_step = ts;
            }

        }

      int timebin;
      
      
      timebins_get_bin_and_do_validity_checks(ti_step, times->Ti_Current, &timebin, 0);
      MPI_Allreduce(MPI_IN_PLACE, &timebin, 1, MPI_INT, MPI_MIN, Comm);
      treg.decrease_timebin(timebin);
    //message(1, "DEBUG... ketju 2580 set_tregbin idx: %d timebin %d \n", timestep_reg_idx, treg.get_timebin());
    timestep_reg_idx ++;
    }

}

void RegionManager::set_output_timebin(DriftKickTimes * times, MPI_Comm Comm) const
{
  // If timesteps for some BHs have already been written into the future, we
  // shouldn't adjust the output timebin before that time has been reached
    if(times->Ti_Current< KetjuData.final_written_Ti)
        return;
    if(regions.empty())
        return;

    int max_timebin = 0;
    for(auto &tr : timestep_regions)
        {
        if(tr.get_timebin() > max_timebin)
            {
            max_timebin = tr.get_timebin();
            }
        }

    double timebase_interval = Dloga_interval_ti(times->Ti_Current);
    double max_phys_dt = (inttime_t(1) << max_timebin) * timebase_interval;

    const double bin_diff        = log2(max_phys_dt / KetjuData.options.output_time_interval);
    KetjuData.output_timebin = max_timebin - bin_diff;
    if(KetjuData.output_timebin < 0)
    {
    // Negative output timebins are a hassle and rarely needed, so limit to non-negative.
    KetjuData.output_timebin = 0;
    }

    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    if(ThisTask == 0)
    {
        ketju_debug_printf("output_timebin = %d\n", KetjuData.output_timebin);
    }
}

int RegionManager::is_particle_timestep_limited(int index) const
{
  if(all_limited_particle_indices.find(index) != all_limited_particle_indices.end())
    return 1;
  else
    return 0;
}

int RegionManager::set_limited_timebins(int min_timebin, int push_down_flag)
{
  // This method works for both timestepping modes.
  // If HIERARCHICAL_GRAVITY is not set, min_timebin and push_down_flag are always 0,
  // and this method is only called once per step.
  // Otherwise this is called each time the hierarchical scheme moves to a lower timebin,
  // and we need to account for the possible global push down of the timebin.

  if(push_down_flag)
    {
      // Called from HIERARCHICAL_GRAVITY loop, push down occurs so set the timebins to at most min_timebin
      for(auto &treg : timestep_regions)
        treg.decrease_timebin(min_timebin);
    }

  // Set the particle timebins to the region bin if possible,
  // but no lower than min_timebin so that HIERARCHICAL_GRAVITY works correctly.
  int max_timebin = 0;
  for(auto &treg : timestep_regions)
    {
      const int timebin = std::max(min_timebin, treg.get_timebin());
      max_timebin       = std::max(max_timebin, timebin);

      for(int target : treg.get_limited_particle_indices())
        {
          //Sp.TimeBinsGravity.timebin_move_particle(target, P[target].TimeBinGravity, timebin);
          P[target].TimeBinGravity = timebin;
        }
    }
  return max_timebin;
}

double RegionManager::region_cost_estimate(int region_index) const
{
  auto elem = region_previous_cost.find(regions[region_index].get_bhs()[0].ID);
  if(elem == region_previous_cost.end())
    {
      // The BH ID didn't have an entry in the map, so base on particle count.
      // The prefactor scales the value to assume around 20 steps which should
      // generally be of the right magnitude, and even if it's very wrong,
      // we only get poor performance for a single step.
      // No need to include timestep factors here.
      return 20. * std::pow(regions[region_index].get_total_particle_count(), 2);
    }
  return elem->second;
}

using index_cost_pair           = std::pair<int, double>;
auto cost_comp_func             = [](const index_cost_pair &a, const index_cost_pair &b) { return a.second < b.second; };
using index_cost_priority_queue = std::priority_queue<index_cost_pair, std::vector<index_cost_pair>, decltype(cost_comp_func)>;

RegionManager::parallel_task_allocation RegionManager::allocate_parallel_run(const std::vector<int> &region_indices,
                                                                             int run_index) const
{
  int NTask;
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  const int N = region_indices.size();
  std::vector<int> task_counts(N, 0);

  std::vector<double> reg_costs;
  index_cost_priority_queue pq(cost_comp_func);
  for(int i = 0; i < N; ++i)
    {
      reg_costs.push_back(region_cost_estimate(region_indices[i]));
      pq.push({i, std::numeric_limits<double>::max()});  // max initial priority ensures >=1 tasks per region
    }

  for(int n = 0; n < NTask && !pq.empty(); ++n)
    {
      int i = pq.top().first;
      pq.pop();
      int reg = region_indices[i];

      task_counts[i] += 1;
      // The integrator can make efficient use of  N_particles / ~10 tasks,
      // at this point the scaling flattens as most of the time gets spent on
      // comms. The exact number depends on the machine, so the exact number
      // is user definable.
      // Adding more tasks than required can be worse for performance due
      // increased comms.
      if(KetjuData.options.minimum_particles_per_task <= double(regions[reg].get_total_particle_count()) / (task_counts[i] + 1))
        {
          pq.push({i, reg_costs[i] / task_counts[i]});
        }
    }

  parallel_task_allocation alloc;
  int allocated = 0;
  for(int i = 0; i < N; ++i)
    {
      alloc[region_indices[i]] = {run_index, allocated, allocated + task_counts[i] - 1};
      allocated += task_counts[i];
    }

  return alloc;
}

std::vector<RegionManager::parallel_task_allocation> RegionManager::allocate_sequential_runs(int num_runs) const
{
  int NTask;
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  
  index_cost_priority_queue region_pq(cost_comp_func);
  index_cost_priority_queue run_pq(cost_comp_func);

  std::vector<std::vector<int>> regions_in_run(num_runs);
  const int num_regions = regions.size();
  for(int i = 0; i < num_regions; ++i)
    {
      region_pq.push({i, region_cost_estimate(i)});
    }
  for(int i = 0; i < num_runs; ++i)
    {
      run_pq.push({i, 0.});
    }

  while(!region_pq.empty())
    {
      int reg         = region_pq.top().first;
      int region_cost = region_pq.top().second;
      region_pq.pop();
      int run      = run_pq.top().first;
      int run_cost = run_pq.top().second;
      run_pq.pop();
      regions_in_run[run].push_back(reg);
      // Prioritize the runs with the least cost assigned,
      // and ensure a run doesn't get overfilled.
      if(regions_in_run[run].size() < NTask)
        run_pq.push({run, run_cost - region_cost});
    }

  std::vector<parallel_task_allocation> alloc;
  for(int i = 0; i < num_runs; ++i)
    {
      alloc.emplace_back(allocate_parallel_run(regions_in_run[i], i));
    }
  return alloc;
}

double RegionManager::estimate_total_time(const std::vector<parallel_task_allocation> &alloc) const
{
  const int particles_per_task_scaling_limit = 7;  // Estimate based on the data in the integrator code paper

  double tot_t = 0;

  for(auto &par_alloc : alloc)
    {
      double max_t = 0;
      for(auto &kv : par_alloc)
        {
          const int reg  = kv.first;
          int task_count = kv.second.final_task_index - kv.second.first_task_index + 1;
          if(task_count > regions[reg].get_total_particle_count() / particles_per_task_scaling_limit)
            {
              // The parallelization efficiency drops dramatically around
              // here, so account for it in the estimation.
              task_count = regions[reg].get_total_particle_count() / particles_per_task_scaling_limit;
              if(task_count < 1)
                task_count = 1;
            }
          const double t = region_cost_estimate(reg) / task_count;
          if(t > max_t)
            {
              max_t = t;
            }
        }

      tot_t += max_t;
    }
  return tot_t;
}

void RegionManager::allocate_compute_tasks()
{
  // Determine the estimated best division of tasks and number of
  // sequential runs by testing out the possibilities and estimating the times
  // they will take.
  const int num_regions = regions.size();
  int NTask;
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);
  int ThisTask;

  // We can't have more regions than tasks computed in parallel
  const int min_num_runs = num_regions / NTask + (num_regions % NTask == 0 ? 0 : 1);

  double min_time = DBL_MAX;

  std::vector<parallel_task_allocation> best_alloc;

  for(int num_runs = min_num_runs; num_runs <= num_regions; ++num_runs)
    {
      std::vector<parallel_task_allocation> current_alloc = allocate_sequential_runs(num_runs);

      const double t = estimate_total_time(current_alloc);
      if(t < min_time)
        {
          min_time = t;
          std::swap(best_alloc, current_alloc);
        }
      else
        {
          // If the estimated performance didn't increase with an additional
          // sequential run, adding another isn't likely to help either.
          break;
        }
    }

  num_sequential_computations = best_alloc.size();

  for(auto &par_alloc : best_alloc)
    {
      for(auto &kv : par_alloc)
        {
          regions[kv.first].set_compute_info(kv.second);
          MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
          if(ThisTask == 0)
            {
              ketju_debug_printf("region %d got tasks %d-%d seq pos %d cost %g\n", kv.first, kv.second.first_task_index,
                                 kv.second.final_task_index, kv.second.compute_sequence_position, region_cost_estimate(kv.first));
            }
        }
    }
}

void RegionManager::set_up_regions_for_integration(inttime_t Ti_Current, const struct UnitSystem units)
{
  //TIMER_START(CPU_KETJU_CONSTRUCTION);
  allocate_compute_tasks();

  for(auto &reg : regions)
    {
      reg.set_up_compute_comms(MPI_COMM_WORLD);
      reg.set_up_integrator(MPI_COMM_WORLD, Ti_Current, units);
    }
  //TIMER_STOP(CPU_KETJU_CONSTRUCTION);
}

void RegionManager::integrate_regions(inttime_t Ti_Current, FILE * FdSingleBH, FILE * FdKetjuRegion, int trace_bhid)
{

  //TIMER_START(CPU_KETJU_INTEGRATION);
  for(int i = 0; i < num_sequential_computations; ++i)
    {
      for(auto &reg : regions)
        {
          if(reg.get_compute_sequence_position() != i)
            continue;
          if(!reg.this_task_is_in_compute_tasks())
            continue;

          // if there is only one BH part in this ketju region, skip it. 
          // if(KetjuData.options.IntegrateBinaryRegion){
          //   if(reg.get_bh_count() < 2){
          //     message(1, "DEBUG...ketju 2853: skip reg_index %d since it only has %d bh\n", reg.get_region_index(), reg.get_bh_count());
          //     continue;
          //   }
          //   }

          const int timebin = timestep_regions[reg.get_timestep_region_index()].get_timebin();
          reg.do_integration_step(timebin, Ti_Current, FdSingleBH, FdKetjuRegion, trace_bhid);
        }
    }
  //TIMER_STOP(CPU_KETJU_INTEGRATION);

  // Store the integration cost for load-balancing
  std::vector<double> region_costs;
  for(auto &reg : regions)
    {
      region_costs.push_back(
          reg.get_normalized_integration_cost_from_compute_root(timestep_regions[reg.get_timestep_region_index()].get_timebin()));
    }
  // Sum the costs so that all the tasks agree on the value.
  MPI_Allreduce(MPI_IN_PLACE, region_costs.data(), region_costs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const int num_regions = regions.size();
  for(int i = 0; i < num_regions; ++i)
    {
      region_previous_cost[regions[i].get_bhs()[0].ID] = region_costs[i];
    }

    
}

std::pair<std::vector<std::vector<mpi_particle>>, std::vector<int>> RegionManager::comm_and_get_output_data_on_root(MPI_Comm Comm) const
{
  // first communicate how many data points each region has
  std::vector<MPI_Request> send_count_reqs, recv_count_reqs;
  std::vector<int> data_counts, send_data_counts;

  int ThisTask;
  MPI_Comm_rank(Comm, &ThisTask);

  const int num_regs           = regions.size();
  const bool this_task_is_root = ThisTask == 0;

  if(this_task_is_root)
    {
      data_counts.resize(num_regs);
      recv_count_reqs.resize(num_regs);
    }

  for(int i = 0; i < num_regs; ++i)
    {
      auto &reg = regions[i];
      if(reg.this_task_is_compute_root())
        {
          send_data_counts.push_back(reg.get_output_bh_data().size());
          send_count_reqs.emplace_back();
          MPI_Isend(&send_data_counts.back(), 1, MPI_INT, 0, i, Comm, &send_count_reqs.back());
        }

      if(this_task_is_root)
        {
          MPI_Irecv(&data_counts[i], 1, MPI_INT, reg.get_compute_root_sim(), i, Comm, &recv_count_reqs[i]);
        }
    }

  if(!send_count_reqs.empty())
    {
      MPI_Waitall(send_count_reqs.size(), send_count_reqs.data(), MPI_STATUSES_IGNORE);
    }

  if(!recv_count_reqs.empty())
    {
      MPI_Waitall(recv_count_reqs.size(), recv_count_reqs.data(), MPI_STATUSES_IGNORE);
    }

  // Then the actual data
  std::vector<std::vector<mpi_particle>> recv_data;
  std::vector<int> writable_reg_indices;
  std::vector<MPI_Request> send_data_reqs, recv_data_reqs;
  for(int i = 0; i < num_regs; ++i)
    {
      auto &reg = regions[i];
      if(this_task_is_root)
        {
          if(data_counts[i] == 0)
            continue;

          recv_data.emplace_back(data_counts[i]);
          recv_data_reqs.emplace_back();
          MPI_Irecv(recv_data.back().data(), data_counts[i], mpi_particle::get_mpi_datatype(), reg.get_compute_root_sim(), i,
                    Comm, &recv_data_reqs.back());
          writable_reg_indices.push_back(i);
        }

      if(reg.this_task_is_compute_root())
        {
          auto &data     = reg.get_output_bh_data();
          const int size = data.size();

          if(size == 0)
            continue;

          send_data_reqs.emplace_back();
          MPI_Isend(data.data(), size, mpi_particle::get_mpi_datatype(), 0, i, Comm, &send_data_reqs.back());
        }
    }

  if(!send_data_reqs.empty())
    {
      MPI_Waitall(send_data_reqs.size(), send_data_reqs.data(), MPI_STATUSES_IGNORE);
    }

  if(!recv_data_reqs.empty())
    {
      MPI_Waitall(recv_data_reqs.size(), recv_data_reqs.data(), MPI_STATUSES_IGNORE);
    }

  return std::make_pair(std::move(recv_data), std::move(writable_reg_indices));
}

std::vector<bh_merger_data> RegionManager::comm_and_get_merger_data_on_root(MPI_Comm Comm) const
{
  std::vector<MPI_Request> send_count_reqs, recv_count_reqs;
  std::vector<int> data_counts, send_data_counts;

  const int num_regs           = regions.size();
  
  int ThisTask;
  MPI_Comm_rank(Comm, &ThisTask);
  const bool this_task_is_root = ThisTask == 0;

  if(this_task_is_root)
    {
      data_counts.resize(num_regs);
      recv_count_reqs.resize(num_regs);
    }

  for(int i = 0; i < num_regs; ++i)
    {
      auto &reg = regions[i];
      if(reg.this_task_is_compute_root())
        {
          send_data_counts.push_back(reg.get_output_merger_data().size());
          send_count_reqs.emplace_back();
          MPI_Isend(&send_data_counts.back(), 1, MPI_INT, 0, i, Comm, &send_count_reqs.back());
        }

      if(this_task_is_root)
        {
          MPI_Irecv(&data_counts[i], 1, MPI_INT, reg.get_compute_root_sim(), i, Comm, &recv_count_reqs[i]);
        }
    }

  if(!send_count_reqs.empty())
    {
      MPI_Waitall(send_count_reqs.size(), send_count_reqs.data(), MPI_STATUSES_IGNORE);
    }

  if(!recv_count_reqs.empty())
    {
      MPI_Waitall(recv_count_reqs.size(), recv_count_reqs.data(), MPI_STATUSES_IGNORE);
    }

  const int total_count = std::accumulate(data_counts.begin(), data_counts.end(), 0);
  std::vector<bh_merger_data> recv_data(total_count);
  std::vector<MPI_Request> send_data_reqs, recv_data_reqs;

  int offset = 0;
  for(int i = 0; i < num_regs; ++i)
    {
      auto &reg = regions[i];
      if(this_task_is_root)
        {
          if(data_counts[i] == 0)
            continue;

          recv_data_reqs.emplace_back();
          MPI_Irecv(recv_data.data() + offset, data_counts[i], bh_merger_data::get_mpi_datatype(), reg.get_compute_root_sim(), i,
                    Comm, &recv_data_reqs.back());
          offset += data_counts[i];
        }

      if(reg.this_task_is_compute_root())
        {
          auto &data = reg.get_output_merger_data();
          int size   = data.size();

          if(size == 0)
            continue;

          send_data_reqs.emplace_back();
          MPI_Isend(data.data(), size, bh_merger_data::get_mpi_datatype(), 0, i, Comm, &send_data_reqs.back());
        }
    }

  if(!send_data_reqs.empty())
    {
      MPI_Waitall(send_data_reqs.size(), send_data_reqs.data(), MPI_STATUSES_IGNORE);
    }

  if(!recv_data_reqs.empty())
    {
      MPI_Waitall(recv_data_reqs.size(), recv_data_reqs.data(), MPI_STATUSES_IGNORE);
    }

  return recv_data;
}

void RegionManager::update_sim_data(inttime_t Ti_Current, MPI_Comm Comm)
{
  //TIMER_START(CPU_KETJU_UPDATE);
  for(auto &reg : regions)
    {
      // if there is only one BH part in this ketju region, skip it. 
      // if(KetjuData.options.IntegrateBinaryRegion){
      //   if(reg.get_bh_count() < 2){ 
      //     message(1, "DEBUG...ketju 3065: skip reg_index %d since it only has %d bh\n", reg.get_region_index(), reg.get_bh_count());
      //     continue;
      //   }
      //   }
      reg.update_sim_data(Ti_Current, Comm); 
    }
  //TIMER_STOP(CPU_KETJU_UPDATE);
}

//////////////////

class OutputFile
{
  const char *bhs_group     = "/BHs";
  const char *timestep_dset = "/timesteps";
  const char *merger_dset   = "/mergers";

  hid_t file_id = -1;

  void init_hdf5_types();
  void init_hdf5_output_file(Cosmology * CP_all, const struct UnitSystem units, inttime_t Ti_current);
  void open_file_on_root_if_needed(MPI_Comm Comm, int RestartSnapNum, const char * OutputDir, Cosmology * CP_all, const struct UnitSystem units, inttime_t Ti_current);
  hid_t open_or_create_bh_dataset(MyIDType ID);

  // Timestep data is written separately from the BHs so that we don't need to
  // duplicate it for each BH, and instead just need a single index into the
  // array of timesteps. This saves some storage, but not that much since we
  // only have two fields to store per timestep at the moment.
  // The separation of the timestep data requires keeping track of the index
  // to the timestep array (All.KetjuData.next_tstep_index),
  // which takes some effort when BHs may have different timesteps and get written at different rates.
  // On the other hand, this allows uniquely identifying which BH data points match each other,
  // even in the case where some timesteps are duplicated after the run has been restarted
  // after crashing for some reason.
  // The design originates from an older GADGET-3 version of the code where we
  // had some other arrays that referenced the timestep data in addition to the
  // BHs, and deduplicating the timestep data was more useful.
  struct timestep_output_data
  {
    double physical_time;
    //< Physical time of the simulation.
    //  All.Time in non-comoving and the cosmic time computed with  Driftfac.get_cosmic_time in comoving integration.
    double scale_factor;
    // Cosmological scale factor, always 1 in non-comoving integration.
  };

  struct bh_output_data
  {
    double gadget_position[3];  // physical position / scale_factor
    double gadget_velocity[3];  // peculiar velocity * scale_factor
    double physical_position[3];
    // Physical position wrapped to the nearest image to the region main BH  position for periodic runs, allowing easy computation of
    // the distances between BHs in the same region. Equals gadget_position * scale_factor modulo periodic box size.
    double physical_velocity[3];
    // Physical velocity including hubble flow contribution
    // (hubble flow part = H * physical_position).
    double spin[3];
    double mass;
    int timestep_index;
    // Index of the current timestep in the global timestep table
    int num_particles_in_region, num_BHs_in_region;

    bh_output_data(const mpi_particle &bh, const double ref_pos[3], int tstep_index, int num_BHs_in_region, int num_particles_in_region, double hubble, double scale_fac);
  };

  // HDF5 library datatype definitions stored here.
  // Note that we use the same datatypes to describe the struct in memory and
  // file, since the conversion overhead can be quite significant.
  // This can cause padding bytes from the structs to be written to the file
  // if they are present, which uses some extra space and causes Valgrind to
  // complain. But the performance impact is not worth using separate packed
  // representations for writing.
  class hdf5_datatypes
  {
    hid_t vector;
    hid_t timestep_data;
    hid_t bh_data;
    hid_t merger_data;

   public:
    hdf5_datatypes();
    ~hdf5_datatypes();

    template <typename T>
    hid_t get() const;
  } datatypes;

  template <typename T>
  hsize_t append_single_element(hid_t dset, const T &data);

  void write_bh_datapoint(MyIDType ID, const bh_output_data &data);
  void write_merger_datapoint(const bh_merger_data &data);
  hsize_t write_timestep_datapoint(double physical_time, double scale_factor);
  void write_timestep_data(const RegionManager &RM, MPI_Comm Comm, const inttime_t Ti_Current, const int RestartSnapNum);
  void write_bh_data(const RegionManager &RM, MPI_Comm Comm, const inttime_t Ti_Current);
  void write_merger_data(const RegionManager &RM, MPI_Comm Comm);

  void flush()
  {
    if(file_id >= 0)
      H5Fflush(file_id, H5F_SCOPE_LOCAL);
  }

 public:
  ~OutputFile()
  {
    if(file_id >= 0)
      {
        H5Fclose(file_id);
      }
  }

  void write_output_data(const RegionManager &RM, MPI_Comm Comm, const inttime_t Ti_Current, int RestartSnapNum, const char * OutputDir, Cosmology * CP_all, const struct UnitSystem units);
};

OutputFile::hdf5_datatypes::hdf5_datatypes()
{
  hsize_t dims[1] = {3};
  vector          = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims);

  timestep_data = H5Tcreate(H5T_COMPOUND, sizeof(timestep_output_data));
  H5Tinsert(timestep_data, "physical_time", HOFFSET(timestep_output_data, physical_time), H5T_NATIVE_DOUBLE);
  H5Tinsert(timestep_data, "scale_factor", HOFFSET(timestep_output_data, scale_factor), H5T_NATIVE_DOUBLE);

  bh_data = H5Tcreate(H5T_COMPOUND, sizeof(bh_output_data));
  H5Tinsert(bh_data, "gadget_position", HOFFSET(bh_output_data, gadget_position), vector);
  H5Tinsert(bh_data, "gadget_velocity", HOFFSET(bh_output_data, gadget_velocity), vector);
  H5Tinsert(bh_data, "physical_position", HOFFSET(bh_output_data, physical_position), vector);
  H5Tinsert(bh_data, "physical_velocity", HOFFSET(bh_output_data, physical_velocity), vector);
  H5Tinsert(bh_data, "spin", HOFFSET(bh_output_data, spin), vector);
  H5Tinsert(bh_data, "mass", HOFFSET(bh_output_data, mass), H5T_NATIVE_DOUBLE);
  H5Tinsert(bh_data, "timestep_index", HOFFSET(bh_output_data, timestep_index), H5T_NATIVE_INT);
  H5Tinsert(bh_data, "num_particles_in_region", HOFFSET(bh_output_data, num_particles_in_region), H5T_NATIVE_INT);
  H5Tinsert(bh_data, "num_BHs_in_region", HOFFSET(bh_output_data, num_BHs_in_region), H5T_NATIVE_INT);

  static_assert(std::is_same<MyIDType, unsigned long int>::value || std::is_same<MyIDType, unsigned long long>::value);
  const auto ID_type = std::is_same<MyIDType, unsigned int>::value ? H5T_NATIVE_UINT : H5T_NATIVE_ULLONG;

  merger_data = H5Tcreate(H5T_COMPOUND, sizeof(bh_merger_data));
  H5Tinsert(merger_data, "ID1", HOFFSET(bh_merger_data, ID1), ID_type);
  H5Tinsert(merger_data, "ID2", HOFFSET(bh_merger_data, ID2), ID_type);
  H5Tinsert(merger_data, "ID_remnant", HOFFSET(bh_merger_data, ID_remnant), ID_type);
  H5Tinsert(merger_data, "m1", HOFFSET(bh_merger_data, m1), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "m2", HOFFSET(bh_merger_data, m2), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "m_remnant", HOFFSET(bh_merger_data, m_remnant), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "chi1", HOFFSET(bh_merger_data, chi1), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "chi2", HOFFSET(bh_merger_data, chi2), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "chi_remnant", HOFFSET(bh_merger_data, chi_remnant), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "kick_velocity", HOFFSET(bh_merger_data, v_kick), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "merger_physical_time", HOFFSET(bh_merger_data, t_merger), H5T_NATIVE_DOUBLE);
  H5Tinsert(merger_data, "merger_redshift", HOFFSET(bh_merger_data, z_merger), H5T_NATIVE_DOUBLE);
}

OutputFile::hdf5_datatypes::~hdf5_datatypes()
{
  H5Tclose(vector);
  H5Tclose(timestep_data);
  H5Tclose(bh_data);
  H5Tclose(merger_data);
}

template <typename T>
hid_t OutputFile::hdf5_datatypes::get() const
{
  static_assert("Invalid type passed");
  return 0;
}

template <>
hid_t OutputFile::hdf5_datatypes::get<OutputFile::bh_output_data>() const
{
  return bh_data;
}

template <>
hid_t OutputFile::hdf5_datatypes::get<bh_merger_data>() const
{
  return merger_data;
}

template <>
hid_t OutputFile::hdf5_datatypes::get<OutputFile::timestep_output_data>() const
{
  return timestep_data;
}

template <>
hid_t OutputFile::hdf5_datatypes::get<double[3]>() const
{
  return vector;
}

void OutputFile::init_hdf5_output_file(Cosmology * CP_all, const struct UnitSystem Units, inttime_t Ti_current)
{
  {  // Local scopes for different datasets/attributes, to prevent reusing the same variables by mistake
    // Cosmology
    struct cosmology_data
    {
      double HubbleParam;
      double Omega0;
      double OmegaLambda;
      double OmegaBaryon;
      int ComovingIntegrationOn;
      int Periodic;
      double BoxSize[3];
    } cosmo = {CP_all->HubbleParam,
               CP_all->Omega0,
               CP_all->OmegaLambda,
               CP_all->OmegaBaryon,
               CP_all->ComovingIntegrationOn,
               0,
               {0, 0, 0}
    };

    hid_t datatype = H5Tcreate(H5T_COMPOUND, sizeof(cosmology_data));
    H5Tinsert(datatype, "HubbleParam", HOFFSET(cosmology_data, HubbleParam), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "Omega0", HOFFSET(cosmology_data, Omega0), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "OmegaLambda", HOFFSET(cosmology_data, OmegaLambda), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "OmegaBaryon", HOFFSET(cosmology_data, OmegaBaryon), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "ComovingIntegrationOn", HOFFSET(cosmology_data, ComovingIntegrationOn), H5T_NATIVE_INT);
    H5Tinsert(datatype, "Periodic", HOFFSET(cosmology_data, Periodic), H5T_NATIVE_INT);
    H5Tinsert(datatype, "BoxSize", HOFFSET(cosmology_data, BoxSize), datatypes.get<double[3]>());

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate(file_id, "cosmology", datatype, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, datatype, &cosmo);

    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(datatype);
  }

  {
    // Units
    double timebase_interval = Dloga_interval_ti(Ti_current);
    struct unit_data
    {
      double unit_time_in_s, unit_length_in_cm, unit_mass_in_g, unit_velocity_in_cm_per_s, G_cgs, c_cgs, timebase_interval;
    } units = {Units.UnitTime_in_s,
               Units.UnitLength_in_cm,
               Units.UnitMass_in_g,
               Units.UnitVelocity_in_cm_per_s,
               CP_all->GravInternal / (Units.UnitMass_in_g / std::pow(Units.UnitVelocity_in_cm_per_s, 2) / Units.UnitLength_in_cm),
               LIGHTCGS,
               timebase_interval};

    hid_t datatype = H5Tcreate(H5T_COMPOUND, sizeof(unit_data));
    H5Tinsert(datatype, "unit_time_in_s", HOFFSET(unit_data, unit_time_in_s), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "unit_length_in_cm", HOFFSET(unit_data, unit_length_in_cm), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "unit_mass_in_g", HOFFSET(unit_data, unit_mass_in_g), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "unit_velocity_in_cm_per_s", HOFFSET(unit_data, unit_velocity_in_cm_per_s), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "G_cgs", HOFFSET(unit_data, G_cgs), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "c_cgs", HOFFSET(unit_data, c_cgs), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "timebase_interval", HOFFSET(unit_data, timebase_interval), H5T_NATIVE_DOUBLE);

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate(file_id, "units", datatype, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, datatype, &units);

    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(datatype);
  }

  {
    // Integrator options
    hid_t datatype = H5Tcreate(H5T_COMPOUND, sizeof(Ketju_Options));
    H5Tinsert(datatype, "minimum_bh_mass", HOFFSET(Ketju_Options, minimum_bh_mass), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "region_physical_radius", HOFFSET(Ketju_Options, region_physical_radius), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "output_time_interval", HOFFSET(Ketju_Options, output_time_interval), H5T_NATIVE_DOUBLE);

    hid_t pn_string = H5Tcopy(H5T_C_S1);
    H5Tset_size(pn_string, sizeof Ketju_Options::PN_terms);
    H5Tinsert(datatype, "PN_terms", HOFFSET(Ketju_Options, PN_terms), pn_string);
    H5Tclose(pn_string);

    H5Tinsert(datatype, "enable_bh_merger_kicks", HOFFSET(Ketju_Options, enable_bh_merger_kicks), H5T_NATIVE_INT);
    H5Tinsert(datatype, "use_star_star_softening", HOFFSET(Ketju_Options, use_star_star_softening), H5T_NATIVE_INT);
    H5Tinsert(datatype, "expand_tight_binaries_period_factor", HOFFSET(Ketju_Options, expand_tight_binaries_period_factor),
              H5T_NATIVE_DOUBLE);

    H5Tinsert(datatype, "integration_relative_tolerance", HOFFSET(Ketju_Options, integration_relative_tolerance), H5T_NATIVE_DOUBLE);
    H5Tinsert(datatype, "output_time_relative_tolerance", HOFFSET(Ketju_Options, output_time_relative_tolerance), H5T_NATIVE_DOUBLE);

    H5Tinsert(datatype, "minimum_particles_per_task", HOFFSET(Ketju_Options, minimum_particles_per_task), H5T_NATIVE_DOUBLE);

    H5Tinsert(datatype, "use_divide_and_conquer_mst", HOFFSET(Ketju_Options, use_divide_and_conquer_mst), H5T_NATIVE_INT);
    H5Tinsert(datatype, "max_tree_distance", HOFFSET(Ketju_Options, max_tree_distance), H5T_NATIVE_INT);
    H5Tinsert(datatype, "steps_between_mst_reconstruction", HOFFSET(Ketju_Options, steps_between_mst_reconstruction), H5T_NATIVE_INT);

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate(file_id, "integrator options", datatype, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, datatype, &KetjuData.options);

    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(datatype);
  }

  {
    // Create data group for BHs
    hid_t group = H5Gcreate(file_id, bhs_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(group);
  }

  {
    // Create dataset for timestep data
    hsize_t dims[1] = {0}, maxdims[1] = {H5S_UNLIMITED};
    hsize_t chunk_dims[1] = {4096 / sizeof(timestep_output_data)};
    hid_t space           = H5Screate_simple(1, dims, maxdims);
    hid_t prop            = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(prop, 1, chunk_dims);
    hid_t dset = H5Dcreate(file_id, timestep_dset, datatypes.get<timestep_output_data>(), space, H5P_DEFAULT, prop, H5P_DEFAULT);
    H5Pclose(prop);
    H5Dclose(dset);
    H5Sclose(space);
  }

  {
    // Create dataset for merger data
    hsize_t dims[1] = {0}, maxdims[1] = {H5S_UNLIMITED};
    hsize_t chunk_dims[1] = {1};
    hid_t space           = H5Screate_simple(1, dims, maxdims);
    hid_t prop            = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(prop, 1, chunk_dims);
    hid_t dset = H5Dcreate(file_id, merger_dset, datatypes.get<bh_merger_data>(), space, H5P_DEFAULT, prop, H5P_DEFAULT);
    H5Pclose(prop);
    H5Dclose(dset);
    H5Sclose(space);
  }
}

void OutputFile::open_file_on_root_if_needed(MPI_Comm Comm, int RestartSnapNum, const char * OutputDir, Cosmology * CP_all, const struct UnitSystem units, inttime_t Ti_current)
{

  int ThisTask;
  MPI_Comm_rank(Comm, &ThisTask);
  if(ThisTask != 0)
    return;

  if(file_id >= 0)
    return;
  
  char * postfix;
  char fname[550];

  if(RestartSnapNum != -1) {
      postfix = fastpm_strdup_printf("-R%03d", RestartSnapNum);
  } else {
      postfix = fastpm_strdup_printf("%s", "");
  }

  //fname = fastpm_strdup_printf("%s/%s%s.hdf5", OutputDir, "ketju_bhs", postfix);
  std::snprintf(fname, sizeof fname, "%s/%s%s.hdf5", OutputDir, "ketju_bhs", postfix);

  // if(All.RestartFlag == RST_RESUME || All.RestartFlag == RST_STARTFROMSNAP)
  //   {
  //     // Try to open the existing file
  //     file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
  //     if(file_id < 0)
  //       {
  //         ketju_printf(
  //             "Could not open file '%s' for appending, will attempt "
  //             "creating a new file\n",
  //             fname);
  //       }
  //   }
  // if(All.RestartFlag == RST_BEGIN || file_id < 0)
  //   {
      // Create a new output file when starting from scratch or when
      // opening the previous output file failed
  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  init_hdf5_output_file(CP_all, units, Ti_current);
  message(0, "KETJU: create the output file %s.\n", fname);
    //}
  if(file_id < 0)
    {
      char msg[700];
      std::snprintf(msg, sizeof msg, "error in opening file '%s'\n", fname);
      endrun(1, msg);
    }
}

hid_t OutputFile::open_or_create_bh_dataset(MyIDType ID)
{
  char dset_name[20];
  std::snprintf(dset_name, sizeof dset_name, "%s/%llu", bhs_group, static_cast<unsigned long long>(ID));
  if(H5Lexists(file_id, dset_name, H5P_DEFAULT))
    {
      hid_t dset = H5Dopen(file_id, dset_name, H5P_DEFAULT);
      return dset;
    }

  // need to create the dataset
  hsize_t dims[1] = {0}, maxdims[1] = {H5S_UNLIMITED};
  hid_t space           = H5Screate_simple(1, dims, maxdims);
  hid_t prop            = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t chunk_dims[1] = {4096 / sizeof(bh_output_data)};
  H5Pset_chunk(prop, 1, chunk_dims);

  hid_t dset = H5Dcreate(file_id, dset_name, datatypes.get<bh_output_data>(), space, H5P_DEFAULT, prop, H5P_DEFAULT);
  H5Pclose(prop);
  H5Sclose(space);
  return dset;
}

template <typename T>
hsize_t OutputFile::append_single_element(hid_t dset, const T &data)
{
  hid_t space = H5Dget_space(dset);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space, dims, NULL);
  H5Sclose(space);
  hsize_t index = dims[0];
  dims[0] += 1;
  H5Dset_extent(dset, dims);
  space = H5Dget_space(dset);

  hsize_t offset[1] = {index};
  hsize_t count[1]  = {1};
  H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, count, NULL);

  // Need to use a separate dataspace for the memory, since the dtype_id
  // constructed at the beginning of the program differs from the one for
  // reopened files/datasets.
  hid_t memspace = H5Screate_simple(1, count, count);

  H5Dwrite(dset, datatypes.get<T>(), memspace, space, H5P_DEFAULT, &data);

  H5Sclose(memspace);
  H5Sclose(space);
  return index;
}

void OutputFile::write_bh_datapoint(MyIDType ID, const bh_output_data &data)
{
  hid_t dset = open_or_create_bh_dataset(ID);
  append_single_element(dset, data);
  H5Dclose(dset);
}

void OutputFile::write_merger_datapoint(const bh_merger_data &data)
{
  hid_t dset = H5Dopen(file_id, merger_dset, H5P_DEFAULT);
  append_single_element(dset, data);
  H5Dclose(dset);
}

// returns the current index to the timestep table
hsize_t OutputFile::write_timestep_datapoint(double physical_time, double scale_factor)
{
  timestep_output_data data = {physical_time, scale_factor};
  hid_t dset                = H5Dopen(file_id, timestep_dset, H5P_DEFAULT);
  hsize_t index             = append_single_element(dset, data);
  H5Dclose(dset);
  return index;
}

void OutputFile::write_timestep_data(const RegionManager &RM, MPI_Comm Comm, const inttime_t Ti_Current, const int RestartSnapNum)
{
  int ThisTask;
  MPI_Comm_rank(Comm, &ThisTask);
  // First set the final_written_Ti value, needed on all tasks for setting the output timebin
  int max_timebin = 0;
  for(auto &tr : RM.get_timestep_regions())
    {
      if(tr.get_timebin() > max_timebin)
        {
          max_timebin = tr.get_timebin();
        }
    }

  const int output_timebin         = KetjuData.output_timebin;
  const inttime_t output_Ti_step = inttime_t(1) << output_timebin;

  // Nothing to do if the next output time is larger than the one reached by
  // the highest current bin
  if(Ti_Current > 0 &&
     (Ti_Current + (((inttime_t)1) << max_timebin)) < (1 + Ti_Current / output_Ti_step) * output_Ti_step)
    return;

  // The final time that is written is set by the highest region timebin,
  // which will be in sync with the output timebin by the above check.
  const inttime_t final_Ti_to_write = Ti_Current + (((inttime_t)1) << max_timebin);

  if(KetjuData.final_written_Ti >= final_Ti_to_write)
    return;

  KetjuData.final_written_Ti = final_Ti_to_write;

  // Actual writing on root only
  const bool this_task_is_root = ThisTask == 0;
  if(!this_task_is_root)
    return;

  int tstep_index        = -10;  // We should always write something, this catches errors if we don't
  int num_tsteps_written = 0;
  // Initial Ti is the next output time we haven't written yet
  inttime_t Ti = (1 + Ti_Current / output_Ti_step) * output_Ti_step;
  if(Ti_Current == 0)
    Ti = 0;
  for(; Ti <= final_Ti_to_write; Ti += output_Ti_step)
    {
      //double current_time, scale_factor;
      //current_time = All.TimeBegin + Ti * All.Timebase_interval;
      double current_time = get_atime(Ti);
      current_time = log(current_time);
      double scale_factor = 1;
    
      tstep_index = write_timestep_datapoint(current_time, scale_factor);
      ++num_tsteps_written;
      //message(0, "DEBUG... ketju.c 3711 in write_timestep Ti_Current: %d written_T %d tstep_index %d num_tsteps_written %d final_ti_to_write %d output_Ti_step %d KetjuData.next_tstep_index %d \n",
      //Ti_Current, Ti, tstep_index, num_tsteps_written, final_Ti_to_write, output_Ti_step, KetjuData.next_tstep_index);
    }

  // Flag to handle the possible error below specially on the first
  // write after restarting.
  static bool first_write = true;

  if(tstep_index - num_tsteps_written + 1 != KetjuData.next_tstep_index)
    {
      char buf[100];
      std::snprintf(buf, sizeof buf,
                    "Ketju: inconsistent output tstep index, expected %d "
                    "actual %d",
                    KetjuData.next_tstep_index, tstep_index - num_tsteps_written + 1);
      if(first_write && RestartSnapNum != -1)
        {
          // It seems the error is most likely caused by restarting
          // after the code crashed (or using restartfiles older than
          // the output file for some other reason).  In this case
          // there's extra data in the file the index isn't aware of,
          // but it can be safely ignored by adjusting the index. The
          // sections of duplicated data will just need to be handled
          // when reading.
          std::printf("%s\n", buf);
          ketju_printf(
              "it looks like this is caused by restarting"
              " after a crash, so adjusting and continuing.\n");
          KetjuData.next_tstep_index = tstep_index - num_tsteps_written + 1;
        }
      else
        {
          // Something else has gone wrong, either a bug or an
          // unaccounted for special case requiring adjustment.
          endrun(1, buf);
        }
    }
  first_write = false;
}

OutputFile::bh_output_data::bh_output_data(const mpi_particle &bh, const double ref_pos[3], 
                                           int tstep_index, int num_BHs_in_region, int num_particles_in_region, double hubble, 
                                           double scale_fac)
    : mass(bh.Mass), 
      timestep_index(tstep_index), 
      num_particles_in_region(num_particles_in_region), 
      num_BHs_in_region(num_BHs_in_region)
{
  //conv.intpos_to_pos(bh.IntPos, gadget_position);

  std::copy(std::begin(bh.Vel), std::end(bh.Vel), std::begin(gadget_velocity));
  std::copy(std::begin(bh.Spin), std::end(bh.Spin), std::begin(spin));

  //conv.nearest_image_intpos_to_pos(bh.IntPos, ref_intpos, physical_position);

  //double ref_pos[3];
  //conv.intpos_to_pos(ref_intpos, ref_pos);

  for(int l = 0; l < 3; ++l)
    {
      physical_position[l] = bh.Pos[l] - ref_pos[l];
      physical_position[l] += ref_pos[l];
      physical_position[l] *= scale_fac;
      physical_velocity[l] = bh.Vel[l] / scale_fac + hubble * physical_position[l];
    }
}

void OutputFile::write_bh_data(const RegionManager &RM, MPI_Comm Comm, const inttime_t Ti_Current)
{
  int ThisTask;
  MPI_Comm_rank(Comm, &ThisTask);
  const auto output_data = RM.comm_and_get_output_data_on_root(Comm);
  const bool this_task_is_root = ThisTask == 0;
  if(!this_task_is_root)
    return;

  const std::vector<std::vector<mpi_particle>> &writable_region_data = output_data.first;
  const std::vector<int> &writable_reg_indices                       = output_data.second;

  const int output_timebin         = KetjuData.output_timebin;
  const inttime_t output_Ti_step = inttime_t(1) << output_timebin;

  const int num_writable_regs = writable_region_data.size();
  for(int i = 0; i < num_writable_regs; ++i)
    {
      const auto &data   = writable_region_data[i];
      const auto &region = RM.get_regions()[writable_reg_indices[i]];

      const int num_initial_bhs   = region.get_bh_count();
      const int num_output_points = data.size() / num_initial_bhs;

      // time of the first datapoint
      const inttime_t first_Ti_out = (Ti_Current / output_Ti_step) * output_Ti_step + (Ti_Current == 0 ? 0 : output_Ti_step);

      int tstep_index = KetjuData.next_tstep_index;
      for(int j = 0; j < num_output_points; ++j, ++tstep_index)
        {
          double scale_fac = 1., hubble = 0.;

          // There may have been a merger during the integration, which is
          // detected from zero data for the merged away BHs.
          int actual_num_bhs = 0;
          for(int k = 0; k < num_initial_bhs; ++k)
            {
              const auto &bh = data[j + num_output_points * k];
              if(bh.Mass > 0)
                ++actual_num_bhs;
            }

          const auto &ref_pos            = data[j].Pos;
          int num_particles_in_region = region.get_total_particle_count() - num_initial_bhs + actual_num_bhs;

          for(int k = 0; k < num_initial_bhs; ++k)
            {
              const auto &bh = data[j + num_output_points * k];
              if(bh.Mass == 0)
                continue;  // Merged
                   //bh_output_data(const mpi_particle &bh, double ref_pos[3], int tstep_index,
                   //int num_BHs_in_region, int num_particles_in_region, double hubble, double scale_fac);
              bh_output_data bh_data(bh, ref_pos, tstep_index, actual_num_bhs, num_particles_in_region, hubble, scale_fac);
              write_bh_datapoint(bh.ID, bh_data);
            }
        }
    }
}

void OutputFile::write_merger_data(const RegionManager &RM, MPI_Comm Comm)
{
  int ThisTask;
  MPI_Comm_rank(Comm, &ThisTask);  
  const auto merger_data = RM.comm_and_get_merger_data_on_root(Comm);

  const bool this_task_is_root = ThisTask == 0;
  if(!this_task_is_root)
    return;

  for(auto &merger : merger_data)
    {
      write_merger_datapoint(merger);
    }
}

void OutputFile::write_output_data(const RegionManager &RM, MPI_Comm Comm, const inttime_t Ti_Current, int RestartSnapNum, const char * OutputDir, Cosmology * CP_all, const struct UnitSystem units)
{
  if(RM.get_regions().empty())
    return;

  open_file_on_root_if_needed(Comm, RestartSnapNum, OutputDir, CP_all, units, Ti_Current);
  write_timestep_data(RM, Comm, Ti_Current, RestartSnapNum);
  write_bh_data(RM, Comm, Ti_Current);
  write_merger_data(RM, Comm);
  flush();
}

///////////////

// next_tstep_index is the index in the output file for the next datapoint to be
// written. To allow for new BHs that are created during the run, it needs to be
// advanced every time the output timebin is synced/active, regardless of if
// there are BHs that are integrated on this step.
// But if there aren't any BHs the index shouldn't be advanced beyond the final
// step index + 1. This is detected from the final_written_Ti check.

// change for mp-gadget: for non-hierarchy mode, the times.mingravtimebin is not updated. 
// So use the min region-timestep as the "lowest occupied timestep" 
void advance_next_tstep_index(DriftKickTimes * times, MPI_Comm Comm, const RegionManager &RM)
{
  //message(0, "DEBUG... ketju.c 3878 in advance_tstep_index Time: %d next_tstep_index %d min_timebin %d \n",
  //times->Ti_Current, KetjuData.next_tstep_index, times->mintimebin);
  int ThisTask;
  MPI_Comm_rank(Comm, &ThisTask);

  if(ThisTask != 0)
    return;  // the value isn't needed

  // An extra datapoint is written for the initial state when Ti_Current == 0,
  // but this is accounted for in the timestep index progress only on the next
  // pass, so need to use this little trick.
  static bool start_point_handled = true;

  if(times->Ti_Current == 0)
    {
      start_point_handled = false;
      return;  // Start of the run, haven't written anything yet
    }

  if(!start_point_handled)
    {
      ++KetjuData.next_tstep_index;
      start_point_handled = true;
    }

  int min_timebin = times->mintimebin;

  // for(auto &tr : RM.get_timestep_regions())
  //   {
  //     if(tr.get_timebin() < min_timebin)
  //       {
  //         min_timebin = tr.get_timebin();
  //       }
  //   }


  inttime_t output_Ti_step = inttime_t(1) << KetjuData.output_timebin;
  if(times->Ti_Current <= KetjuData.final_written_Ti && (times->Ti_Current % output_Ti_step == 0))
    {
      if( min_timebin > KetjuData.output_timebin)
        {
          const int substeps = 1 << (min_timebin - KetjuData.output_timebin);

          KetjuData.next_tstep_index += substeps;
          //message(0, "DEBUG... ketju.c 3920 in advance_tstep_index addsubstep %d min_timebin:%d output_timebin: %d next_tstep_index %d \n", substeps, min_timebin, KetjuData.output_timebin, KetjuData.next_tstep_index);
        }
      else
        {
          KetjuData.next_tstep_index += 1;
          //message(0, "DEBUG... ketju.c 3925 in advance_tstep_index min_timebin:%d output_timebin: %d next_tstep_index %d \n", min_timebin, KetjuData.output_timebin, KetjuData.next_tstep_index);
        }
    }
}


///////////////////

// Objects that we only need (and can only have) a single one each.
// Only used directly in the interface functions below.

RegionManager RM;
OutputFile OF;

// }  // namespace

//////////////////

// Interface functions, these are the only thing the rest of the code sees.

void Ketju_init_ketjuRM(Cosmology * CP, const double TimeBegin, const struct UnitSystem units){
  RM.RM_init(CP, TimeBegin, units);
}

void Ketju_find_regions(const ActiveParticles * act, DriftKickTimes * times, MPI_Comm Comm, int FastParticleType, double asmth, UnitSystem units)
{
  //TIMER_START(CPU_KETJU_CONSTRUCTION);
  RM.find_regions(act, times->Ti_Current, units);
  RM.find_timestep_regions(act, times->Ti_Current);
  RM.set_region_timebins(FastParticleType, asmth, times, Comm);
  advance_next_tstep_index(times, Comm, RM);
  RM.set_output_timebin(times, Comm); 

  //TIMER_STOP(CPU_KETJU_CONSTRUCTION);
}

int Ketju_is_particle_timestep_limited(int index) { return RM.is_particle_timestep_limited(index); }
int Ketju_set_limited_timebins(int min_timebin, int push_down_flag)
{
  //TIMER_START(CPU_KETJU_MISC);
  const int max_bin = RM.set_limited_timebins(min_timebin, push_down_flag);
  //TIMER_STOP(CPU_KETJU_MISC);
  return max_bin;
}

void Ketju_run_integration(DriftKickTimes * times, const struct UnitSystem units, MPI_Comm Comm, FILE * FdSingleBH, FILE * FdKetjuRegion, int trace_bhid, Cosmology * CP, int RestartSnapNum, const char * OutputDir)
{
  //TIMER_START(CPU_KETJU_MISC);
  RM.set_up_regions_for_integration(times->Ti_Current, units);

  double newatime = get_atime(times->Ti_Current);  
  double floattime = log(newatime);
  //trace_singleblackhole(FdSingleBH, floattime, "Ketju_setup_integration", trace_bhid, PartManager);

  RM.integrate_regions(times->Ti_Current, FdSingleBH, FdKetjuRegion, trace_bhid);
  

  OF.write_output_data(RM, Comm, times->Ti_Current, RestartSnapNum, OutputDir, CP, units);
  RM.update_sim_data(times->Ti_Current, Comm);
  //trace_singleblackhole(FdSingleBH, floattime, "ketju_update_sim_data", trace_bhid, PartManager);

 


  RM.clear_data();

  //TIMER_STOP(CPU_KETJU_MISC);
}

void Ketju_set_final_velocities(const ActiveParticles * act)
{


  //TIMER_START(CPU_KETJU_UPDATE);  
  #pragma omp parallel for
  for(int i = 0; i < act->NumActiveParticle; ++i)
    {
      const int target = act->ActiveParticle ? act->ActiveParticle[i] : i;

      auto &p = P[target];
      if(p.KetjuIntegrated)
        {
          std::copy(std::begin(p.KetjuFinalVel), std::end(p.KetjuFinalVel), std::begin(P[target].Vel));
        }
        

    }
  //TIMER_STOP(CPU_KETJU_UPDATE);
}

void Ketju_finish_step(const ActiveParticles * act, DriftKickTimes * times, UnitSystem units, int NumCurrentTiStep, MPI_Comm Comm)
{
  //message(0, "DEBUG... ketju.c 3226 treenode no %d current ChildType %d sibling %d father %d \n", tree_no, tree->Nodes[tree_no].f.ChildType, tree->Nodes[tree_no].sibling, tree->Nodes[tree_no].father);
  //TIMER_START(CPU_KETJU_UPDATE);

  // if (NumCurrentTiStep == 0){
  //     RM.find_regions(act, times->Ti_Current, units);
  //     RM.set_up_regions_for_integration(times->Ti_Current, units);
  //     RM.update_sim_data(times->Ti_Current, Comm);
  //     //RM.find_regions(Sp);
  //     //RM.set_up_regions_for_integration(Sp);
  //     //RM.update_sim_data(Sp);
  //     RM.clear_data();
  //     Ketju_set_final_velocities(act);
  //     //set_final_velocities(Sp);    
  // }

  #pragma omp parallel for
  for(int i = 0; i < act->NumActiveParticle; ++i)
    {
      
      const int target = act->ActiveParticle ? act->ActiveParticle[i] : i;

      if(P[target].KetjuIntegrated){
          P[target].Potential += P[target].KetjuPotentialEnergyCorrection;
          P[target].KetjuIntegrated = 0;
      }
        

    }
  //TIMER_STOP(CPU_KETJU_UPDATE);
}


void ketju_Region::Region_info(FILE * FdKetjuRegion, const double Time, double timestep, int timebin)
{
    if(!FdKetjuRegion)
        return;


    fprintf(FdKetjuRegion, "Time %g Region %d around BH ID %llu with %d BH(s), %d star particles, %d total particle(s) on timebin %d (physical timestep %g) ",
           Time, get_region_index(), static_cast<unsigned long long>(get_bhs()[0].ID), get_bh_count(), get_star_count(), get_total_particle_count(), timebin, timestep);
    
    // fprintf(FdKetjuRegion, "Particle List: ");

    // for(int p : local_member_indices)
    //   {
    //     fprintf(FdKetjuRegion, "%lu(%d), ", P[p].ID, P[p].Type);
        // ketju_printf("DEBUG...ketju 1893: region %d %d members: type %d ID %ld \n", get_region_index(), p, P[p].Type, P[p].ID);
        // message(0, "DEBUG...ketju 1894: %d region members: type %d ID %ld, %d, \n", p, P[p].Type, P[p].ID, P[p].Type);
        // ketju_printf("DEBUG...ketju 1894: %d region members: type %d ID %ld, %d, \n", p, P[p].Type, P[p].ID, P[p].Type);
      // }
    fprintf(FdKetjuRegion, "\n");

    fflush(FdKetjuRegion);
} 


