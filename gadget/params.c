#include <libgadget/gravity.h>
#include <libgadget/densitykernel.h>
#include <libgadget/timebinmgr.h>
#include <libgadget/timestep.h>
#include <libgadget/utils.h>
#include <libgadget/treewalk.h>
#include <libgadget/cooling_rates.h>
#include <libgadget/winds.h>
#include <libgadget/sfr_eff.h>
#include <libgadget/blackhole.h>
#include <libgadget/density.h>
#include <libgadget/hydra.h>
#include <libgadget/fof.h>
#include <libgadget/secondfof.h>
#include <libgadget/init.h>
#include <libgadget/run.h>
#include <libgadget/timebinmgr.h>
#include <libgadget/petaio.h>
#include <libgadget/cooling_qso_lightup.h>
#include <libgadget/metal_return.h>
#include <libgadget/uvbg.h>
#include <libgadget/stats.h>
#include <libgadget/plane.h>
#include <libgadget/tidalfield.h>

static int
BlackHoleFeedbackMethodAction (ParameterSet * ps, const char * name, void * data)
{
    int v = param_get_enum(ps, name);
    if(HAS(v, BH_FEEDBACK_TOPHAT) == HAS(v, BH_FEEDBACK_SPLINE)) {
        message(1, "error BlackHoleFeedbackMethod contains either tophat or spline, but both\n");
        return 1;
    }
    if(HAS(v, BH_FEEDBACK_MASS) ==  HAS(v, BH_FEEDBACK_VOLUME)) {
        message(1, "error BlackHoleFeedbackMethod contains either volume or mass, but both\n");
        return 1;
    }
    return 0;
}

static int
StarformationCriterionAction(ParameterSet * ps, const char * name, void * data)
{
    int v = param_get_enum(ps, name);
    if(!HAS(v, SFR_CRITERION_DENSITY)) {
        message(1, "error: At least use SFR_CRITERION_DENSITY\n");
        return 1;
    }
    return 0;
}

static ParameterSet *
create_gadget_parameter_set()
{
    ParameterSet * ps = parameter_set_new();

    param_declare_string(ps, "InitCondFile", REQUIRED, NULL, "Path to the Initial Condition File");
    param_declare_string(ps, "OutputDir",    REQUIRED, NULL, "Prefix to the output files");

    static ParameterEnum DensityKernelTypeEnum [] = {
        {"cubic", DENSITY_KERNEL_CUBIC_SPLINE},
        {"quintic", DENSITY_KERNEL_QUINTIC_SPLINE},
        {"quartic", DENSITY_KERNEL_QUARTIC_SPLINE},
        {NULL, DENSITY_KERNEL_QUARTIC_SPLINE},
    } ;
    param_declare_enum(ps,    "DensityKernelType", DensityKernelTypeEnum, OPTIONAL, "quintic", "SPH density kernel to use. Supported values are cubic, quartic and quintic.");
    param_declare_string(ps, "SnapshotFileBase", OPTIONAL, "PART", "Base name of the snapshot files, _%03d will be appended to the name.");
    param_declare_string(ps, "FOFFileBase", OPTIONAL, "PIG", "Base name of the fof files, _%03d will be appended to the name.");
    param_declare_string(ps, "EnergyFile", OPTIONAL, "energy.txt", "File to output energy statistics.");
    param_declare_int(ps,    "OutputEnergyDebug", OPTIONAL, 0, "Should we output energy statistics to energy.txt");
    param_declare_string(ps, "CpuFile", OPTIONAL, "cpu.txt", "File to output cpu usage information");
    param_declare_string(ps, "OutputList", REQUIRED, NULL, "List of output scale factors.");

    /*Potential plane parameters*/
    param_declare_string(ps, "PlaneOutputList", OPTIONAL, NULL, "List of potential plane output scale factors.");
    param_declare_int(ps, "PlaneResolution", OPTIONAL, 256, "Number of pixels per dimension in the potential plane (should be an even number).");
    param_declare_double(ps, "PlaneThickness", OPTIONAL, -1, "Thickness of the potential plane in the normal direction in internal gadget units (kpc/h by default).");
    param_declare_string(ps, "PlaneCutPoints", OPTIONAL, NULL, "List of potential plane cut points in the normal direction in internal gadget units (kpc/h by default).");
    param_declare_string(ps, "PlaneNormals", OPTIONAL, "\"0, 1, 2\"", "List of potential plane normal directions (0=x, 1=y, 2=z).");

    /*Cosmology parameters*/
    param_declare_double(ps, "Omega0", REQUIRED, 0.2814, "Total matter density at z=0");
    param_declare_double(ps, "CMBTemperature", OPTIONAL, 2.7255,
            "Present-day CMB temperature in Kelvin, default from Fixsen 2009; affects background if RadiationOn is set.");
    param_declare_double(ps, "OmegaBaryon", OPTIONAL, -1, "Baryon density at z=0");
    param_declare_double(ps, "OmegaLambda", OPTIONAL, -1, "Dark energy density at z=0");
    param_declare_double(ps, "Omega_fld", OPTIONAL, 0, "Energy density of dark energy fluid.");
    param_declare_double(ps, "w0_fld", OPTIONAL, -1., "Dark energy equation of state.");
    param_declare_double(ps, "wa_fld", OPTIONAL, 0, "Dark energy evolution parameter.");
    param_declare_double(ps, "Omega_ur", OPTIONAL, 0, "Extra radiation density, eg, a sterile neutrino");

    param_declare_double(ps, "HubbleParam", OPTIONAL, -1, "Hubble parameter. Does not affect gravity. Used only for cooling and star formation.");
    /*End cosmology parameters*/

    param_declare_int(ps,    "OutputPotential", OPTIONAL, 1, "Save the potential in snapshots.");
    param_declare_int(ps,    "OutputTimebins", OPTIONAL, 0, "Save the particle timebins in snapshots, for debugging.");
    param_declare_int(ps,    "OutputHeliumFractions", OPTIONAL, 0, "Save the helium ionic fractions in snapshots.");
    param_declare_int(ps,    "OutputDebugFields", OPTIONAL, 0, "Save a large number of debug fields in snapshots.");
    param_declare_int(ps,    "ShowBacktrace", OPTIONAL, 1, "Print a backtrace on crash. Hangs on stampede.");
    param_declare_double(ps,    "MaxMemSizePerNode", OPTIONAL, 0.6, "Pre-allocate this much memory per computing node/ host, in MB. Passing < 1 allocates a fraction of total available memory per node, defaults to 0.6 available memory.");
    param_declare_double(ps, "AutoSnapshotTime", OPTIONAL, 0, "Seconds after which to automatically generate a snapshot if nothing is output.");

    param_declare_double(ps, "TimeMax", OPTIONAL, 1.0, "Scale factor to end run.");
    param_declare_double(ps, "TimeLimitCPU", REQUIRED, 0, "CPU time to run for in seconds. Code will stop if it notices that the time to end of the next PM step is longer than the remaining time.");

    param_declare_int   (ps, "MaxDomainTimeBinDepth", OPTIONAL, 8, "Forces a domain decompositon every 2^MaxDomainTimeBinDepth timesteps.");
    param_declare_int   (ps, "DomainOverDecompositionFactor", OPTIONAL, -1, "Create on average this number of sub domains on a MPI rank. Higher numbers improve the load balancing. For optimal tree building efficiency, use one domain per thread (the default).");
    param_declare_double(ps, "RandomParticleOffset", OPTIONAL, 8., "Internally shift the particles within a periodic box by a random fraction of a PM grid cell each domain decomposition, ensuring that tree openings are decorrelated between timesteps. This shift is subtracted before particles are saved.");

    param_declare_int   (ps, "DomainUseGlobalSorting", OPTIONAL, 1, "Determining the initial refinement of chunks globally. Enabling this produces better domains at costs of slowing down the domain decomposition.");
    param_declare_double(ps, "ErrTolIntAccuracy", OPTIONAL, 0.02, "Controls the length of the short-range timestep. Smaller values are shorter timesteps.");
    param_declare_double(ps, "ErrTolForceAcc", OPTIONAL, 0.002, "Force accuracy required from tree. Controls tree opening criteria. Lower values are more accurate.");
    param_declare_double(ps, "BHOpeningAngle", OPTIONAL, 0.175, "Barnes-Hut opening angle. Alternative purely geometric tree opening angle. Lower values are more accurate.");
    param_declare_double(ps, "MaxBHOpeningAngle", OPTIONAL, 0.9, "Barnes-Hut opening angle, applied in addition to the relative aceleration criterion. Lower values are more accurate.");
    param_declare_double(ps, "TreeRcut", OPTIONAL, 6, "Number of mesh cells at which we cease walking.");
    param_declare_int(ps, "TreeUseBH", OPTIONAL, 2, "If 1, use Barnes-Hut opening angle rather than the standard Gadget acceleration based opening angle. If 2, use BH criterion for the first timestep only, before we have relative accelerations.");
    param_declare_int(ps, "SplitGravityTimestepsOn", OPTIONAL, 1, "This flag enables the momentum conserving hierarchical timestepping, where only active particles gravitate, from Gadget 4, for the short-range gravity, and splits the hydro and gravitational timesteps.");

    param_declare_double(ps, "Asmth", OPTIONAL, 1.5, "The scale of the short-range/long-range force split in units of FFT-mesh cells."
                                                      "Larger values suppresses grid anisotropy. ShortRangeForceWindowType = erfc supports any value. 'exact' only supports 1.5. ");
    param_declare_int(ps,    "Nmesh", OPTIONAL, -1, "Size of the PM grid on which to compute the long-range force.");

    static ParameterEnum ShortRangeForceWindowTypeEnum [] = {
        {"exact", SHORTRANGE_FORCE_WINDOW_TYPE_EXACT},
        {"erfc", SHORTRANGE_FORCE_WINDOW_TYPE_ERFC },
        {NULL, SHORTRANGE_FORCE_WINDOW_TYPE_EXACT },
    };
    param_declare_enum(ps,    "ShortRangeForceWindowType", ShortRangeForceWindowTypeEnum, OPTIONAL, "exact", "type of shortrange window, exact or erfc (default is exact) ");

    param_declare_double(ps, "MinGasHsmlFractional", OPTIONAL, 0, "Minimal gas Hsml as a fraction of gravity softening.");
    param_declare_double(ps, "MaxGasVel", OPTIONAL, 3e5, "Maximal limit on the gas velocity in km/s. By default speed of light.");

    /*Setting MaxSizeTimestep = 0.05 increases the power on large scales by a constant factor of 1.002.*/
    param_declare_double(ps, "MaxSizeTimestep", OPTIONAL, 0.1, "Maximum size of the PM timestep (as delta-a).");
    param_declare_double(ps, "MinSizeTimestep", OPTIONAL, 0, "Minimum size of the PM timestep.");
    param_declare_int(ps, "ForceEqualTimesteps", OPTIONAL, 0, "Force all (tree) timesteps to be the same, and equal to the smallest required.");

    /* MaxRMSDisplacementFac = 0.1 increases the power on large scales by a small constant factor of 1.0005. */
    param_declare_double(ps, "MaxRMSDisplacementFac", OPTIONAL, 0.2, "Controls the length of the PM timestep. Max RMS displacement per timestep in units of the mean particle separation.");
    param_declare_double(ps, "ArtBulkViscConst", OPTIONAL, 0.75, "Artificial viscosity constant for SPH.");
    param_declare_double(ps, "CourantFac", OPTIONAL, 0.15, "Courant factor for the timestepping.");
    param_declare_double(ps, "DensityResolutionEta", OPTIONAL, 1.0, "Resolution eta factor (See Price 2008) 1 = 33 for Cubic Spline");

    param_declare_double(ps, "DensityContrastLimit", OPTIONAL, 100, "Has an effect only if DensityIndepndentSphOn=1. If = 0 enables the grad-h term in the SPH calculation. If > 0 also sets a maximum density contrast for hydro force calculation.");
    param_declare_double(ps, "MaxNumNgbDeviation", OPTIONAL, 2, "Maximal deviation from the desired number of neighbours for each SPH particle.");
    param_declare_double(ps, "HydroCostFactor", OPTIONAL, 1, "Unused.");

    param_declare_int(ps, "BytesPerFile", OPTIONAL, 1024 * 1024 * 1024, "number of bytes per file");
    param_declare_int(ps, "NumWriters", OPTIONAL, 0, "Max number of concurrent writer processes. 0 implies Number of Tasks; ");
    param_declare_int(ps, "MinNumWriters", OPTIONAL, 1, "Min number of concurrent writer processes. We increase number of Files to avoid too few writers. ");
    param_declare_int(ps, "WritersPerFile", OPTIONAL, 8, "Number of Writer groups assigned to a file; total number of writers is capped by NumWriters.");

    param_declare_int(ps, "EnableAggregatedIO", OPTIONAL, 1, "Reduces the number of open files in snapshots so that each file has size BytesPerFile.");
    param_declare_int(ps, "AggregatedIOThreshold", OPTIONAL, 256, "Max size (in MB) on a writer before reverting to throttled IO.");

    /*Parameters of the cooling module*/
    param_declare_int(ps, "CoolingOn", REQUIRED, 0, "Enables cooling");
    param_declare_string(ps, "TreeCoolFile", OPTIONAL, "", "Path to the Cooling Table");
    param_declare_string(ps, "MetalCoolFile", OPTIONAL, "", "Path to the Metal Cooling Table. Empty string disables metal cooling. Refer to cooling.c");
    param_declare_string(ps, "ReionHistFile", OPTIONAL, "", "Path to the file containing the helium III reionization table. Used if QSOLightupOn = 1.");
    param_declare_string(ps, "UVFluctuationFile", OPTIONAL, "", "Path to the UVFluctation Table. Refer to cooling.c.");
    param_declare_double(ps, "HIReionTemp", OPTIONAL, 0, "Boost the particle temperature to this value during the timestep when it undergoes HI reionization. Do not boost star-forming gas. 1807.09282 suggests a boost of 20000.");
    param_declare_double(ps, "UVRedshiftThreshold", OPTIONAL, -1.0, "Earliest Redshift that UV background is enabled. This modulates UVFluctuation and TreeCool globally. Default -1.0 means no modulation.");
    static ParameterEnum CoolingTypeTable [] = {
        {"KWH92", KWH92 },
        {"Enzo2Nyx", Enzo2Nyx },
        {"Sherwood", Sherwood },
        {NULL, Cen92 },
    };
    static ParameterEnum RecombTypeTable [] = {
        {"Cen92", Cen92 },
        {"Verner96", Verner96 },
        {"Badnell06", Badnell06},
        {NULL, Cen92 },
    };
    param_declare_enum(ps, "CoolingRates", CoolingTypeTable, OPTIONAL, "Sherwood", "Which cooling rate table to use. Options are KWH92 (old gadget default), Enzo2Nyx and Sherwood (new default).");
    param_declare_enum(ps, "RecombRates", RecombTypeTable, OPTIONAL, "Verner96", "Which recombination rate table to use. Options are Cen92 (old gadget default), Verner96 (new default), Badnell06");
    param_declare_int(ps, "SelfShieldingOn", OPTIONAL, 1, "Enable a correction in the cooling table for self-shielding.");
    param_declare_double(ps, "PhotoIonizeFactor", OPTIONAL, 1, "Scale the TreeCool table by this factor.");
    param_declare_int(ps, "PhotoIonizationOn", OPTIONAL, 1, "Should PhotoIonization be enabled.");
    /* End cooling module parameters*/

    param_declare_int(ps, "HydroOn", OPTIONAL, 1, "Enables hydro force");
    param_declare_int(ps, "DensityOn", OPTIONAL, 1, "Enables SPH density computation.");
    param_declare_int(ps, "DensityIndependentSphOn", REQUIRED, 1, "Enables density-independent (pressure-entropy) SPH.");
    param_declare_int(ps, "LightconeOn", OPTIONAL, 0, "Enables a wildly experimental lightcone algorithm that writes particles crossing a lightcone boundary to a file. May not work!");
    param_declare_int(ps, "TreeGravOn", OPTIONAL, 1, "Enables tree gravity");
    param_declare_int(ps, "RadiationOn", OPTIONAL, 1, "Include radiation density in the background evolution.");
    param_declare_int(ps, "FastParticleType", OPTIONAL, 2, "Particles of this type will not decrease the long-range timestep. Default neutrinos.");
    param_declare_double(ps, "PairwiseActiveFraction", OPTIONAL, 0, "Pairwise gravity instead of tree gravity is used if N(active particles) / N(particles) is less than this. Currently unimplemented as slower.");

    param_declare_double(ps, "GravitySoftening", OPTIONAL, 1./30., "Gravitational Softening. Units of mean separation of DM. ForceSoftening is 2.8 times this.");
    param_declare_int(ps, "GravitySofteningGas", OPTIONAL, 1, "Unused. Previously was for adaptive softening.");

    param_declare_int(ps, "GasTidalField", OPTIONAL, 0, "If 1, compute local tidal field eigenvalues for gas particles before each snapshot.");

    param_declare_double(ps, "ImportBufferBoost", OPTIONAL, 2., "Memory factor to allow for there being more particles imported during treewlk than exported. Increase this if code crashes during treewalk with out of memory.");
    param_declare_double(ps, "PartAllocFactor", OPTIONAL, 1.5, "Over-allocation factor of particles. The load can be imbalanced to allow for the work to be more balanced.");
    param_declare_double(ps, "TopNodeAllocFactor", OPTIONAL, 0.5, "Initial TopNode allocation as a fraction of maximum particle number.");
    param_declare_double(ps, "SlotsIncreaseFactor", OPTIONAL, 0.01, "Percentage factor to increase slot allocation by when requested.");

    param_declare_double(ps, "InitGasTemp", OPTIONAL, -1, "Initial gas temperature. By default set to CMB temperature at starting redshift.");
    param_declare_double(ps, "MinGasTemp", OPTIONAL, 5, "Minimum gas temperature");

    param_declare_int(ps, "ParticlesAlwaysSorted", OPTIONAL, 0, "If enabled, peano-sort all particles after domain exchange. Much slower, but good for testing.");

    param_declare_int(ps, "SnapshotWithFOF", REQUIRED, 0, "Enable Friends-of-Friends halo finder.");
    param_declare_int(ps, "FOFPrimaryLinkTypes", OPTIONAL, 2, "2^ particle types to use as primary FOF targets.");
    param_declare_int(ps, "FOFSecondaryLinkTypes", OPTIONAL, 1+16+32, "2^ particle types to link to nearest primaries.");
    param_declare_int(ps, "FOFSaveParticles", OPTIONAL, 1, "Save particles in the FOF catalog.");
    param_declare_double(ps, "FOFHaloLinkingLength", OPTIONAL, 0.2, "Linking length for Friends of Friends halos.");
    param_declare_int(ps, "FOFHaloMinLength", OPTIONAL, 32, "Minimum number of particles per FOF Halo.");
    param_declare_int(ps, "FOFMinPrimaryLength", OPTIONAL, 0, "Minimum number of primary-linking-type particles per FOF Halo. Halos with fewer primary particles are not counted as a FOF (excluded from the catalog and from BH seeding). 0 disables the filter.");
    param_declare_int(ps, "FOFPotentialMin", OPTIONAL, 0, "Track the position of minimum gravitational potential in FOF groups.");
    param_declare_double(ps, "MinFoFMassForNewSeed", OPTIONAL, 2, "Minimal halo mass for seeding tracer particles in internal mass units.");
    param_declare_double(ps, "MinMStarForNewSeed", OPTIONAL, 5e-4, "Minimal stellar mass in halo for seeding black holes in internal mass units.");
    param_declare_double(ps, "TimeBetweenSeedingSearch", OPTIONAL, 1.04, "Scale factor fraction increase between Seeding Attempts.");

    /* Second FOF (star-primary) */
    param_declare_int(ps, "SecondFOFOn", OPTIONAL, 0, "Enable second FOF pass for star clusters.");
    param_declare_int(ps, "SecondFOFPrimaryLinkTypes", OPTIONAL, 16, "2^ particle types for primary linking in second FOF.");
    param_declare_int(ps, "SecondFOFSecondaryLinkTypes", OPTIONAL, 1, "2^ particle types for secondary linking in second FOF.");
    param_declare_double(ps, "SecondFOFLinkingLength", OPTIONAL, 0.1, "Linking length for the second FOF, in units of the mean DM interparticle separation (same convention as FOFHaloLinkingLength). Multiplied internally by BoxSize / NTotalInit[1]^(1/3) to get the comoving code-unit length.");
    param_declare_int(ps, "SecondFOFMinLength", OPTIONAL, 32, "Minimum particle count per group in second FOF.");
    param_declare_int(ps, "SecondFOFMinPrimaryLength", OPTIONAL, 0, "Minimum number of primary-linking-type particles per group in second FOF. Groups with fewer primary particles are dropped from the SecPIG catalog. 0 disables the filter.");
    param_declare_string(ps, "SecondFOFFileBase", OPTIONAL, "SecPIG", "Base name of the second FOF catalog files.");
    param_declare_int(ps, "SecondFOFSize", OPTIONAL, 0, "Compute group size properties (R50, R90, Rmax) in second FOF.");
    param_declare_int(ps, "SecondFOFGridLinking", OPTIONAL, 0, "Primary-linking algorithm for the second FOF only; the halo FOF always uses the treewalk. 0 = the original treewalk (default). 1 = grid/cell-collapse linker: identical groups, but avoids the O(N*n_ngb) neighbour enumeration that makes the star FOF the dominant cost of the run. 2 = run both and abort on any label mismatch (validation; deliberately pays the full cost of the slow path). The default is 0 until one production snapshot has been run at 2 and the resulting SecPIG catalogue and seeded particle IDs compared against a treewalk run; switch to 1 after that. A collective preflight falls back to the treewalk on its own when the grid path is unsafe: garbage/swallowed primary particles present, MPI int count limits, or insufficient free memory.");
    param_declare_int(ps, "SecFOFonly", OPTIONAL, 0, "If 1, skip saving the primary FOF catalog (PIG) and only save the second FOF catalog (SecPIG). Primary FOF still runs internally.");
    param_declare_int(ps, "SeedInSecFOFasStarCluster", OPTIONAL, 0, "If 1, use StarCluster BH-seeding in the secondary FOF catalog. Requires SecondFOFOn=1 and StarClusterOn=1.");
    param_declare_int(ps, "SeedSecFOFcomSample", OPTIONAL, 0, "If 1, seed BHs in the secondary FOF by one combined star-cluster sampling per group (mass function cutoff = group total unseeded stellar mass); seed mass set by clusters > 1e4 Msun. Requires SeedInSecFOFasStarCluster=1 and MinMscForBHseed>0.");
    param_declare_int(ps, "SeedSecFOFcomSampleParticle", OPTIONAL, 0, "Only used with SeedSecFOFcomSample=1. If 1, the combined per-secFOF sampling is replaced by a per-star-particle sampling: each unseeded star draws its own cluster population (mass function cutoff = min(M_cstar, group total unseeded stellar mass)), the >1e4 Msun clusters are summed per star and over the group into tot_msc_fof, which (optionally capped, then required >= MinMscForBHseed) sets the seed mass and is redistributed into each unseeded star's ClusterMass weighted by stellar mass.");
    param_declare_int(ps, "SCmasscapSecFOFstarmass", OPTIONAL, 0, "Only used with SeedSecFOFcomSample=1. If 1, a group cannot turn more stellar mass into star clusters than it has unseeded stars: the sampled cluster population is truncated in DRAW ORDER at Mcut (= the group's total unseeded stellar mass, the same quantity used as the ICMF cutoff). Clusters are accepted until the running total reaches Mcut, the cluster that crosses it is shortened to the remaining budget so the total lands exactly on Mcut, and every later cluster is dropped. Truncating in draw order rather than after any sort is deliberate: with StarClusterICMFcutoff=0 the pure m^-2 ICMF has no knowledge of the host, so a small group can draw a cluster heavier than all its stars, and deleting a prefix of the draw leaves the survivors an unbiased ICMF sample, whereas truncating a sorted list would only ever delete the lightest clusters. Applies in all three seeding modes: the summed mode (SecFOFseedsumover=1), the per-cluster mode (SecFOFseedsumover=0, where it also lowers the number of clusters that reach MinMscForBHseed and hence the number of seeds), and the per-particle mode (SeedSecFOFcomSampleParticle=1, where the per-star draws use a per-star cutoff so the cap is instead applied to the group-summed total). If 0, no cap.");
    param_declare_int(ps, "SeedInSecFOFMultipleSeeds", OPTIONAL, 0, "Only used with SeedInSecFOFasStarCluster star-cluster seeding. If 1, a secondary-FOF group whose seeding cluster mass M_SC exceeds 1e8 Msun seeds N_seed=floor(M_SC/1e8) equal-mass BHs (capped by the number of unseeded stars) instead of one: seed 1 is the largest-m*Gamma star, the others are the next-largest unseeded stars lying farther than 2*GravitySoftening from seed 1. If 0, always one seed per group.");
    param_declare_int(ps, "SecFOFseedsumover", OPTIONAL, 1, "Only used with SeedSecFOFcomSample=1. If 1 (default), the combined per-secFOF star-cluster draw is summed into ONE BH seed per group (the classic SeedSecFOFcomSample behavior). If 0, per-cluster seeding: every sampled cluster with mass >= MinMscForBHseed seeds its own BH (mass SeedBlackHoleMass*m_sc when BHseedMassScaleMsc=1, else SeedBlackHoleMass), each hosted on a distinct unseeded star of the group chosen per SeedInSecFOFRandomStarParticle. If the eligible clusters outnumber the unseeded stars, a message is printed and the remaining clusters in that group are skipped. SecFOFseedsumover=0 requires SeedSecFOFcomSample=1 and is incompatible with SeedSecFOFcomSampleParticle, SeedInSecFOFMultipleSeeds and SeedSeedFOFMassiveBoundStar. SCmasscapSecFOFstarmass does apply in this mode, as a draw-order truncation of the cluster population at the group's unseeded stellar mass.");
    param_declare_int(ps, "SeedInSecFOFRandomStarParticle", OPTIONAL, 0, "Only used in the per-cluster secFOF seeding mode (SecFOFseedsumover=0; ignored with a message when SecFOFseedsumover=1). Selects the unseeded star particles that host the per-cluster BH seeds: if 1, distinct unseeded stars are randomly sampled; if 0, the N unseeded stars with the largest cluster-forming mass ClusterMass host the N seeds (generalizing the sum-over mode's largest-ClusterMass host pick).");
    param_declare_double(ps, "SecFOFseedHostZBeta", OPTIONAL, 0, "Exponent beta of the metallicity-dependent host-star ranking proxy for star-cluster BH seeding. If > 0, every RANKED host-star choice -- which unseeded star particle is converted into the BH -- uses the key Gamma*m_star * Z^-beta instead of the raw Gamma*m_star, preferring metal-poor stars at fixed cluster-forming mass (motivated by the CW-model seeding yield: the minimum cluster mass that leaves a BH scales as m_crit ~ Z^beta with beta ~ 0.3). Z is the star's frozen BirthMetallicity (absolute mass fraction; any constant normalisation cancels in a ranking), floored at 10^-7 -- the same pristine-star floor the unseeded-star metallicity statistics use -- so Z <= 0 stars get a large but finite boost. If <= 0 (the default), the ranking is the raw Gamma*m_star one, bit-for-bit the historical behavior. The weight applies to the sum-over host pick, the per-cluster ordinary and MinBHSeedInSC compensating host ordering, the SeedInSecFOFMultipleSeeds extra hosts, and the bound-host repointing of BHseedSecFOFbound / SeedSeedFOFMassiveBoundStar; it does NOT apply when SeedInSecFOFRandomStarParticle=1 draws random hosts. RANKING ONLY: the seeding budgets, cluster draws, gates and seed masses are untouched -- but note the group reference star (SeedStarID) moves to the largest-proxy star, and since that ID seeds the per-group cluster-sampler RNG, the per-group draw REALIZATION changes (same statistics, different random numbers).");
    param_declare_double(ps, "SecFOFseedHostZFloor", OPTIONAL, 0, "Metallicity floor of the SecFOFseedHostZBeta host-star ranking proxy, in units of solar metallicity (Zsun = 0.0134, the same value the CW seed-mass model uses). If > 0 and SecFOFseedHostZBeta > 0, the ranking key becomes Gamma*m_star * max(Z, Zfloor)^-beta: every unseeded star with BirthMetallicity below the floor gets the same weight Zfloor^-beta, so among the sub-floor stars the raw Gamma*m_star decides, and only stars above the floor are penalised for their metals. Motivated by the post-hoc host-rule tests: without a floor the Z^-beta key is dominated by the rare very-metal-poor stars in the outskirts (Gamma*m_star varies by only ~10% across a galaxy while Z spans 2-3 dex in its tail), so seeds land at 2-3 R50 even at z ~ 11-13; a floor of ~0.1 Zsun keeps the seeds central while the galaxy is metal-poor and lets them move outward only once the local Z rises above the floor. If <= 0 (the default), only the pristine-star floor of 10^-7 absolute applies (the historical SecFOFseedHostZBeta behavior). Has no effect when SecFOFseedHostZBeta <= 0 (a message is printed) or for SeedInSecFOFRandomStarParticle=1 random hosts. RANKING ONLY, like SecFOFseedHostZBeta: budgets, cluster draws, gates and seed masses are untouched.");
    param_declare_int(ps, "SecFOFseedHostZcrit", OPTIONAL, 0, "Only used in the per-cluster secFOF seeding mode with MbhMscRelationCWmodel=1 (SeedSecFOFcomSample=1, SecFOFseedsumover=0, SeedSecFOFcomSampleParticle=0; anything else is an error when this is 1). If 1, the star particle that hosts each ordinary BH seed is restricted by a per-cluster critical metallicity: only unseeded stars with frozen BirthMetallicity <= Z_crit = Z_cl * (M_VMS / SeedBlackHoleMass)^(2.1/0.74) may host the seed, where Z_cl is the metallicity fed to the CW model for that cluster (CWmodelMetallicity mode) and M_VMS its model seed mass. Z_crit is the metallicity at which the SAME cluster (mass, radius, age fixed) would just have reached the seed-mass floor -- from the Vink-2018 wind equilibrium (1-f_vms) Mdot_in = C M^2.1, C ~ Z^0.74, so M_VMS ~ Z^-0.352 at fixed cluster -- i.e. the mask puts the seed on a star whose environment could have produced it, which brings the seeds back towards the galaxy centre instead of the metal-poor outskirts. Among the stars that pass, the usual host ranking decides: raw Gamma*m_star, or Gamma*m_star * max(Z, Zfloor)^-beta when SecFOFseedHostZBeta > 0 (the two combine: mask first, Z-weighted ranking among the survivors), or the random key when SeedInSecFOFRandomStarParticle=1. If no unseeded star of the group passes, the lowest-metallicity unseeded star hosts the seed (fallback; counted in the per-pass message). To make the masked pick reachable the per-cluster host gather keeps EVERY unseeded (bound) star of a requesting group instead of the top n_request by key, so the gathered candidate list is larger (one entry per unseeded star of the requesting groups, replicated on every rank). Density-capped clusters (0.01 M_cl bypass) and clusters whose M_VMS was capped at the cluster mass use the same formula on the capped M_VMS, which only lowers Z_crit. MinBHSeedInSC compensating seeds carry no per-cluster M_VMS and are NOT masked. If 0 (the default), the host choice is the historical one, bit-for-bit. MASK ONLY: seeding budgets, cluster draws, gates and seed masses are untouched. Whenever this is 1 or SecFOFseedHostZBeta > 0 the startup log describes the full host-search criterion.");
    param_declare_int(ps, "MbhMscRelationCWmodel", OPTIONAL, 0, "Only used in the per-cluster secFOF seeding mode (requires SeedSecFOFcomSample=1 and SecFOFseedsumover=0). If 1, the seed mass of each sampled cluster >= MinMscForBHseed is set by the Williams et al. 2026 stellar-collision VMS model (M_BH = M_VMS from the cluster mass, its size-mass-relation virial radius r_max=1.4*Reff, the host secFOF's unseeded-star metal mass ratio, and the age of the universe at seeding). BHseedMassScaleMsc is ignored in this mode; SeedBlackHoleMass instead acts as the lower seed-mass limit: a BH is only seeded when M_VMS >= SeedBlackHoleMass (which also drops clusters with no net inflow, M_VMS=0). M_VMS is capped at the cluster mass. The seeding budget Gamma*m_star carries no metallicity weighting, so the CW model's own Vink-wind Z dependence is the only metallicity dependence in the seeding -- unless SecFOFseedHostZBeta > 0, which Z-weights the HOST-STAR ranking (where the BH is placed), still leaving the budget unweighted.");
    param_declare_int(ps, "SeedSeedFOFMassiveBoundStar", OPTIONAL, 0, "Only used with SeedSecFOFcomSample=1 (and NOT SeedSecFOFcomSampleParticle, NOT SeedInSecFOFMultipleSeeds). If 1, a secondary-FOF group whose unseeded star-cluster mass Sum(m*Gamma) exceeds 1e8 Msun is restricted to the unseeded star particles gravitationally bound to the secFOF before seeding: the softened potential at each star is summed over ALL secFOF members, the rest frame is the deepest-potential member's velocity, and only the bound unseeded stars' Sum(m*Gamma) and stellar mass feed the combined sampler (and thus the seed mass when BHseedMassScaleMsc=1). If 0, the whole structure is used.");
    param_declare_int(ps, "BHseedSecFOFbound", OPTIONAL, 0, "Only used with SeedInSecFOFasStarCluster=1. Restricts the star particles that contribute Gamma*m_star to the BH-seeding budget of a secondary-FOF group to the ones gravitationally bound to that group. 0 = off (every member star contributes, the default). 1 = keep only stars bound to (all member stars + all DM inside Rmax). 2 = keep only stars bound to (all member stars + the DM inside min(2*R50, Rmax)); Rmax is set by the single most distant member star and follows a percolating FoF, so 2*R50 keeps the DM test on the object itself. The potential is the spherically-averaged profile about SecPotMinPos (softened, O(N log N)); the rest frame is the mass-weighted centre-of-mass VELOCITY of the same system that potential is built from -- all member stars PLUS the DM inside the sphere -- so the kinetic and potential terms refer to one frame. (Stars alone would be the wrong frame here: the DM dominates the well being tested, and this is the Eq.-2 test of Williams et al. 2025 Sec 2.3, whose stars-only Eq.-1 counterpart is a different test in a different frame.) The frame is computed once over ALL member stars, including the ones the test then declares unbound; there is no iterative unbinding. Everything the seeding decision reads off the unseeded stars is restricted together: the cluster-mass budget, the M_cut cutoff, the host pool and seed count, AND the unseeded-star metallicity summary the CW seed-mass model draws each cluster's Z from (CWmodelMetallicity 'ave' / 'lognormal' / 'uniform' all read it), so an unbound star can no longer move M_VMS and hence the BH seed mass while contributing no mass and hosting no seed. The unseeded-star COUNT is restricted with it, since it is the denominator of those equal-weight sums; that also tightens the multi-seed host cap, which is correct as an unbound star can never host a seed. The SecPIG catalogue and StarClusterDetails still describe ALL member stars, with the bound subset reported in the added SecBound* / Bound* fields (0 when this is off) -- except the StarClusterDetails MetUnseeded* columns and Metallicity, which are written at seeding time and therefore describe the bound unseeded stars, matching the Z that was actually used. Those diagnostics use the SAME definitions as the totals beside them, so every bound fraction is a ratio of like for like: SecBoundStarMass/SecMassByType[4], SecBoundSCMass/SecSCMass (cluster mass over all bound stars) and SecBoundSCMass_unseeded/(SecSCMass - SecSCMass_seeded) (over the bound unseeded ones). The restriction is applied at GROUP level (the summed budget and the single seed host). Per-cluster seeding (SecFOFseedsumover=0) is also supported: it picks its own hosts, so it is given a transient per-particle bound flag that restricts its host pool, and with it the seed count (a group places min(n_request, bound hosts) seeds). That flag lives only inside the seeding call, so the modes that re-scan the unseeded stars outside it remain incompatible and abort at startup: SeedSecFOFcomSampleParticle=1 (its per-star sampler draws for every unseeded star, unbound ones included, so they keep feeding tot_msc_fof), SeedInSecFOFMultipleSeeds=1 (extra hosts are not boundedness-tested) and SeedSeedFOFMassiveBoundStar=1 (same job, would apply twice). The same flag also restricts which stars a seeded group marks Seeded=1, in BOTH seeding paths: only the bound stars are spent, so an unbound star keeps its cluster mass and contributes at a later seeding search if it becomes bound. The restriction therefore DEFERS the unbound stars rather than consuming them.");
    param_declare_int(ps, "SecFOFStarCluster", OPTIONAL, 1, "Flag that second FOF groups are star clusters. Requires SecondFOFOn=1 and StarClusterOn=1. Writes StarClusterMass/Metallicity/MetalElemMass to SecPIG.");
    param_declare_int(ps, "SecFOFUnseededPart", OPTIONAL, 0, "If 1, the second FOF uses only unseeded star particles (Type 4 with Seeded==0) as primary-linking particles; seeded stars are dropped from the primary-linking set (other configured primary types such as gas are unaffected). Requires StarClusterOn=1.");

    /*Black holes*/
    param_declare_int(ps, "BlackHoleOn", REQUIRED, 1, "Master switch to enable black hole formation and feedback. If this is on, type 5 particles are treated as black holes.");
    param_declare_int(ps, "MetalReturnOn", REQUIRED, 1, "Enable the return of metals from star particles to the gas.");

    param_declare_double(ps, "BlackHoleAccretionFactor", OPTIONAL, 100, "BH accretion boosting factor relative to the rate from the Bondi accretion model.");
    param_declare_double(ps, "BlackHoleEddingtonFactor", OPTIONAL, 2.1, "Maximum Black hole accretion as a function of Eddington.");
    param_declare_double(ps, "SeedBlackHoleMass", OPTIONAL, 2e-5, "Mass of initial black hole seed in internal mass units. If this is too much smaller than the gas particle mass, BH will not accrete. When MaxSeedBlackHoleMass >0, this is the lower limit of the BHseed mass. When BHseedMassScaleMsc = 1, this is in the unit of star cluster mass.");
    param_declare_double(ps, "MaxSeedBlackHoleMass", OPTIONAL, 0, "Black hole seed masses are drawn from a power law. This is the upper limit on the BH seed mass. If <= 0 then all BHs have the SeedBlackHoleMass and the power law is disabled.");
    param_declare_double(ps, "SeedBlackHoleMassIndex", OPTIONAL, -2, "Power law index of the seed mass distribution");


    param_declare_int(ps, "BlackHoleSeedHaloBased", OPTIONAL, 1, "Used halo-based black hole seeding prescriptions");
    param_declare_int(ps, "BlackHoleSeedStarCluster", OPTIONAL, 0, "Enable StarCluster-based black hole seeding.");
    param_declare_int(ps, "BlackHoleSeedGasBased", OPTIONAL, 0, "Used gas-based black hole seeding prescriptions");
    param_declare_int(ps, "BlackholeSeedSCparticle", OPTIONAL, 0, "If 1, seed black holes from individual star particles whose star cluster mass exceeds MinMscForBHseed. Requires StarClusterOn=1.");
    param_declare_int(ps, "BHseedEveryTimestep", OPTIONAL, 0, "If 1, seed BH from SC particles every timestep after star formation, not just during PM steps. Requires BlackholeSeedSCparticle=1.");
    param_declare_double(ps, "BlackHoleSeedsfmpGas", OPTIONAL, 0.001, "The mass of star-forming, metal-poor gas for seeding blackhole. in the unit of 1e10/hh soloarmass. Only used when BlackHoleSeedGasBased = 1");
    param_declare_double(ps, "BlackHoleseedsMetalThres", OPTIONAL, 0.0001, "The threshold of the metal-poor gas to seed blackhole, in the unit of solar metallicity (0.0127) Only used when BlackHoleSeedGasBased = 1");


    param_declare_double(ps, "BlackHoleNgbFactor", OPTIONAL, 2, "Factor by which to increase the number of neighbours for a black hole.");

    param_declare_double(ps, "BlackHoleMaxAccretionRadius", OPTIONAL, 99999., "NO EFFECT. Was maximum search radius for black holes.");
    param_declare_double(ps, "BlackHoleFeedbackFactor", OPTIONAL, 0.05, " Fraction of the black hole luminosity to turn into thermal energy");
    param_declare_double(ps, "BlackHoleFeedbackRadius", OPTIONAL, 0, "NO EFFECT. Was the comoving radius at which the black hole feedback energy was deposited. Did not affect accretion so had odd behaviour.");
    param_declare_int(ps, "BlackHoleRepositionEnabled", OPTIONAL, 0, "Enables Black hole repositioning to the potential minimum.");

    param_declare_int(ps, "BlackHoleKineticOn", OPTIONAL, 0, "Switch to AGN kinetic feedback when Eddington accretion is low.");
    param_declare_double(ps,"BHKE_EddingtonThrFactor",OPTIONAL, 0.05, "Threshold of the Eddington rate for the kinetic feedback");
    param_declare_double(ps,"BHKE_EddingtonMFactor",OPTIONAL, 0.002, "Factor for mbh-dependent Eddington threshold for the kinetic feedback");
    param_declare_double(ps,"BHKE_EddingtonMPivot",OPTIONAL, 0.05, "Pivot MBH for mbh-dependent Eddington threshold for the kinetic feedback");
    param_declare_double(ps,"BHKE_EddingtonMIndex",OPTIONAL, 2, "Powlaw index for mbh-dependent Eddington threshold for the kinetic feedback");
    param_declare_double(ps,"BHKE_EffRhoFactor",OPTIONAL, 0.05, "Factor1 for kinetic feedback efficiency, compare with BH density");
    param_declare_double(ps,"BHKE_EffCap",OPTIONAL, 0.05, "Factor2 for kinetic feedback efficiency, sets the maximum factor that converts accretion energy to kinetic feedback");
    param_declare_double(ps,"BHKE_InjEnergyThr",OPTIONAL, 5, "Factor for Minimum KineticFeedbackEnergy injection, controls the burstiness of kinetic feedback");

    param_declare_double(ps, "BlackHoleFeedbackRadiusMaxPhys", OPTIONAL, 0, "Unused.");
    param_declare_int(ps,"WriteBlackHoleDetails",OPTIONAL, 1, "If set, output BH details at every time step.");
    param_declare_int(ps, "MaxBlackHoleDetails", OPTIONAL, 50, "Max number of GB to write to bh details file before opening a new one.");
    param_declare_int(ps, "StarClusterDetails", OPTIONAL, 0, "If 1 and a star-cluster BH seeding mode is active (BlackHoleSeedStarCluster / SeedInSecFOFasStarCluster / SeedSecFOFcomSample), write one binary record per seeded star cluster to per-rank files under OutputDir/StarClusterDetails, capturing seeding steps between checkpoints.");

    param_declare_int(ps,"BH_DynFrictionMethod",OPTIONAL, 1, "If set to non-zero, dynamical friction is applied through this method. Setting BH_DynFrictionMethod = 1, = 2, = 3 uses stars only (=1), dark matter + stars (=2), all mass (=3) to compute the DF force.");
    param_declare_int(ps,"BH_DFBoostFactor",OPTIONAL, 1, "If set, dynamical friction is boosted by this factor.");
    param_declare_double(ps,"BH_DFbmax",OPTIONAL, 20, "Maximum impact range for dynamical friction. We use 20 pkpc as default value.");
    param_declare_int(ps,"BH_DRAG",OPTIONAL, 1, "Add drag force to the BH dynamic");
    param_declare_int(ps,"MergeGravBound",OPTIONAL, 1, "If set to 1, apply gravitational bound criteria for merging event. This criteria would be automatically turned off if reposition is enabled.");
    param_declare_double(ps, "SeedBHDynMass", OPTIONAL, -1, "The initial dynamic mass of BH, default -1 will use the mass of gas particle. Larger Mdyn would help to stablize the BH in the early phase if turning off reposition.");
    param_declare_int(ps, "BlackholeTidalField", OPTIONAL, 0, "If 1, compute the tidal tensor eigenvalues and tidal field strength for BH particles and record the strength in BH detail files. The tensor needs a tree of all particles: with SplitGravityTimestepsOn=0 it is recomputed at every BH gravity step, with SplitGravityTimestepsOn=1 only at PM steps and held in between.");
    param_declare_int(ps, "GWRecoilVelocityKick", OPTIONAL, 0, "If 1, apply gravitational wave recoil kick velocity to BH merger remnants assuming non-spinning BHs.");
    param_declare_int(ps, "GWRecoilSCKick", OPTIONAL, 0, "If 1, check if GW recoil kick ejects BH from its host star cluster (zero SC mass if v_kick > v_esc). Requires StarClusterOn=1.");
    param_declare_int(ps, "BHVorticity", OPTIONAL, 0, "If 1, compute dimensionless SPH vorticity of surrounding gas for each BH and record in BH detail files.");

    /*Star cluster parameters*/
    param_declare_int(ps, "StarClusterOn", OPTIONAL, 0, "Enable star-cluster bh seeding formation.");
    param_declare_int(ps, "StarClusterSampling", OPTIONAL, 0, "Enable sampling of individual star cluster masses from the mass function. Only used when StarClusterOn = 1. If on, Msc_ave, Nsc_sample, NumStarCluster, StarClusterMass_sample are computed per star particle, and StarClusterMassSample is used for BH seeding. If off, these are skipped and StarClusterMass is used instead.");
    param_declare_int(ps, "StarClusterICMFcutoff", OPTIONAL, 1, "Shape of the star-cluster initial mass function used for BH seeding. Only used with star-cluster-based seeding (StarClusterOn=1, e.g. StarClusterSampling / SeedInSecFOFasStarCluster / BlackholeSeedSCparticle). If 1 (default), n(m) ~ m^-2 exp(-m/Mcut) with the exponential cutoff at Mcut (group/M_cstar). If 0, the exponential cutoff is dropped and the pure power law p(M) ~ M^-2 is used over the [1e2, 1e8] Msun range.");
    param_declare_int(ps, "StarClusterCFEsigma", OPTIONAL, 0, "How the cluster formation efficiency Gamma of the gas (and of every new star, from its parent) is evaluated. Only used when StarClusterOn = 1. 0 (default): the tabulated Kruijssen (2012) Gamma(P/k_B) at the thermodynamic pressure, i.e. sigma_loc = sqrt(P/rho) as in E-MOSAICS. 1: Kruijssen (2012) eq. 26 evaluated directly from the physical density and the 1-D velocity dispersion of the star-forming gas neighbours measured in the density treewalk at every step (VDispGasSF; requires SCgasVDisp = 1); a particle with fewer than 8 star-forming neighbours has VDispGasSF = 0 and gets the no-turbulence limit of the model, Gamma = 0.012 * 3 Myr / (0.5 t_ff) ~ 1e-3 (n_H/cm^-3)^0.5. 2: as 1 with the dispersion of all gas neighbours (VDispGasAll). Modes 1-2 use the same model constants as the table (c_s = 0.3 km/s, eps_core = 0.5, sSFR_ff = 0.012, t_sn = 3 Myr, phi_fb = 0.16 cm^2 s^-3, t = 10 Myr, b = 0.5); only Gamma changes, the M_cstar cloud-mass scale keeps sigma_loc = sqrt(P/rho).");
    param_declare_int(ps, "SCgasVDisp", OPTIONAL, 0, "Enable gas and stellar velocity dispersion calculation for gas particles at every PM step (VDispGas/VDispStar), and the per-step 1D gas dispersions accumulated in the density treewalk from all gas neighbours (VDispGasAll) and from star-forming neighbours only (VDispGasSF, with the count VDispGasNSF; 0 below 8 star-forming neighbours).");
    param_declare_int(ps, "BHseedMassScaleMsc", OPTIONAL, 0, "When star-cluster bh seeding formation is enabled, whether the seed mass is scaled by the star cluster mass. If so, parameter SeedBlackHoleMass is in unit of Msc. If not, it is in mass unit.");
    param_declare_double(ps, "MinMscForBHseed", OPTIONAL, 0.0001, "When star-cluster bh seeding formation is enabled, the minimum star cluster mass for seeding black hole. Only used when BHseedMassScaleMsc = 1.");
    param_declare_double(ps, "MinBHSeedInSC", OPTIONAL, 0, "Only used with MbhMscRelationCWmodel=1 (per-cluster secFOF seeding). Lowest BH seed mass, in internal mass units, that a star cluster can physically produce: in the Williams et al. 2026 model only a VMS above 200 Msun collapses to a BH seed, so the value actually used is max(200 Msun, MinBHSeedInSC) and 0 (the default) disables the feature entirely. When enabled, every sampled cluster whose model M_VMS reaches this limit but which seeds no BH particle -- either because the cluster is below the resolution-set MinMscForBHseed, or because its M_VMS falls below SeedBlackHoleMass -- has its M_VMS accumulated per secFOF group into a missed BH seed mass. That mass is then compensated by seeding N = floor(missed_mass / SeedBlackHoleMass) extra BHs of mass SeedBlackHoleMass each, placed after the ordinary seeds on the remaining unseeded stars in the same ClusterMass order. Each compensating BH carries StarClusterMass = (summed cluster mass of the missed clusters)/N, and is written to StarClusterDetails with Flag=3. NOTE this evaluates the CW model over the whole sub-MinMscForBHseed cluster population (as MinMscForSCdetail=0 does), which costs order 1% of wall time, and it can multiply the number of seeded BHs several-fold.");
    param_declare_int(ps, "SecFOFseedSpendOnPlaced", OPTIONAL, 0, "Only used in the per-cluster secFOF seeding mode (SeedSecFOFcomSample=1 and SecFOFseedsumover=0). Chooses what marks a secFOF group's seedable star population spent (Seeded=1), which is what removes it from every later seeding draw. 0 (default, historical behavior) marks it as soon as the group REQUESTED a seed, i.e. drew at least one cluster >= MinMscForBHseed. 1 marks it only once the group has actually PLACED a BH particle, ordinary or MinBHSeedInSC compensating. The two differ because placement carries gates the request does not see -- with MbhMscRelationCWmodel=1 each cluster additionally needs M_VMS >= SeedBlackHoleMass, and every seed needs an unseeded host star -- so under 0 a group whose drawn clusters all fall below the seed-mass floor spends its entire cluster-mass budget without ever making a BH, and is sterilised permanently: the budget is also the ICMF cutoff SCcomMcut, so its later draws get weaker rather than stronger. Under 1 that group keeps its stars unseeded and retries at the next seeding step with a budget grown by the stars formed since, until a cluster clears the floor. NOTE the cost: a group that never seeds redraws an ever-larger cluster population at every seeding step, so seeding wall time grows over the run, and with MinMscForSCdetail lowered below MinMscForBHseed the StarClusterDetails volume grows with it.");
    param_declare_double(ps, "MinMscForSCdetail", OPTIONAL, -1, "Only used with StarClusterDetails=1 in the per-cluster secFOF seeding mode (SecFOFseedsumover=0). Lowest sampled star-cluster mass (code units) written to the StarClusterDetails files. The default -1 means MinMscForBHseed, i.e. only clusters that can seed a BH are recorded. A smaller value additionally records every sampled cluster down to it with Flag=2 and Mbh_seed=0; these carry the host group's reference star as ID/Pos since they have no host of their own. 0 records the whole drawn population down to the 100 Msun mass-function limit -- with the m^-2 ICMF that is O(MinMscForBHseed/100 Msun) times more records than the default, so the detail files can grow by several orders of magnitude.");
    param_declare_double(ps, "StarClusterFixReff", OPTIONAL, 0, "Only used in the per-cluster secFOF seeding mode (SecFOFseedsumover=0). Effective radius (in pc) assigned to every seeded star cluster. If > 0, this fixed value replaces the size-mass relation selected by StarClusterReffRelation (with its scatter) for all clusters. Default 0 uses the size-mass relation.");
    param_declare_string(ps, "StarClusterReffRelation", OPTIONAL, "BG21", "Only used with SecondFOFOn=1 (the secFOF star-cluster seeding is the only place cluster radii are sampled). Size-mass relation that sets the effective radius of every sampled star cluster, R_eff = R4 pc * (M_cl/1e4 Msun)^beta, together with the lognormal scatter drawn about it: 'BG21' (default) is the Brown & Gnedin (2021) fit to the 1-10 Myr LEGUS clusters, R4 = 2.365 pc, beta = 0.180, scatter 0.32 dex; 'GBF' is the relation the code used before this parameter existed, R4 = 1.4 pc, beta = 0.25, scatter 0.5 dex. Both are projected effective radii; the CW seed-mass model applies its own r_max = 1.4 R_eff. The radius enters the Williams et al. 2026 seed-mass model (MbhMscRelationCWmodel=1) and the StarClusterDetails Reff records; log10(R/pc) is clipped to [-1, 2] as before. StarClusterReffScatter >= 0 replaces the relation's scatter. StarClusterFixReff > 0 overrides the relation. Any other value is an error when SecondFOFOn=1 and is ignored otherwise.");
    param_declare_double(ps, "StarClusterReffScatter", OPTIONAL, -1, "Only used where a star-cluster size-mass relation is actually sampled, i.e. with SecondFOFOn=1 and StarClusterFixReff=0 (a fixed radius has no scatter). Lognormal scatter, in dex of log10(R_eff/pc), drawn about the median relation selected by StarClusterReffRelation. Negative (default -1) keeps the scatter that belongs to the selected relation, 0.32 dex for 'BG21' and 0.5 dex for 'GBF'. A value >= 0 replaces it for either relation, e.g. 0 places every cluster exactly on the median relation. The draw is otherwise unchanged (same host-star-keyed Box-Muller Gaussian, same [-1, 2] clip of log10(R/pc)), so only the width of the radius distribution changes; the active value is printed at start-up.");
    param_declare_double(ps, "CWmodelAlpha", OPTIONAL, 1.2, "Only used with MbhMscRelationCWmodel=1. Density power-law index alpha of the CW-model cluster profile rho(r) ~ r^-alpha (0 < alpha < 3). Default 1.2. The Rose et al. 2020 eccentricity functions are recomputed for this alpha, so the collision rate stays consistent.");
    param_declare_string(ps, "CWmodelMetallicity", OPTIONAL, "ave", "Only used with MbhMscRelationCWmodel=1. Metallicity fed to the CW seed-mass model for each sampled cluster. 'ave' (default): every cluster uses the host secFOF's unseeded-star metal mass ratio. 'lognormal': each cluster's log10(Z) is drawn from a normal distribution with the mean and standard deviation of the unseeded stars' log10(BirthMetallicity) (equal weight per star, NOT mass-weighted; log10(Z) floored at SC_MET_HIST_LOGMIN = -7 for pristine stars; the draw is clipped to the group's [min,max] log10(Z)). 'uniform': log10(Z) is drawn uniformly between the group's min and max unseeded-star log10(BirthMetallicity). 'starsample': each cluster takes the metallicity of a uniformly-random unseeded star of the group -- an inverse-CDF draw off the group's own log10(Z) histogram, i.e. a sample of the EMPIRICAL stellar metallicity distribution instead of a fitted shape. Sampling is with replacement, which is what drawing from an MDF means; the star supplies a metallicity only, and is unrelated to the star that is later converted into the BH particle. Prefer this over 'lognormal' when the metal-poor tail matters: a Gaussian in log10(Z) fits the bulk of the stellar MDF but overestimates the high-Z end and underestimates the low-Z end, and it is the low-Z end that produces the massive seeds (M_VMS ~ (Z/Zsun)^-0.35). The histogram spans ALL the group's unseeded stars, so the draw is unbiased -- note this is deliberately NOT taken from the gathered host-star candidate list, which is pre-truncated to the top n_request by ClusterMass and would therefore be a mass-biased subset of the stars whenever a group has more unseeded stars than requested clusters. Resolution is one histogram bin, 8/62 = 0.129 dex, except that the underflow and overflow bins return the group's exact min and max -- so raising SC_MET_HIST_LOGMIN would buy resolution at the cost of collapsing more of the metal-poor tail onto a single value (see the measured trade-off table in fof.h; do not raise it). Draws are keyed on the host star ID, so they are reproducible across ranks/restarts. Any other value is an error when MbhMscRelationCWmodel=1.");
    param_declare_double(ps, "CWmodelMetallicityMin", OPTIONAL, 0, "Only used with MbhMscRelationCWmodel=1. Lower limit, in units of Zsun = 0.0134, on the stellar metallicities that may be used when a sampled star cluster is assigned its metallicity (the Z fed to the CW seed-mass model). Default 0: every unseeded star of the host secFOF contributes, as before. If > 0, unseeded stars with BirthMetallicity below the limit are masked out of the group's metallicity statistics -- the metal-mass ratio ('ave'), the log10(Z) mean/std ('lognormal'), the min/max ('uniform') and the log10(Z) histogram ('starsample'), and therefore also the MetUnseeded* columns of the StarClusterDetails records -- so e.g. with 0.001 and CWmodelMetallicity=starsample only stars with Z >= 1e-3 Zsun are sampled. A group with no star above the limit gives every one of its clusters exactly the limit as metallicity. The mask acts on the cluster metallicity ONLY: the choice of the host star that is converted into the BH particle (Gamma*m_star ranking, SecFOFseedHostZBeta, SecFOFseedHostZcrit) is unchanged, so a star below the limit can still host the seed.");
    param_declare_double(ps, "CWmodelSeedMassCap", OPTIONAL, 1e6, "Only used with MbhMscRelationCWmodel=1. Upper limit, in solar masses (physical Msun, no h), on the CW-model seed mass of a single star cluster. A cluster whose model M_VMS (after the existing cap at the cluster mass) exceeds this value does NOT seed a BH of mass M_VMS: its seed mass is replaced by CWmodelSeedMassCapFrac times the cluster mass instead -- the same fixed-fraction fallback the model already applies to clusters above its mean-density cap. Default 1e6 Msun; 0 disables the cap and keeps the raw model value. Applied at the single chokepoint shared by the seeding path, the StarClusterDetails Mbh_seed records and the MinBHSeedInSC missed-mass sum, so all three see the same number; the SecFOFseedHostZcrit host mask is evaluated on the capped value (a stricter mask, as for density-capped clusters).");
    param_declare_double(ps, "CWmodelSeedMassCapFrac", OPTIONAL, 0.01, "Only used with MbhMscRelationCWmodel=1 and CWmodelSeedMassCap > 0. Fraction of the star cluster mass used as the BH seed mass when the cluster's CW-model M_VMS exceeds CWmodelSeedMassCap. Default 0.01 (1 per cent of the cluster, matching the model's mean-density-cap fallback). Must lie in (0, 1].");
    param_declare_int(ps, "StarClusterBHDyn", OPTIONAL, 0, "If 1, include the evolving star cluster mass in the BH dynamical mass, P[i].Mass = max(Mtrack + StarClusterMass, SeedBHDynMass); BH+SC treated as one body for dynamics. If 2, the star-cluster mass that seeded the BH (init_Msc) acts as a frozen per-BH dynamical-mass floor replacing SeedBHDynMass: P[i].Mass = max(Mtrack, init_Msc); no SC payload is attached (no SC evolution or merger SC transfer, like mode 0), the floor never changes (mergers keep the accretor's own init_Msc), and non-star-cluster seeds keep the SeedBHDynMass floor. Requires StarClusterOn=1. If 0, P[i].Mass = max(Mtrack, SeedBHDynMass) without star cluster contribution.");
    param_declare_int(ps, "SCEvolutionStellar", OPTIONAL, 0, "If 1, the star clusters attached to BH particles lose mass by stellar evolution: StarClusterMass is the birth mass times one minus the cumulative AGB + SNII + SNIa mass-return fraction of the cluster SSP at its age (same yield tables and Chabrier IMF as metal_return), updated at every active BH step. The lost mass is NOT deposited into the gas: the cluster mass is part of the star particles, whose own return (MetalReturnOn) already puts it there, so this is a mass update only and does not need MetalReturnOn. The lost mass leaves StarClusterMass (and P.Mass when StarClusterBHDyn=1) and is recorded in StarClusterTotalMassReturned. Requires StarClusterOn=1. Replaces the former StarClusterEvolution (stellar-evolution part).");
    param_declare_int(ps, "SCEvolutionRelaxation", OPTIONAL, 0, "Two-body relaxation mass loss of the star clusters attached to BH particles, applied at every active BH step. 0: off. 1: E-MOSAICS law (Kruijssen et al. 2011; Pfeffer et al. 2018 eq. 13), dM/dt = -M/t_dis with t_dis = t0_sun (M/Msun)^0.62 (T/T_sun)^-1/2, t0_sun = 21.3 Myr, T_sun = 7.01e2 Gyr^-2. 2: Gieles & Baumgardt (2008) / Alexander & Gieles (2012) law as in EMP-Pathfinder (Reina-Campos et al. 2022 eqs. 53-56), dM/dt = -xi M / t_rh with the Spitzer half-mass relaxation time (mean stellar mass 0.42 Msun, Coulomb log ln(0.11 N)) and xi = xi0 (1-P) + (3/5) zeta P (xi0 = 0.0142, zeta = 0.1), P set by r_h/r_t; r_h = (4/3) SC_Reff, the cluster's current effective radius (its initial radius SC_initReff -- the drawn R_eff, or the StarClusterReffRelation median at the cluster mass when the seeding path drew none -- unless StarClusterSizeEvolution=1 evolves it). Both use the E-MOSAICS tidal strength T = max(lambda) + Omega^2 of the BH tidal tensor (mode 2 through r_t = (G M / T)^1/3), so they require BlackholeTidalField=1: with SplitGravityTimestepsOn=0 the field is recomputed every BH gravity step, with SplitGravityTimestepsOn=1 the value from the last PM step is held. A BH without a field yet (seeded after the gravity of a PM step, it gets its first field at the next PM step when SplitGravityTimestepsOn=1) defers its relaxation, and the whole interval is applied with its first field. A cluster falling below 100 Msun is dissolved. The stripped mass leaves StarClusterMass (and P.Mass when StarClusterBHDyn=1) and is not returned to the gas. Requires StarClusterOn=1.");
    param_declare_int(ps, "StarClusterSizeEvolution", OPTIONAL, 0, "If 1, evolve the effective radius SC_Reff of the star clusters attached to BH particles with their mass loss, following Guerra et al. (2026) sect. 2.2.3 (no tidal-shock term, which is not modelled). Their half-mass-radius prescription is applied to R_eff = (3/4) r_h: at fixed profile shape both change by the same fraction. Stellar-evolution term (only when SCEvolutionStellar=1): adiabatic expansion by the inverse ratio of the cluster mass after and before each stellar-evolution mass loss. Two-body-relaxation term (only when SCEvolutionRelaxation=2, GB08): dR_eff/R_eff = (2 - zeta/xi) dm_rlx/m (their eq. 22), expanding in isolation and contracting when tidally limited; the GB08 relaxation then uses the evolved radius. The E-MOSAICS relaxation (SCEvolutionRelaxation=1) has no size term. If neither mass-evolution model is on, size evolution is switched off with a warning. The start-up log states which terms are active.");

    static ParameterEnum BlackHoleFeedbackMethodEnum [] = {
        {"mass", BH_FEEDBACK_MASS},
        {"volume", BH_FEEDBACK_VOLUME},
        {"tophat", BH_FEEDBACK_TOPHAT},
        {"spline", BH_FEEDBACK_SPLINE},
        {NULL, BH_FEEDBACK_SPLINE | BH_FEEDBACK_MASS},
    };
    param_declare_enum(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodEnum,
            OPTIONAL, "spline, mass", "");
    /*End black holes*/

    /*Star formation parameters*/
    static ParameterEnum StarformationCriterionEnum [] = {
        {"density", SFR_CRITERION_DENSITY}, /* SH03 density model for star formation*/
        {"h2", SFR_CRITERION_MOLECULAR_H2}, /* Form stars depending on the computed
                                               molecular gas fraction as a function of metallicity. */
        {"selfgravity", SFR_CRITERION_SELFGRAVITY}, /* Form stars only when the gas is self-gravitating. From Phil Hopkins.*/
        {"convergent", SFR_CRITERION_CONVERGENT_FLOW}, /* Modify self-gravitating star formation to form stars only when the gas flow is convergent. From Phil Hopkins.*/
        {"continuous", SFR_CRITERION_CONTINUOUS_CUTOFF}, /* Modify self-gravitating star formation to smooth the star formation threshold. From Phil Hopkins.*/
        {NULL, SFR_CRITERION_DENSITY},
    };

    static ParameterEnum WindModelEnum [] = {
        {"subgrid", WIND_SUBGRID}, /* If this is true, winds are spawned from the star forming gas.
                                      If false, they are spawned from neighbours of the star particle.*/
        {"decouple", WIND_DECOUPLE_SPH}, /* Specifies that wind particles are created temporarily decoupled from the gas dynamics */
        {"halo", WIND_USE_HALO}, /* Wind speeds depend on the halo circular velocity*/
        {"fixedefficiency", WIND_FIXED_EFFICIENCY}, /* Winds have a fixed efficiency and thus fixed wind speed*/
        {"sh03", WIND_SUBGRID | WIND_DECOUPLE_SPH | WIND_FIXED_EFFICIENCY} , /*The canonical model of Spring & Hernquist 2003*/
        {"vs08", WIND_FIXED_EFFICIENCY},
        {"ofjt10", WIND_USE_HALO | WIND_DECOUPLE_SPH},
        {"isotropic", WIND_ISOTROPIC}, /*Does nothing: wind direction is always random and isotropic.*/
        {NULL, WIND_USE_HALO | WIND_DECOUPLE_SPH }, /* Default is ofjt10*/
    };

    param_declare_int(ps, "StarformationOn", REQUIRED, 0, "Enables star formation");
    param_declare_int(ps, "WindOn", REQUIRED, 0, "Enables wind feedback");
    param_declare_enum(ps, "StarformationCriterion",
            StarformationCriterionEnum, OPTIONAL, "density", "Extra star formation criteria to use. Default is density which corresponds to the SH03 model.");

    /*See Springel & Hernquist 2003 for the meaning of these parameters*/
    param_declare_double(ps, "CritOverDensity", OPTIONAL, 57.7, "Threshold over-density (in units of the critical density) for gas to be star forming.");
    param_declare_double(ps, "CritPhysDensity", OPTIONAL, 0, "Threshold physical density (in protons/cm^3) for gas to be star forming. If zero this is worked out from CritOverDensity.");

    param_declare_int(ps, "BoostSFDenseGas", OPTIONAL, 1, "Reduce sfr timescale for ultra-dense gas above BoostSFOverDenseFactor of the CritPhysDensity");
    param_declare_double(ps, "BoostSFOverDenseFactor", OPTIONAL, 1000, "Threshold overdensity with respect to the SF threshold, TNG50 uses 230, but this is too aggressive for our larger halos.");

    param_declare_int(ps, "BHFeedbackUseTcool", OPTIONAL, 1, "Control how BH feedback interacts with the SFR. If 0, star-forming gas which is heated by a BH remains pressurized (and thus does not cool). If 1, it cools exponentially to the EEQOS using the cooling time rather than the relaxation time. If 2, gas more than 0.3 dex above the EOS temp just cools normally. If 3 all star forming gas cools normally. 1 and 2 give similar BH output, but 1 is 50% faster due to the smaller timebins populated by 2.");
    param_declare_double(ps, "FactorSN", OPTIONAL, 0.1, "Fraction of the gas energy which is locally returned as supernovae on star formation.");
    param_declare_double(ps, "FactorEVP", OPTIONAL, 1000, "Parameter of the SH03 model, controlling the energy of the hot gas.");
    param_declare_double(ps, "TempSupernova", OPTIONAL, 1e8, "Temperature of the supernovae remnants in K.");
    param_declare_double(ps, "TempClouds", OPTIONAL, 1000, "Temperature of the cold star forming clouds in K.");
    param_declare_double(ps, "MaxSfrTimescale", OPTIONAL, 1.5, "Maximum star formation time in units of the density threshold.");
    param_declare_int(ps, "Generations", OPTIONAL, 4, "Number of stars to create per gas particle.");
    param_declare_enum(ps, "WindModel", WindModelEnum, OPTIONAL, "ofjt10", "Wind model to use. Default is the varying wind velocity model with isotropic winds.");

    /* The following two are for VS08 and SH03*/
    param_declare_double(ps, "WindEfficiency", OPTIONAL, 2.0, "Fraction of the stellar mass that goes into a wind. Needs sh03 or vs08 wind models.");
    param_declare_double(ps, "WindEnergyFraction", OPTIONAL, 1.0, "Fraction of the available energy that goes into winds.");

    /* The following two are for OFJT10*/
    param_declare_double(ps, "WindSigma0", OPTIONAL, 353, "Square root of energy ejection rate for winds (controls mass loading) in km/s. Needs ofjt10 wind model.");
    param_declare_double(ps, "WindSpeedFactor", OPTIONAL, 3.7, "Factor connecting wind speed to local particle velocity dispersion. ofjt10 wind model.");

    param_declare_double(ps, "WindFreeTravelLength", OPTIONAL, 20, "Expected decoupling distance for the wind in internal distance units. Small effect because the other recoupling conditions dominate.");
    param_declare_double(ps, "WindFreeTravelDensFac", OPTIONAL, 0.1, "If the density of the wind particle drops below this factor of the star formation density threshold, the gas will recouple.");
    param_declare_double(ps, "MinWindVelocity", OPTIONAL, 0, "Minimum velocity of the kicked particle in the wind, in internal units (physical km/s).");
    param_declare_double(ps, "WindThermalFactor", OPTIONAL, 0, "Fraction of the wind energy which comes thermally rather than kinetic.");

    param_declare_double(ps, "MaxWindFreeTravelTime", OPTIONAL, 60, "Maximum time in Myrs for the wind to be decoupled.");

    param_declare_int(ps, "RandomSeed", OPTIONAL, 42, "Random number generator seed. Combined with the current integer time to seed a separate random table each timestep.");

    /*These parameters are Lyman alpha forest specific*/
    param_declare_double(ps, "QuickLymanAlphaProbability", OPTIONAL, 0, "Probability gas is turned directly into stars, irrespective of pressure. One is equivalent to quick lyman alpha star formation.");
    param_declare_double(ps, "QuickLymanAlphaTempThresh", OPTIONAL, 1e5, "Temperature threshold for gas to be star forming in the quick lyman alpha model, in K. Gas above this temperature does not form stars.");
    param_declare_double(ps, "HydrogenHeatAmp", OPTIONAL, 1, "Density-independent heat boost to hydrogen.");
    /* Enable model for helium reionisation which adds extra photo-heating to under-dense gas.
     * Extra heating has the form: H = Amp * (rho / rho_c(z=0))^Exp
     * but is density-independent when rho / rho_c > Thresh. */
    param_declare_int(ps, "HeliumHeatOn", OPTIONAL, 0, "Change photo-heating rate to model helium reionisation on underdense gas.");
    param_declare_double(ps, "HeliumHeatThresh", OPTIONAL, 10, "Overdensity above which heating is density-independent.");
    param_declare_double(ps, "HeliumHeatAmp", OPTIONAL, 1, "Density-independent heat boost. Changes mean temperature.");
    param_declare_double(ps, "HeliumHeatExp", OPTIONAL, 0, "Density dependent heat boost (exponent). Changes gamma.");
    /*End of star formation parameters*/
    /* Parameters for the QSO lightup model for helium reionization*/
    param_declare_int(ps, "QSOLightupOn", OPTIONAL, 0, "Enable the quasar lighup model for helium reionization");
    /* Default QSO BH masses correspond to the Illustris BHs hosted in halos between 2x10^12 and 10^13 solar masses.
     * In small boxes this may be too small.*/
    param_declare_double(ps, "QSOMaxMass", OPTIONAL, 1000, "Maximum mass of a halo potentially hosting a quasar in internal mass units.");
    param_declare_double(ps, "QSOMinMass", OPTIONAL, 100, "Minimum mass of a halo potentially hosting a quasar in internal mass units.");
    param_declare_double(ps, "QSOMeanBubble", OPTIONAL, 20000, "Mean size of the ionizing bubble around a quasar. By default 20 Mpc/h = 28 Mpc. 0807.2799");
    param_declare_double(ps, "QSOVarBubble", OPTIONAL, 0, "Variance of the ionizing bubble around a quasar. By default zero so all bubbles are the same size");
    param_declare_double(ps, "QSOHeIIIReionFinishFrac", OPTIONAL, 0.995, "Reionization fraction at which all particles are flash-reionized instead of having quasar bubbles placed.");

    /* Parameters for the metal return model*/
    param_declare_double(ps, "MetalsSn1aN0", OPTIONAL, 1.3e-3, "Overall rate of SN1a per Msun");
    param_declare_double(ps, "MetalsMaxNgbDeviation", OPTIONAL, 5., "Maximum variance in the number of neighbours metals are returned to.");
    param_declare_int(ps, "MetalsSPHWeighting", OPTIONAL, 1, "If true, return metals to gas with a volume-weighted SPH kernel. If false use a volume-weighted uniform kernel.");

    /*Parameters for the massive neutrino model*/
    param_declare_int(ps, "MassiveNuLinRespOn", REQUIRED, 0, "Enables linear response massive neutrinos of 1209.0461. Make sure you enable radiation too.");
    param_declare_int(ps, "HybridNeutrinosOn", OPTIONAL, 0, "Enables hybrid massive neutrinos, where some density is followed analytically, and some with particles. Requires MassivenuLinRespOn");
    param_declare_double(ps, "MNue", OPTIONAL, 0, "First neutrino mass in eV.");
    param_declare_double(ps, "MNum", OPTIONAL, 0, "Second neutrino mass in eV.");
    param_declare_double(ps, "MNut", OPTIONAL, 0, "Third neutrino mass in eV.");
    param_declare_double(ps, "Vcrit", OPTIONAL, 500., "For hybrid neutrinos: Critical velocity (in km/s) in the Fermi-Dirac distribution below which the neutrinos are particles in the ICs.");
    param_declare_double(ps, "NuPartTime", OPTIONAL, 0.3333333, "Scale factor at which to turn on hybrid neutrino particles.");
    /*End parameters for the massive neutrino model*/

    /*Parameters for the Excursion Set Algorithm*/
    param_declare_int(ps, "ExcursionSetReionOn", OPTIONAL, 0, "Use the excursion set instead of the global UV field");
    param_declare_int(ps, "UVBGdim", OPTIONAL, 64, "Number of cells on a side of the excursion set grid. Resolution = BoxSize/UVBGdim");
    param_declare_int(ps, "ReionFilterType", OPTIONAL, 0, "Filter type for Excursion set: 0 = real-space top-hat, 1 = k-space top-hat, 2 = gaussian");
    param_declare_int(ps, "RtoMFilterType", OPTIONAL, 0, "Filter type for radius to mass calculation: 0 = top-hat, 1 = gaussian");
    param_declare_double(ps, "ReionRBubbleMax", OPTIONAL, 20340., "Maximum radius of excursion set filters in internal units");
    param_declare_double(ps, "ReionRBubbleMin", OPTIONAL, 406.8, "Minimum radius of excursion set filters in internal units");
    param_declare_double(ps, "ReionDeltaRFactor", OPTIONAL, 1.1, "Fractional difference between excursion set bubble sizes.");
    param_declare_double(ps, "ReionGammaHaloBias", OPTIONAL, 2.0, "Halo Bias for calculating J21.");
    param_declare_double(ps, "ReionNionPhotPerBary", OPTIONAL, 4000., "Photons produced per stellar baryon.");
    param_declare_double(ps, "AlphaUV", OPTIONAL, 3., "Spectral slope of ionising radiation above the Hydrogen ionisation threshold.");
    param_declare_double(ps, "EscapeFractionNorm", OPTIONAL, 0.2, "Normalisation of escape fraction at 1e10 solar masses.");
    param_declare_double(ps, "EscapeFractionScaling", OPTIONAL, 0.5, "Power law scaling of escape fraction with halo mass.");
    param_declare_double(ps, "UVBGTimestep", OPTIONAL, 10., "Time in Myr between UVBG calculations.");
    param_declare_string(ps, "J21CoeffFile", OPTIONAL, "", "Rate coefficient table for converting J21 to photo ion/heating rates at a certain spectral slope");
    param_declare_double(ps, "ExcursionSetZStop", OPTIONAL, 5., "Redshift at which we stop the excursion set and use global UVBG");
    param_declare_double(ps, "ExcursionSetZStart", OPTIONAL, 25., "Redshift at which we start the excursion set");
    param_declare_int(ps, "ReionUseParticleSFR", OPTIONAL, 0, "Use the gas particle SFR instead of the usual excursion set stellar mass / timescale");
    param_declare_double(ps, "ReionSFRTimescale", OPTIONAL, 0.1, "timescale to calculate the SFR from stellar mass filtered grids (units of Hubble time)");
    /*End Parameters for the Excursion Set Algorithm*/

    param_set_action(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodAction, NULL);
    param_set_action(ps, "StarformationCriterion", StarformationCriterionAction, NULL);

    return ps;
}

/*! This function parses the parameterfile in a simple way.  Each paramater is
 *  defined by a keyword (`tag'), and can be either of type douple, int, or
 *  character string.  The routine makes sure that each parameter appears
 *  exactly once in the parameterfile, otherwise error messages are
 *  produced that complain about the missing parameters.
 */
void read_parameter_file(char *fname, int * ShowBacktrace, double * MaxMemSizePerNode)
{
    ParameterSet * ps = create_gadget_parameter_set();

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(0 != param_parse_file(ps, fname)) {
        endrun(1, "Parsing %s failed.\n", fname);
    }
    if(0 != param_validate(ps)) {
        endrun(1, "Validation of %s failed.\n", fname);
    }

    message(0, "----------- Running with Parameters ----------\n");
    if(ThisTask == 0)
        param_dump(ps, stdout);
    message(0, "----------------------------------------------\n");

    *ShowBacktrace = param_get_int(ps, "ShowBacktrace");
    *MaxMemSizePerNode = param_get_double(ps, "MaxMemSizePerNode");
    if(*MaxMemSizePerNode <= 1) {
        *MaxMemSizePerNode *= get_physmem_bytes() / (1024. * 1024.);
    }

    /*Initialize per-module parameters.*/
    set_all_global_params(ps);
    set_plane_params(ps);
    set_init_params(ps);
    set_petaio_params(ps);
    set_timestep_params(ps);
    set_cooling_params(ps);
    set_uvf_params(ps);
    set_density_params(ps);
    set_hydro_params(ps);
    set_qso_lightup_params(ps);
    set_treewalk_params(ps);
    set_gravshort_tree_params(ps);
    set_tidalfield_params(ps);
    set_domain_params(ps);
    set_sfr_params(ps);
    set_sync_params(ps);
    set_uvbg_params(ps);
    set_winds_params(ps);
    set_fof_params(ps);
    set_secondfof_params(ps);
    set_blackhole_params(ps);
    set_metal_return_params(ps);
    set_stats_params(ps);
    parameter_set_free(ps);
}
