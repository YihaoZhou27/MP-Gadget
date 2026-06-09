# MP-Gadget Development Log

## 2026-06-07 — secFOF multi-seeding: surplus-aware N_seed and "first-capped, rest-equal" seed masses

**Branch:** SecFOFCombined

Reworked how the per-secFOF multi-seeding splits mass among the N_seed black holes. N_seed is now floor(M_SC/1e8) **plus one extra** when the leftover above floor(M_SC/1e8)×1e8 is itself seedable (≥ MinMscForBHseed), still capped by the number of unseeded stars. The seeds are no longer equal mass: seed 1 (the largest-m·Γ star) carries exactly the threshold mass (1e8 Msun-equivalent, i.e. its BH mass is 1e8×SeedBlackHoleMass under `BHseedMassScaleMsc=1`), and the remaining N_seed−1 extra seeds split the surplus (M_SC − 1e8) equally. This removes the old discontinuity where a single BH in the 1e8–2e8 Msun range could be seeded above the threshold mass. The attached StarClusterMass payload and the init_Msc / init_Msc_sample records follow the seed-mass share (proportional to each seed's mass when `BHseedMassScaleMsc=1`, otherwise an equal 1/N_seed split); total seed mass and payload remain conserved across the N_seed seeds. Only affects groups with `SeedInSecFOFMultipleSeeds=1`.

**Files modified:** `libgadget/fof.c`

## 2026-06-07 — Fix: non-combined secFOF star-cluster seeding could convert gas instead of a star

**Branch:** SecFOFCombined

Fixed a bug in non-combined (`SeedSecFOFcomSample=0`) secondary-FOF star-cluster seeding: the seed particle could be a gas particle instead of the intended star. When gas is a secondary linking type, the group carries a densest-gas `seed_index >= 0`, and the star override only replaced it when `seed_index < 0`, so the gas index was kept and that gas particle was converted into the BH (with the star-cluster payload attached). Now the star override is unconditional (always uses the largest-ClusterMass star), and the SC seeding mask additionally requires that a star seed exists — matching the combined-sample path, which already overrode gas unconditionally. Star-cluster seeding now always converts a star. (This also makes the multi-seed feature trigger correctly for such groups.)

**Files modified:** `libgadget/fof.c`

## 2026-06-07 — secFOF multi-seeding: gating parameter + per-group logging

**Branch:** SecFOFCombined

Put the per-secFOF multi-seeding feature behind a new int parameter `SeedInSecFOFMultipleSeeds` (default 0). When 0 the behaviour is the original single seed per group; when 1 the multi-seeding (below) runs. `SeedInSecFOFMultipleSeeds=1` requires the effective `SeedInSecFOFasStarCluster=1` (i.e. SecondFOFOn=1, StarClusterOn=1, SecFOFStarCluster=1); otherwise the run aborts with an error (checked in both `set_secondfof_params` and the `run.c` init, mirroring the `SeedSecFOFcomSample` validation). Also added a rank-0 log message, emitted per multi-seed group (M_SC > 1e8 Msun), reporting M_SC (solar masses and code units), N_seed, the number of seeds actually placed, and the position (and ID) of every seed — seed 1 plus each extra seed — followed by a one-line global summary.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/secondfof.c`, `libgadget/run.c`

## 2026-06-06 — secFOF star-cluster seeding: multiple BH seeds in very massive groups

**Branch:** SecFOFCombined

Added per-secFOF multi-seeding for the `SeedInSecFOFasStarCluster` path. When a secondary-FOF group's seeding star-cluster mass M_SC exceeds 1e8 Msun (converted to code units), it now seeds N_seed = floor(M_SC/1e8) black holes (capped by the number of unseeded stars in the group) instead of one. All N seeds have equal mass: with `BHseedMassScaleMsc=1` each is M_SC*SeedBlackHoleMass/N_seed, and the attached StarClusterMass payload (when `StarClusterBHDyn=1`) is split equally (Σmγ/N_seed). Seed 1 is the largest-m·Γ unseeded star (as before); seeds 2..N are the next-largest-m·Γ unseeded stars lying farther than 2×(gravitational softening ε) from seed 1, each converted in place into a BH. Stars are considered in descending m·Γ order; the separation is measured from seed 1 only. If too few stars are far enough, the remaining seeds are skipped and a summary is logged. The selection is a collective operation across MPI ranks (multi-seed groups and seed-1 positions are gathered, candidates are globally ranked, and each rank converts the chosen stars it owns); BH slots for the extra seeds are pre-reserved up front. Groups below 1e8 Msun are unaffected (single seed, identical to before). (Gated by `SeedInSecFOFMultipleSeeds` as of 2026-06-07.)

**Files modified:** `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`, `libgadget/fof.h`, `libgadget/fof.c`

## 2026-06-06 — Star-cluster BH seeding: convert the parent star in-place instead of spawning

**Branch:** SecFOFCombined

Changed star-cluster-based BH seeding (both the per-star `BlackholeSeedSCparticle` path and the secondary-FOF `SeedInSecFOFasStarCluster` path, including the combined-sample modes) to **convert the parent star particle in-place into the black hole** instead of spawning a new BH next to it. The star is now consumed and disappears from the simulation, exactly as a gas particle is consumed under `BlackHoleSeedHaloBased=1`. The BH keeps the star's ID and full mass, and its initial `Mtrack` is the parent star mass (mass-conserving). The attached `StarClusterMass` payload and seed-mass scaling are unchanged. Removed the now-unnecessary base-particle capacity checks (no new particles are created; only BH slots are still reserved) and the post-seed star `Seeded` flagging on the consumed star.

**Files modified:** `libgadget/blackhole.c`, `libgadget/fof.c`, `libgadget/secondfof.c`

## 2026-06-05 — SeedSecFOFcomSampleParticle: per-star-particle star-cluster sampling for secFOF BH seeding

**Branch:** SecFOFCombined

Added a per-star-particle variant of the combined per-secFOF star-cluster sampling, gated by a new int parameter `SeedSecFOFcomSampleParticle` (default 0; requires `SeedSecFOFcomSample=1`). When on, the single per-group draw is replaced by an independent draw for each unseeded star: cluster mass function n(m) ~ m^-2 exp(-m/m_cut) with cutoff m_cut = min(M_cstar, group total unseeded stellar mass), expected count n = Gamma*m_star/<m>, Poisson-sampled, and the >1e4 Msun clusters summed per star and over the group into tot_msc_fof (optionally capped at the group unseeded stellar mass via SCmasscapSecFOFstarmass). Seeding keeps the two gates (pre-filter on the group's summed Gamma*m_star, then seed iff tot_msc_fof >= MinMscForBHseed); tot_msc_fof sets the seed mass. For seeded groups, tot_msc_fof is redistributed into each unseeded star's ClusterMass weighted by stellar mass (sum conserved), and those stars are flagged Seeded. The per-particle sampling runs collectively (Allgather candidate groups, sample local stars, Allreduce). The combined (non-particle) mode and the StarClusterMass_sample block are unchanged.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/secondfof.c`, `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`, `libgadget/run.c`

## 2026-05-10 — Fix BH dynamical mass (P.Mass) to track BHP.Mass + StarClusterMass

**Branch:** SC_ParticleSeeding

Changed P.Mass for blackholes to be a derived quantity: `P.Mass = max(BHP.Mass + StarClusterMass, SeedBHDynMass)`. Previously, P.Mass was tracked independently through stochastic gas swallowing and accumulated artificial SeedBHDynMass mass through BH mergers (each merged post-seed-regime BH contributed ~SeedBHDynMass to the survivor's P.Mass while only adding tiny BHP.Mass). With the new approach, P.Mass is set directly at the end of each accretion postprocess step, preventing the compounding. Also removed the `max(I->Mass, BHP.Mass)` clamp in `blackhole_accretion_copy` that was preventing gas swallowing after seed regime.

**Files modified:** `blackhole.c`, `bhinfo.c`

## 2026-04-21 — Gas and Stellar Velocity Dispersion Module

**Branch:** StarCluster

Added a new module (`gasveldisp.c/h`) that computes, for each gas particle at every PM step, the gas and stellar velocity dispersion, total mass, and neighbor count within the gas particle's SPH smoothing length (Hsml).

**New output fields (gas particles, type 0):**
- `VDisp_gas`, `VDisp_star` — 1D gas/stellar velocity dispersion
- `VDisp_mgas`, `VDisp_mstar` — total gas/stellar mass within Hsml
- `VDisp_Ngas`, `VDisp_Nstar` — number of gas/star neighbors within Hsml

**Key design choices:**
- Uses two separate treewalks for efficiency: the gas walk reuses the existing gasTree (avoiding a costly tree rebuild), while the star walk builds a small star-only tree (cheap since N_star << N_gas).
- Uses predicted velocities (`SPH_VelPred` / `DM_VelPred`) to ensure consistent velocity comparisons across hierarchical timebins.
- Runs unconditionally at PM steps (gated on `GasEnabled && is_PM`, not tied to `CoolingOn`).
- All 6 output fields are zeroed each PM step to prevent stale values when neighbor count is zero.

**Files modified:** `slotsmanager.h`, `run.c`, `petaio.c`, `Makefile`
**Files added:** `gasveldisp.c`, `gasveldisp.h`

## 2026-04-22 — Second FOF (Star-Primary) Pass

**Branch:** SecondFOF

- Created and revised implementation plan for Second FOF (star-primary) pass to identify small stellar structures (star clusters). Key features: separate `secondfof.c` file, post-processing via RestartFlag=3, group size properties (R50/R90/Rmax), potential-minimum center. Plan saved at `plan/second_fof_plan.md`.
- Implemented second FOF: new files `secondfof.c`/`secondfof.h`, parameters registered in `params.c`, `fof_get/set_params` accessors in `fof.c`, `SecGrNr` field in `partmanager.h`, `SecGroupID` snapshot block in `petaio.c`, integration in `run.c` (live loop + runfof post-processing). SecPIG catalog with PotMinPos, R50/R90/Rmax. Compiles cleanly.
- Fixed `cmp_double` UB bug (was sorting `int64_t` array with `double` comparator).
- Split `secondfof_save()` into `secondfof_run()`/`secondfof_write()`/`secondfof_finish()` for I/O safety (write SecPIG after checkpoint, not before).

## 2026-04-23 — Second FOF Refinements and LIFO Fixes

**Branch:** SecondFOF

- Moved PotMin/PotMinPos computation from separate `secondfof_compute_potmin()` (~220 lines) into `fof_compile_catalogue` (`add_particle_to_group`/`fof_reduce_group` in `fof.c`). Added `PotMin`/`PotMinPos` fields to `struct Group` in `fof.h`. Simplified `SecondGroupExtra` to only R50/R90/Rmax. Falls back to CM when potential unavailable. Compiles cleanly.
- Added `FOFPotentialMin` parameter (default 0) to control whether PotMin tracking runs in the FOF engine. PotMin tracking in `add_particle_to_group`/`fof_reduce_group` is now guarded by this flag. Second FOF sets it automatically based on `OutputPotential`. Compiles cleanly.
- Fixed stack allocator LIFO violation in `secondfof.c`: deferred `SecFOF_Result` allocation until after `SecFOF_SavedGrNr` is freed, using a local `FOFGroups` variable to hold the intermediate fof_fof() result.
- Fixed another stack allocator LIFO violation: reordered `SecFOF_Output` allocation before `SecFOF_Extra` so that `SecFOF_Extra` can be freed while on top of the stack.
- Fixed latent LIFO violation in `secondfof_compute_sizes`: deferred frees of `dm_local`, `rcounts`, `rdispls` until after `dm_global` is freed (would crash when `SecondFOFSize` is enabled).
- Fixed stack allocator LIFO violation in `secondfof_finish`: `Group` was freed before `SecFOF_Result`, violating LIFO order. Fixed by copying `fof` to a local variable, freeing `result` first, then calling `fof_finish`.
- Fixed stack allocator LIFO violation in `run.c`: `fof_finish` (freeing primary Group) was called before `secondfof_finish`, but second FOF allocations sat on top of the primary Group in the stack. Reordered both call sites to finish second FOF before primary FOF.
- Fixed MPI_Type_free double-free crash: `secondfof_finish` called `fof_finish` which freed the static `MPI_TYPE_GROUP` datatype, then the outer `fof_finish` for the primary FOF tried to free it again. Fixed by having `secondfof_finish` free only the Group array directly instead of calling `fof_finish`.
- Added particle catalog saving to SecPIG output. Only primary-linked particles are saved (ordered by secondary FOF group number). Extracted `fof_save_particles_to_bigfile()` from `fofpetaio.c` as a reusable function. In `secondfof_write()`, temporarily swaps `SecGrNr` into `GrNr` for primary particles, calls the particle saver, then restores `GrNr`. Updated header to count only primary-linked particles. Compiles cleanly.

## 2026-04-25 — Star Cluster Mass (M_cstar) and Second FOF MPI Fix

**Branch:** gas-StarCluster

Added a new star particle property `Mcstar` computed from the Toomre mass model at star formation. `Mcstar = 0.1 * CFE * f_coll * M_T`, where M_T is the Toomre mass from gas surface density and tidal field eigenvalues, and f_coll is the feedback-regulated collapse fraction. The existing `ClusterMass = Mass * CFE` is preserved unchanged.

The calculation uses parent gas particle properties at birth: density, internal energy, tidal field eigenvalues, cluster formation efficiency, and gas/stellar velocity dispersions (phi_P pressure correction). Constants t_sn (3 Myr) and phi_fb (0.16 cm^2/s^3) are precomputed in code units at initialization.

Also added `Msc_ave`: the average mass of the cluster mass function n(m) ~ m^-2 exp(-m/Mcstar) over [1e2, 1e8] Msun, computed analytically using the exponential integral E1 (GSL). Added `NumStarCluster = ClusterMass / Msc_ave`.

**New output fields (star particles, type 4):**
- `Msc_ave`, `StarClusterMass_sample` — always written
- `Mcstar`, `NumStarCluster`, `Nsc_sample` — written only when `OutputDebugFields = 1`

**Files modified:** `sfr_eff.c`, `slotsmanager.h`, `petaio.c`

**Branch:** SecondFOF

- Fixed MPI deadlock in `secondfof_compute_sizes` when `SecondFOFSize=1`. The old code called MPI collectives (Allgather/Allgatherv) inside a per-group loop, but different ranks owned different numbers of groups, causing ranks with fewer groups to exit the loop early while others waited. Replaced with two single Allgatherv calls: one for group centers, one for all particle (dist, mass, GrNr) data. Each rank then computes R50/R90/Rmax locally for its own groups from the complete global data. No per-group MPI collectives needed.
- Added `SecFOFonly` parameter (default 0). When set to 1, skips saving the primary FOF catalog (PIG files) while still running the primary FOF internally and saving only the second FOF catalog (SecPIG). Works for both live simulation (RestartFlag=1) and post-processing (RestartFlag=3).

## 2026-04-26 — FOF Group Properties and SeedInSecFOF

**Branch:** gas-StarCluster

Added two new FOF group properties: `StarClusterMassSample` (sum of `StarClusterMass_sample` over all hosted stars) and `NscSample` (sum of `Nsc_sample` over all hosted stars). These are accumulated per-particle and reduced across MPI ranks following the same pattern as `StarClusterMass`. Changed the BH seeding condition with `StarClusterOn`: now compares `StarClusterMassSample` (instead of `StarClusterMass`) against `MinMscForBHseed`.

Both new FOF group properties are written to FOF output snapshots.

**Files modified:** `fof.h`, `fof.c`, `fofpetaio.c`

**Branch:** SecondFOF

- Added `SeedInSecFOF` parameter (default 0). When set to 1 (and `SecondFOFOn=1`), black hole seeding uses the secondary FOF catalog instead of the primary. A new `secondfof_seed()` function temporarily swaps FOF params to secondary values, runs FOF, calls `fof_seed`, then restores params. Avoids MPI_TYPE_GROUP double-free by manually freeing the Group array instead of calling `fof_finish`.
- Fixed bug where `4/GroupID` and `4/SecGroupID` in the SecPIG particle catalog saved identical values. The GrNr-to-SecGrNr swap for particle selection was clobbering the primary FOF ID. Now both fields are swapped before distribution, and `fof_save_particles_to_bigfile` swaps them back on the distributed particles before writing IO, so `GroupID` = primary FOF ID and `SecGroupID` = secondary FOF ID.

## 2026-04-27 — StarClusterSampling Parameter and SecPIG Enhancements

**Branch:** gas-StarCluster

Added new integer parameter `StarClusterSampling` (default 0, requires `StarClusterOn = 1`). Controls whether individual star cluster masses are sampled from the mass function.

- **When on:** Msc_ave, NumStarCluster, Nsc_sample, StarClusterMass_sample are computed per star particle. FOF groups accumulate StarClusterMassSample and NscSample. BH seeding threshold and seed mass (when BHseedMassScaleMsc=1) use StarClusterMassSample.
- **When off:** Those sampling properties are set to zero/initial values and the sampling computation is skipped. BH seeding threshold and seed mass use StarClusterMass (the CFE-based total) instead.

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`, `libgadget/fof.c`, `libgadget/blackhole.c`

**Branch:** SecondFOF

- Added `PrimaryFOFNum` and `PrimaryFOFID` properties to the SecPIG catalog. For each secondary FOF group, `PrimaryFOFNum` counts how many distinct primary FOF groups its primary-linked particles belong to, and `PrimaryFOFID` records the primary FOF group that hosts the largest fraction. Computed via Allgatherv pattern (same as sizes). Written as `i4` blocks in SecPIG output. Compiles cleanly.
- Fixed SecPIG particle catalog metadata inconsistency: moved `petaio_build_selection` before the GrNr/SecGrNr swap in `fof_save_particles_to_bigfile`. Previously, particles in a secondary FOF but not in any primary FOF (original GrNr = -1) were counted in the header but excluded from the particle blocks after the swap. Now selection happens while GrNr still holds the secondary FOF ID (>= 0), so all grouped particles are included.

## 2026-04-28 — SecFOFStarCluster Parameter and BH Seeding Reorganization

**Branch:** StarCluster

Added `SecFOFStarCluster` integer parameter (default 1, requires `SecondFOFOn=1` and `StarClusterOn=1`). Asserts at startup that `SecondFOFPrimaryLinkTypes` is 16 (type 4 stars). Writes three star cluster group properties to the SecPIG catalog: `StarClusterMass`, `StarClusterMetallicity`, `StarClusterMetalElemMass[NMETALS]`. These values are already accumulated by the FOF engine in the Group struct during catalogue compilation.

Renamed `SeedInSecFOF` to `SeedInSecFOFasStarCluster`. If `SecondFOFOn=0`, `StarClusterOn=0`, or `SecFOFStarCluster=0`, the parameter is forced to 0 with a log message.

Added `BlackHoleSeedStarCluster` integer parameter (default 0) for StarCluster-based BH seeding. In `fof_seed`, the `SC_Mask` (StarClusterMass-based seeding) is now gated on `BlackHoleSeedStarCluster` instead of `StarClusterOn`. Updated the BH seeding validation: if `BlackHoleOn=1`, at least one of `SeedInSecFOFasStarCluster`, `BlackHoleSeedHaloBased`, `BlackHoleSeedStarCluster`, or `BlackHoleSeedGasBased` must be enabled.

`secondfof_seed` now temporarily overrides seeding params to only use StarCluster criteria (`BlackHoleSeedStarCluster=1`, `BlackHoleSeedHaloBased=0`, `BlackHoleSeedGasBased=0`) before calling `fof_seed`, then restores them. Added `fof_get_seed_params`/`fof_set_seed_params` accessors in `fof.c`/`fof.h`. Currently, seeding in secondary FOF and seeding in primary FOF cannot be turned on in the same run.

Fixed `MPI_TYPE_GROUP` handle leak: the static datatype in `fof.c` is now initialized to `MPI_DATATYPE_NULL`. `fof_fof()` frees any existing handle before creating a new one, and `fof_finish()` resets it to `MPI_DATATYPE_NULL` after freeing. This prevents leaks when `fof_fof()` is called multiple times per step (primary + secondary FOF, secondary seeding).

Added parameter enforcement (temporary): when `SeedInSecFOFasStarCluster=1`, `BlackHoleSeedStarCluster`, `BlackHoleSeedHaloBased`, and `BlackHoleSeedGasBased` are forced to 0 in both `run.c` (`All` struct) and `secondfof.c` (via `fof_set_seed_params`). Seeding in secondary FOF and primary FOF cannot be on in the same run.

Added `seeded_by_starcluster` flag to `blackhole_make_one`. The BH seed mass scaling (`BHP.Mass *= StarClusterMass` when `BHseedMassScaleMsc=1`) now only applies when the BH is seeded by star-cluster criteria (either primary or secondary FOF), not halo-based or gas-based. In `fof_seed_make_one`, `seeded_by_starcluster` is determined by checking `BlackHoleSeedStarCluster && StarClusterMass >= MinMscForBHseed`. Since `secondfof_seed` temporarily sets `BlackHoleSeedStarCluster=1`, this correctly identifies secondary-FOF-seeded BHs as star-cluster-seeded.

Added `MinMscForBHseed` validation in `set_secondfof_params`: when `SeedInSecFOFasStarCluster=1` and `BHseedMassScaleMsc=1`, asserts `MinMscForBHseed > 0`. This covers the case where `secondfof_seed` forcibly enables star-cluster criteria at runtime but `BlackHoleSeedStarCluster=0` in the param file, so the existing validation in `set_fof_params` would not catch it.

**Files modified:** `gadget/params.c`, `libgadget/secondfof.c`, `libgadget/run.c`, `libgadget/fof.c`, `libgadget/fof.h`

Optimized the PM seeding block in `run.c` to skip the primary FOF (`fof_fof`) when `SeedInSecFOFasStarCluster=1` and neither helium reionization nor excursion-set reionization needs the primary catalog. The secondary FOF seeding builds its own FOF internally, so the primary run was redundant.

**Files modified:** `libgadget/run.c`

Optimized `secondfof_compute_primary_fof_info`: replaced the O(Ngroups * Npairs) full-scan approach with a single sort of all pairs by (SecGrNr, GrNr) followed by binary search per group. Reduces per-group lookup to O(log(Npairs) + group_size) and eliminates per-group malloc/free of temporary arrays.

**Files modified:** `libgadget/secondfof.c`

Fixed star-cluster BH seeding in secondary FOF: seed particle selection now uses the star at the potential minimum (`seed_index_star`/`seed_task_star` fields in `struct Group`) instead of the densest gas particle (which doesn't exist in star-only groups). Added parameter checks: `SeedInSecFOFasStarCluster` now requires type 5 in `SecondFOFSecondaryLinkTypes` and `FOFPotentialMin = 1`. `secondfof_seed` enables `FOFPotentialMin` when overriding FOF params.

**Files modified:** `libgadget/fof.h`, `libgadget/fof.c`, `libgadget/blackhole.c`, `libgadget/blackhole.h`, `libgadget/secondfof.c`

Fixed primary FOF star-cluster seeding (`BlackHoleSeedStarCluster=1`): `SC_Mask` in `fof_seed` previously required `seed_index_star >= 0`, which is never populated in primary FOF (default `FOFPrimaryLinkTypes=2`, DM-only). Changed to accept either `seed_index >= 0` (gas particle at highest density) or `seed_index_star >= 0` (star at potential minimum). The `seed_index_star → seed_index` fallback copy now only triggers when `seed_index < 0` (no gas seed available), naturally distinguishing primary vs. secondary FOF paths. Primary FOF star-cluster seeding converts the densest gas particle to a BH (old behavior); secondary FOF seeding spawns a new BH from the star at the potential minimum.

Added base particle capacity pre-check in `fof_seed`: before the seeding loop, counts how many import groups have star-type seed particles (`Nspawn`) and verifies `NumPart + Nspawn <= MaxPart`. Aborts with a clear message suggesting `PartAllocFactor` increase if insufficient, preventing partial-spawn crashes mid-loop. Follows the same pattern as `slots_split_particle` in `slotsmanager.c` (MP-Gadget does not support growing `MaxPart` at runtime).

Merged `blackhole_make_one` and `blackhole_make_one_from_star` into a single unified `blackhole_make_one`. The function now determines the creation path internally via `spawn_from_star = seeded_by_starcluster && (P[index].Type == 4)`. When `spawn_from_star` is true (secondary FOF), it allocates a new particle slot, copies base properties from the star, generates a unique child ID, sets gravitational mass, then converts to type 5. When false (primary FOF / halo / gas-based), it converts the gas particle in-place. All BH-specific initialization (accretion mass, star-cluster payload, MinPotPos, DFAccel, Mtrack, etc.) is shared. Removed the separate `blackhole_make_one_from_star` declaration from `blackhole.h` and the dispatch logic in `fof_seed_make_one`. Net reduction: ~91 lines of duplicated code.

**Files modified:** `libgadget/fof.c`, `libgadget/blackhole.c`, `libgadget/blackhole.h`

## 2026-05-03 — Fix StarClusterSampling mismatch in BH seeding

**Branch:** StarCluster

Fixed two mismatches in `fof_seed_make_one` when `StarClusterSampling=1`:

1. The `seeded_by_starcluster` flag always checked `StarClusterMass` against `MinMscForBHseed`, while the marking code used `StarClusterMassSample`. This caused groups to be marked for seeding but then fall into the gas-conversion path (crash: "Only Gas turns into blackholes"). Now uses `sc_mass` (sampling-aware) for the threshold.

2. The star cluster mass passed to `blackhole_make_one` was always `g->StarClusterMass` (CFE-based total), not `sc_mass`. When `BHseedMassScaleMsc=1`, this caused the seed mass to scale by the raw (smaller) mass instead of the sampled mass, producing BHs below the intended minimum mass. Now passes `sc_mass` to `blackhole_make_one`, which is also stored as the BH's `StarClusterMass`.

**Files modified:** `libgadget/fof.c`

## 2026-05-05 — Fix BH-from-star ID collision in star cluster seeding

**Branch:** StarCluster

Fixed a particle ID uniqueness collision in `blackhole_make_one` when spawning a BH from a star particle (star cluster seeding). The old code incremented the star's Generation counter and used it as the generation byte in the child BH's ID, but sibling stars from the same gas parent already occupied those generation numbers (1..Generations). Now uses the star's own generation (from its ID top byte) plus a fixed offset of 128, guaranteeing no collision with star-formation generations (max 14).

**Files modified:** `libgadget/blackhole.c`

Fixed SecPIG particle catalog to include secondary-linked particles (e.g., BH type 5). Previously only primary-linked particle types were saved and counted in the header, so BH blocks in SecPIG were always empty despite BHs being assigned to secondary FOF groups.

**Files modified:** `libgadget/secondfof.c`

## 2026-05-07 — Add Missing SecPIG Group and Particle Blocks

**Branch:** StarCluster

Added three missing FOF group output blocks to the SecPIG catalog: `SecGasSfmpMass`, `SecStarClusterMassSample`, and `SecNscSample`. These fields were already accumulated in the Group struct during secondary FOF compilation but were not being written to the SecPIG output.

Added `OutputDebugFields` support to the SecPIG particle catalog. Previously `fof_save_particles_to_bigfile` (used by SecPIG) did not register debug IO blocks, so particle-level fields like `NumStarCluster`, `Nsc_sample`, and `Mcstar` were missing from SecPIG even when `OutputDebugFields=1`. Threaded the `OutputDebugFields` parameter through `secondfof_write` and `fof_save_particles_to_bigfile`.

Changed star-cluster BH seeding (`SeedInSecFOFasStarCluster`) to spawn the BH from the star particle with the largest `ClusterMass` (or `StarClusterMass_sample` when `StarClusterSampling=1`) instead of the star at the potential minimum. Added `MaxStarClusterMass` field to `struct Group` to track this across particles and MPI ranks. `seed_index_star`/`seed_task_star` are now decoupled from PotMin tracking.

After a BH is seeded by star-cluster criteria in secondary FOF, `ClusterMass` and `StarClusterMass_sample` are zeroed for all star particles in the seeded group (since the star cluster mass has been transferred to `BHP.StarClusterMass` on the new BH). Uses Allgatherv to broadcast seeded group numbers across MPI ranks.

Added two new star particle fields: `initClusterMass` and `initStarClusterMass_sample`. These are set at star formation to the same values as `ClusterMass`/`StarClusterMass_sample` and are never modified afterwards, preserving the original birth values. Written as snapshot output blocks. Pre-zeroed for all type-4 particles before snapshot reading so that restarts from older snapshots (lacking these blocks) default to 0 instead of uninitialised memory.

Removed the `LenType[5] == 0` check from the `SC_Mask` star-cluster BH seeding condition. A secondary FOF group can now seed multiple BHs across timesteps: after each seeding the star cluster mass is zeroed, so subsequent seedings require new accumulation and no star particle is double-counted.

Refactored seeded-group marking: `fof_seed` now optionally returns the GrNr of each locally-seeded group via output parameters (`seeded_grnr_out`, `n_seeded_out`). `secondfof_seed` uses these directly instead of inferring seeded groups from new particle indices. Primary FOF callers pass NULL to skip collection. Replaced the `saved_GrNr` heap allocation (8 bytes/particle) in `secondfof_seed` with `P[i].SecGrNr` as temporary storage for primary GrNr. `SecGrNr` is reset to -1 after restore; `secondfof_run` recomputes it before any snapshot output.

**Files modified:** `libgadget/secondfof.c`, `libgadget/secondfof.h`, `libgadget/fofpetaio.c`, `libgadget/fof.h`, `libgadget/fof.c`, `libgadget/run.c`, `libgadget/slotsmanager.h`, `libgadget/sfr_eff.c`, `libgadget/petaio.c`

---

## 2026-05-08 — Per-particle Star Cluster BH Seeding (BlackholeSeedSCparticle)

**Branch:** SC_ParticleSeeding

Added a new BH seeding method that creates black holes from individual star particles whose star cluster mass exceeds `MinMscForBHseed`, without requiring FOF group finding. New parameter `BlackholeSeedSCparticle` (int, default 0) controls this mode; requires `StarClusterOn=1`. The seeding runs every PM step alongside existing FOF-based seeding. Uses `StarClusterMass_sample` or `ClusterMass` depending on `StarClusterSampling`. The parent star's cluster mass is zeroed after seeding. When this is the only active seeding method, the primary FOF catalog is skipped for efficiency.

---

## 2026-05-09 — Every-timestep BH Seeding from SC Particles (BHseedEveryTimestep)

**Branch:** SC_ParticleSeeding

Added `BHseedEveryTimestep` parameter (int, default 0). When enabled, `blackhole_seed_sc_particle` is moved from the PM-step seeding block to after star formation, so it runs every timestep. This catches newly formed stars whose cluster mass exceeds `MinMscForBHseed` immediately. Requires `BlackholeSeedSCparticle=1`. All other seeding methods (FOF-based, gas-based, etc.) remain PM-only.

---

## 2026-05-10 — Add StarClusterBHDyn parameter to control SC contribution to BH dynamical mass

**Branch:** StarClusterEvolution

Added `StarClusterBHDyn` (int, default 1) parameter. When on (and StarClusterOn=1), star cluster mass is included in BH dynamical mass: `P.Mass = max(BHP.Mass + SC, SeedBHDynMass)`. When off, SC is excluded: `P.Mass = max(BHP.Mass, SeedBHDynMass)`. Affects seeding, accretion postprocess, drag force, and accretion tree walk mass input.

**Files modified:** `blackhole.c`, `params.c`

---

## 2026-05-10 — Add PMstep and StarClusterMass to BH detail output

**Branch:** StarClusterEvolution

Added two new fields to the BH detail binary output: `int PMstep` (whether the current timestep is a PM step) and `double StarClusterMass` (the star cluster mass of the BH). Passed `is_PM` flag from `run.c` through `blackhole()` into `BHPriv`, then into the detail struct.

**Files modified:** `blackhole.h`, `blackhole.c`, `bhinfo.c`, `run.c`

---

## 2026-05-10 — Tidal field strength for BH particles (BlackholeTidalField)

**Branch:** StarClusterEvolution

Added `BlackholeTidalField` parameter (int, default 0). When enabled, computes the gravitational tidal tensor for active BH particles every timestep by reusing the existing gravity tree walk infrastructure. The PM (long-range) tidal tensor is stored on `bh_particle_data.TidalTensorPM[6]` and updated every PM step; the tree (short-range) part is accumulated during `grav_short_tree()` and combined with PM in postprocessing. Tidal field strength is the Frobenius norm of eigenvalues, stored as `BhP.TidalFieldStrength` and recorded in BH detail files. Requires `SplitGravityTimestepsOn=0` (hierarchical gravity trees only contain active particles, producing incomplete tidal tensors). Independent of `GasTidalField`.

**Files modified:** `params.c`, `blackhole.h`, `blackhole.c`, `slotsmanager.h`, `gravpm.c`, `gravshort.h`, `gravshort-tree.c`, `tidalfield.h`, `tidalfield.c`, `bhinfo.c`
## 2026-05-10 — Optimize BH seeding from SC particles: use NewStars list

**Branch:** SC_ParticleSeeding

When `BHseedEveryTimestep` is enabled, `blackhole_seed_sc_particle` now receives the `NewStars` list from `cooling_and_starformation` and only checks newly formed stars, instead of scanning all particles every timestep. Since star cluster mass is fixed at formation, only newly formed stars can qualify for seeding. The PM-step fallback path (when `BHseedEveryTimestep=0`) retains the full scan.

---

## 2026-05-11 — Mtrack-based dynamical mass and StarClusterOn seeding regime

**Branch:** StarClusterEvolution

Two separate P.Mass/Mtrack regimes depending on StarClusterOn:

When `StarClusterOn=1`: Mtrack always accumulates swallowed mass (gas + BH mergers). Seeding regime defined by `Mtrack + StarClusterMass < SeedBHDynMass`. `P.Mass = max(Mtrack + SC, SeedBHDynMass)`, enforced at all update points: gas swallowing, BH mergers, seeding initialization, and star cluster evolution (SC decrease). BH merger othermass for seed-regime secondaries includes StarClusterMass (`Mtrack + SC`).

When `StarClusterOn=0`: master-branch mass-conservation behavior. `P.Mass` starts at `SeedBHDynMass`, `Mtrack` grows via swallowing. At transition (`Mtrack` reaches `SeedBHDynMass`), `P.Mass` is set to accumulated mass. In regular regime, both `P.Mass` and `Mtrack` grow by swallowed mass. `SeedBHDynMass` serves as floor.

**Files modified:** `blackhole.c`, `blackhole.h`, `starcluster_evolution.c`

---

## 2026-05-12 — Unified Mtrack and StarClusterBHDyn cleanup

**Branch:** StarClusterEvolution

Unified the Mtrack mass-conservation tracker: Mtrack is now always active regardless of SeedBHDynMass. P.Mass is always derived from `max(Mtrack [+ SC if StarClusterBHDyn], SeedBHDynMass)`. Removed the three-branch accretion logic (StarClusterOn/seed-regime/SeedBHDynMass==0) in favor of a single path: `Mtrack += dynaccmass`, then recompute P.Mass. Gas swallowing now always compares BHP.Mass vs Mtrack directly. BH merger othermass always uses `Mtrack + SC` (true physical mass). StarClusterBHDyn now cleanly controls whether SC mass is in P.Mass; BH drag and dynamical friction use P.Mass directly (no SC subtraction). StarClusterEvolution (SC mass return) simplified since it requires StarClusterBHDyn=1.

**Files modified:** `blackhole.c`, `blackhole.h`, `slotsmanager.h`, `starcluster_evolution.c`

---

## 2026-05-14 — GW recoil kick for BH mergers (GWRecoilKickOn)

**Branch:** StarClusterEvolution

Added `GWRecoilKickOn` parameter (int, default 0). When enabled, BH merger remnants receive a gravitational wave recoil kick velocity assuming non-spinning BHs. The kick magnitude follows the Fitchett/Gonzalez+ fitting formula as a function of mass ratio. The kick direction is random within the orbital plane (perpendicular to the angular momentum vector of the merging pair). Applied after momentum-conserving velocity update in feedback postprocess. When `StarClusterOn=1`, the kick is compared against the escape velocity of the combined star cluster (MK12 half-mass radius model); if the kick exceeds v_esc, the BH is ejected from the star cluster by zeroing all SC state (mass, metallicity, metals, mass returned) with sentinels `StarClusterFormationTime=1e6`, `StarClusterLastEnrichmentMyr=-1`.

Also added SC property merging during BH mergers: `StarClusterFormationTime` is set to the min of swallower and swallowed (oldest component), and `StarClusterLastEnrichmentMyr` is set to the max (most advanced enrichment). These are propagated via min/max reduce through two new priv arrays.

**Files modified:** `blackhole.c`, `blackhole.h`, `params.c`

---

## 2026-05-15 — Switch GW recoil escape velocity model from MK12 to BG21

**Branch:** StarClusterEvolution

Changed the star cluster half-mass radius model used for escape velocity in the GW recoil kick from MK12 (Marks & Kroupa 2012) to BG21 (Brown & Gnedin 2021) with the full LEGUS sample (age="all"). The BG21 model uses `Reff = 2.55 * (M/1e4)^0.242` projected to 3D half-mass radius via `rh = (4/3) * Reff`.

**Files modified:** `blackhole.c`

---

## 2026-05-16 — Split GWRecoilKickOn into two independent parameters

**Branch:** StarClusterEvolution

Replaced `GWRecoilKickOn` with two independent parameters: `GWRecoilVelocityKick` (applies the velocity kick to the merger remnant) and `GWRecoilSCKick` (checks if kick exceeds escape velocity and zeros star cluster mass). Both default to 0. The kick velocity is calculated if either is enabled. `GWRecoilSCKick=1` requires `StarClusterOn=1` (endrun if not).

**Files modified:** `blackhole.c`, `params.c`

---

## 2026-05-20 — Fix GSL underflow crash in star cluster mass sampling

**Branch:** StarClusterEvolution

Fixed a crash caused by `gsl_sf_expint_E1` triggering a fatal GSL underflow error when `Mcstar` is very small (making the argument `msc_max_code / Mcstar` very large). Replaced all `gsl_sf_expint_E1()` calls in `sfr_eff.c` with a `safe_expint_E1()` wrapper that uses the error-returning variant `gsl_sf_expint_E1_e()` and treats underflow as zero (mathematically correct since E1 decays exponentially for large arguments).

**Files modified:** `sfr_eff.c`

---

## 2026-05-26 — Add BH surrounding gas dimensionless vorticity calculation

**Branch:** StarClusterEvolution

Add a dimensionless vorticity calculation for the gas surrounding each BH. The SPH curl estimator ω = (1/ρ) Σⱼ mⱼ (vⱼ - v_BH) × ∇Wᵢⱼ is computed during the BH accretion tree walk, then converted to dimensionless form ω* = ω G M_BH / c_s³ (where c_s is the local sound speed). Both the dimensionless vorticity (`BH_DimlessVorticity`) and the sound speed (`BH_SoundSpeed`) are recorded in the BH details file. A new integer parameter `BHVorticity` (default 0) controls the vorticity calculation; when `BHVorticity=0`, the vorticity output is explicitly zeroed every step so no stale values leak. The sound speed is always recorded. When `BHP(i).Density <= 0` or `soundspeed <= 0`, the vorticity is set to zero.

**Files modified:** `blackhole.c`, `blackhole.h`, `bhinfo.c`, `params.c`

---

## 2026-05-26 — Fix Bondi radius in BH vorticity calculation to include relative velocity

**Branch:** StarClusterEvolution

Fixed the dimensionless vorticity formula to use the correct Bondi radius `R_bondi = G * M_BH / (c_s² + v_rel²)`. Previously the formula used `G * M_BH / c_s³`, missing the BH-gas relative velocity term. Now uses `(c_s² + v_rel²)^1.5` (the `norm` variable already computed for the accretion rate) as the denominator.

**Files modified:** `blackhole.c`

---

## 2026-05-30 — SeedSecFOFcomSample: combined per-secFOF star-cluster sampling for BH seeding

**Branch:** SecFOFCombined

Added `SeedSecFOFcomSample` parameter (int, default 0; requires `SeedInSecFOFasStarCluster=1`
and `MinMscForBHseed>0`, else the run exits). In this mode, BH seeding in the secondary FOF
draws one combined star-cluster sample per group instead of summing per-star samples: the mass
function n(m)~m^-2 exp(-m/M_cut) uses M_cut = group total unseeded stellar mass, the number of
clusters is n = Σ(m_star·Γ)/<m> (over unseeded stars), and a Poisson-then-mass draw yields
`bhseed_msc` = summed mass of sampled clusters above 1e4 Msun. A group seeds a BH (spawned from
its largest-ClusterMass star) only if `bhseed_msc ≥ MinMscForBHseed`. The seed mass reuses the
existing scaling (`SeedBlackHoleMass·bhseed_msc` when `BHseedMassScaleMsc=1`, else fixed
`SeedBlackHoleMass`); the attached star-cluster mass is the full Σ(m_star·Γ) only when
`StarClusterBHDyn=1`. Per-star cluster-mass sampling is skipped in this mode. A new per-star
`Seeded` flag excludes stars that already contributed to a seed from later seeding sums (set on
all stars of a group after it seeds; replaces ClusterMass-zeroing in this mode). The combined
draw runs serially (the GSL E1 wrapper is not OpenMP-safe). Plan at
`plan/seed_secfof_combined_sample_plan.md`.

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`,
`libgadget/fof.c`, `libgadget/fof.h`, `libgadget/blackhole.c`, `libgadget/blackhole.h`,
`libgadget/secondfof.c`, `libgadget/run.c`, `libgadget/slotsmanager.h`, `libgadget/petaio.c`

---

## 2026-06-02 — Consistent seeded-star handling + total/seeded star-cluster mass in catalogs

Made the two star-cluster BH seeding paths behave consistently and exposed the consumed
cluster mass in the group catalogs. `BlackholeSeedSCparticle` now flags a seeded star with
`Seeded=1` and keeps its `ClusterMass` (instead of zeroing it), and skips already-seeded
stars — matching the `SeedSecFOFcomSample` convention. The `Seeded` flag is now an
unconditional exclusion from every star-cluster seeding sum (no longer only in com mode),
which also prevents double-seeding when the per-particle and FOF-based paths run together.

FOF/secFOF group star-cluster mass and metallicity are now accumulated over **all** member
stars (total), while seeding internally uses new unseeded-only accumulators
(`StarClusterMassUnseeded`, `StarClusterMassSampleUnseeded`). A new group field
`SCMass_seeded` records the cluster mass of already-seeded stars and is written to both the
PIG (`SCMass_seeded`) and SecPIG (`SecSCMass_seeded`) catalogs; the existing `SCMass`/
`StarClusterMass` output is now the group total (`total = unseeded + SCMass_seeded`). The
seed-mass payload metallicity is the total-mass-weighted group mean.

**Files modified:** `libgadget/fof.h`, `libgadget/fof.c`, `libgadget/blackhole.c`,
`libgadget/fofpetaio.c`, `libgadget/secondfof.c`, `libgadget/slotsmanager.h`

---

## 2026-06-02 — Record seeding star-cluster mass on BH (init_Msc, init_Msc_sample)

Added two black-hole properties recording the star-cluster mass that seeded each BH:
`init_Msc` (cluster-forming mass, Σ star_mass·Γ of the consumed stars) and `init_Msc_sample`
(the mass drawn from the cluster mass function that triggered the seed). They are set once at
seeding and never modified by mergers, so an accretor retains its own seed value. Per seeding
path: per-particle (`BlackholeSeedSCparticle`) uses the star's ClusterMass and
StarClusterMass_sample; combined per-secFOF (`SeedSecFOFcomSample`) uses the group's unseeded
cluster-forming mass and the full sampled cluster mass of all consumed stars (the whole draw —
not just the >1e4 Msun part that sets the seed mass; the combined sampler now also returns this
total); group star-cluster seeding (`BlackHoleSeedStarCluster`) uses the group's unseeded
cluster and sampled masses; gas/halo seeds leave both 0. Both are written (non-fatal) to the
type-5 block of the snapshot (PART), PIG, and SecPIG catalogs, and pre-zeroed when reading
older snapshots.

**Files modified:** `libgadget/slotsmanager.h`, `libgadget/blackhole.h`,
`libgadget/blackhole.c`, `libgadget/fof.c`, `libgadget/petaio.c`

---

## 2026-06-02 — Cap combined-sampled SC mass at hosting stellar mass (SCmasscapSecFOFstarmass)

Added int parameter `SCmasscapSecFOFstarmass` (default 0; only relevant with
`SeedSecFOFcomSample=1`). When 1, the combined per-secFOF star-cluster sampling caps its
sampled cluster mass at the group's total unseeded stellar mass (the mass-function cutoff
M_cut), so the sampled SC mass cannot exceed the stellar mass that hosts it. The cap is applied
to both the full sampled draw (recorded as the BH `init_Msc_sample`) and the seed-driving
> 1e4 Msun sum that sets the seed mass, preserving bhseed_msc <= total_sampled <= M_cut. When
0, no cap (previous behaviour).

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`

---

## 2026-06-03 — Fix epicyclic frequency and feedback-limited M_cstar (StarCluster)

Corrected the epicyclic frequency used in the star-cluster Toomre-mass model to match the
E-MOSAICS prescription (Pfeffer et al. 2018, eq. A6): the model now uses the smallest tidal
eigenvalue (radial direction) rather than the largest, and converts the (comoving) tidal field
to physical units before forming kappa^2, consistent with the analysis notebook `code/SC_MF.ipynb`.
Replaced the explicit Toomre-mass × collapse-fraction product with the equivalent
min(Toomre-limited, feedback-limited) cloud mass: this reproduces the previous result where gas
is rotationally supported, restores the missing quartic feedback dependence, and now assigns a
finite feedback-limited cluster mass (instead of zero) in compressive regions that have no
centrifugal support (kappa^2 <= 0).

**Files modified:** `libgadget/sfr_eff.c`

---

## 2026-06-04 — Fix OpenMP race in safe_expint_E1 (spurious GSL underflow abort)

Fixed a crash (`GSL_ERROR ... expint.c ... errno:15 underflow`, MPI_Abort 2001) that occurred
during star-cluster sampling. `safe_expint_E1` suppressed the expected large-x E1 underflow by
toggling GSL's global error handler off/on around the call, but it is invoked from
`make_particle_star` inside an OpenMP parallel for, so concurrent threads raced on the shared
global handler and intermittently aborted. The wrapper now reproduces GSL's own underflow
threshold and returns 0 for those x without calling GSL, so the global handler is never invoked —
making the function thread-safe with no global side effects.

**Files modified:** `libgadget/sfr_eff.c`

---

## 2026-06-04 — FOFMinPrimaryLength / SecondFOFMinPrimaryLength: minimum primary-particle count per FOF group

**Branch:** SecFOFCombined

Added two integer parameters (both default 0 = disabled) that drop FOF groups with too few primary-linking-type particles, so such groups are no longer counted as a FOF at all — excluded from the catalog and from BH seeding:
- `FOFMinPrimaryLength` — applies to the primary FOF.
- `SecondFOFMinPrimaryLength` — applies to the secondary FOF (both the SecPIG catalog and secondary-FOF BH seeding).

The filter is applied in the FOF engine's small-group elimination stage (alongside the existing min-length cut), so the existing group-number assignment keeps group and particle catalogs consistent and particles in dropped groups get GrNr=-1. The secondary value is recorded as a `SecondFOFMinPrimaryLength` attribute in the SecPIG header.

**Files modified:** `gadget/params.c`, `libgadget/fof.h`, `libgadget/fof.c`, `libgadget/secondfof.c`

---

## 2026-06-05 — secondary-FOF seeding: flag Seeded instead of zeroing ClusterMass (SeedSecFOFcomSample OFF)

**Branch:** SecFOFCombined

Made the post-seeding bookkeeping in secondary-FOF BH seeding consistent across all modes. Previously, with `SeedSecFOFcomSample` OFF, stars in a just-seeded group had their `ClusterMass`/`StarClusterMass_sample` zeroed, while the combined-sample mode and the per-star seeder only set the `Seeded` flag. Now all paths set `Seeded = 1` and keep `ClusterMass`/`StarClusterMass_sample` as a record. This is safe because the group-property accumulation already excludes `Seeded` stars from the unseeded cluster-mass sums that drive seeding (no behavior change to the seeding decision).

**Files modified:** `libgadget/secondfof.c`, `libgadget/slotsmanager.h`

---

## 2026-06-07 — fix "Mismatched Free: SlotsBase" crash during BH seeding

**Branch:** SecFOFCombined

Fixed a memory-allocator (LIFO) crash that aborted multi-seeding runs when BH slots had to grow during seeding. The live gas/BH force tree from the main loop sits on the allocator's bottom stack above the slots block, so growing the slots violated LIFO. The seeding routines now temporarily relocate the force tree (and active-particle list) off the bottom stack around the slot growth and restore it afterwards — the same pattern already used by star formation. Also tightened the multi-seed BH-slot pre-reservation so it no longer over-counts by the full number of member stars (now capped per group at the actual number of extra seeds), avoiding unnecessary slot growth. The tree stays valid because seeding only converts particles in place.

**Files modified:** `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/secondfof.c`, `libgadget/secondfof.h`, `libgadget/blackhole.c`, `libgadget/blackhole.h`, `libgadget/run.c`

---

## TODO

- Allow seeding in primary FOF and secondary FOF to be on in the same run.
- SecPIG particle catalog: currently forces a separate partmanager copy (no PartManager reuse) to avoid GrNr/SecGrNr corruption during the save-restore cycle in `secondfof_write`. If star fractions grow large at low redshift and the >25% threshold is hit, this will use significantly more memory. Implement proper PartManager-reuse support for SecPIG (handle domain exchange and avoid the saved-array restore) when this becomes an issue.
