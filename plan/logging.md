# MP-Gadget Development Log

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

## TODO

- Allow seeding in primary FOF and secondary FOF to be on in the same run.
- SecPIG particle catalog: currently forces a separate partmanager copy (no PartManager reuse) to avoid GrNr/SecGrNr corruption during the save-restore cycle in `secondfof_write`. If star fractions grow large at low redshift and the >25% threshold is hit, this will use significantly more memory. Implement proper PartManager-reuse support for SecPIG (handle domain exchange and avoid the saved-array restore) when this becomes an issue.
