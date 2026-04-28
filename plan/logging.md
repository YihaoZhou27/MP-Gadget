# Implementation Log

## 2026-04-22
- Created and revised implementation plan for Second FOF (star-primary) pass to identify small stellar structures (star clusters). Key features: separate `secondfof.c` file, post-processing via RestartFlag=3, group size properties (R50/R90/Rmax), potential-minimum center. Plan saved at `plan/second_fof_plan.md`.
- Implemented second FOF: new files `secondfof.c`/`secondfof.h`, parameters registered in `params.c`, `fof_get/set_params` accessors in `fof.c`, `SecGrNr` field in `partmanager.h`, `SecGroupID` snapshot block in `petaio.c`, integration in `run.c` (live loop + runfof post-processing). SecPIG catalog with PotMinPos, R50/R90/Rmax. Compiles cleanly.
- Fixed `cmp_double` UB bug (was sorting `int64_t` array with `double` comparator).
- Split `secondfof_save()` into `secondfof_run()`/`secondfof_write()`/`secondfof_finish()` for I/O safety (write SecPIG after checkpoint, not before).

## 2026-04-23
- Moved PotMin/PotMinPos computation from separate `secondfof_compute_potmin()` (~220 lines) into `fof_compile_catalogue` (`add_particle_to_group`/`fof_reduce_group` in `fof.c`). Added `PotMin`/`PotMinPos` fields to `struct Group` in `fof.h`. Simplified `SecondGroupExtra` to only R50/R90/Rmax. Falls back to CM when potential unavailable. Compiles cleanly.
- Added `FOFPotentialMin` parameter (default 0) to control whether PotMin tracking runs in the FOF engine. PotMin tracking in `add_particle_to_group`/`fof_reduce_group` is now guarded by this flag. Second FOF sets it automatically based on `OutputPotential`. Compiles cleanly.
- Fixed stack allocator LIFO violation in `secondfof.c`: deferred `SecFOF_Result` allocation until after `SecFOF_SavedGrNr` is freed, using a local `FOFGroups` variable to hold the intermediate fof_fof() result.
- Fixed another stack allocator LIFO violation: reordered `SecFOF_Output` allocation before `SecFOF_Extra` so that `SecFOF_Extra` can be freed while on top of the stack.
- Fixed latent LIFO violation in `secondfof_compute_sizes`: deferred frees of `dm_local`, `rcounts`, `rdispls` until after `dm_global` is freed (would crash when `SecondFOFSize` is enabled).
- Fixed stack allocator LIFO violation in `secondfof_finish`: `Group` was freed before `SecFOF_Result`, violating LIFO order. Fixed by copying `fof` to a local variable, freeing `result` first, then calling `fof_finish`.
- Fixed stack allocator LIFO violation in `run.c`: `fof_finish` (freeing primary Group) was called before `secondfof_finish`, but second FOF allocations sat on top of the primary Group in the stack. Reordered both call sites to finish second FOF before primary FOF.
- Fixed MPI_Type_free double-free crash: `secondfof_finish` called `fof_finish` which freed the static `MPI_TYPE_GROUP` datatype, then the outer `fof_finish` for the primary FOF tried to free it again. Fixed by having `secondfof_finish` free only the Group array directly instead of calling `fof_finish`.
- Added particle catalog saving to SecPIG output. Only primary-linked particles are saved (ordered by secondary FOF group number). Extracted `fof_save_particles_to_bigfile()` from `fofpetaio.c` as a reusable function. In `secondfof_write()`, temporarily swaps `SecGrNr` into `GrNr` for primary particles, calls the particle saver, then restores `GrNr`. Updated header to count only primary-linked particles. Compiles cleanly.

## 2026-04-25
- Fixed MPI deadlock in `secondfof_compute_sizes` when `SecondFOFSize=1`. The old code called MPI collectives (Allgather/Allgatherv) inside a per-group loop, but different ranks owned different numbers of groups, causing ranks with fewer groups to exit the loop early while others waited. Replaced with two single Allgatherv calls: one for group centers, one for all particle (dist, mass, GrNr) data. Each rank then computes R50/R90/Rmax locally for its own groups from the complete global data. No per-group MPI collectives needed.
- Added `SecFOFonly` parameter (default 0). When set to 1, skips saving the primary FOF catalog (PIG files) while still running the primary FOF internally and saving only the second FOF catalog (SecPIG). Works for both live simulation (RestartFlag=1) and post-processing (RestartFlag=3).

## 2026-04-26
- Added `SeedInSecFOF` parameter (default 0). When set to 1 (and `SecondFOFOn=1`), black hole seeding uses the secondary FOF catalog instead of the primary. A new `secondfof_seed()` function temporarily swaps FOF params to secondary values, runs FOF, calls `fof_seed`, then restores params. Avoids MPI_TYPE_GROUP double-free by manually freeing the Group array instead of calling `fof_finish`.
- Fixed bug where `4/GroupID` and `4/SecGroupID` in the SecPIG particle catalog saved identical values. The GrNr-to-SecGrNr swap for particle selection was clobbering the primary FOF ID. Now both fields are swapped before distribution, and `fof_save_particles_to_bigfile` swaps them back on the distributed particles before writing IO, so `GroupID` = primary FOF ID and `SecGroupID` = secondary FOF ID.

## 2026-04-27
- Added `PrimaryFOFNum` and `PrimaryFOFID` properties to the SecPIG catalog. For each secondary FOF group, `PrimaryFOFNum` counts how many distinct primary FOF groups its primary-linked particles belong to, and `PrimaryFOFID` records the primary FOF group that hosts the largest fraction. Computed via Allgatherv pattern (same as sizes). Written as `i4` blocks in SecPIG output. Compiles cleanly.
- Fixed SecPIG particle catalog metadata inconsistency: moved `petaio_build_selection` before the GrNr/SecGrNr swap in `fof_save_particles_to_bigfile`. Previously, particles in a secondary FOF but not in any primary FOF (original GrNr = -1) were counted in the header but excluded from the particle blocks after the swap. Now selection happens while GrNr still holds the secondary FOF ID (>= 0), so all grouped particles are included.
