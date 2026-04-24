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
