# Second FOF (Star-Primary) Implementation Plan

## Goal

Add a second, independent FOF pass that uses **star particles** (or any user-chosen type) as the primary link type, with a user-specified comoving linking length. This identifies small stellar structures (star clusters) without modifying the existing halo FOF. The second FOF code lives in its own file (`secondfof.c`) and reuses the existing FOF engine by overloading its parameters.

---

## 1. New Parameters

All parameters are registered in `gadget/params.c : create_gadget_parameter_set()` and read/broadcast in `secondfof.c`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `SecondFOFOn` | int | 0 | Master switch: run the second FOF pass |
| `SecondFOFPrimaryLinkTypes` | int | 16 | Bitmask for primary link types (16 = type 4 = stars) |
| `SecondFOFSecondaryLinkTypes` | int | 1 | Bitmask for secondary link types (1 = type 0 = gas) |
| `SecondFOFLinkingLength` | double | 0.01 | Comoving linking length in code units (kpc/h) -- **not** in units of mean particle separation |
| `SecondFOFMinLength` | int | 32 | Minimum particle count per group |
| `SecondFOFFileBase` | string | "SecPIG" | Base name for second FOF catalog files |
| `SecondFOFSize` | int | 0 | Compute group size properties (r50, r90, rmax) |

### Key design choice -- linking length

The standard halo FOF expresses its linking length as a fraction of the DM mean separation and then converts to comoving code units via `fof_init()`. For the second FOF the user supplies the linking length directly in comoving code units (kpc/h), because the relevant scale is a physical size (star cluster radius), not the mean inter-particle distance.

---

## 2. Architecture: Separate `secondfof.c` / `secondfof.h`

### Design principle

Keep all second-FOF-specific logic in new files. The existing `fof.c` is modified minimally: only to expose internal helpers so that `secondfof.c` can call them.

### What goes where

**`libgadget/secondfof.h`** -- public interface:
```c
void set_secondfof_params(ParameterSet * ps);
void secondfof_save(DomainDecomp * ddecomp, int snapnum, double atime,
                    Cosmology * CP, const double * MassTable, int MetalReturnOn,
                    MPI_Comm Comm);
int get_secondfof_on(void);       /* so run.c can check without seeing the struct */
```

**`libgadget/secondfof.c`** -- implementation:
- Holds its own `struct SecondFOFParams` (static, like `fof_params`).
- `set_secondfof_params()`: reads the 7 parameters, broadcasts.
- `secondfof_save()`: the main entry point called from `run.c` and `runfof()`. This function:
  1. Temporarily swaps the relevant fields in the existing `fof_params` (linking length, primary/secondary types, min length) with the second FOF values.
  2. Calls `fof_fof(ddecomp, 1, Comm)` -- this reuses the entire existing FOF machinery unchanged.
  3. Copies the resulting `P[i].GrNr` into `P[i].SecGrNr` for all particles.
  4. Computes extra group properties (potential minimum, r50/r90/rmax) if enabled.
  5. Saves the catalog via a dedicated `secondfof_save_groups()`.
  6. Restores the original `fof_params` values.
  7. Restores `P[i].GrNr` to its halo-FOF value (saved before step 1).

### Why "temporary swap" instead of refactoring `fof_fof()`

Refactoring `fof_fof()` and all its internal helpers (`fof_label_primary`, `fof_label_secondary`, `fof_compile_base`, treewalks, etc.) to accept a parameter struct would touch ~20 functions and 500+ lines deep inside `fof.c`. The temporary-swap approach requires:
- Exposing `fof_params` (or providing getter/setter functions) -- a few lines in `fof.c`/`fof.h`.
- No changes to any internal FOF function.
- The swap is safe because FOF is single-threaded at the top level (no concurrent calls).

### What needs to be exposed from `fof.c` / `fof.h`

Add to `fof.h`:
```c
/* Allow secondfof.c to temporarily override FOF parameters */
void fof_get_params(int *PrimaryLinkTypes, int *SecondaryLinkTypes,
                    double *ComovingLinkingLength, int *MinLength);
void fof_set_params(int PrimaryLinkTypes, int SecondaryLinkTypes,
                    double ComovingLinkingLength, int MinLength);
```

These are thin accessors for the static `fof_params` struct fields.

---

## 3. New Particle Field: `SecGrNr`

**File: `libgadget/partmanager.h`**

Add to `struct particle_data`:

```c
int64_t SecGrNr;   /* Second FOF group number; -1 if not in any group */
```

This travels with the particle during domain exchange, just like `GrNr`.

---

## 4. Extended Group Structure for Second FOF

**File: `libgadget/secondfof.c`**

Define a local extended group structure:

```c
struct SecondGroup {
    struct Group base_group;   /* Reuse the standard Group struct for mass, CM, etc. */
    /* Extra properties for the second FOF */
    double PotMinPos[3];       /* Position of primary particle with minimum potential */
    float  PotMin;             /* Minimum potential value */
    float  R50;                /* Half-mass radius of primary particles (from PotMinPos) */
    float  R90;                /* 90%-mass radius of primary particles (from PotMinPos) */
    float  Rmax;               /* Max separation of any primary particle from PotMinPos */
};
```

The `PotMinPos` is computed during group compilation by tracking the primary particle with the lowest `P[i].Potential`. The size properties (R50, R90, Rmax) are computed in a second pass after group compilation, using `PotMinPos` as the center.

---

## 5. Potential Minimum Computation (Feature 3)

During group property compilation (analogous to `add_particle_to_group` in `fof.c`), for each particle added to a second-FOF group:

```c
/* Only consider primary-linked particles for potential minimum */
if (is_primary_type(P[index].Type)) {
    if (P[index].Potential < gdst->PotMin) {
        gdst->PotMin = P[index].Potential;
        for (d = 0; d < 3; d++)
            gdst->PotMinPos[d] = P[index].Pos[d];
    }
}
```

### Post-processing check (RestartFlag=3)

When running as post-processing, particles are loaded from a snapshot. If `OutputPotential` was enabled when the snapshot was written, `P[i].Potential` will have valid values. The code should:
1. Check whether the `Potential` block exists in the snapshot (via BigFile block existence check, or just check whether `OutputPotential` is set in the parameter file).
2. If potential is not available, skip the potential-minimum computation and set `PotMinPos = CM` (center of mass) as fallback.
3. Log a warning if `SecondFOFSize` is enabled but potential is not available, since the size properties will use CM as center instead.

### MPI reduction

The potential minimum across ranks requires a custom reduction: when merging group data across tasks, keep the entry with the lower `PotMin` value, similar to how `MaxDens`/`seed_index`/`seed_task` is reduced in the existing FOF.

---

## 6. Group Size Computation (Feature 2, controlled by `SecondFOFSize`)

After all group properties are compiled and reduced across MPI ranks (so `PotMinPos` is finalized), compute R50, R90, Rmax:

### Algorithm

For each group, on each MPI rank:
1. Collect all **primary-linked** particles belonging to this group on this rank.
2. Compute distances from `PotMinPos` (with periodic wrapping).
3. Gather `(distance, mass)` pairs across all ranks holding particles of this group.
4. Sort by distance.
5. Accumulate mass; find r at 50% and 90% of total primary mass.
6. Rmax = max distance.

### Implementation approach

This is done **after** `fof_compile_catalogue` + `fof_reduce_groups` (which gives us the finalized `PotMinPos` and total primary mass per group). The steps are:

1. **Local pass:** For each local particle with `SecGrNr >= 0` and primary type, compute distance to its group's `PotMinPos`. Store `(SecGrNr, distance, mass)` triples.
2. **Sort** the triples by `(SecGrNr, distance)`.
3. **Reduce** across ranks: use the same MPI reduction framework as `fof_reduce_groups`. Each rank contributes sorted distance/mass lists per group. The reduction merges sorted lists and computes R50/R90/Rmax.

Alternatively, since star cluster groups are typically small and entirely local, a simpler approach:
1. After the group catalogue is finalized, loop over local primary particles.
2. For each group on this rank, collect distances, sort locally, compute partial mass sums.
3. Do an MPI_Allreduce for R50/R90/Rmax per group (using a binned approach or exact gather for small groups).

The exact implementation will depend on typical group sizes. For star clusters (tens to hundreds of particles), gathering all distances on one rank per group is feasible.

---

## 7. Post-Processing via RestartFlag=3 (Feature 1)

**File: `libgadget/run.c`**

The existing `runfof()` function (line 814) runs the halo FOF on a snapshot and saves the PIG catalog. When `SecondFOFOn` is enabled, add the second FOF call after the halo FOF:

```c
void runfof(const int RestartSnapNum, const inttime_t Ti_Current,
            const struct header_data * header)
{
    /* ... existing setup: domain decompose, SFR recompute ... */

    /* Existing halo FOF */
    FOFGroups fof = fof_fof(ddecomp, 1, MPI_COMM_WORLD);
    fof_save_groups(&fof, All.OutputDir, All.FOFFileBase, RestartSnapNum,
                    &All.CP, header->TimeSnapshot, header->MassTable,
                    All.MetalReturnOn, MPI_COMM_WORLD);
    fof_finish(&fof);

    /* NEW: Second FOF if enabled */
    if (get_secondfof_on()) {
        secondfof_save(ddecomp, RestartSnapNum, header->TimeSnapshot,
                       &All.CP, header->MassTable, All.MetalReturnOn,
                       MPI_COMM_WORLD);
    }
}
```

So `MP-Gadget paramfile 3 N` with `SecondFOFOn = 1` in the parameter file will produce **both** `PIG_N` and `SecPIG_N`.

---

## 8. Second FOF Catalog Output

**File: `libgadget/secondfof.c`**

Create a dedicated `secondfof_save_groups()` that writes the SecPIG catalog. This is similar to `fof_save_particles()` in `fofpetaio.c` but:

1. Registers a different set of IO blocks (includes R50, R90, Rmax, PotMinPos; excludes BH seeding fields).
2. Uses `SecGrNr` instead of `GrNr` for particle selection and sorting.
3. Writes to `SecPIG_NNN` directory.

### Catalog fields

**Standard fields (inherited from `struct Group`):**
- `GroupID` (u4) -- the second FOF group number
- `Mass` (f4) -- total group mass
- `MassCenterPosition` (f8, 3) -- center of mass
- `MassCenterVelocity` (f4, 3) -- CM velocity
- `LengthByType` (u4, 6) -- particle count per type
- `MassByType` (f4, 6) -- mass per type
- `Imom` (f4, 9) -- inertia tensor
- `Jmom` (f4, 3) -- angular momentum
- `StarFormationRate` (f4)

**New fields specific to second FOF:**
- `PotMinPos` (f8, 3) -- position of the primary particle with minimum potential
- `PotMin` (f4, 1) -- minimum potential value
- `R50` (f4, 1) -- half-mass radius of primary particles (only if `SecondFOFSize = 1`)
- `R90` (f4, 1) -- 90%-mass radius of primary particles (only if `SecondFOFSize = 1`)
- `Rmax` (f4, 1) -- max primary particle distance from center (only if `SecondFOFSize = 1`)

### Particle saving

For the SecPIG catalog, `FOFSaveParticles` behavior should use `SecGrNr` instead of `GrNr` to select particles. This requires either:
- A separate save function that checks `SecGrNr >= 0` instead of `GrNr >= 0`, or
- Temporarily copying `SecGrNr` into `GrNr` before calling the existing save machinery, then restoring.

The temporary-copy approach is simpler and consistent with the swap strategy.

---

## 9. Snapshot Output: `SecGroupID`

**File: `libgadget/petaio.c`**

Add a new getter and register it:

```c
SIMPLE_GETTER(GTSecGroupID, SecGrNr, uint32_t, 1, struct particle_data)

/* In register_io_blocks(), alongside the GroupID block: */
if (WriteGroupID && SecondFOFOn) {
    /* Register for all particle types, like GroupID */
    IO_REG_WRONLY(SecGroupID, "u4", 1, i, IOTable);
}
```

Particles not in any second-FOF group will have `SecGrNr = -1`, which maps to `UINT32_MAX` in unsigned output -- same convention as `GroupID`.

---

## 10. Execution Order in `run.c` (Live Simulation)

**File: `libgadget/run.c`**, around lines 696-722:

```
if (WriteFOF) {
    fof = fof_fof(ddecomp, 1, MPI_COMM_WORLD);       // Halo FOF -> P[i].GrNr
}
if (WriteFOF && get_secondfof_on()) {
    secondfof_save(ddecomp, snapnum, atime, ...);     // Second FOF -> P[i].SecGrNr
}
if (WriteSnapshot)
    write_checkpoint(snapnum, WriteFOF, ...);          // Snapshot with GroupID + SecGroupID
if (WriteFOF) {
    fof_save_groups(&fof, ..., "PIG", ...);            // Halo PIG catalog
    fof_finish(&fof);
}
```

The second FOF must run **before** `write_checkpoint` so that `SecGrNr` is available for snapshot output. The `secondfof_save()` function handles the second FOF catalog (SecPIG) internally, so it can be called either before or after the snapshot write. But for consistency, run it before snapshot write.

Note: `secondfof_save()` internally saves/restores `GrNr`, so the halo FOF's `GrNr` values survive.

---

## 11. Detailed File-by-File Changes

### New files

#### `libgadget/secondfof.h`
- Declare `set_secondfof_params()`, `secondfof_save()`, `get_secondfof_on()`.
- Define `struct SecondGroup` (extended group with PotMinPos, R50, R90, Rmax).

#### `libgadget/secondfof.c`
- `struct SecondFOFParams`: holds all 7 parameters.
- `set_secondfof_params(ParameterSet * ps)`: read and broadcast.
- `get_secondfof_on()`: accessor.
- `secondfof_save()`: main entry point.
  - Save current `fof_params` via `fof_get_params()`.
  - Save `P[i].GrNr` for all particles into a temporary array.
  - Override `fof_params` via `fof_set_params()` with second FOF values.
  - Call `fof_fof(ddecomp, 1, Comm)` to run FOF.
  - Copy `P[i].GrNr` -> `P[i].SecGrNr`.
  - Restore `P[i].GrNr` from saved array.
  - Restore `fof_params` via `fof_set_params()`.
  - Compute extended properties (PotMinPos, R50/R90/Rmax) using the FOFGroups result.
  - Save SecPIG catalog.
  - `fof_finish()`.
- `secondfof_compile_extra()`: compute PotMinPos, R50, R90, Rmax from particle data.
- `secondfof_register_io_blocks()`: register IO blocks for SecPIG catalog.
- `secondfof_save_groups()`: write SecPIG files (similar to `fof_save_particles`).

### Modified files

#### `gadget/params.c`
- Register 7 new parameters: `SecondFOFOn`, `SecondFOFPrimaryLinkTypes`, `SecondFOFSecondaryLinkTypes`, `SecondFOFLinkingLength`, `SecondFOFMinLength`, `SecondFOFFileBase`, `SecondFOFSize`.

#### `libgadget/fof.h`
- Declare `fof_get_params()` and `fof_set_params()`.

#### `libgadget/fof.c`
- Implement `fof_get_params()` and `fof_set_params()` (trivial accessors, ~10 lines each).

#### `libgadget/partmanager.h`
- Add `int64_t SecGrNr;` to `struct particle_data`.

#### `libgadget/petaio.c`
- Add `GTSecGroupID` getter.
- Register `SecGroupID` block in `register_io_blocks()` when `SecondFOFOn && WriteGroupID`.
- Need to pass `SecondFOFOn` flag to `register_io_blocks()` (add parameter or check global).

#### `libgadget/run.c`
- `#include "secondfof.h"`
- Call `set_secondfof_params()` during init.
- In live simulation loop: call `secondfof_save()` between halo FOF and snapshot write.
- In `runfof()`: call `secondfof_save()` after halo FOF.

#### `libgadget/Makefile` (or `CMakeLists.txt`)
- Add `secondfof.c` to the build.

---

## 12. Implementation Order

1. **Add `SecGrNr` to `particle_data`** in `partmanager.h`.
2. **Register new parameters** in `params.c`.
3. **Add `fof_get_params`/`fof_set_params`** to `fof.c`/`fof.h` -- minimal change to existing code.
4. **Create `secondfof.c`/`secondfof.h`** with basic structure: params, swap-and-call, catalog output (without size properties first).
5. **Integrate into `run.c`**: live simulation and `runfof()` post-processing.
6. **Add `SecGroupID` snapshot output** in `petaio.c`.
7. **Add potential-minimum tracking** in `secondfof_compile_extra()`.
8. **Add size properties** (R50/R90/Rmax) controlled by `SecondFOFSize`.
9. **Add build system entry** for `secondfof.c`.
10. **Test** with a small simulation and post-processing mode.

---

## 13. Notes and Caveats

- **Performance:** The second FOF builds its own tree from primary (star) particles. Stars are typically far fewer than DM, so this should be cheap.
- **Temporary GrNr swap:** The swap of `P[i].GrNr` requires allocating a temporary array of `NumPart` int64_t values. For large simulations this is ~8 bytes/particle, acceptable.
- **Linking length units:** The standard FOF linking length is in units of mean DM separation. The second FOF linking length is in comoving code units (kpc/h) directly.
- **No BH seeding:** The second FOF does not participate in BH seeding.
- **Backward compatibility:** When `SecondFOFOn = 0` (default), no extra work is done, no extra fields are written.
- **Post-processing potential:** When running `MP-Gadget paramfile 3 N`, the snapshot must have been written with `OutputPotential = 1` for the potential-minimum center to work. Otherwise falls back to CM.
- **SecPIG particle saving:** Uses the same `FOFSaveParticles` setting as the halo FOF. Could add a separate flag if needed.
- **Size properties are comoving:** R50, R90, Rmax are in comoving code units (kpc/h), consistent with the linking length and positions.
