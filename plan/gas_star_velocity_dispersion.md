# Plan: Gas and Stellar Velocity Dispersion for Gas Particles

## Goal

At every PM step, compute the gas velocity dispersion (`VDisp_gas`) and stellar velocity dispersion (`VDisp_star`) for each gas particle, using the gas particle's SPH smoothing length (`Hsml`) as the search radius. Also record the total gas mass (`VDisp_mgas`), total stellar mass (`VDisp_mstar`), number of gas neighbors (`VDisp_Ngas`), and number of star neighbors (`VDisp_Nstar`) within that region.

## Overview of Approach

Use **two separate treewalks** for efficiency:
1. **Gas velocity dispersion**: reuse the existing `gasTree` (GASMASK | BHMASK) that is already built for density/hydro/source-terms. Filter to gas-only neighbors in the ngbiter (ignore BH neighbors). This avoids rebuilding an expensive gas-sized tree.
2. **Stellar velocity dispersion**: build a small `STARMASK`-only tree and run a second treewalk. Since N_star << N_gas typically, this tree build is cheap.

### Performance rationale

| | Old (single GASMASK\|STARMASK tree) | New (reuse gasTree + small star tree) |
|---|------|------|
| Tree build | O((N_gas + N_star) * log(N_gas + N_star)) | **0** (gas, reused) + O(N_star * log N_star) |
| Treewalk | 1 walk, 1 MPI round | 2 walks, 2 MPI rounds |
| Neighbor processing | identical total | identical total |

With N_star/N_gas ~ 0.1, tree build savings are ~90%. The extra MPI round for the second treewalk is far cheaper than rebuilding a gas-sized tree from scratch.

## New Properties (6 fields on `sph_particle_data`)

| Field | Type | Description |
|-------|------|-------------|
| `VDisp_gas` | `MyFloat` | 1D gas velocity dispersion within Hsml |
| `VDisp_star` | `MyFloat` | 1D stellar velocity dispersion within Hsml |
| `VDisp_mgas` | `MyFloat` | Total gas mass within Hsml |
| `VDisp_mstar` | `MyFloat` | Total stellar mass within Hsml |
| `VDisp_Ngas` | `int` | Number of gas neighbors within Hsml |
| `VDisp_Nstar` | `int` | Number of star neighbors within Hsml |

## Files to Modify

### 1. `libgadget/slotsmanager.h` — Add fields to `sph_particle_data`

Add the 6 new fields to `struct sph_particle_data` (after the existing `VDisp` field):

```c
MyFloat VDisp_gas;   /* 1D gas velocity dispersion within Hsml */
MyFloat VDisp_star;  /* 1D stellar velocity dispersion within Hsml */
MyFloat VDisp_mgas;  /* Total gas mass within Hsml */
MyFloat VDisp_mstar; /* Total stellar mass within Hsml */
int VDisp_Ngas;      /* Number of gas neighbors within Hsml */
int VDisp_Nstar;     /* Number of star neighbors within Hsml */
```

### 2. `libgadget/gasveldisp.c` (NEW) — Two treewalk modules

Create a new file implementing **two treewalks**: one for gas neighbors (reusing the existing gasTree), one for star neighbors (using a dedicated star-only tree).

Both treewalks share the same pattern but differ in:
- The tree they walk
- The neighbor mask (GASMASK vs STARMASK)
- The velocity prediction function (SPH_VelPred for gas, DM_VelPred for stars)
- Which output fields they write

**Predicted velocities:** Both treewalks use predicted velocities at the current force computation time via `SPH_VelPred` (for gas particles, both query and neighbor) and `DM_VelPred` (for star neighbors). The `struct kick_factor_data` is initialized once via `init_kick_factor_data()`.

**Preprocess:** A single preprocess callback (used by the first treewalk) zeroes all 6 output fields to avoid stale values from previous PM steps.

**Main function signature:**
```c
void gas_star_veldisp(const ActiveParticles * act, Cosmology * CP,
                      const DriftKickTimes * times,
                      const ForceTree * gasTree,
                      DomainDecomp * ddecomp, const char * OutputDir);
```

The function will:
1. Initialize kick factors via `init_kick_factor_data(&kf, times, CP)`
2. Run gas treewalk using the provided gasTree (GASMASK | BHMASK; ngbiter filters to gas only)
3. Build a STARMASK-only tree
4. Run star treewalk using the star tree
5. Free the star tree and temporary arrays

### 3. `libgadget/gasveldisp.h` (NEW) — Header

```c
#ifndef GASVELDISP_H
#define GASVELDISP_H

#include "forcetree.h"
#include "timestep.h"
#include "cosmology.h"
#include "domain.h"

void gas_star_veldisp(const ActiveParticles * act, Cosmology * CP,
                      const DriftKickTimes * times,
                      const ForceTree * gasTree,
                      DomainDecomp * ddecomp, const char * OutputDir);

#endif
```

### 4. `libgadget/run.c` — Call the module at PM steps

Insert the call **inside** the `GasEnabled` block, just before the final `force_tree_free(&gasTree)` at line 684, so the gasTree is still alive. Gate on `is_PM` only — **not** on `CoolingOn`.

```c
/* Gas+Star velocity dispersion: runs every PM step (not gated on CoolingOn).
 * Reuses the existing gasTree for gas neighbors; builds a small star-only tree internally. */
if(is_PM)
    gas_star_veldisp(&Act, &All.CP, &times, &gasTree, ddecomp, All.OutputDir);
```

### 5. `libgadget/petaio.c` — Register I/O blocks

Add getter functions and IO registrations for the 6 new fields (write-only):

```c
SIMPLE_GETTER_PI(GTVDispGas, VDisp_gas, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTVDispStar, VDisp_star, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTVDispMgas, VDisp_mgas, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTVDispMstar, VDisp_mstar, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTVDispNgas, VDisp_Ngas, int, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTVDispNstar, VDisp_Nstar, int, 1, struct sph_particle_data)

IO_REG_WRONLY(VDispGas,   "f4", 1, 0, IOTable);
IO_REG_WRONLY(VDispStar,  "f4", 1, 0, IOTable);
IO_REG_WRONLY(VDispMgas,  "f4", 1, 0, IOTable);
IO_REG_WRONLY(VDispMstar, "f4", 1, 0, IOTable);
IO_REG_WRONLY(VDispNgas,  "i4", 1, 0, IOTable);
IO_REG_WRONLY(VDispNstar, "i4", 1, 0, IOTable);
```

### 6. `libgadget/Makefile` — Add the new object file

Add `gasveldisp.o` to `GADGET_OBJS`.

## Velocity Dispersion Calculation Details

### Predicted Velocities

Particles in MP-Gadget store their velocities at the last kick time, not the current force computation time. Since different particles may have been kicked at different times (due to hierarchical timestepping), raw `P[i].Vel` values are not synchronized and cannot be directly compared.

To get velocities at a consistent time (the current drift/force computation time), we must use the velocity prediction functions from `density.h`:

- **`SPH_VelPred(i, VelPred, kf)`** — for gas particles (Type 0). Predicts velocity using both gravity and hydro kick factors.
- **`DM_VelPred(i, VelPred, kf)`** — for collisionless particles (DM, stars). Predicts using gravity kicks only.

### Computation Steps (same for both gas and star treewalks)

1. For each gas particle `i`, predict its velocity: `SPH_VelPred(i, Vel_i, &kf)` (in fill callback)
2. For each neighbor `j` within `P[i].Hsml`:
   - Gas: `SPH_VelPred(j, VelPred_j, &kf)`
   - Star: `DM_VelPred(j, VelPred_j, &kf)`
3. Accumulate: `V1sum[d] += v_rel[d]`, `V2sum += v_rel[d]^2`, `M += Mass`, `N += 1`
4. Postprocess: `sigma_1D = sqrt(max(V2sum/N - |V1sum/N|^2, 0) / 3)`

## Execution Order in the Time Loop

```
PM step:
  1. Domain decomposition
  2. gasTree built (GASMASK | BHMASK)
  3. Density (updates Hsml)
  4. Hydro forces
  5. gasTree freed for memory (is_PM)
  6. Gravity (PM + tree)
  7. Kicks
  8. gasTree rebuilt (GASMASK | BHMASK) for source terms
  9. Metal return / FOF / BH / Cooling / SF
  10. Gas VDisp treewalk (reuses gasTree)     <-- NEW
  11. Star VDisp treewalk (small star tree)    <-- NEW
  12. gasTree freed
  13. Snapshot output
```

## Testing Considerations

- Verify struct alignment: ensure `sizeof(Result) % 8 == 0` and `sizeof(Query) % 8 == 0`
- For a particle with no star neighbors, `VDisp_star = 0`, `VDisp_mstar = 0`, `VDisp_Nstar = 0` (zeroed by preprocess)
- For a particle with no gas neighbors (unlikely), `VDisp_gas = 0`, etc.
- The gas treewalk uses GASMASK | BHMASK tree but filters to gas-only in ngbiter — verify BH particles are skipped
- Check that the output appears in the BigFile snapshot under `0/VDispGas`, `0/VDispStar`, etc.
