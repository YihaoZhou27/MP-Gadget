# Blackhole Tidal Field Implementation Plan

## Goal

Compute the gravitational tidal field strength for active black hole particles at every timestep, and record it in the BH detail output file. Controlled by a new `BlackholeTidalField` parameter.

## Background: Existing Gas Tidal Field

The gas tidal field (`GasTidalField` parameter) computes the tidal tensor T_ij = d²Φ/(dx_i dx_j) for gas particles. It has two components:

1. **PM (long-range):** Computed in `gravpm.c` via FFT. The PM potential is differentiated twice in Fourier space (T_ij(k) = -K_i K_j Φ(k)). Results stored in `SphP[PI].TidalTensorPM[6]`. Updated every PM step. Currently only reads out for Type 0 (gas).

2. **Tree (short-range):** Accumulated during `grav_short_tree()` in `gravshort-tree.c` via `apply_tidal_to_output()`. Uses the analytical second derivative of the softened gravity kernel with a short-range window function. Stored in a temporary `TidalTensorStore[NumPart][6]` array allocated per tree walk. Currently only accumulated for Type 0 (`P[i].Type == 0` checks in `grav_short_reduce` and `grav_short_postprocess`).

3. **Combination:** In `grav_short_postprocess()` (in `gravshort.h`), the tree tensor (without G) is added to PM tensor (divided by G to match units), then eigenvalues are computed via `tidal_field_store_eigenvalues()` and stored in `SphP[PI].TidalFieldEigenvalues[3]`.

Currently, the tidal tensor is ONLY computed when `full_particle_tree_flag == 1` (all particles in the tree), which in the non-hierarchical path happens every timestep (since `force_tree_full()` is called every timestep at `run.c:592`).

## Design Decision: Hierarchical Timestepping Compatibility

**Assertion: `BlackholeTidalField` requires `SplitGravityTimestepsOn = 0`.**

When `SplitGravityTimestepsOn = 1` (hierarchical gravity), the gravity tree only contains active particles at each sub-step (`full_particle_tree_flag = 0`). The tidal tensor from such a tree would be incomplete — missing contributions from all inactive particles. This would give incorrect tidal field values.

**Is there a feasible workaround?** No practical one:

- **Option A: Build a separate full tree for BH tidal.** This would require an extra `force_tree_full()` + tree walk every timestep just for the (typically few) BH particles. This defeats the purpose of hierarchical timestepping (which avoids building full trees) and would be very expensive — the full tree build and moment computation is O(N log N) over ALL particles regardless of how few BHs need tidal fields.

- **Option B: Cache tree contributions from previous full-tree steps and update incrementally.** This is theoretically possible but extremely complex: you'd need to track which particles changed since the last full tree step and compute differential tidal contributions. Not worth the complexity.

- **Option C: Use only the PM tidal tensor (updated every PM step) and skip the tree part.** This would miss the short-range (< a few PM cells) contribution, which is often the dominant contribution for the local tidal field around a BH. Not physically useful.

**Conclusion:** The simplest and correct approach is to require `SplitGravityTimestepsOn = 0`. With this setting, `force_tree_full()` is called every timestep (`run.c:592`), producing `full_particle_tree_flag = 1`, and the tidal tensor from the tree walk includes all source particles.

## Design Decision: PM Tidal Tensor Update Frequency

**PM tidal tensor updated every PM step only — this is fine.**

The PM force captures long-range (> a few PM cells) gravitational interactions. These change slowly compared to the short-range forces, and are already only updated at PM steps for gravity itself (`P[i].GravPM`). The short-range tree part, which captures local interactions (the dominant tidal contribution near a BH), is updated every timestep. So the total tidal field will be accurate at every timestep with a slowly-varying long-range baseline.

## Implementation Steps

### 1. Parameter Declaration

**File:** `gadget/params.c`

Add parameter declaration near the other BH parameters:
```c
param_declare_int(ps, "BlackholeTidalField", OPTIONAL, 0, 
    "If 1, compute tidal field strength for BH particles every timestep. "
    "Requires SplitGravityTimestepsOn=0.");
```

### 2. Parameter Storage and Validation

**File:** `libgadget/blackhole.c`

- Add `int BlackholeTidalField;` to `struct BlackholeParams`.
- In `set_blackhole_params()`, read with `param_get_int(ps, "BlackholeTidalField")`.
- Assert: if `BlackholeTidalField && SplitGravityTimestepsOn`, `endrun()` with a clear error message.

Need to check: `SplitGravityTimestepsOn` is stored as `All.HierarchicalGravity` in `run.c`. The BH params are set in `set_blackhole_params()` which is called from `read_parameter_file()`. We can either:
- (a) Pass the `HierarchicalGravity` value into `set_blackhole_params` — but this changes the interface.
- (b) Read `SplitGravityTimestepsOn` directly from the ParameterSet in `set_blackhole_params`, since `ps` is available.
- (c) Do the validation in `run.c` after all params are read.

**Recommended: Option (b)** — read `SplitGravityTimestepsOn` from `ps` directly in `set_blackhole_params`:
```c
if(blackhole_params.BlackholeTidalField && param_get_int(ps, "SplitGravityTimestepsOn"))
    endrun(1, "BlackholeTidalField requires SplitGravityTimestepsOn=0.\n");
```

### 3. BH Particle PM Tidal Tensor Storage

**File:** `libgadget/slotsmanager.h`

Add to `struct bh_particle_data`:
```c
MyFloat TidalTensorPM[6]; /* PM long-range tidal tensor: xx, yy, zz, xy, xz, yz */
```

### 4. PM Tidal Tensor Readout for BH Particles

**File:** `libgadget/gravpm.c`

Modify the tidal readout functions to also accumulate for Type 5 (BH) particles. Currently they filter `if(P[i].Type != 0) return;`. Change to also handle Type 5:

```c
static void readout_tidal_xx(PetaPM * pm, int i, double * mesh, double weight) {
    if(P[i].Type == 0) {
        SphP[P[i].PI].TidalTensorPM[0] += weight * mesh[0];
    } else if(P[i].Type == 5) {
        BhP[P[i].PI].TidalTensorPM[0] += weight * mesh[0];
    }
}
```

Similarly for all 6 components. The `if(P[i].Type == 5)` path should be gated on `BlackholeTidalField` being enabled. However, the gravpm readout functions don't have access to `blackhole_params`. Options:
- (a) Add a getter function `get_bh_tidalfield_on()` in `blackhole.c` (similar to `get_tidalfield_on()` in `tidalfield.c`).
- (b) Use a module-level flag in gravpm.c set during initialization.

**Recommended: Option (a)** — add `int get_bh_tidalfield_on(void)` to `blackhole.h`/`blackhole.c`.

Also need to initialize `BhP[PI].TidalTensorPM` to zero at the start of each PM step in `gravpm_force()`, similar to how gas is initialized:
```c
if(bh_tidal_on) {
    for(i = 0; i < SlotsManager->info[5].size; i++)
        memset(BhP[i].TidalTensorPM, 0, sizeof(BhP[i].TidalTensorPM));
}
```

### 5. Tree Tidal Tensor for BH Particles

**File:** `libgadget/gravshort.h`

The tidal tensor from the tree walk is already computed for all particles (the `apply_tidal_to_output()` call in `force_treeev_shortrange` in `gravshort-tree.c` is called for all particle types). However, the **reduce** and **postprocess** functions filter on `P[i].Type == 0`.

Modify `grav_short_reduce()` to also reduce tidal tensor for Type 5:
```c
if(TidalStore && (P[place].Type == 0 || P[place].Type == 5)) {
    ...
}
```

Modify `grav_short_postprocess()` to also combine and compute eigenvalues for Type 5:
```c
if(GRAV_GET_PRIV(tw)->TidalTensorStore && P[i].Type == 5) {
    int PI = P[i].PI;
    int k;
    for(k = 0; k < 6; k++)
        GRAV_GET_PRIV(tw)->TidalTensorStore[i][k] += BhP[PI].TidalTensorPM[k] / G;
    /* Compute eigenvalues and store tidal field strength on BH */
    bh_tidal_field_store(i, GRAV_GET_PRIV(tw)->TidalTensorStore[i], G);
}
```

**Important:** The tidal computation in the tree walk (`apply_tidal_to_output` call in `force_treeev_shortrange`) currently checks `do_tidal && P[i].Type == 0` (line ~395 in `gravshort-tree.c`). This needs to be changed to `do_tidal && (P[i].Type == 0 || P[i].Type == 5)`.

Wait — actually let me re-check. The `do_tidal` flag in the tree walk function is set based on `TidalTensorStore != NULL`. The `P[i].Type == 0` check is only in the reduce and postprocess, not in the tree walk kernel itself. Let me verify...

Looking at `gravshort-tree.c:395`: the `apply_tidal_to_output` is called based on `do_tidal` which is `priv->TidalTensorStore != NULL`. The Type filtering only happens in `grav_short_reduce` (line 118) and `grav_short_postprocess` (line 79). So the tree walk kernel already computes tidal for all types — it's only the reduce/store that filters.

So we need to:
1. Change `grav_short_reduce` type check to include Type 5
2. Change `grav_short_postprocess` to handle Type 5

### 6. Tidal Tensor Allocation Gating

**File:** `libgadget/gravshort-tree.c`

Currently (line 117):
```c
if(get_tidalfield_on() && tree->full_particle_tree_flag) {
```

Change to also allocate when BH tidal is on:
```c
if((get_tidalfield_on() || get_bh_tidalfield_on()) && tree->full_particle_tree_flag) {
```

### 7. BH Tidal Field Strength Computation and Storage

**File:** `libgadget/slotsmanager.h`

Add to `struct bh_particle_data`:
```c
MyFloat TidalFieldStrength; /* Tidal field strength: largest eigenvalue of tidal tensor */
```

**File:** `libgadget/tidalfield.c` (or a new helper in `blackhole.c`)

Add a function to compute eigenvalues from the tidal tensor and store the strength on the BH particle:
```c
void bh_tidal_field_store(int i, const MyFloat tensor[6], double G) {
    double T[6];
    for(int k = 0; k < 6; k++)
        T[k] = tensor[k] * G;
    double eig[3];
    eigen_symmetric_3x3(T[0], T[1], T[2], T[3], T[4], T[5], eig);
    /* Store the largest eigenvalue as tidal field strength.
     * lambda1 >= lambda2 >= lambda3, lambda1 is most compressive. */
    BhP[P[i].PI].TidalFieldStrength = eig[0];
}
```

**Question for user:** What definition of "tidal field strength" is desired?
- (a) Largest eigenvalue: `eig[0]` (most compressive direction)
- (b) Frobenius norm: `sqrt(sum of eig[k]^2)`
- (c) Difference: `eig[0] - eig[2]` (tidal stretching)
- (d) All three eigenvalues (store 3 floats instead of 1)

**Default assumption: Store the largest eigenvalue `eig[0]`.** Can be changed easily.

### 8. BH Detail File Output

**File:** `libgadget/bhinfo.c`

Add `float TidalFieldStrength;` to `struct BHinfo`:
```c
float TidalFieldStrength;
```

In `collect_BH_info()`, populate:
```c
info->TidalFieldStrength = BHManager[PI].TidalFieldStrength;
```

### 9. Initialization of BH Tidal Fields

**File:** `libgadget/blackhole.c`

In `blackhole_make_one()`, initialize:
```c
if(blackhole_params.BlackholeTidalField) {
    memset(BHP(child).TidalTensorPM, 0, sizeof(BHP(child).TidalTensorPM));
    BHP(child).TidalFieldStrength = 0;
}
```

## Summary of Files to Modify

| File | Changes |
|------|---------|
| `gadget/params.c` | Declare `BlackholeTidalField` parameter |
| `libgadget/blackhole.c` | Add param to struct, read it, validate against `SplitGravityTimestepsOn`, add getter, init in `blackhole_make_one` |
| `libgadget/blackhole.h` | Add `get_bh_tidalfield_on()` declaration |
| `libgadget/slotsmanager.h` | Add `TidalTensorPM[6]` and `TidalFieldStrength` to `bh_particle_data` |
| `libgadget/gravpm.c` | Extend tidal readout functions for Type 5, init BH tidal tensor to zero |
| `libgadget/gravshort.h` | Extend reduce/postprocess to handle Type 5 tidal |
| `libgadget/gravshort-tree.c` | Gate tidal allocation on BH tidal too |
| `libgadget/tidalfield.c` | Add `bh_tidal_field_store()` (or put in `blackhole.c`) |
| `libgadget/bhinfo.c` | Add `TidalFieldStrength` to BHinfo struct and collection |

## Resolved Questions

1. **Tidal field strength definition:** Frobenius norm of eigenvalues: `sqrt(eig[0]^2 + eig[1]^2 + eig[2]^2)`. Implemented via `tidal_field_norm()` in `tidalfield.c`.
2. **`BlackholeTidalField` and `GasTidalField` are independent.** The tree walk tidal allocation is gated on `get_tidalfield_on() || get_bh_tidalfield_on()`. PM readout functions handle both types. Each is controlled by its own parameter.
3. **`do_tidal` in `force_treeev_shortrange`:** The Type==0 filter IS in the kernel at line 340: `do_tidal = (TidalTensorStore != NULL) && (input->Type == 0)`. Changed to `(input->Type == 0 || input->Type == 5)`. The reduce and postprocess also had Type==0 filters, both updated to include Type 5.

## Implementation Status

**IMPLEMENTED** — all changes compiled cleanly. See logging entry in `plan/logging.md`.
