# Plan: `SeedSeedFOFMassiveBoundStar` — bound-star restriction for massive secFOF groups

**Status:** PLAN ONLY (no code changed yet). Author: pre-implementation design.
**Target tree:** `/home1/09475/yihaoz/work2/Data/SCmodel/MP-Gadget-development/MP-Gadget-dev2`
(resolves to `/scratch3/09475/yihaoz/Software/MP-Gadget-dev2`).

---

## 1. Goal (user spec)

In the combined-sample secondary-FOF seeding path (`SeedSecFOFcomSample=1`):

> When a secFOF group's **unseeded star-cluster mass** Σ(m·Γ) exceeds **1e8 M⊙**, do
> **not** use the whole structure. Instead, determine which unseeded star particles are
> **gravitationally bound** to the secFOF (using **all** particles in the secFOF to compute
> the potential), and use **only the bound unseeded stars** to seed the black hole.
> If `BHseedMassScaleMsc=1`, the BH seed mass = `SeedBlackHoleMass` × (combined-sampled mass
> from the **bound** unseeded stars).

Controlled by a new integer parameter **`SeedSeedFOFMassiveBoundStar`** (default `0`;
feature active only when `=1`).

### Confirmed design decisions (from earlier Q&A)
- **Seed mass:** *Re-run the combined sampler on the bound stars* (do not just rescale).
- **Trigger quantity:** the **star-cluster mass Σ(m·Γ)** = `StarClusterMassUnseeded` (NOT stellar mass).
- **Potential:** summed over **all** secFOF member particles, **softened** (Plummer-equivalent).
- **Rest frame:** velocity of the **potential-minimum particle** (deepest member of the well).

---

## 2. Scope & gating

The feature is active **only** when ALL of:
1. `SeedSeedFOFMassiveBoundStar == 1`
2. `SeedSecFOFcomSample == 1`
3. `BlackHoleSeedStarCluster` is in effect (always true in `secondfof_seed`, which forces it on)
4. **NOT** `SeedSecFOFcomSampleParticle` (v1 explicitly excludes the per-particle sampler — see §9)

Per-group activation additionally requires the trigger:
- `StarClusterMassUnseeded > get_msc_multiseed_thresh_code()` (= 1e8 M⊙ in code units).

For groups below the trigger, behaviour is **unchanged** (whole structure used).

When `SeedSeedFOFMassiveBoundStar == 0`, the entire feature is a no-op and the code path is
identical to today.

---

## 3. Physics: the boundedness test

For each **unseeded star** *i* in a qualifying group, decide bound/unbound using specific
energy E = KE + PE (per unit mass), evaluated in the rest frame of the deepest-potential
member and against the potential of **all** members.

### Units (MP-Gadget comoving convention)
- `P[i].Pos` are **comoving** code lengths; pairwise separations `r_ij` below are comoving.
- `P[i].Vel` is the code velocity; **physical peculiar velocity = `P.Vel / a`**
  (confirmed from `stats.c`: KE = `0.5·m·Vel²/a²`).
- `G = CP->GravInternal` (internal units).
- Softening `eps = FORCE_SOFTENING() / 2.8` (Plummer-equivalent; same idiom as `fof.c:1978`
  and `gravshort.h:72`). Softening is comoving, matching `r_ij`.

### Specific potential magnitude at member k (softened, over ALL members j≠k)
```
pot_mag(k) = Σ_{j≠k} m_j / sqrt(r_kj² + eps²)        (comoving r_kj, code masses)
```
Physical specific potential Φ(k) = − G · pot_mag(k) / a.

### Rest frame
Compute `pot_mag(k)` for **every** member k (all types), pick `k* = argmax pot_mag(k)`
(deepest well). Rest-frame velocity `V_ref = P[k*].Vel`. (Computed self-consistently with
the same softened sum used for the bound test — see §10 note vs. `Group.PotMinPos`.)

### Bound criterion for unseeded star i
```
E_i = 0.5·|Vel_i − V_ref|² / a²  −  G·pot_mag(i) / a
```
Bound ⇔ E_i ≤ 0. Multiply through by a² to avoid a division:
```
bound(i)  ⇔  0.5·|Vel_i − V_ref|²  ≤  atime · G · pot_mag(i)
```
(Marginally-bound, E_i = 0, counts as bound — consistent with the `check_grav_bound`
convention in `blackhole.c:234`.)

### Outputs per qualifying group
```
M_bound_mGamma = Σ_{bound unseeded i} STARP(i).ClusterMass     (Σ m·Γ, bound only)
M_bound_starmass = Σ_{bound unseeded i} P[i].Mass             (Σ m_*, bound only)
```

---

## 4. Distributed algorithm

A secFOF group's member particles are spread across MPI ranks, so each rank cannot, on its
own, see all members of a group it owns. We mirror the proven Allgatherv pattern in
`secondfof_compute_sizes()` (`secondfof.c:222-411`), which gathers per-group member data
globally, keyed by group number, then has each rank process **only the groups it owns**.

> **Context note:** this runs inside `fof_seed` (called from `secondfof_seed`), where
> `fof_fof(ddecomp, 1, …)` has written the **secondary** group id into `P[i].GrNr`
> (and the primary id was saved into `SecGrNr`). So we key the gather on **`P[i].GrNr`**,
> matching `fof->Group[g].base.GrNr` — exactly as the multi-seed code does at `fof.c:1984`.
> (This differs from `compute_sizes`, which keys on `SecGrNr` because it runs in a different
> phase.)

### Step 0 — build the set of qualifying group numbers (optimization)
Each rank owns `fof->Group[g]` with the **global** per-group `StarClusterMassUnseeded`
already accumulated. Build a local list of `GrNr` where
`StarClusterMassUnseeded > thresh_code`, `MPI_Allgather`-v it into a global sorted set
`qual_grnr[]`. Only these groups need member gathering / O(N²) work. (Massive complexes are
rare, so this keeps both memory and compute bounded.)

### Step 1 — pack local members of qualifying groups
For every local particle `i` with `P[i].GrNr >= 0` **and** `P[i].GrNr ∈ qual_grnr`
(binary search), pack one record:
```
struct bound_member {
    int64_t GrNr;
    double  Pos[3];
    double  Vel[3];
    double  Mass;
    double  mGamma;          /* STARP(i).ClusterMass if unseeded star, else 0 */
    int     is_unseeded_star;/* 1 if Type==4 && !STARP(i).Seeded, else 0 */
};
```
ALL particle types are packed (potential uses *all* members); only unseeded stars carry
`mGamma>0` / `is_unseeded_star=1`.

### Step 2 — Allgatherv to all ranks
`MPI_Allgather` the per-rank counts, build byte counts/displs, `MPI_Allgatherv` the records
into `bm_global[]`. (Same MPI_BYTE idiom as `compute_sizes`.)

### Step 3 — sort + per-group processing (owning rank only)
`qsort(bm_global)` by `GrNr` so each group is a contiguous block. For each **local** group `g`
that qualifies, binary-search its `[start,end)` block in `bm_global`, then:
1. **Pass A (rest frame):** for each member k in the block compute `pot_mag(k)` over the
   other members (O(N²)); track `k*` = argmax. `V_ref = bm_global[k*].Vel`.
   - Reuse: store `pot_mag(k)` in a scratch array so unseeded stars don't recompute it.
2. **Pass B (bound sums):** for each member i with `is_unseeded_star`, test
   `0.5·|Vel_i − V_ref|² ≤ atime·G·pot_mag(i)`; if bound, add `mGamma` to `M_bound_mGamma`
   and `Mass` to `M_bound_starmass`.
3. **Overwrite in place** (see §5):
   `fof->Group[g].StarClusterMassUnseeded = M_bound_mGamma;`
   `fof->Group[g].SCcomMcut             = M_bound_starmass;`

### Step 4 — strict LIFO frees
Free every `mymalloc2` buffer in exact reverse allocation order (scratch `pot_mag`,
`bm_global`, displs/counts, local pack buffer, `qual_grnr`, …), following the discipline and
inline comments in `compute_sizes` (`secondfof.c:401-410`). **Every `myfree` will be traced
against the live stack before writing.**

### Complexity / memory
- Gather is bounded to qualifying (massive) groups only.
- O(N²) per qualifying group, N = members of that group. Massive complexes can be large;
  this is the dominant cost but only for the rare >1e8 M⊙ groups. (Future optimization:
  tree/Barnes-Hut potential — out of scope for v1; noted in §11.)

---

## 5. Where it plugs into the pipeline (overwrite-in-place)

The new routine — call it `fof_secfof_bound_massive_restrict(fof, atime, CP, Comm)` — is
invoked at the **top** of the existing `SeedSecFOFcomSample` serial block in `fof_seed`,
i.e. immediately *before* the loop at `fof.c:2357-2393`, and *before*
`fof_secfof_particle_sample` (which we disallow combining with anyway):

```c
if(fof_params.SeedSecFOFcomSample && fof_params.BlackHoleSeedStarCluster) {
    if(fof_params.SeedSeedFOFMassiveBoundStar)
        fof_secfof_bound_massive_restrict(fof, atime, CP, Comm);   /* NEW */
    if(fof_params.SeedSecFOFcomSampleParticle)
        fof_secfof_particle_sample(fof, rnd, Comm);
    for(i = 0; i < fof->Ngroups; i++) { ... }   /* unchanged */
}
```

**Why overwrite `StarClusterMassUnseeded` and `SCcomMcut` in place** rather than add new
fields and branch everywhere:
- `StarClusterMassUnseeded` is read as **Gate 1** (`fof.c:2367`) and as the sampler
  `sum_mGamma` (`fof.c:2373`).
- `SCcomMcut` is the sampler `Mcut` (`fof.c:2372-2374`).
- Downstream consumers — Gate 1, the combined sampler (`starcluster_combined_bhseed_msc`),
  `secfof_compute_nseed` (uses `BHSeedMsc`, `fof.c:1569`), and `fof_seed_make_one`
  (payload/init/scaling) — all flow from these two fields and from `BHSeedMsc` (which the
  sampler recomputes). Overwriting the two inputs makes the **entire** seed decision and seed
  **mass** use the bound subset automatically, including the `BHseedMassScaleMsc=1` seed-mass
  scaling the user asked for ("re-run sampler on bound stars"). No change needed in the
  sampler, `make_one`, or `compute_nseed`.

**Safety of overwriting** (verified): `StarClusterMassUnseeded` and `SCcomMcut` are used
**only inside `fof.c`** and are **not** written to the SecPIG catalogue (confirmed: no
references in `fofpetaio.c`, `secondfof.c`, or any IO block; the catalogue mass is the
separate `StarClusterMass`/`SCMass`). So in-place modification does not corrupt output.

**Seed location unchanged:** `seed_index_star` / `SeedStarID` stay as today (the seed is still
placed at the largest-m·Γ unseeded star). Only the seed **mass** and the seed **decision**
(via the bound-restricted `BHSeedMsc`) change. A massive group could in principle drop below
the seed gate if too few stars are bound — that is the intended physical behaviour.

---

## 6. New data structures

In `fof.c` (file-local, near the other secFOF helpers):
```c
struct bound_member { int64_t GrNr; double Pos[3]; double Vel[3];
                      double Mass; double mGamma; int is_unseeded_star; };
```
Plus the static helper `fof_secfof_bound_massive_restrict(...)` implementing §4.

No change to `struct Group` is required (overwrite-in-place reuses existing fields). No new
output blocks.

---

## 7. File-by-file changes

> All edits are **targeted**; existing files are `Read` immediately before editing (hand-edit
> safety per CLAUDE.md). Line numbers below are current-on-disk references, not literal.

### 7.1 `gadget/params.c` (after line 233, the `SeedInSecFOFMultipleSeeds` declaration)
Add:
```c
param_declare_int(ps, "SeedSeedFOFMassiveBoundStar", OPTIONAL, 0,
  "Only used with SeedSecFOFcomSample=1 (and NOT SeedSecFOFcomSampleParticle). If 1, a "
  "secondary-FOF group whose unseeded star-cluster mass Sum(m*Gamma) exceeds 1e8 Msun is "
  "restricted to the unseeded star particles gravitationally bound to the secFOF before "
  "seeding: the softened potential at each star is summed over ALL secFOF members, the rest "
  "frame is the deepest-potential member's velocity, and only the bound unseeded stars' "
  "Sum(m*Gamma) and stellar mass feed the combined sampler (and thus the seed mass when "
  "BHseedMassScaleMsc=1). If 0, the whole structure is used.");
```

### 7.2 `libgadget/fof.c`
- **`struct FOFParams`** (~line 69): add `int SeedSeedFOFMassiveBoundStar;`.
- **`set_fof_params`** (~line 103): add
  `fof_params.SeedSeedFOFMassiveBoundStar = param_get_int(ps, "SeedSeedFOFMassiveBoundStar");`.
- **`fof_seed` signature** (line 2273): add a trailing `Cosmology * CP` parameter (needed for
  `CP->GravInternal`). Update the call sites (§7.4, §7.5).
- **New static helper** `fof_secfof_bound_massive_restrict(FOFGroups * fof, double atime,
  Cosmology * CP, MPI_Comm Comm)` implementing §4.
- **Call site:** insert the guarded call at the top of the `SeedSecFOFcomSample` block
  (~line 2357), as shown in §5.
- **Include:** add `#include "cosmology.h"` (needed to dereference `CP->GravInternal`).
  *Verify at implementation time whether it is already transitively included via
  `fof.h`→`timestep.h`; add explicitly regardless for clarity.*

### 7.3 `libgadget/fof.h`
- Update the `fof_seed` prototype (lines 146-148) to add `Cosmology * CP`.
  `Cosmology` is already a known type here (used by `fof_save_groups`, line 152), so **no new
  include** is required in the header.

### 7.4 `libgadget/secondfof.c` / `secondfof.h`
- **`secondfof.h`** (line 17): add `Cosmology * CP` to the `secondfof_seed` prototype.
- **`secondfof_seed`** (line 811): add `Cosmology * CP` parameter; pass `CP` through to
  `fof_seed(...)` at line 856.
- **`set_secondfof_params`** (after the `SeedSecFOFcomSampleParticle` validation, ~line 86):
  add validation:
  ```c
  if(param_get_int(ps, "SeedSeedFOFMassiveBoundStar")) {
      if(!sfof_params.SeedSecFOFcomSample)
          endrun(1, "SeedSeedFOFMassiveBoundStar=1 requires SeedSecFOFcomSample=1.\n");
      if(sfof_params.SeedSecFOFcomSampleParticle)
          endrun(1, "SeedSeedFOFMassiveBoundStar=1 is incompatible with "
                    "SeedSecFOFcomSampleParticle=1 (v1 supports the combined sampler only).\n");
  }
  ```

### 7.5 `libgadget/run.c`
- **`secondfof_seed` call** (line 702): pass `&All.CP` →
  `secondfof_seed(ddecomp, &Act, &gasTree, atime, &rnd, &All.CP, MPI_COMM_WORLD);`
- **`fof_seed` call** (line 704): pass `&All.CP` →
  `fof_seed(&fof, &Act, &gasTree, atime, &rnd, NULL, NULL, NULL, NULL, &All.CP, MPI_COMM_WORLD);`
- **Param validation mirror** (the block at lines 204-217): add the same two checks as §7.4
  (require `SeedSecFOFcomSample`, forbid `SeedSecFOFcomSampleParticle`).

> `&All.CP` is confirmed to be a `Cosmology *` (used as such throughout `run.c`).

---

## 8. Control-flow summary (qualifying group, feature ON)

```
fof_seed (SeedSecFOFcomSample block)
  └─ fof_secfof_bound_massive_restrict           ← NEW
       gather all members of >1e8 groups (Allgatherv, keyed on GrNr)
       per owned qualifying group:
         pot_mag(k) ∀k  → V_ref = Vel[argmax]
         bound test ∀ unseeded star → M_bound_mGamma, M_bound_starmass
         StarClusterMassUnseeded ← M_bound_mGamma   (overwrite)
         SCcomMcut               ← M_bound_starmass (overwrite)
  └─ existing serial loop (unchanged code):
       Gate 1: StarClusterMassUnseeded (bound) ≥ MinMscForBHseed
       sampler: starcluster_combined_bhseed_msc(SCcomMcut_bound, StarClusterMassUnseeded_bound)
                → BHSeedMsc (bound)
       Gate 2: BHSeedMsc ≥ MinMscForBHseed → mark for seeding
  └─ make_one: scaling_mass = BHSeedMsc (bound) → seed mass = SeedBlackHoleMass·BHSeedMsc
```

---

## 9. Why v1 excludes `SeedSecFOFcomSampleParticle`

The per-particle variant computes `BHSeedMsc`/`SCcomMcut` via `fof_secfof_particle_sample`
**before** the serial loop and redistributes per-star shares; restricting that to bound stars
would require threading the bound mask into the per-particle pass and its redistribution.
Out of scope for v1; an `endrun` guard makes the unsupported combination an explicit, loud
error rather than silent wrong behaviour. (Can be lifted in a follow-up.)

---

## 10. Notable decisions / edge cases

- **Rest frame "potential-minimum particle":** computed self-consistently here as the member
  with the deepest **softened** potential (over all members), using its `Vel`. This is *not*
  necessarily identical to `Group.PotMinPos` (which is the **tree** potential min among
  **primary-link** particles and stores only a position, no velocity). Using the
  self-computed min keeps the test internally consistent and avoids adding a velocity field to
  `struct Group` + editing the FOF compile path. **Flag for user:** acceptable, or prefer to
  reuse `Group.PotMinPos`'s particle? (Would need a new gathered/stored velocity.)
- **Trigger uses Σ(m·Γ)** = `StarClusterMassUnseeded`, compared to
  `get_msc_multiseed_thresh_code()` (1e8 M⊙ in code units) — the same threshold helper the
  multi-seed feature uses, so "1e8 M⊙" is defined identically across features.
- **Zero bound stars:** `M_bound_mGamma = 0` → Gate 1 fails → group does not seed. Intended.
- **Interaction with `SeedInSecFOFMultipleSeeds`:** `secfof_compute_nseed` uses the
  bound-restricted `BHSeedMsc` (`fof.c:1569`), so N_seed is computed from bound mass —
  consistent, no extra work.
- **Self-pair excluded** (j≠k) in `pot_mag` to avoid the 1/eps self term.
- **Periodicity:** use `NEAREST(dx, BoxSize)` for separations (as in `compute_sizes`/multi-seed).

---

## 11. Performance considerations

- Gather + O(N²) restricted to >1e8 M⊙ groups only (rare).
- `pot_mag(k)` cached in a scratch array so Pass B does not recompute the potential.
- For very large complexes the O(N²) could dominate; a tree-based potential is the natural
  optimization but is deferred (correctness-first v1). Will be noted as a TODO in the code.

---

## 12. Compile & logging

- Compile with:
  ```
  module load intel/19.1.1
  module load gsl/2.6
  module load impi/19.0.9
  ```
- After a clean build, append a **brief, dated** feature summary (no code detail) to
  `plan/logging.md`, per CLAUDE.md.

---

## 13. Testing suggestions (post-implementation)

1. **Off-switch parity:** `SeedSeedFOFMassiveBoundStar=0` reproduces current results bit-for-bit
   (the new routine is never called).
2. **Sub-threshold parity:** with the feature on but no group above 1e8 M⊙, results unchanged.
3. **Validation errors:** confirm `endrun` fires for `SeedSecFOFcomSample=0` and for
   `SeedSecFOFcomSampleParticle=1` combinations (both in `set_secondfof_params` and `run.c`).
4. **Physics sanity:** in a run with a >1e8 M⊙ complex, log per-group N_total, N_unseeded,
   N_bound, M_bound_mGamma/StarClusterMassUnseeded(old) and confirm 0 ≤ bound ≤ total and that
   the bound fraction is physically reasonable.
5. **MPI invariance:** identical seeding decisions/masses on 1 vs many ranks (the gather makes
   the per-group computation rank-count-independent).

---

## 14. Resolved decisions (confirmed by user 2026-06-14)

1. **Rest frame** (§10): **self-consistent** softened-potential minimum computed in this
   routine (use that member's `Vel`). No change to `struct Group` or the FOF compile path.
2. **Trigger threshold:** **reuse** `get_msc_multiseed_thresh_code()` (1e8 M⊙ in code units);
   no new parameter — "1e8" is defined identically to the multi-seed feature.
3. **`SeedSecFOFcomSampleParticle`:** **hard-error** the combination in v1 (both in
   `set_secondfof_params` and the `run.c` validation mirror). May be lifted in a follow-up.
