# SeedSecFOFcomSample — Combined Per-secFOF Star-Cluster Sampling for BH Seeding

**Branch:** `SecFOFCombined`  (MP-Gadget-dev2)

**Status:** PLAN (no code changed yet)

---

## 1. Goal

Add a parameter flag `SeedSecFOFcomSample` (int, default `0`). When enabled it
changes how black holes are seeded inside the **secondary FOF** star-cluster
catalogue (`SeedInSecFOFasStarCluster` path):

Instead of summing per-star pre-sampled cluster masses, the seeding module draws
**one combined sample of the star-cluster population per secFOF group** at seed
time, using the group's *total stellar mass* as the mass-function cutoff. The BH
seed mass is set from the total mass of the *massive* (> 1e4 M⊙) sampled clusters.

This requires `SeedInSecFOFasStarCluster = 1`; otherwise the run aborts.

---

## 2. Definitions (per secFOF group)

For a secondary-FOF group, considering only its **type-4 star particles**:

- `m_star`  : star particle mass (code units)
- `Gamma`   : `STARP.ClusterFormationEfficiency` (CFE) of that star
- `ClusterMass = m_star * Gamma` (already stored as `STARP.ClusterMass`)
- **Σmγ** = Σ `ClusterMass` over the group's stars **with `Seeded == 0`**
  (this is `Group.StarClusterMass`, accumulation restricted to unseeded stars in
  this mode — see §6).
- **M_cut** = Σ `m_star` over the group's stars **with `Seeded == 0`** (new
  accumulator `Group.SCcomMcut`, see §6). *Used only as the exponential cutoff of
  the mass function.* (`MassType[4]`, which counts all stars, is left untouched
  for its other consumers.)

### Mass function (seed-mass PDF)
```
N(M) dM  ∝  M^-2 * exp(-M / M_cut) dM ,     M ∈ [1e2, 1e8] M⊙
```
This is the **same** functional form already used per-star in `sfr_eff.c`, but
with the cutoff set to the group's `M_cut` (instead of the per-star Toomre mass
`Mcstar`).

---

## 3. Seeding algorithm (per secFOF group)

1. **Cheap pre-filter (Gate 1):** if `Σmγ < MinMscForBHseed` → do nothing.
   (Faithful to the original spec; also bounds sampling cost to eligible groups,
   since `bhseed_msc ≤ total sampled mass ≈ Σmγ`.)
2. Compute `m_ave` = mean of `N(M)` over `[1e2,1e8]` with cutoff `M_cut`
   (analytic E1 formula — reuse `msc_ave_from_cutoff`, §5).
3. `n = Σmγ / m_ave`  (expected number of clusters).
4. Draw `N ~ Poisson(n)` (reuse existing Poisson sampler: normal approx for
   `n>30`, Knuth otherwise).
5. Sample `N` cluster masses from `N(M)` (inverse-CDF bisection, reuse existing).
6. `bhseed_msc` = Σ of those sampled masses that exceed `1e4 M⊙`.
   *(merger remnant of all first-generation seeds from massive clusters.)*
7. **Seeding criterion (Gate 2):** seed **iff** `bhseed_msc ≥ MinMscForBHseed`.
   Otherwise do nothing to this secFOF (stars stay `Seeded=0`; can seed on a
   later PM step once more cluster mass accumulates / a different draw lands).
8. If seeding:
   - Seed mass:
     - `BHseedMassScaleMsc = 1` → `BHP.Mass = SeedBlackHoleMass * bhseed_msc`
       (so `SeedBlackHoleMass` plays the role of the "0.01" fraction, tunable).
     - `BHseedMassScaleMsc = 0` → `BHP.Mass = SeedBlackHoleMass` (fixed), same as
       the original SC-based seeding.
   - Attached star-cluster mass (`BHP.StarClusterMass`):
     - `StarClusterBHDyn = 1` → `BHP.StarClusterMass = Σmγ` (the **full**
       cluster-forming mass of the consumed stars).
     - `StarClusterBHDyn = 0` → `BHP.StarClusterMass = 0` (no SC attached).
   - **Seed location:** spawn the new BH from the star with the **largest
     `ClusterMass = m_star*Gamma`** among the group's unseeded stars
     (`seed_index_star`, already tracked).
   - **Mark consumed:** set `Seeded = 1` for **all** type-4 stars in the seeded
     group (prevents double-counting their cluster mass in any later seeding).
     `ClusterMass` itself is **kept** (not zeroed) as a record.

---

## 4. New / changed parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `SeedSecFOFcomSample` | int | 0 | Master switch for this mode. |

**Validation (hard `endrun` on failure):**
- `SeedSecFOFcomSample = 1` requires the **effective** `SeedInSecFOFasStarCluster = 1`
  (i.e. after its own prerequisite checks: `SecondFOFOn`, `StarClusterOn`,
  `SecFOFStarCluster`). Checked in `set_secondfof_params` (where the effective
  value of `SeedInSecFOFasStarCluster` is resolved) and mirrored in `run.c` init.
- `SeedSecFOFcomSample = 1` requires `MinMscForBHseed > 0` (it is now the seeding
  criterion regardless of `BHseedMassScaleMsc`).

**Reused parameters:** `MinMscForBHseed`, `SeedBlackHoleMass`,
`BHseedMassScaleMsc`, `StarClusterBHDyn`, `MaxSeedBlackHoleMass`
(power-law seed mass should be off, i.e. `MaxSeedBlackHoleMass ≤ 0`, for a clean
`SeedBlackHoleMass * bhseed_msc`; if left on, the power-law base mass is scaled
by `bhseed_msc` — document, do not forbid).

---

## 5. New / refactored code in `sfr_eff.c` (+ `sfr_eff.h`)

`SeedSecFOFcomSample` read into `sfr_params`. New code-unit constant
`msc_seed_thresh_code = 1e4 * SOLAR_MASS / UnitMass_in_g` (the "massive cluster"
cutoff), initialised next to `msc_min_code`/`msc_max_code`.

**Refactor** the existing analytic average-mass formula (currently inline at
lines ~782–800) into a reusable static helper:
```c
static double msc_ave_from_cutoff(double Mcut_code);   /* <m> over [msc_min,msc_max] */
```
Used by both the per-star path and the new combined sampler.

**New exposed helper** (declared in `sfr_eff.h`, called from `fof.c`):
```c
/* Combined per-group star-cluster sampling.
 *  Mcut_code     : mass-function cutoff = group total stellar mass (code units)
 *  sum_mGamma    : Σ m_star*Gamma over unseeded stars (code units)
 *  rand_id       : RNG seed (seed star's particle ID)
 * Returns bhseed_msc = Σ of sampled cluster masses > 1e4 M⊙ (code units).
 * Returns 0 if Mcut<=0, m_ave<=0, or sum_mGamma<=0. */
double starcluster_combined_bhseed_msc(double Mcut_code, double sum_mGamma,
                                       uint64_t rand_id, const RandTable * rnd);
```
Internals: `m_ave = msc_ave_from_cutoff(Mcut)`, `lambda = sum_mGamma/m_ave`,
Poisson draw → `N`, then `N` inverse-CDF samples (cutoff = `Mcut`), accumulating
those `> msc_seed_thresh_code`. Reuses the existing `safe_expint_E1` and the same
bisection/Poisson code as the per-star sampler. RNG offsets reuse the per-star
convention (`+10,+11` Poisson; `+300+s` masses) — collision-free because the seed
star does **not** run per-star sampling in this mode.

**`make_particle_star` change:** skip per-star sampling when
`SeedSecFOFcomSample = 1` (CFE and `ClusterMass = m_star*CFE` are still computed;
`StarClusterMass_sample` and friends set to 0). Implement by widening the guard:
`if (StarClusterSampling && !SeedSecFOFcomSample) { …sample… } else { …zeros… }`.
Also initialise `STARP.Seeded = 0` at formation.

---

## 6. Changes in `fof.c` (+ `fof.h`)

### 6.1 `struct Group` — new fields (`fof.h`)
```c
MyFloat  SCcomMcut;   /* Σ m_star over UNSEEDED stars (mass-function cutoff M_cut) */
MyFloat  BHSeedMsc;   /* combined-sampled seed cluster mass (bhseed_msc), 0 default */
MyIDType SeedStarID;  /* ID of the max-ClusterMass unseeded star (RNG seed + record) */
```
Both travel via `MPI_TYPE_GROUP` (a contiguous byte copy of `struct Group` —
adding fields is automatically safe).

### 6.2 `add_particle_to_group` (group accumulation)
Read `SeedSecFOFcomSample` into `fof_params`. For type-4 stars, define
`contribute = !(fof_params.SeedSecFOFcomSample && STARP(index).Seeded)`:
- Accumulate `StarClusterMass`, `StarClusterMetallicity`,
  `StarClusterMetalElemMass[]`, and the new `SCcomMcut += m_star` **only when
  `contribute`** (so both `Σmγ` and `M_cut` exclude already-seeded stars in this
  mode; unchanged when the flag is off).
- `MaxStarClusterMass` / `seed_index_star` / `seed_task_star` tracking: in this
  mode use `ClusterMass` (not `StarClusterMass_sample`) and only consider
  `contribute` stars; also store `gdst->SeedStarID = STARP(index).ID` when the
  max updates.
- `MassType[4]`, `StellarMetalMass`, etc. remain over **all** stars (other
  consumers unaffected); only `SCcomMcut` carries the unseeded stellar mass.

### 6.3 `fof_reduce_group` (MPI reduction)
In the `MaxStarClusterMass` reduction block, also copy `SeedStarID` alongside
`seed_index_star`/`seed_task_star`. `BHSeedMsc` is computed post-reduction
(§6.4) so it does not need a reduce rule (stays 0 until then).

### 6.4 `fof_seed` — marking
The existing parallel loop computes `Gas_Mask`/`Halo_Mask` and the non-com
`SC_Mask`. For the **com** mode the sampler calls `safe_expint_E1`, which toggles
the **global** GSL error handler and is therefore **not OpenMP-safe**. So:

- In the parallel loop, for com mode set only the cheap **Gate-1** candidate flag
  `cand = (Group.StarClusterMass >= MinMscForBHseed) && (seed_index_star >= 0)`.
- Add a **serial** pass over candidates:
  ```c
  bhseed_msc = starcluster_combined_bhseed_msc(Group.SCcomMcut,
                  Group.StarClusterMass, Group.SeedStarID, rnd);
  Group.BHSeedMsc = bhseed_msc;
  SC_Mask = (bhseed_msc >= MinMscForBHseed);   /* Gate 2 */
  ```
- For groups that pass Gate 2, set `seed_index = seed_index_star` /
  `seed_task = seed_task_star` **unconditionally**. Do NOT reuse the non-com
  "only if seed_index < 0" fallback: if gas is a secondary link type the group
  can carry a densest-gas `seed_index >= 0`, which must never be used here (it
  would convert gas in place instead of spawning a BH from the largest-`m·Γ`
  star). The seed location in this mode is always the star.

### 6.5 `fof_seed_make_one`
For com mode:
- `scaling_mass = g->BHSeedMsc`
- `payload_mass = (StarClusterBHDyn ? g->StarClusterMass : 0)`
- `payload_metallicity / metals` from `g->StarClusterMetallicity / g->StarClusterMass`
- `seeded_by_starcluster = 1` (Gate 2 already passed)
- call `blackhole_make_one(index, …, seeded_by_starcluster, payload_mass,
  scaling_mass, payload_metallicity, payload_metals)`  (see §7).

For non-com modes the call is unchanged (`scaling_mass == payload_mass == sc_mass`).

---

## 7. Change in `blackhole.c` (+ `blackhole.h`)

`blackhole_make_one` currently uses one `StarClusterMass` argument for **both**
the seed-mass scaling and the attached SC payload. Decouple by adding a separate
scaling argument:
```c
void blackhole_make_one(int index, double atime, const RandTable * rnd,
                        int seeded_by_starcluster,
                        MyFloat StarClusterMass,    /* payload → BHP.StarClusterMass */
                        MyFloat ScalingMass,        /* NEW: used by BHseedMassScaleMsc */
                        MyFloat StarClusterMetallicity, const float * StarClusterMetals);
```
- Scaling line becomes `if (seeded_by_starcluster && BHseedMassScaleMsc) BHP.Mass *= ScalingMass;`
- Payload logic (`BHP.StarClusterMass = StarClusterMass`, formation time,
  metallicity) is unchanged — it now receives the (possibly 0) payload.
- **Existing callers** (`fof_seed_make_one` non-com path, `blackhole_seed_sc_particle`)
  pass `ScalingMass = StarClusterMass` → identical behaviour, no regression.

This is the only `blackhole.c` change for the seeding logic; the
spawn-from-star, Mtrack, and `P.Mass = max(Mtrack[+SC], SeedBHDynMass)` paths are
reused unchanged.

---

## 8. Change in `secondfof.c`

- Read `SeedSecFOFcomSample`; add the §4 validation (`endrun` if
  `SeedSecFOFcomSample && !SeedInSecFOFasStarCluster`, and if `MinMscForBHseed<=0`).
- In `secondfof_seed`, the post-seed "consume" step currently zeroes
  `ClusterMass`/`StarClusterMass_sample` for stars in seeded groups. When
  `SeedSecFOFcomSample = 1`, instead set `STARP(i).Seeded = 1` for those stars
  (keep `ClusterMass`). The seeded-group set is already gathered via Allgatherv +
  binary search — only the per-star action changes.

---

## 9. Star property + IO (`slotsmanager.h`, `petaio.c`)

- `slotsmanager.h`: add `int Seeded;` to `struct star_particle_data`.
- `petaio.c`:
  - `SIMPLE_PROPERTY_PI(Seeded, Seeded, int, 1, struct star_particle_data)`
  - `IO_REG_NONFATAL(Seeded, "i4", 1, 4, IOTable);`  (so it persists across restarts)
  - pre-zero in the snapshot-read init block (next to `initClusterMass`):
    `STARP(i).Seeded = 0;`  (old snapshots lacking the block default to 0)

---

## 10. `gadget/params.c`

`param_declare_int(ps, "SeedSecFOFcomSample", OPTIONAL, 0, "If 1, seed BHs in the
secondary FOF by one combined star-cluster sampling per group (cutoff = group
total stellar mass); seed mass from clusters > 1e4 Msun. Requires
SeedInSecFOFasStarCluster=1.");`

---

## 11. File-by-file summary

| File | Change |
|---|---|
| `gadget/params.c` | declare `SeedSecFOFcomSample` |
| `libgadget/sfr_eff.c` | read flag; `msc_seed_thresh_code`; refactor `msc_ave_from_cutoff`; new `starcluster_combined_bhseed_msc`; skip per-star sampling in this mode; init `Seeded=0` |
| `libgadget/sfr_eff.h` | declare `starcluster_combined_bhseed_msc` |
| `libgadget/fof.h` | `struct Group`: `SCcomMcut`, `BHSeedMsc`, `SeedStarID`; read flag into params |
| `libgadget/fof.c` | `#include "sfr_eff.h"`; read flag; unseeded-restricted accumulation + `SeedStarID`; reduce `SeedStarID`; serial com sampling + Gate 1/2 in `fof_seed`; decoupled call in `fof_seed_make_one` |
| `libgadget/blackhole.c` | add `ScalingMass` arg to `blackhole_make_one`; update internal scaling; update `blackhole_seed_sc_particle` caller |
| `libgadget/blackhole.h` | update `blackhole_make_one` prototype |
| `libgadget/secondfof.c` | read flag; validation; set `Seeded=1` instead of zeroing in this mode |
| `libgadget/run.c` | mirror validation at init |
| `libgadget/slotsmanager.h` | `int Seeded` on star slot |
| `libgadget/petaio.c` | register + pre-zero `Seeded` |

---

## 12. Open assumptions / notes for review

1. **M_cut and Σmγ are both over UNSEEDED stars** (`Seeded==0`), via the new
   `Group.SCcomMcut` and the restricted `Group.StarClusterMass` accumulation.
   (Confirmed 2026-05-30.)
2. **Per-PM-step re-draw:** a group that passes Gate 1 but fails Gate 2 keeps its
   stars unseeded and is re-sampled next PM step. The per-step seeding
   probability therefore depends on PM cadence (a known feature of "sample at
   seed time" vs the per-star "sample at birth" model). With `MinMscForBHseed`
   large enough that `n ≫ 1`, groups seed on the first eligible step, making this
   negligible. Flagged as a modelling caveat — see §13.
3. **GSL thread-safety:** the combined sampler must run serially (it toggles the
   global GSL error handler). Done via the serial candidate pass in `fof_seed`.
4. **Gate-1 pre-filter** can, in rare upward Poisson fluctuations, exclude a
   group whose `bhseed_msc` would have reached `MinMscForBHseed` while
   `Σmγ < MinMscForBHseed`. Considered acceptable (matches the original spec's
   Σmγ gate and bounds cost). Removable if undesired.

---

## 13. Suggestions for the model (future work — NOT in v1)

v1 implements the **simple stochastic per-PM-step draw** as specified. The notes
below are recorded for a possible later refinement and are not implemented now.

- **Analytic mean/variance of `bhseed_msc` (closed form).** Because `bhseed_msc`
  is a compound-Poisson sum restricted to `M > M_th = 1e4 M⊙`, both its mean and
  variance reduce to `E1`:
  ```
  E[bhseed_msc]   = Σmγ · [E1(M_th/M_cut) − E1(M_max/M_cut)] / [E1(M_min/M_cut) − E1(M_max/M_cut)]
  Var[bhseed_msc] = Σmγ · M_cut · (e^{−M_th/M_cut} − e^{−M_max/M_cut}) / [E1(M_min/M_cut) − E1(M_max/M_cut)]
  ```
  (`M_min=1e2`, `M_max=1e8`.) Useful for a deterministic threshold, sanity checks,
  or a moment-matched draw.

- **Cadence-independence with scatter — the right method.** The cadence
  dependence comes from giving the *same* stellar population a fresh random draw
  on every PM step. The robust fix is to **freeze each star's randomness at its
  birth** (derive its uniforms deterministically from its own permanent particle
  ID), and at seed time rebuild the group's clusters from its member stars'
  frozen uniforms using the group cutoff `M_cut`. Adding stars only adds clusters,
  and a rising `M_cut` only pushes masses up, so `bhseed_msc` rises monotonically
  → unique threshold crossing → cadence-stable, while remaining a genuine random
  sample (full scatter). By the Poisson superposition property (sum of
  independent Poissons is Poisson), this per-star construction yields the **exact
  same model** as the single per-group draw (same `n = Σmγ/m_ave`, same mass
  distribution). This avoids the (discarded) "stable group anchor" idea, which
  fails because star IDs are inherited from parent gas (spatial, *not*
  time-ordered — `slotsmanager.c:117`) and secFOF membership shifts every step.

- **Cost note:** for very large `n` the O(N) bisection sampling could be reduced
  via the analytic moments above; kept stochastic in v1 per spec.

---

## 14. Logging

On implementation, append a dated summary to `plan/logging.md` (feature-level,
no code detail), per project convention.
