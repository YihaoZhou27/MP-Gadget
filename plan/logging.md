# MP-Gadget Development Log

## 2026-08-18 (later) — `SecondFOFLinkingLength` now in units of the mean DM particle separation

**Breaking parameter-file change.** `SecondFOFLinkingLength` used to be an absolute comoving length in code units (ckpc/h); it is now a dimensionless multiple of the mean DM interparticle separation `BoxSize / NTotalInit[1]^(1/3)`, exactly the convention `FOFHaloLinkingLength` already uses. The conversion happens once at startup in a new `secondfof_init()` called next to `fof_init()`, and the derived comoving length is what the FOF engine receives; a startup message prints parameter, mean separation and resulting comoving length. Default raised from 0.01 to 0.1.

Every existing parameter file must be converted: divide the old value by the mean DM separation of that run. For the 12.5 Mpc/h, 512^3 paper runs the separation is 24.4140625 ckpc/h, so the production `SecondFOFLinkingLength = 2.44` becomes `0.1` (and 1.22 -> 0.05) — the same physical linking as before. Runs left at an unconverted absolute value would link on a scale ~24x too large.

Catalogue headers are unchanged for existing analysis: the `SecondFOFLinkingLength` attribute in `SecPIG` still holds the comoving code-unit length, with the new attribute `SecondFOFLinkingLengthMeanSep` recording the dimensionless parameter value alongside it.

Compiles clean. Not committed.

## 2026-08-18 — Cluster formation efficiency capped at Gamma <= 1

`get_cluster_formation_efficiency()` (Kruijssen 2012 P/k_B -> CFE table, log-log linear with extrapolation) now caps its return value at 1. Beyond the last tabulated point (P/k_B = 6.3e9 K cm^-3, CFE = 0.952) the extrapolated last segment kept rising and gave Gamma > 1 for birth pressures above 2.2e10 K cm^-3 — in the SC-norecoil paper run ~0.3% of all star particles at z=6 (up to Gamma = 1.12), all born from the densest nuclear gas at z < 8.3. Gamma is a mass fraction, so it is now clamped; the change touches every user of the function (star `ClusterFormationEfficiency` / `ClusterMass` at formation and the gas-side CFE used for `SumSFRdtCFE`). Effect on the star-cluster mass budget is < 0.1%; runs whose gas never exceeds that pressure are bit-for-bit unchanged.

Compiles clean. Not committed.

## 2026-08-16 (later) — `SecFOFseedHostZcrit`: per-cluster critical-metallicity mask on the seed host star

New optional parameter `SecFOFseedHostZcrit` (0/1, default 0). When 1, in the per-cluster CW-model seeding path the star particle that hosts each ordinary BH seed must have frozen BirthMetallicity at or below the cluster's own critical metallicity `Z_crit = Z_cl (M_VMS / SeedBlackHoleMass)^(2.1/0.74)` — the metallicity at which that same cluster (mass, radius, age fixed) would just have reached the seed-mass floor, from the Vink-wind equilibrium behind the CW model (`M_VMS ~ Z^-0.352` at fixed cluster). Among the passing stars the existing ranking decides (raw `Gamma*m_star`, or the `SecFOFseedHostZBeta`/`SecFOFseedHostZFloor` Z-weighted key — the two features combine, mask first then Z-weighted ranking among the survivors — or the random key), and if no unseeded star of the group passes, the lowest-metallicity unseeded star hosts the seed (fallback). Motivated by the post-hoc host-rule notebook: the Z-weighted ranking alone puts seeds in the metal-poor outskirts, while the per-seed Z_crit mask brings them back to ~R50 with the host consistent with its own cluster's gate.

Mechanics worth knowing: the exponents 2.1/0.74 now live in `cwmodel.h` (`CW_WIND_MEXP`, `CW_WIND_ZEXP`, used by the model's own equilibrium, so the two cannot drift apart) together with the helper `cw_host_zcrit_massfrac()`. With the mask on, the per-cluster host gather keeps EVERY unseeded (bound) star of a requesting group instead of the top n_request by key, since the masked pick can lie below the ranking cut (in runs with a tiny MinMscForBHseed the gather already held every star, so nothing changes there; with the default MinMscForBHseed it grows to the requesting groups' unseeded-star count). Density-capped clusters and clusters with M_VMS capped at the cluster mass use the same formula on the capped M_VMS, which only lowers Z_crit. MinBHSeedInSC compensating seeds carry no per-cluster M_VMS and are not masked. Requires the per-cluster mode with `MbhMscRelationCWmodel=1` (an error otherwise). MASK ONLY: budgets, cluster draws, gates and seed masses are untouched; the group reference star (SeedStarID / RNG seed) is unchanged. Whenever `SecFOFseedHostZcrit=1` or `SecFOFseedHostZBeta>0` the startup log now prints one message describing the full host-search criterion (ranking key, floor, mask, fallback), and each seeding pass reports how many placed seeds the mask moved off the largest-key star and how many fell back.

Compiles clean; `test_fof` passes; default-off runs are bit-for-bit unaffected. Not committed.

## 2026-08-16 — `SecFOFseedHostZFloor`: metallicity floor for the Z-weighted host ranking

New optional parameter `SecFOFseedHostZFloor` (default 0, in units of solar metallicity with Zsun = 0.0134, the CW seed-mass model's value). When > 0 together with `SecFOFseedHostZBeta` > 0, the host-star ranking key becomes `Gamma*m_star * max(Z, Zfloor)^-beta`: every unseeded star below the floor gets the same weight, so the raw `Gamma*m_star` decides among the sub-floor stars and only stars above the floor are penalised for their metals. When <= 0 only the pristine 10^-7 floor applies and the ranking is unchanged from the existing `SecFOFseedHostZBeta` behaviour; the parameter is inert (with a message) when `SecFOFseedHostZBeta` <= 0. Implemented at the single ranking-weight chokepoint, so all host-selection sites pick it up; ranking only, budgets/draws/gates/seed masses untouched.

Motivation from the post-hoc host-rule notebook: without a floor the Z^-beta key is dominated by the rare very-metal-poor outskirt stars (Gamma*m_star varies by ~10% across the galaxy, Z by 2-3 dex in its tail), so seeds land at 2-3 R50 even at z ~ 11-13; a ~0.1 Zsun floor keeps them central while the galaxy is metal-poor and lets them drift outward as the local Z rises above the floor.

Compiles clean; default-off runs are unaffected. Not committed.

## 2026-08-13 (late) — `SecFOFseedHostZBeta`: Z-dependent host-star ranking for BH seeding

New optional parameter `SecFOFseedHostZBeta` (default 0). When > 0, every *ranked* choice of which unseeded star particle converts into a BH seed uses the proxy `Gamma*m_star * Z^-beta` (Z = the star's frozen birth metallicity, pristine stars floored at the metallicity histogram's floor) instead of the raw `Gamma*m_star`, so metal-poor stars are preferred as seed hosts at fixed cluster-forming mass. When <= 0 the ranking — and the whole run — is bit-for-bit the historical behavior.

The weight covers all five host-selection sites: the per-group seed-1 argmax, the per-cluster ordinary and compensating host orderings, the multi-seed extra hosts, and the bound-host repointing of both bound-restriction passes. It is **ranking only**: seeding budgets, cluster draws, gates and seed masses are untouched (unlike the deleted f(Z) budget factor below, which this feature replaces in spirit but not in mechanism). One knock-on to be aware of: the group's reference star / RNG seed moves with the ranking, so per-group cluster-draw realizations differ from a raw-ranked run (same statistics). Motivated by the notebook diagnostics: the CW-model seeding gate scales as m_crit ~ Z^beta with beta ≈ 0.27–0.38, flat over the whole threshold range, and the raw sort picks hosts ~0.4 dex *above* their group's mean log Z while packing multi-seed BHs ~10x closer together than the Z-weighted proxy would.

Compiles clean; default-off runs are unaffected.

## 2026-08-13 — Removed the metallicity-dependent seeding factor f(Z)

The star-cluster BH-seeding budget is no longer weighted by star metallicity. The per-star cluster-forming mass `Gamma*m_star` now enters every seeding path raw: the group budgets, the per-star Poisson cluster count, the host-star ranking and the host-eligibility test. Metal-rich stars are neither down-weighted nor barred from hosting a seed.

`StarClusterSeedMetallicityMin` and `StarClusterSeedMetallicityMax` are **deleted**, not defaulted off — a paramfile that still lists either one will abort at startup with "Parameter is unknown", so those lines must be removed from existing paramfiles before running this build.

Metallicity still enters the seed **mass** through the CW model's own Vink-wind Z scaling (`MbhMscRelationCWmodel`); only the weighting of the seeding budget is gone. Since f(Z) was off by default — and was force-disabled whenever the CW model was on — runs configured that way are bitwise unaffected. Full libgadget test suite passes.

## 2026-08-08 — Grid primary linker for the second FOF (`SecondFOFGridLinking`)

The star FOF was 62% of the run's wall clock and got no faster with more nodes: ~90% of stars land on one rank because the domain is balanced by particle count, and the treewalk enumerates every neighbour pair when the median star has 1.6e5 neighbours inside the linking length. New alternative primary linker for the **second FOF only** — the halo FOF is untouched and still uses the treewalk.

FOF groups are just the connected components of the "within l" graph, so a spanning subgraph is enough. Particles are binned into cells of side `a <= l/sqrt(3)`, whose body diagonal is therefore at most `l`: every pair inside a cell is linked with no distance test, so the cell collapses to one union-find node. Only nearby cell pairs are probed, each stopping at its first hit, and pairs whose roots already agree are skipped outright — which is where most of the saving comes from once a component has grown. The primary set is small enough (~1e5–1e6 stars) to gather to every rank and process identically, so the convergence loop and the imbalance both disappear.

**Same groups, not similar ones.** Validated offline against a brute-force O(N^2) FOF on live `PART` snapshots at three linking lengths plus edge cases, and in-code by five new `test_fof_grid*` cases covering DM and star primaries, secondary attachment, seeded-star exclusion, garbage/swallowed fallback, empty ranks, the periodic wrap, exact linking-length boundaries and the stencil corners. Two details are load-bearing and easy to get backwards: the cell size must be rounded **down** (rounding up lets the diagonal exceed `l` and merges particles the treewalk leaves apart), and the stencil bound must be **inclusive**, since the neighbour search accepts `r2 == l^2` — which is what keeps the (±2,±2,±2) corner cells in the stencil.

**The traversal order is worth a factor of ~4000 and is not free to change.** The stencil must be walked **offset-major** — every cell at the nearest offset, then every cell at the next — so the dense component forms immediately and all later pairs collapse to O(1) root comparisons. The natural cell-major nesting costs 1.82e8 distance evaluations against 4.41e4 on the same z=8.75 data, because it probes far offsets before any component exists and those probes fail at full `N_A × N_B` cost. The answer is identical either way; only the work changes. The loop carries a comment with the measurement so it is not "tidied" back.

Selection is by call context rather than by the type mask, set transiently around the second FOF's own `fof_fof` calls. A collective preflight falls back to the treewalk on its own when the grid path cannot be proven equivalent or affordable: any garbage/swallowed primary particle present (the treewalk excludes those asymmetrically — as initiating targets but not as passive neighbours — and that is not worth reproducing in a symmetric graph), MPI `int` count limits, or insufficient free memory. Every fallback logs its reason.

**`SecondFOFGridLinking` defaults to 0 (treewalk), so this change is inert until deliberately enabled.** 1 selects the grid linker, 2 runs both and aborts on any mismatch. The gate to flipping the default is one `paper_runs` restart at mode 2 covering a seeding step, with the resulting SecPIG catalogue **and seeded particle IDs** compared against a treewalk run — the seeding link is the one step no test has yet executed, and code inspection alone is not enough to promote it. Criteria are written down in `plan/secfof_grid_linking_plan.md` Section 13.3.

Measured on the unit test: 0.42 s against 24.3 s for the treewalk on the same data, labels identical at 1, 2 and 4 ranks; full libgadget suite passes. Note the replicated design is a large constant-factor win, **not** strong scaling — per-rank cost is independent of rank count — which is what the memory gate is there to bound. Rationale, measurements and validation are in `plan/secfof_grid_linking_plan.md` and `plan/secfof_scaling_report.md`.

## 2026-08-07 — `SC_MET_HIST_LOGMIN`: −6 evaluated and reverted, stays at −7

Raising the metallicity histogram's lower bound would improve bin resolution from 0.129 to 0.113 dex. That range stopped being cosmetic once `CWmodelMetallicity = 'starsample'` began drawing per-cluster metallicities off the same histogram, so it was measured against paper_runs rather than assumed. Per-group error on ⟨Z^−0.352⟩ (the moment that sets M_VMS) over the groups that actually seed: the **median is unchanged** (0.0052 at −6 vs 0.0054 at −7), but the **tail degrades sharply** — max error 0.44 against 0.027 — because the underflow bin has no width to interpolate across and returns the group's exact minimum, and 0.69% of stars land there instead of 0.045%. The groups this hurts are the metal-poor ones that make the massive seeds. −5.0 is far worse still: 4.9% underflow, median error 0.090, max 3.34.

Reverted; the value is unchanged from before this session. The measured trade-off table now sits in `fof.h` beside the define so the experiment isn't repeated. Note the constant is shared: it also clamps the log-sums behind `met_logmean`/`met_logstd` and the `[min,max]` clipping bounds, so changing it moves `lognormal` and `uniform` too, not just `starsample`. The clean way to make the floor nearly free is to have the underflow bin interpolate between `zmin` and `10^LOGMIN` instead of returning `zmin` flat — not done.

## 2026-08-07 — Per-cluster seeding: test every qualifying cluster, spend a host star only on a seed

The per-cluster secFOF seeding pass used to cap the number of clusters it would even *look at* to the group's unseeded-star count, and then pair cluster `s` with host star `s` positionally. Because the pairing advanced whether or not the cluster seeded anything, a cluster rejected by the `M_VMS < SeedBlackHoleMass` gate still consumed a host star it never converted. In this run **99% of tested clusters fail that gate** (66,492 NOVMS vs 691 SEEDED), so a star-limited group spent almost its entire budget on rejections while qualifying clusters behind them were never tested at all.

Now the star supply limits only what it physically limits — how many star particles can be turned into BHs. Three coupled changes:

1. **The cluster-count cap is gone.** The draw keeps every cluster at or above `MinMscForBHseed`. The mass buffer is sized exactly by a count-only pre-pass of the same sampler (documented to walk an identical RNG sequence, and it allocates nothing), so this is an exact size and not an estimate. Costs one extra ICMF draw per owned group — `SCdraw` is 0.19 s over a whole run next to 1254 s in the FOF walk.
2. **A host cursor replaces the positional pairing.** It advances only when a BH is actually placed, so a rejected cluster costs nothing. The loop is bounded by the cluster count alone: running out of host stars stops the *seeding*, not the *modelling*, so every qualifying cluster still gets its M_VMS computed, recorded, and counted towards the MinBHSeedInSC missed mass. A cluster that clears `SeedBlackHoleMass` with no star left is simply lost — no BH and no detail row, since that would need a fifth `SC_FLAG` value and a schema change for every analysis script; it is reported at group level by the shortage message instead, which now counts *unhosted* clusters rather than the old cluster-count cap.
3. **The per-cluster radius and metallicity draws are re-keyed.** They were keyed on the paired host star, which is circular once the host is chosen only after `M_VMS` is known. They now key on `SeedStarID` mixed with the cluster's draw index — the same construction the sub-threshold pass uses — XORed with a stream tag, because for `SeedInSecFOFRandomStarParticle=0` the reference star *is* the SeedStarID star and the two keyspaces would otherwise coincide and correlate two independent cluster populations.

**Behaviour changes to expect.** Seed counts are unchanged wherever the host supply was never binding, which is every group in the current run (the shortage message never fired). The re-keying does shift results everywhere: per-cluster radii and metallicities are redrawn, statistically identical but individually different, in all `CWmodelMetallicity` modes. Rejected clusters no longer carry a distinct host in `StarClusterDetails` — like the sub-threshold records, Flag-1 (NOVMS) rows now carry the group's reference star as ID/Pos, so that column no longer identifies a unique host for them. The BH-slot reservation is untouched and still correct, since placements remain bounded by the host count. Builds clean; full libgadget suite passes.

## 2026-08-07 — `CWmodelMetallicity = "starsample"`: per-cluster Z from the empirical stellar MDF

New fourth mode for the per-cluster metallicity fed to the CW seed-mass model. Each sampled cluster takes the metallicity of a uniformly-random **unseeded star of its host secFOF**, drawn by inverse CDF off the group's existing log10(Z) histogram — i.e. a sample of the group's *empirical* metal distribution rather than of a fitted shape. The other three modes (`ave`, `lognormal`, `uniform`) are unchanged, and `ave` remains the default.

**Why.** `lognormal` matches the bulk of the stellar MDF but overestimates the high-Z end and underestimates the low-Z end, and the low-Z end is what makes massive seeds (`M_VMS ~ (Z/Zsun)^-0.35`). Sampling the histogram keeps the real metal-poor tail with no distributional assumption.

**Two points that decide the implementation.** The draw is taken from the per-group histogram, which spans *all* unseeded stars, and deliberately **not** from the gathered host-star candidate list, which is pre-truncated to the top `n_request` by `f(Z)*ClusterMass` and would therefore be metallicity-biased in any group holding more unseeded stars than requested clusters. And sampling is **with replacement**, which is simply what drawing from an MDF means: this star supplies a metallicity only, and has nothing to do with the star later converted into the BH particle, so reuse is correct rather than merely tolerable.

Resolution is one histogram bin (0.129 dex); the underflow and overflow bins return the group's exact min and max. Because the metallicity floor was removed on 2026-08-06, a group whose stars are all pristine now draws Z = 0 and gets the model's zero-wind limit (M_VMS = cluster mass) — not reachable in the current runs, whose minimum stellar Z is 3.5e-6 Zsun, but worth knowing. Draws stay keyed on the host star ID, so they remain reproducible across ranks and restarts. Histogram cost is one float array per eligible group (groups number in the hundreds). Builds clean; full libgadget suite passes.

## 2026-08-06 — Chabrier IMF fix, CW model f_IMF, and removal of the CW metallicity floor

Three related changes, all touching how the IMF and the metallicity enter the seed model.

**1. `metal_return.c` Chabrier IMF: natural log -> log10.** The low-mass lognormal branch read `exp(-(log(m/0.079)/0.69)^2/2)`, but Chabrier (2003) defines the width sigma = 0.69 in log10. The pair of normalisations already in the function is the proof: at m = 1 Msun the two branches agree to 1.0000 under log10 and differ by a factor 242 under natural log. The old form made the lognormal 2.303x too narrow, collapsing the [0.3, 1] Msun mass fraction from 0.262 to 0.023.

Every dying-star window lies above 1 Msun, where both versions share the same `m^-2.3` power law, so this is a **pure normalisation correction**: `compute_imf_norm` over [MINMASS, MAXMASS] = [0.1, 40] rises 0.624632 -> 0.936977 and **every mass and metal yield per unit stellar mass formed drops by exactly a factor 1.500**. No yield changes shape. The total returned mass fraction in the unit test moves 0.63 -> 0.42, which is where a Chabrier SSP should sit. SNIa are untouched (fixed `N0 = 1.3e-3` per Msun, not IMF-derived).

`tests/test_metal_return.c` pinned the old buggy value (0.624632); updated to 0.936977. That the test's hard-coded number matched the buggy normalisation to six digits is itself confirmation of the diagnosis.

This does **not** change the SN/wind feedback energy: `EgySpecSN`, `FactorSN`, `WindEnergyFraction` and the effective EOS are paramfile quantities and reference no IMF. It reaches feedback only indirectly, through metal-line cooling.

**2. CW model now uses MP-Gadget's own IMF.** `CW_FIMF` 0.0649 -> **0.0969**: the mass fraction in [1, 1.5] Msun for the (now corrected) Chabrier normalised over [0.1, 40], replacing modelv2.py's Salpeter (alpha = 2.35, 0.1-100 Msun) value. The seed model and the metal return now assume the same IMF.

Note f_IMF is **not** a rescaling of M_VMS: it multiplies `cw_mdot_df_anti` and `cw_mdot_dep_anti` but not `cw_mdot_bin_anti`, so `Mdot_in = (1-f_vms)[f_IMF (A_df - A_dep) - A_bin]` is affine in it. Clusters that only marginally beat binary heating gain far more than the naive `(0.0969/0.0649)^(1/2.1) = 1.21x`; the measured per-cluster median shift is 1.43x. The constant is tied to both `chabrier_imf` and `[MINMASS, MAXMASS]` — recompute it if either changes.

**3. `CW_ZRATIO_FLOOR = 1e-4` removed.** The floor was never part of modelv2.py, which runs at a single fixed Z = 0.1 solar and so never meets a low metallicity; it existed only to stop `C -> 0` diverging once per-cluster simulation Z was fed in. It was load-bearing in the wrong way: 25.5% of seeds were being evaluated at the floor rather than at their own Z, while only 0.04% of actual star particles in the star FOF are below 1e-4 Zsun (min over `SecPIG_021` is Z/Zsun = 3.50e-6, and nothing anywhere in the run is below 1e-6 Zsun or pristine).

Metallicity is now used as given. The one case still handled explicitly is Z <= 0: the wind vanishes, the equilibrium VMS mass is unbounded, and the model's own limit is that the VMS consumes the cluster — so `M_msun` is returned directly rather than letting an infinity propagate to the `cw_seed_mass_code` cap. Same answer, finite.

Effect at fixed cluster (3e5 Msun, r_max = 0.7 pc): Z = 0.1 Zsun 2155 -> 2611 Msun (f_IMF only); at the run's actual minimum Z the seed goes 2.46e4 -> 9.71e4 Msun, since both the floor removal (x3.25) and f_IMF now apply.

Verified: full `libgadget` test suite passes (`test_mpsort` needed a rebuild — its binary dated 2026-08-01 was linked against a `libgsl.so.25` that no longer exists on the system, unrelated to these changes). Built with `gsl/2.8` via explicit `GSL_INCL`/`GSL_LIBS`, since the tree's pkg-config lookup finds no gsl.

## 2026-08-03 — SecPIG: bound-star R50 / R90 / Rmax

Three new SecPIG group blocks — `SecR50Bound`, `SecR90Bound`, `SecRmaxBound` — over the member stars the `BHseedSecFOFbound` pass flagged (`STARP.Bounded == 1`), about the same `SecPotMinPos` centre as `SecR50`/`R90`/`Rmax`, so bound/total is a ratio of like for like. Identically 0 when `BHseedSecFOFbound = 0` or a group has no bound star. Registered under `ComputeSize && SecFOFStarCluster`, so the catalogue schema does not depend on the bound switch.

The one difference from the totals that remains is the population: these are stars-only by construction, since boundness is only defined for stars, while `SecR50` covers every primary-linked type. Under `SecondFOFPrimaryLinkTypes = 17` the totals are gas+star half-mass radii, not stellar — worth keeping in mind when comparing across the `PriStar` (17) and `boundSC` (16) runs.

Sharing the centre keeps this nearly free. `struct dist_mass_grp` gains a `bound` flag (24 -> 32 B) and the existing per-group scan accumulates a second cumulative sum: because every entry of a group is measured from one centre, the bound entries are already in ascending distance order inside the sorted block, so no second gather and no second sort are needed. The alternative that was built first — centring the bound radii on the bound-star COM — forced a second gather and sort, since the bound stars order differently about a different centre; it was measured at ~2x this function and dropped in favour of the shared centre.

Measured baseline for that decision (`output_l0.1_BH5e3_bound_seededfix`, `cpu.txt-R013` + `R020`): `/SecondFOF/Compute` is **4.08 s of a 12,023 s run (0.034%)**, 0.51 s per snapshot step, against `/FOF/Primary` at 5600 s. The whole choice of centre was worth ~4 s, so it was settled on physics: `PotMinPos` keeps `SecR50` comparable with every catalogue already on disk, and stays on the dominant clump when the SecFOF percolates at large linking length, where a COM would drift into empty space between clumps.

Enabled by the `STARP.Bounded` field added earlier today: the flag survives in the star slot from the Step 5b bound pass to the size pass, with no reordering in between.

Compiles clean; `test_fof`, `test_slotsmanager`, `test_exchange`, `test_memory` pass on 4 ranks. No production run yet.

**Files modified:** `libgadget/secondfof.c`

---

## 2026-08-03 — `4/SeedBHTime`: record when each star's cluster mass was spent on a BH seed

New star property written to every snapshot — `PART_`, `PIG_` and `SecPIG_` alike: the scale factor at which `Seeded` flipped 0 → 1, and `-1` while the star is still unseeded. It marks the whole seedable population of a seeded group, not only the host star that became the BH; a star already converted to a BH is type 5 and carries no star slot, so it is not recorded.

Kept as a separate field rather than converting `Seeded` from an int flag to a float scale factor. Repurposing looked free but is not: bigfile silently casts dtypes on read, so an existing snapshot's `i4` `4/Seeded` would load into an `f4` field as 1.0, restarting every previously seeded star as "seeded at a = 1.0" with no error. On top of that, ~20 sites use `.Seeded` as a boolean and a `-1` sentinel is truthy, so each would have inverted for unseeded stars, and every analysis notebook reads `4/Seeded` as 0/1.

Backward compatible. A restart from a snapshot without the block warns and initialises from `Seeded`: 0 for already-seeded stars (real seeding time unrecoverable, and distinct from any real scale factor, which is always > 0), -1 for unseeded ones. Same mechanism as the existing `4/BirthMetallicity` fallback.

Costs: memory zero — the `float` lands in the padding hole between `FormationTime` and `ClusterFormationEfficiency`, so `sizeof(struct star_particle_data)` is still 168 B (verified by compiling). CPU is two extra stores at the only two places `Seeded` is set (`fof.c` per-cluster marking, `secondfof.c` single-seed marking). Output grows by one `f4` array of N_star per snapshot, ~1 MB at 265k stars.

Compiles clean; `test_fof`, `test_slotsmanager`, `test_exchange` pass on 4 ranks. No production run yet.

**Files modified:** `libgadget/slotsmanager.h`, `libgadget/petaio.c`, `libgadget/fof.c`, `libgadget/secondfof.c`, `libgadget/sfr_eff.c`

---

## 2026-08-03 — SecPIG: per-star `4/Bounded` flag from the BHseedSecFOFbound pass

The secondary-FOF catalogue now records boundness per star particle, not just as the per-group `SecBound*` totals. New SecPIG block `4/Bounded`, tri-state: `-1` not a member of any secFOF group in this catalogue (also the value everywhere when `BHseedSecFOFbound=0`), `0` member star that failed the bound test, `1` member star bound to its group. It flags bound stars whether or not they are seeded — the unseeded restriction is left to whoever consumes it.

Written only to SecPIG, and write-only: it means nothing in `PART_`/`PIG_`, where it would be a stale leftover of whichever secondary FOF ran last, and it is never read back, so a restart starts from `-1`. The catalogue pass (`apply=0`) is the only writer, so the flag always describes the catalogue it is written beside, at that catalogue's linking length and `MinPrimaryLength`.

Cost is close to nothing. The boundness computation already existed and already produced this flag internally; the catalogue path simply stopped throwing it away. Requesting it does make every rank sweep every group instead of only its own, but the expensive shared work (member-star replication, DM binning, the two `Allreduce`s) was paid either way, and one group holds the large majority of the member stars, so its owner was already doing that work while the other ranks waited at the barrier. Memory is genuinely zero: the new `int` lands in the trailing padding after `Seeded`, leaving `sizeof(struct star_particle_data)` at 168 B (verified by compiling both). Output adds one `int32` array of length N_star, ~1 MB at 265k stars.

The flag has to live in the star slot rather than stay a transient array: the SecPIG writer copies particles into a fresh part/slot manager and re-sorts them by `(Type, GrNr)` across ranks, so a `P[]`-indexed mask would no longer line up. Slots are copied whole, so the field travels with its particle.

Compiles clean; no run yet.

**Files modified:** `libgadget/slotsmanager.h`, `libgadget/secondfof.c`, `libgadget/petaio.c`, `libgadget/petaio.h`, `libgadget/fofpetaio.c`, `libgadget/sfr_eff.c`

---

## 2026-08-02 — cpu.txt: break BH seeding down into phases

BH seeding was already timed in `cpu.txt`, but as a single `FOF/Seeding` lump covering the whole of `fof_seed`. It is now split into ten leaves under `FOF/Seeding`, so the total is unchanged and the phases are visible: `Bound` (the `BHseedSecFOFbound` boundness pass), `Gate` (seeding gates plus the `SeedSecFOFcomSample` combined draw), `Export` (marked-group exchange), `CountUB` (the upper-bound seed counts, which redraw each group's cluster population), `Slots` (BH slot reservation and the tree relocation around it), `MakeOne` (the one-seed-per-group placement loop), `Extra` (`SeedInSecFOFMultipleSeeds`), `SCdraw` and `SCplace` (the per-cluster mode's ICMF draw and its host gather / placement / marking), and `Misc`.

`SCdraw` and `CountUB` are the two that scale with a group's seeding budget, so they are what to watch when running with `SecFOFseedSpendOnPlaced=1`.

Hierarchy and summation verified against the real `walltime.c` with a standalone harness (parent totals its leaves, no double counting under `FOF`); `test_fof` passes.

**Files modified:** `libgadget/fof.c`

---

## 2026-08-02 — SecFOFseedSpendOnPlaced: spend a secFOF group's seeding budget only when a BH is actually placed

New optional parameter `SecFOFseedSpendOnPlaced` (default 0 = unchanged behavior) for the per-cluster secFOF seeding mode. It changes what marks a group's seedable stars `Seeded=1`, i.e. what permanently removes them from the seeding budget: 0 marks them as soon as the group *requested* a seed, 1 marks them only once the group has actually *placed* a BH particle (ordinary or `MinBHSeedInSC` compensating).

The two differ because placement carries gates the request does not see — with `MbhMscRelationCWmodel=1` each cluster additionally needs `M_VMS >= SeedBlackHoleMass`, and every seed needs an unseeded host star. Under the old rule a group whose drawn clusters all fall below the seed-mass floor spends its entire cluster-mass budget without ever making a BH, and is sterilised for good: that budget is also the ICMF cutoff `SCcomMcut`, so its later draws get weaker rather than stronger. This is what leaves ASTRID-eligible haloes unseeded in `output_l0.1_BH3e5_bound` (only 1.8% of drawn clusters clear the floor there). Under the new rule such a group keeps its stars unseeded and retries at the next seeding step with a budget grown by the stars formed since.

The count of placed seeds is accumulated per group during the two placement phases, which already run the same selection on every rank, so it needs no extra communication and the marking loop is unchanged in cost. The seeding log line now also reports how many groups requested a seed but placed none and kept their budget.

**Cost, measured on `output_l0.1_BH3e5_bound`:** a group that never seeds redraws an ever-larger cluster population every step. At z=9 the full carried-over budget is ~37x the per-step drawn cluster mass, so the sampling cost and (with `MinMscForSCdetail` lowered) the StarClusterDetails volume grow correspondingly over a run. Keep `MinMscForSCdetail` at its default when enabling this.

Rejected at startup outside the per-cluster mode. Compiles clean, `test_fof` passes, parameter parsing and the guard verified end to end; no production run yet.

**Files modified:** `libgadget/fof.c`, `gadget/params.c`

---

## 2026-08-01 — SCmasscapSecFOFstarmass: drop the truncated cluster when its remainder is sub-ICMF

The draw-order truncation added earlier today shortened the cluster crossing `Mcut` to whatever stellar-mass budget was left, so the draw landed exactly on `Mcut`. That remainder is not an ICMF sample and can be arbitrarily small, and `output_l0.1_BH3e5_bound` duly produced a **26.8 Msun** star cluster — below the 100 Msun lower limit of the mass function itself, so its size, VMS mass and seed mass would all be extrapolated off the bottom of their fitted relations.

The crossing cluster is now dropped whole, and the budget closed, whenever the remainder is below `msc_min_code` (100 Msun); otherwise it is shortened as before. Every emitted mass is therefore either a full ICMF draw or a remainder of at least 100 Msun. The draw lands on `Mcut` when it can and under it by less than 100 Msun when it cannot — the cap stays a cap either way, since it can now only remove mass. All three samplers share `msc_budget_take` and treat a 0 return as end-of-draw, so the identical-population invariant holds unchanged.

Measured on the run that exposed it (9.62M cluster records, 2062 secFOF draws keyed on the host `(a, GrNr)`, compensation records excluded): 63 draws (3.1%) hit the cap, and in 62 of them the shortened cluster was already >= 100 Msun and is untouched. Exactly one record changes — the last of the 44 clusters of `GrNr=20` at `a=0.096975`, the only sub-100-Msun cluster in the whole file — and it disappears. Effect on that group: total drawn falls from 212,398.10 Msun (exactly its bound unseeded stellar mass) to 212,371.33 Msun. Nothing downstream changes; that cluster had `Mbh_seed = 0`.

**Files modified:** `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`

---

## 2026-08-01 — BHseedSecFOFbound: restrict the unseeded-star metallicity to the bound subset

The bound pass restricted every *mass* the seeding decision reads — the cluster-mass budget, `SCcomMcut`, the host pool, the seed host — but deliberately left the unseeded-star metallicity summary (`SCMetalMassUnseeded`, `SCClusterMassUnseededInit`, `SCMetUnseeded{Min,Max,Sum,Sum2,LogSum,LogSum2,Hist}`) accumulated over ALL unseeded stars, because `NStarUnseeded` is its denominator and could not be recomputed there. That left a real hole: those fields are not diagnostics. Per-cluster CW seeding draws each cluster's metallicity from them — `CWmodelMetallicity` 'ave' from the metal-mass ratio, 'lognormal' from the log sums, 'uniform' from the min/max — and Z feeds the Vink winds in the collision model, so an unbound star could still shift `M_VMS` and hence the BH seed mass while contributing no mass and hosting no seed.

The whole summary is now rebuilt over the bound unseeded stars and written back under `apply=1`, term for term as `add_particle_to_group` builds the unrestricted one (same sentinels, same `SC_MET_HIST_LOGMIN` floor, same equal-weight-per-star convention), so the bound version is the unrestricted one with the unbound stars removed and nothing else. `NStarUnseeded` is restricted with it — required, since it is the particle count those equal-weight sums are divided by in `sc_met_unseeded_stats`, and restricting the sums alone would divide bound sums by an unrestricted count.

**Second-order effect, deliberate:** `NStarUnseeded` also caps the host count in the multi-seed modes and sizes their slot reservation, so both tighten. That is correct — an unbound star can never host a seed — and remains a valid upper bound, since the host pool `(bound && unseeded && f(Z)>0)` is a subset of the bound unseeded stars counted here. A group with bound stars but no bound *unseeded* star now ends with `NStarUnseeded = 0` and is dropped by `secfof_random_group_eligible`, which it would have been anyway on a zero budget.

**Meaning change:** in a `BHseedSecFOFbound` run the StarClusterDetails `MetUnseeded*` columns and `Metallicity` now describe the bound unseeded stars — matching the Z actually used for the seed mass. The SecPIG catalogue path is untouched (it calls the bound pass with `apply=0`).

`struct sb_star` gained the frozen `initClusterMass` and `BirthMetallicity`, 120 -> 128 B, which lowers the replicated-gather ceiling from 17.9M to 16.8M member stars (still 2.00 GB/rank); the header memory table, the shrink-the-record note and the abort message were updated, and the size/ceiling verified by compiling the struct standalone (128 B, INT_MAX/128 = 16,777,215, 2.000 GB). Compiles with no warnings, `test_fof` passes; no end-to-end run.

**Files modified:** `libgadget/fof.c`, `libgadget/fof.h`, `gadget/params.c`

---

## 2026-08-01 — One deterministic seed-host ordering everywhere: (scm desc, ID asc)

The group's seed host — the largest-`scm` unseeded star, whose ID becomes `SeedStarID` — was picked by two different rules. `add_particle_to_group` and `fof_reduce_group` used a bare `>`, i.e. "keep the first maximum encountered", while the `BHseedSecFOFbound` pass broke equal-`scm` ties on the smallest ID. All three now share one `sc_seed_host_better()` helper implementing **(scm descending, ID ascending)**.

The tie-break is not cosmetic. `SeedStarID` seeds the per-group cluster-sampler RNG, so two orderings that disagree on a tie do not give slightly different answers — they give a completely different Poisson draw and cluster population. The old `>` form made that depend on local particle index order and on the rank-merge order, i.e. on the domain decomposition, so the same physical configuration could seed differently at different `NTask`; and it let `BHseedSecFOFbound` change the seeding decision even when every star is bound.

How exposed the existing runs are, measured on `output_l0.1_BH1e3_compensate/PIG_021`: **zero** tied maxima across all 47 groups with an unseeded star. With `SeedSecFOFcomSample=1` the key is `f(Z)*Gamma*m_star`, and both factors are continuous in practice — 246,333 distinct `Mass` values and 265,543 distinct CFE values among 275,756 stars, and the CFE table has no plateau. Completed runs are unaffected. The other branch is the exposed one: when `StarClusterSampling=1 && !SeedSecFOFcomSample` the key is the discrete `StarClusterMass_sample`, which is 0 for 100% of stars in that snapshot, so ties there are expected rather than hypothetical.

Also aligned the bound pass's sentinel (`best_scm` −1 -> 0) with `MaxStarClusterMass`'s, so a group all of whose bound unseeded stars have `scm == 0` ends with no host in both paths instead of an arbitrary one. Safe: in either configuration "every scm is 0" implies a zero seeding budget, so the gates drop the group regardless.

**Files modified:** `libgadget/fof.c`

---

## 2026-08-01 — BHseedSecFOFbound: correct the stale rest-frame description (doc only)

The `BHseedSecFOFbound` parameter text claimed the binding test's rest frame was "the member stars' mass-weighted centre of mass". It is not, and never was: the implementation seeds the COM with the DM momentum and mass inside the sphere and then adds the member stars. The code is right and the description was wrong — the potential in the same loop is `(Min + Mdm_in)/sk + Tout + Tdm_out`, i.e. stars *and* DM, so a stars-only frame would leave the DM that dominates the well streaming through it; both `BHseedSecFOFbound` modes are DM-inclusive by definition; and this is the Eq.-2 test of Williams et al. 2025 Sec 2.3, whose convention (also used by `script/secpig_star_dm_binding.py`) is to refer velocities to the COM of *the system being tested* — stars only for Eq. 1, stars+DM for Eq. 2. A stars-only frame here would have disagreed with that pipeline. Wording corrected, and the description now also states that the frame is computed once over all member stars with no iterative unbinding.

Unrelated but noted while checking: the older `SeedSeedFOFMassiveBoundStar` uses the deepest-potential member's velocity as its rest frame, which is a genuinely different (and weaker) convention — a single particle's velocity carries an offset of order the velocity dispersion, the same scale the test operates at. Left alone; it is incompatible with `BHseedSecFOFbound` and unused.

**Files modified:** `gadget/params.c` (description only, no behaviour change)

---

## 2026-08-01 — SCmasscapSecFOFstarmass: cap the cluster draw in draw order, in every mode

The cap only ever existed in the summed mode, where it clamped the *total* after the fact; the per-cluster mode (`SecFOFseedsumover=0`) drew its population with no such constraint at all, so a group could emit a single cluster heavier than every star available to form it. Measured on the v4 test runs (`output_l0.1_BH1e3_compensate`, 2074 group draws, 10.9M sampled clusters): ~4% of draws overshoot the group's unseeded stellar mass, ~0.6% of the BH-seeding clusters are individually heavier than it, and overshoots reach 300x. Rare, but concentrated — 10 draws carry 96% of the affected seed mass — and those are the draws seeding the most massive BHs.

The cap is now a **draw-order truncation**: clusters are accepted until the running total reaches `Mcut` (the group's unseeded stellar mass), the cluster that crosses it is shortened to the remaining budget so the total lands exactly on `Mcut`, and every later cluster is dropped. Draw order rather than sorted order is the whole point — with `StarClusterICMFcutoff=0` the pure m^-2 ICMF knows nothing about the host, and truncating a descending-sorted list would only ever delete the lightest clusters, leaving the offending heavy one untouched. Shortening the crossing cluster rather than dropping it keeps a group whose first draw already exceeds `Mcut` from seeding nothing at all.

All three samplers share one `struct msc_budget`, charged for **every** drawn cluster before any mass-window or `min_seed_mass` test, so the seeding pass, the slot-reservation pass and the `MinMscForSCdetail` / `MinBHSeedInSC` detail pass all truncate at the identical index however they filter — the existing "identical population" invariant is preserved. In the per-cluster mode this also lowers `n_qualify`, hence the number of BHs seeded. Breaking out of the draw loop early cannot desynchronise anything: the per-cluster deviate is indexed (`rand_id + 300 + s`), not sequential.

**Behaviour change:** the summed mode (`SecFOFseedsumover=1`) now truncates in draw order too instead of clamping the sum, so runs with `SCmasscapSecFOFstarmass=1` will not reproduce earlier results. Deliberate — the parameter now means one thing in every mode. The per-particle mode (`SeedSecFOFcomSampleParticle=1`) is unaffected: its per-star cutoff is not the group budget, so it still passes `allow_cap=0` and the cap stays on the group-summed total in `fof.c`.

Invariants checked against the functions extracted verbatim from the source: cap off / loose budget / `allow_cap=0` truncate nothing; a tight cap lands the total on `Mcut` to 1e-15 relative; survivors are a bit-identical prefix of the uncapped draw with only the crossing cluster shortened; a first cluster heavier than the budget yields exactly one cluster of exactly `Mcut`; two different mass windows truncate at the same index and total. Compiles with no warnings, `test_fof` passes; no end-to-end run yet.

**Files modified:** `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`, `gadget/params.c`

---

## 2026-08-01 — StarClusterDetails: record the host group's full mass matrix

The detail record carried the host group's totals and its bound subset but not the plain *unseeded* sums, so the natural denominators had to be inferred (or, for the unseeded stellar mass, could not be recovered at all — `SCcomMcut` holds it but the bound restriction overwrites it in place). Three fields added, and each record now pins down every cell of

    quantity              all stars             unseeded          bound          bound+unseeded
    Sum m_star            StellarMassTotal   StellarMassUnseeded  BoundStarMass  BoundStarMassUnseeded
    Sum Gamma*m_star   StarClusterMassTotal  SCMass_unseeded      BoundSCMass    BoundSCMassUnseeded

None of the six carry the f(Z) seeding factor, so every ratio between them is meaningful and `SCMass_seeded + SCMass_unseeded == StarClusterMassTotal` exactly. `SCMass_unseeded` is derived from the two raw catalogue sums at record time rather than accumulated again; `StellarMassUnseeded` needed a new Group field, accumulated beside `SCcomMcut` and never overwritten.

The third field, **`SCMassSeedBudget`**, is the one number that actually drove the seed: the f(Z)-*weighted* unseeded sum, already bound-restricted when `BHseedSecFOFbound > 0`, read straight out of `StarClusterMassUnseeded` at record time. It coincides with `BoundSCMassUnseeded` (bound modes) or `SCMass_unseeded` (feature off) precisely when f(Z)=1 for every contributing star, i.e. `StarClusterSeedMetallicityMax <= Min`; otherwise it is strictly smaller. Recording it removes the only remaining place where the file's meaning depended on the metallicity settings. The three travel in a new `struct SCgroupmass` rather than three more positional arguments to the already 15-argument record functions, filled by `sc_group_mass()` mirroring the existing `sc_bound_stats()`, and carried in the two gathered per-group message structs alongside `boundinfo`.

Record 224 -> 248 bytes (payload marker 216 -> 240), converter updated and verified field-by-field against the compiled struct; the marker-160 and older layouts still resolve unchanged, so existing archives convert as before. Markers 208 and 216 are deliberately not readable — both were same-week intermediates that no completed run wrote, and 208's `BoundSCMass` meant something else. Compiles and links cleanly with no warnings, `test_fof` passes; still no end-to-end run.

**Files modified:** `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/scinfo.c`, `libgadget/scinfo.h` (+ `script/scdetails_raw2bf.py`, outside the repo)

---

## 2026-08-01 — BHseedSecFOFbound: make the bound cluster-mass diagnostics comparable to the totals

`SecBoundSCMass` (and the detail file's `BoundSCMass`) were the *f(Z)-weighted* sum over the bound *unseeded* stars — the seeding budget itself — while the totals they would naturally be divided by, `SecSCMass` / `StarClusterMassTotal`, are the plain `Sum(Gamma*m_star)` over *all* member stars. Two differences at once, so the obvious ratio was not a bound fraction of anything. They now use the same definition as those totals: **no f(Z) factor, over all bound stars**, with a new `SecBoundSCMass_unseeded` / `BoundSCMassUnseeded` for the bound *and* unseeded subset. Every bound fraction is now a ratio of like for like:

    SecBoundStarMass          / SecMassByType[4]
    SecBoundSCMass            / SecSCMass
    SecBoundSCMass_unseeded   / (SecSCMass - SecSCMass_seeded)

The seeding budget is unchanged — it is still the f(Z)-weighted unseeded sum, written to `StarClusterMassUnseeded` in the apply pass and never a catalogue output. It coincides with the new unseeded block only when f(Z) = 1 everywhere (`StarClusterSeedMetallicityMax <= Min`), which is why the diagnostics and the budget are now accumulated separately rather than sharing one number. `struct sb_star` grew a raw `ClusterMass` field for this, 112 -> 120 B, which lowers the replicated-gather ceiling from 19.2M to 17.9M member stars (still ~2.0 GB/rank); the header table and the abort message were updated.

The detail record grew by one double, 216 -> 224 bytes (payload marker 208 -> 216), and `script/scdetails_raw2bf.py` was updated to match — verified field-by-field against the C struct, not just by record size (`BoundSCMass` at offset 176, `BoundSCMassUnseeded` at 184, `Flag` at 216 in both). The short-lived marker-208 layout is deliberately **not** kept as a readable older layout: its `BoundSCMass` meant something else, so reading it under the new names would silently mislead, and no completed run ever wrote one (an unknown marker raises a clear error listing the accepted set). Compiles and links cleanly with no warnings, `test_fof` passes; still no end-to-end run.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/secondfof.c`, `libgadget/scinfo.c`, `libgadget/scinfo.h` (+ `script/scdetails_raw2bf.py`, outside the repo)

---

## 2026-07-31 — BHseedSecFOFbound is now compatible with per-cluster seeding (SecFOFseedsumover=0)

Lifted the startup refusal of `BHseedSecFOFbound > 0` together with `SecFOFseedsumover = 0`. The group-level restriction could not reach that mode because per-cluster seeding ignores the single `seed_index_star` the restriction repoints and draws its own host for every sampled cluster from the group's unseeded stars. `fof_secfof_bound_restrict` now optionally fills a **transient per-particle bound flag**, a plain `char` array over local particles that the per-cluster path uses to filter its host pool. Only the budget was ever restricted correctly in that mode; the seeds themselves could land on unbound stars, which is the defect this removes.

Filtering the host pool is sufficient on its own: the pool feeds `navail`, each group places `min(n_request, navail)` seeds, and so the seed count is capped at the number of bound hosts even though `n_request` is still derived from the unrestricted `NStarUnseeded`. The same filter automatically covers the `MinBHSeedInSC` compensating seeds and the BH-slot reservation, which draw from the same pool. Producing the mask requires every rank to evaluate every group rather than only the ones it owns, since a group's member stars are spread over all ranks — but this replaces communication with redundant compute on ranks that were idle in that loop anyway, and because one group holds 93-96% of the member stars, sweeping all groups costs a few percent over sweeping just the largest. The mask lives entirely inside `fof_seed`, allocated at the bottom of its `mymalloc2` stack and released immediately after `ImportGroups` so the caller-owned return buffers stay LIFO-correct.

**Unbound stars are now deferred rather than consumed.** Previously a seeded group flagged *every* seedable star `Seeded = 1`, so under the restriction an unbound star was spent without having contributed to the budget and could never contribute again. The same mask now filters that marking in **both** seeding paths, so an unbound star keeps `Seeded = 0` and its cluster mass and contributes at a later seeding search if it falls in and becomes bound. This has no runaway: a re-visited group's budget still counts only its bound *unseeded* stars, so it can seed again only from stars that newly became bound or newly formed, and a star that does contribute is marked at that point. Reaching the single-seed path's marking, which happens in `secondfof_seed` after `fof_seed` returns, required extending the mask's lifetime past `fof_seed` — it is now returned through a new `bound_mask_out` argument as the oldest of the four buffers the caller owns, freed after `mcut`/`totmsc`/`grnr` and before `fof_finish` (whose `Group` array predates the call). The mask is consequently produced whenever `BHseedSecFOFbound` is on, not only in per-cluster mode.

`SeedSecFOFcomSampleParticle = 1` and `SeedInSecFOFMultipleSeeds = 1` are still refused, but the first is now half-fixed as a side effect: its redistribution in `secondfof_seed` sums over the same restricted loop, so `totmsc * m_star / SCcomMcut` again conserves the group total. Only its sampler (`fof_secfof_particle_sample`, which draws for every unseeded star) remains unfiltered, so lifting that guard now needs one more filter rather than a lifetime change. Compiles and links cleanly with no warnings, `test_fof` passes; still no end-to-end run.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/secondfof.c`

---

## 2026-07-31 — BHseedSecFOFbound: restrict the BH-seeding budget to gravitationally bound stars

Added a new `int` parameter `BHseedSecFOFbound` (default 0 = off; requires `SeedInSecFOFasStarCluster=1`, incompatible with `SeedSeedFOFMassiveBoundStar=1`). When set, only the star particles gravitationally bound to their secondary-FOF group contribute `Gamma*m_star` to that group's BH-seeding budget; the whole structure is used when it is 0. Mode 1 tests boundedness against the member stars plus every DM particle inside `Rmax`; mode 2 caps that sphere at `min(2*R50, Rmax)`. The cap matters because `Rmax` is set by the single most distant member star and therefore tracks a filament once the FoF percolates — measured on the z=9 star-primary catalogues, `Rmax` of the dominant group grows 11x between linking lengths 0.244 and 7.32 ckpc/h while `R50` grows 1.4x, and the enclosed DM/stellar mass ratio goes 2 -> 24, so Eq.-2 boundedness stops being a statement about the object. The `min()` form is deliberate: for marginal few-star groups `2*R50` exceeds `Rmax`, and enlarging the sphere there would make the test more permissive exactly where it should be strict.

The potential is the spherically averaged profile about `SecPotMinPos` (softened, member stars sorted by radius with the DM carried as a 128-bin radial histogram) rather than the exact double sum, which is O(N log N) instead of O(N^2). This is a deliberate accuracy-for-cost trade: 93-96% of `Sum(N_star^2)` sits in a single group, already 3.4e10 pairs at z=9 / Ng=512 and growing as (stellar mass)^2, so the exact sum would be ~64x more expensive at Ng=1024 and ~25x more by z=6. Against the exact per-pair potential on the same snapshot the approximation reproduces the bound stellar mass to -0.05% (mode 1) and -0.21% (mode 2) with 99.9% / 99.7% per-star flag agreement. Distribution follows the existing `secondfof_compute_sizes` pattern — group geometry and member stars replicated by Allgatherv, the DM never moves (each rank bins its own into per-group histograms summed by one Allreduce) — so each group is written only by its owner and the result does not depend on the rank count.

Every quantity the seeding gates read off the unseeded stars is replaced by its bound-only counterpart (`StarClusterMassUnseeded`, `StarClusterMassSampleUnseeded`, `SCcomMcut`) and the seed host is repointed to the best bound unseeded star by the same key the unrestricted run uses, so a BH is never seeded at an unbound star. `NStarUnseeded` is deliberately left alone since it is the denominator of the unseeded-star metallicity histogram; the multi-seed host caps derived from it therefore still count every unseeded star. **The catalogue and the detail records are unchanged in meaning:** `SecSCMass`, `SecMassByType` and `SecLengthByType` still describe all member stars in every mode, and the bound subset is reported separately in six new SecPIG blocks (`SecBoundStarMass`, `SecBoundStarMassUnseeded`, `SecBoundSCMass`, `SecBoundStarNum`, `SecBoundRdm`, `SecBoundDMMass`) and seven new StarClusterDetails fields, all identically zero when the feature is off. The on-disk detail record grows from 168 to 216 bytes (payload marker 160 -> 208), and `script/scdetails_raw2bf.py` has been given the new layout (six auto-detected markers now: 208/160/152/148/100/92). Verified by round-tripping a C-written record with a distinct value per field through the numpy dtype -- field order, not just record size -- and by re-reading two existing archives (2.0M records at marker 160, 5.9M at marker 152), which still convert and are now labelled as older layouts. Readers should check the new `BoundMode` block before using the other `Bound*` blocks: all-zero means the feature was off, not that nothing was bound.

Two follow-up fixes on the same day. (1) Startup now refuses the parameter combinations the group-level restriction cannot honour (see the commit for the reasoning). (2) The catalogue pass took its group centre from `PotMinPos`, which `secondfof_run` leaves at (0,0,0) when `OutputPotential = 0`, because it passes that flag through as the FOF `PotentialMin` and both `add_particle_to_group` and `fof_reduce_group` gate the PotMin update on it; the CM fall-back that repairs it sat *after* the bound pass. Centred on the box corner every group would have taken box-scale radii and a DM lookup grid collapsed to one cell. The fall-back has been moved ahead of the bound pass (new Step 5a). The seeding path was never affected -- `secondfof_seed` hardcodes `PotentialMin = 1` -- so only the reported SecBound* blocks could have been wrong, and only with `OutputPotential = 0` (default 1). (3) The Allgatherv byte counts and displacements are `int`, so the replicated star gather has a hard ceiling at INT_MAX/sizeof(sb_star) = 19.2M member stars (112 B each, ~2.0 GB/rank) beyond which a displacement wraps negative and MPI reads outside the buffer; the local and global counts are now checked and the run aborts with a clear message instead. This is a real ceiling for the Ng=1024 rung, not a theoretical one: the replication is per-GLOBAL-count and does not improve with more ranks. Raising it means either shrinking the record (float offsets from the group centre + float scalars, 112 -> ~72 B, ~30M stars) or replacing the replication with a distributed gather, which removes the ceiling but faces severe load imbalance since 93-96% of the member stars are in one group. The ceiling and both options are documented in the function header. Separately, the geometry pass (which runs over every group on every rank, since a rank needs Rdm for groups it does not own in order to bin its local DM into them) no longer sorts in mode 1: there Rdm = Rmax, a plain O(N) maximum, and the sort was pure waste. Measured on the l=0.244 catalogue that pass drops from 0.041 s to 0.001 s, and the saving grows as N log N. Mode 2 still sorts, since the half-mass radius is a mass-weighted median. The mode-1 result is bit-for-bit identical to the old sort-then-take-the-last form (verified over ties, all-zero radii and n=1). `secondfof_run` gained `atime`/`CP` arguments so it can fill the catalogue blocks. Default 0 leaves every existing run bit-for-bit unchanged. Compiles and links cleanly; the numerical scheme was validated against the exact potential on `codetest/SecondFOF/output_PriStar/PART_021`, but **no end-to-end MP-Gadget run has been made yet** — the 512^3 restart does not fit on a single node and batch submission was unavailable from the session.

Startup refuses the parameter combinations the group-level restriction cannot honour. The restriction rewrites the group's summed budget and repoints the single seed host, but carries no per-star bound flag, so any mode that afterwards re-scans the unseeded stars individually would still see the unbound ones: `SeedSecFOFcomSampleParticle=1` (the per-star sampler draws for every unseeded star, and the redistribution would then divide by the bound stellar mass while summing over all unseeded stars, inflating the group cluster mass by M_unseeded/M_bound instead of conserving it), `SeedInSecFOFMultipleSeeds=1` and `SecFOFseedsumover=0` (extra / per-cluster hosts are not boundedness-tested). These are the same modes `SeedSeedFOFMassiveBoundStar` already excludes, for the same reason. Supporting them needs a per-star bound flag persisted out of `fof_secfof_bound_restrict`.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/secondfof.c`, `libgadget/secondfof.h`, `libgadget/scinfo.c`, `libgadget/scinfo.h`, `libgadget/run.c`

---

## 2026-07-30 — Star cluster model: missing factor of h in every solar-mass constant

Fixed a units bug in the star cluster model: the four hard-coded solar-mass yardsticks converted a physical solar-mass constant to code units as `M * SOLAR_MASS / UnitMass_in_g`, but the code mass unit is 1e10 Msun/h, so the conversion needed an extra factor of `HubbleParam` — as the physical time and velocity constants set immediately beside them already carried one. Every code-to-solar conversion in the model is a ratio against one of these four yardsticks, so all of them were off by 1/h in the same direction: the cluster mass function ran from 147.6 Msun to 1.48e8 Msun instead of 1e2 to 1e8, the "massive cluster" and per-secFOF multi-seed thresholds sat at 1.48e4 and 1.48e8 Msun, the size-mass relation returned radii ~9% too small, and — most consequentially — the cluster masses handed to the CW model were 0.68x their physical value, systematically under-predicting M_VMS and hence the BH seed masses. All four now carry the h. **This changes results:** at h=0.6774 the mass function's mean cluster mass drops by a factor h, so a given cluster-mass budget is split into ~1.48x as many clusters, and every cluster's M_VMS is re-evaluated at its true physical mass. Runs made before this fix are not comparable to runs made after it. Compiles and links cleanly with no warnings; `test_fof` passes; no end-to-end simulation has been run yet.

**Files modified:** `libgadget/sfr_eff.c` (+ a comment in `libgadget/fof.c`)

---

## 2026-07-30 — MinBHSeedInSC: compensate the BH seeds the resolution cannot make

Added a new `double` parameter `MinBHSeedInSC` (default 0 = off, only usable with `MbhMscRelationCWmodel=1`), the lowest BH seed mass a star cluster can physically produce. It is read in internal mass units and the value actually used is `max(200 Msun, given)`, since in the Williams et al. 2026 model only a VMS above 200 Msun collapses to a BH seed; the clamp is applied at first use rather than at parameter time because the code-to-solar conversion is only established once the units are known. The run itself only creates a BH for a sampled cluster that is both at or above `MinMscForBHseed` (a resolution limit) and whose model M_VMS reaches `SeedBlackHoleMass`, so every other cluster whose M_VMS reaches `MinBHSeedInSC` is a seed the simulation should have had and does not. When enabled, those missed M_VMS values are summed per secFOF group — over the sub-`MinMscForBHseed` clusters and over the candidate clusters whose M_VMS fell below the seed floor alike — and paid back by seeding `N = floor(missed_mass / SeedBlackHoleMass)` extra BHs of exactly `SeedBlackHoleMass` each, placed after the ordinary seeds on the group's remaining unseeded stars in the same `f(Z)*ClusterMass` order. Each compensating BH carries `StarClusterMass` = the summed cluster mass of the missed clusters shared evenly over the N seeds, so the missed cluster mass is conserved as well as the missed seed mass; they are written to StarClusterDetails under a new `Flag=3` (`SC_FLAG_COMPENSATE`), and a group that receives only compensating seeds now has its seedable stars consumed like any other seeded group. Implementation notes: the missed mass is accumulated on each group's reference-star owner and shared in a single Allreduce, then a second candidate gather places the seeds (the first gather is truncated to the ordinary seed count and by then largely converted); BH slots are reserved up front from a CW-model-free upper bound that exploits M_VMS being capped at the cluster mass. **This is a large physical change, not a small correction:** measured against the recorded clusters of `BH-L12.5-R1.0-Ng512-v4/output_l0.1_BH2e2_Mscgas`, it multiplies the number of seeded BHs by roughly 8x at `SeedBlackHoleMass = 200 Msun` (16,320 compensating seeds vs 1,938 real ones) and 4x at 5e3 Msun. It also evaluates the CW model over the whole sub-threshold cluster population, as `MinMscForSCdetail=0` does, costing order 1% of wall time. Default 0 leaves every existing run bit-for-bit unchanged. Compiles and links cleanly; parameter parsing and the startup validation were exercised, `test_fof` passes, but no end-to-end simulation has been run yet.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/scinfo.h`, `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`

---

## 2026-07-29 — StarClusterDetails: record the host group's total stellar mass

Added a `StellarMassTotal` field to the StarClusterDetails record: the host (secondary) FOF group's total stellar mass Sum(m_star) over all member stars, taken from the group's `MassType[4]` and written in code units. The file previously had no stellar mass at all — the closest field, `StarClusterMassTotal`, is the cluster mass Sum(Gamma*m_star), and since Gamma is the per-star birth-pressure CFE the two are not related by any single factor, so the stellar mass could not be recovered from the records. Their ratio is now directly usable as the group's mass-weighted mean CFE. The field is filled at all five recording call sites (the primary/secFOF seed path, the per-secFOF multi-seed extras, and the per-cluster `SecFOFseedsumover=0` seeded / no-VMS / below-threshold records), carried per host group through the existing gathered metadata structs, so it adds no MPI communication. On-disk record grows from 160 to 168 bytes (payload marker 152 -> 160); `script/scdetails_raw2bf.py` auto-detects the new layout and still converts all four older ones, verified against existing raw shards. Compiles and links cleanly.

**Files modified:** `libgadget/fof.c`, `libgadget/scinfo.c`, `libgadget/scinfo.h` (+ `script/scdetails_raw2bf.py` in the analysis pipeline)

---

## 2026-07-28 — StarClusterDetails: record the star clusters that seed no BH (MinMscForSCdetail)

Added a new `double` parameter `MinMscForSCdetail` (default -1, only used with `StarClusterDetails=1` in the per-cluster secFOF mode `SecFOFseedsumover=0`), the lowest sampled star-cluster mass written to the StarClusterDetails files. The default -1 means `MinMscForBHseed`, i.e. the previous behavior of recording only clusters that can seed a BH. Setting it lower additionally records every sampled cluster down to that mass — the whole drawn population when it is 0 — even though those clusters never seed. Since they have no host star of their own, they all carry their host group's reference star (the host of the group's most massive cluster) as ID/Pos, and are written by that star's owner rank only, so each cluster is recorded exactly once; the population is redrawn locally from the group's deterministic (cutoff, mass, RNG seed) triple, so the feature adds no MPI communication and buffers nothing. Each cluster still gets its own effective radius and CW-model metallicity, drawn as for the seeding clusters but keyed on its index within the group draw. A new `Flag` field in the record distinguishes the three record types (BH seeded / CW-model M_VMS below the seed floor / below MinMscForBHseed), growing the on-disk record from 156 to 160 bytes (payload marker 148 -> 152); `script/scdetails_raw2bf.py` auto-detects the new layout and still converts all three older ones. Startup aborts if the parameter is lowered outside the per-cluster mode or without `StarClusterDetails=1`, and logs the expected record inflation otherwise. NOTE the m^-2 ICMF makes sub-threshold clusters roughly `MinMscForBHseed / MinMscForSCdetail` times more numerous than seeding ones, so the detail files can grow by orders of magnitude; the bulk records are therefore written through a 1 MB buffered stream flushed once per seeding step rather than per record. Also refactored the Poisson count and the per-cluster inverse-CDF draw, previously copied in each star-cluster sampler, into shared helpers; verified bit-identical over 41M sampled masses in both `StarClusterICMFcutoff` modes. Compiles and links cleanly.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/scinfo.c`, `libgadget/scinfo.h`, `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h` (+ `script/scdetails_raw2bf.py` in the analysis pipeline)

---

## 2026-07-28 — StarClusterDetails: CW-model seed mass for every recorded cluster

The `Mbh_seed` field now carries the CW model's M_VMS for every record when `MbhMscRelationCWmodel=1`, not just for clusters that actually seeded a BH. Previously it was the new BH's subgrid mass for seeded clusters and 0 otherwise; now the clusters recorded via `MinMscForSCdetail` (below `MinMscForBHseed`) and those whose M_VMS fell below `SeedBlackHoleMass` also store the mass the model predicts for them, evaluated from the same cluster mass, size-mass-relation radius and per-cluster metallicity a seeding cluster would use. **This changes nothing about seeding: BHs are still created only for clusters at or above `MinMscForBHseed` whose M_VMS reaches `SeedBlackHoleMass`.** The recorded model mass deliberately has neither the seed-mass floor nor Gate 1 applied, so `Flag` (not `Mbh_seed`) is what says whether a BH particle exists in the run — only `Flag=0` records correspond to one. The seeding path and the record path now share one `cw_seed_mass_code()` helper so the two masses cannot drift apart; verified bit-identical to the previous inline computation over 384k clusters. Measured cost of the extra model evaluations: ~16 us per cluster averaged over the ICMF population (~8.6 us below 1e3 Msun, where the model exits early and returns 0, up to ~270 us above 1e4 Msun), i.e. about 1% of wall time for a 12.5 Mpc/h box to z=9 with `MinMscForSCdetail=0`, dominated by load imbalance since a group's clusters are all evaluated on one rank. Compiles and links cleanly.

**Files modified:** `libgadget/fof.c`, `libgadget/scinfo.c`, `libgadget/scinfo.h` (+ `script/scdetails_raw2bf.py` in the analysis pipeline)

---

## 2026-07-16 — BH seeding: start Mtrack at the seed mass when the seed outweighs its parent

Changed the initial `Mtrack` of a newly seeded BH. Previously it was always the parent particle's mass (the in-place-converted star or gas particle). Now, when the seed BH mass is at or above that parent mass, `Mtrack` starts at the seed mass instead. This removes the birth mass deficit that would otherwise make the BH stochastically swallow neighbouring gas to catch up — a catch-up that overshoots badly, since the deficit is usually a fraction of a gas particle while the smallest available bite is a whole one. The trade-off is that the BH is credited with mass never removed from the gas, so a warning naming the seed mass, the BH ID and the parent mass is emitted on every such seed. Seeds lighter than their parent (the large majority) are unaffected. Applies to every seeding path, though it is a no-op for gas-based seeding in the usual configuration. Compiles and links cleanly.

**Files modified:** `libgadget/blackhole.c`

---

## 2026-07-04 — SecFOFseedsumover: split the secFOF combined-draw seeding controls

Reorganized the features previously bundled in `SeedInSecFOFRandomStarParticle` into two parameters. New int parameter `SecFOFseedsumover` (default 1, only used with `SeedSecFOFcomSample=1`) controls the seed aggregation: 1 = the combined per-secFOF draw is summed into ONE BH seed per group (former `SeedInSecFOFRandomStarParticle=0` behavior); 0 = per-cluster seeding, one BH per sampled cluster >= MinMscForBHseed (former `=1` behavior). `SeedInSecFOFRandomStarParticle` now only selects the host stars of the per-cluster mode: 1 = randomly sampled distinct unseeded stars (as before); 0 = NEW behavior, the N unseeded stars with the largest f(Z)-scaled cluster-forming mass f(Z)*ClusterMass host the N seeds (generalizing the sum-over mode's largest-f(Z)*ClusterMass host pick). It is ignored (with a startup message) when `SecFOFseedsumover=1`. The mutual-exclusion checks moved to the new parameter, and `MbhMscRelationCWmodel=1` now requires `SeedSecFOFcomSample=1` and `SecFOFseedsumover=0` (either host-star mode is allowed). NOTE: existing parameter files that used `SeedInSecFOFRandomStarParticle=1` for per-cluster seeding must now also set `SecFOFseedsumover=0`. Compiles and links cleanly.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`, `libgadget/scinfo.h` (last three: comments only)

---

## 2026-07-04 — CW model: mean-density cap with 1%-of-cluster-mass seeding

Added the mean-density system exclusion of the Williams et al. 2026 model (scmodel.py `rho_mean_cap`, paper Table 1) to the CW seed-mass path: a cluster whose mean density inside r_max, rho_mean = M/(4/3 pi r_max^3), is at or above 6e7 Msun/pc^3 now bypasses the collision-inflow M_VMS calculation and directly gets a BH seed mass of 0.01 x the cluster mass. Clusters below the cap are unchanged. The high-density seed still passes through the existing SeedBlackHoleMass lower-limit check at the call site. Verified with a standalone test (above-cap clusters return exactly 0.01 M_cl; below-cap clusters reproduce the previous CW values); full code compiles and links cleanly.

**Files modified:** `libgadget/cwmodel.c`, `libgadget/cwmodel.h`, `libgadget/fof.c` (comment only)

---

## 2026-07-03 — CWmodelMetallicity: per-cluster metallicity mode for the CW seed-mass model

Added a new string parameter `CWmodelMetallicity` (default `"ave"`, only used with `MbhMscRelationCWmodel=1`) selecting the metallicity fed to the CW model for each sampled cluster. `"ave"` keeps the current behavior (host secFOF's unseeded-star metal mass ratio for every cluster). `"lognormal"` draws each cluster's log10(Z) from a normal distribution with the mean and standard deviation of the group's unseeded-star log10(BirthMetallicity) (equal weight per star, not mass-weighted; log10(Z) floored at -7 for pristine stars; the draw is clipped to the group's [min, max] log10(Z)). `"uniform"` draws log10(Z) uniformly between the group's min and max. Draws are keyed on the host star ID so they are reproducible across ranks/restarts. Any other value aborts at startup when `MbhMscRelationCWmodel=1`. The StarClusterDetails record's metallicity field now stores the Z actually fed to the CW model (unchanged in `"ave"` mode). Compiles and links cleanly.

**Files modified:** `gadget/params.c`, `libgadget/fof.h`, `libgadget/fof.c`

---

## 2026-07-03 — StarClusterDetails: unseeded-star metallicity distribution

Added six new fields to the StarClusterDetails record (`struct SCseedinfo`): the distribution of the host secFOF's unseeded-star BirthMetallicity (equal weight per star, absolute Z) — exact min and max, standard deviation, and the median / 25th / 75th percentiles. Min/max/std come from running per-group accumulators; the percentiles come from a fixed per-group log10(Z) histogram (64 bins, ~0.13 dex; validated against exact sorted percentiles to <0.01 dex), so everything folds into the existing additive group reduction with no new MPI communication and negligible cost. All six are 0 when the host group has no unseeded star. The on-disk record grew to 156 bytes (payload marker 100 -> 148); the raw->BigFile converter `script/scdetails_raw2bf.py` was updated to auto-detect the new layout (and still read the two older ones). Compiles and links cleanly.

**Files modified:** `libgadget/fof.h`, `libgadget/fof.c`, `libgadget/scinfo.h`, `libgadget/scinfo.c` (+ `script/scdetails_raw2bf.py` in the analysis pipeline)

---

## 2026-07-03 — CWmodelAlpha: density profile index of the CW seed-mass model as an input

Made the CW-model density power-law index alpha (rho ~ r^-alpha), previously hard-coded to 1.2, a new `double` parameter `CWmodelAlpha` (default 1.2, only used with `MbhMscRelationCWmodel=1`; aborts at startup if outside (0,3)). The Rose et al. 2020 eccentricity functions f1/f2, which depend on alpha and were frozen as alpha=1.2 constants, are now recomputed at runtime via a Gauss hypergeometric (2F1) implementation so the collision rate stays consistent for any alpha (verified against scipy.special.hyp2f1 to 10 digits; the default alpha=1.2 reproduces the old constants exactly). Eccentricity e stays fixed at 0.5. Compiles and links cleanly.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/cwmodel.c`, `libgadget/cwmodel.h`

---

## 2026-07-02 — StarClusterFixReff: optional fixed effective radius for seeded clusters

Added a new `double` parameter `StarClusterFixReff` (default 0, in pc). When > 0, every seeded star cluster in the `SeedInSecFOFRandomStarParticle=1` path is assigned this fixed effective radius instead of the size-mass relation (which is bypassed along with its 0.5 dex scatter); the fixed Reff feeds both the StarClusterDetails record and the CW seed-mass model. Default 0 preserves the size-mass relation. A negative value aborts at startup. Compiles cleanly.

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`

---

## 2026-07-02 — StarClusterBHDyn=2: warn when MinMscForBHseed < DM particle mass

Added a startup check: when `StarClusterBHDyn=2` and `MinMscForBHseed` is below the dark matter particle mass (header `MassTable[1]`), a warning is printed and the run continues (no abort). This flags the case where BH seeds can be lighter than the background DM particles. Compiles and links cleanly.

**Files modified:** `libgadget/blackhole.c`, `libgadget/blackhole.h`, `libgadget/run.c`

---

## 2026-07-02 — MbhMscRelationCWmodel: disable the f(Z) seeding factor

When `MbhMscRelationCWmodel=1`, the metallicity-dependent seeding factor f(Z) is now disabled at startup by forcing `StarClusterSeedMetallicityMax = StarClusterSeedMetallicityMin` (i.e. f(Z)=1 for all Z, everywhere the factor is used: cluster-formation rate, seeding sums, host-star eligibility). The CW model's own Vink-wind metallicity dependence of M_VMS supplies the Z scaling instead. A startup message is printed when user-set thresholds are overridden. Parameter help text updated. Compiles cleanly.

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`

---

## 2026-07-02 — StarClusterBHDyn=2: frozen seed-cluster-mass dynamical floor

Added mode 2 to `StarClusterBHDyn`. The BH dynamical mass becomes P.Mass = max(Mtrack, init_Msc): the star-cluster mass that seeded the BH acts as a frozen per-BH dynamical-mass floor replacing SeedBHDynMass, and Mtrack takes over once it grows above it. Unlike mode 1, no StarClusterMass payload is attached (no SC stellar evolution, metal return, or merger SC transfer — like mode 0), and the floor never changes: it uses init_Msc, which is frozen at seeding and kept by the accretor through mergers. Non-star-cluster seeds (init_Msc=0) keep the SeedBHDynMass floor. Applied at seeding and at the post-swallow dynamical-mass update; invalid values of StarClusterBHDyn now abort at startup. Parameter help text updated. Compiles cleanly.

**Files modified:** `gadget/params.c`, `libgadget/blackhole.c`, `libgadget/blackhole.h`, `libgadget/bhinfo.c`

---

## 2026-07-02 — StarClusterDetails: Mbh_seed field (+ records for CW-skipped clusters)

Added a `Mbh_seed` field to each StarClusterDetails record: the subgrid mass (code units) of the BH seeded inside that star cluster, read from the just-created BH at all three seeding record sites. When `MbhMscRelationCWmodel=1`, a record is now also written for each sampled cluster that seeds no BH because its M_VMS < SeedBlackHoleMass — there Mbh_seed = 0 and the record's ID/Pos refer to the candidate host star (which remains a star). The record grows from 100 to 108 bytes (payload marker 92 -> 100). The conversion script `script/scdetails_raw2bf.py` now writes the new Mbh_seed block and auto-detects the record layout from the shard payload marker, so detail files written before this change still convert (without the Mbh_seed block). Script verified on synthetic shards in both layouts. Compiles cleanly.

**Files modified:** `libgadget/scinfo.c`, `libgadget/scinfo.h`, `libgadget/fof.c`, `script/scdetails_raw2bf.py` (in the SCmodel script folder)

---

## 2026-07-02 — MbhMscRelationCWmodel: SeedBlackHoleMass as lower seed-mass limit

Follow-up to the `MbhMscRelationCWmodel` feature below. In this mode `BHseedMassScaleMsc` is now explicitly ignored, and `SeedBlackHoleMass` acts as the lower seed-mass limit instead of the seed mass: a BH is only seeded when the model's M_VMS >= SeedBlackHoleMass, so clusters whose VMS is lighter (including the previous M_VMS = 0 no-inflow case) seed no BH. The skipped clusters are counted in the seeding log message. When both `MbhMscRelationCWmodel=1` and `BHseedMassScaleMsc=1` are set, a startup message states that BHseedMassScaleMsc is ignored and the CW model applies. Parameter help text updated. Compiles cleanly.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/blackhole.c`, `libgadget/blackhole.h`

---

## 2026-07-01 — MbhMscRelationCWmodel: Williams et al. 2026 VMS seed-mass model

Added a new parameter `MbhMscRelationCWmodel` (default 0, requires `SeedInSecFOFRandomStarParticle=1`). When enabled, each sampled star cluster above `MinMscForBHseed` gets its BH seed mass from the Williams et al. 2026 stellar-collision VMS model (new module `libgadget/cwmodel.c`, a C port of `code_v2/WilliamModel/scmodel.py` with default parameters and the kappa=5 inflow normalization) instead of the `SeedBlackHoleMass(*m_sc)` prescription. Model inputs per cluster: the sampled cluster mass, the virial radius r_max = 1.4 * Reff from the size-mass relation, the host secFOF's unseeded-star metal mass ratio (Zsun=0.0134, Z/Zsun floored at 1e-4), and the age of the universe at seeding (simulation cosmology). M_VMS is capped at the cluster mass; clusters with no net inflow (M_VMS = 0) seed no BH (counted in the seeding log message). The C port was verified against the Python model to ~1e-11 relative accuracy. Compiles cleanly.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/blackhole.c`, `libgadget/blackhole.h`, `libgadget/Makefile`; **added:** `libgadget/cwmodel.c`, `libgadget/cwmodel.h`

---

## 2026-07-01 — StarClusterDetails: effective-radius field

Added an effective-radius (`Reff`, in pc) field to each StarClusterDetails record. When `SeedInSecFOFRandomStarParticle=1`, every seeded star cluster is assigned a radius sampled from the size-mass relation R_eff = 1.4 pc (M_cl/1e4 Msun)^0.25 with a 0.5 dex lognormal scatter, reproducibly keyed on the host star ID, with log10(R/pc) clipped to [-1, 2] (0.1-100 pc). The cluster mass is converted to solar masses using the same factor as the mass-function thresholds. For the other seeding paths (`SeedInSecFOFRandomStarParticle=0`) the recorded radius is 0.

**Files modified:** `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`, `libgadget/scinfo.c`, `libgadget/scinfo.h`, `libgadget/fof.c`

---

## 2026-07-01 — StarClusterDetails: per-seeded-star-cluster detail files

Added a new parameter `StarClusterDetails` (int, default 0). When it is 1 and a star-cluster BH seeding mode is active (`BlackHoleSeedStarCluster` / `SeedInSecFOFasStarCluster` / `SeedSecFOFcomSample`), the code writes one binary record for every seeded star cluster to per-rank files under `OutputDir/StarClusterDetails`, mirroring the BlackholeDetails mechanism (packed record with leading/trailing size guards). This captures seeding events that happen on steps between checkpoints and are therefore not present in the snapshots. Each record stores: the seed cluster mass (the sampled >1e4 Msun mass in the combined-sample mode, otherwise the mode's seeding SC mass), the seeding scale factor, the host group total star-cluster mass (sum of Gamma*m_star over all member stars), the host group's already-consumed cluster mass (SCMass_seeded, as-is at seed time), the unseeded-star metal mass ratio (mass-weighted BirthMetallicity), and the number of black holes already in the host group before the seed. Records are emitted from all three star-cluster seed paths (single per-group seed, the massive-group extra seeds, and the random-star seeds). All seed paths covered; the per-star `BlackholeSeedSCparticle` path (no host group) is not recorded.

**Files modified:** `gadget/params.c`, `libgadget/stats.c`, `libgadget/stats.h`, `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/Makefile`; **added:** `libgadget/scinfo.c`, `libgadget/scinfo.h`

---

## 2026-06-25 — SeedInSecFOFRandomStarParticle: one BH per sampled cluster at a random star

Added a new parameter `SeedInSecFOFRandomStarParticle` (default 0), only used with `SeedSecFOFcomSample=1` (and mutually exclusive with `SeedSecFOFcomSampleParticle`, `SeedInSecFOFMultipleSeeds`, and `SeedSeedFOFMassiveBoundStar`, enforced at startup). When on, the combined per-secFOF cluster draw is no longer summed into a single seed; instead every sampled cluster with mass >= MinMscForBHseed seeds its own BH (mass SeedBlackHoleMass*m_sc when BHseedMassScaleMsc=1, else SeedBlackHoleMass), each hosted on a randomly chosen distinct unseeded star of the group with a positive metallicity-dependent seeding factor f(Z) (the host pool, seed-cap and Seeded-flagging all require f(Z)>0, consistent with the group sampling mass Sum(f(Z)*ClusterMass); f(Z)=0 stars never host). If the eligible clusters outnumber the group's seedable (f(Z)>0) stars, a message is printed and the remaining (smallest) clusters in that group are skipped. The whole group is then flagged as seeded. Seed placement is distributed across MPI ranks (per-group cluster-mass lists and candidate stars gathered to all ranks, deterministic random selection keyed by star ID). Each BH carries m_sc as its star-cluster mass and the host star's BirthMetallicity. BH slots are pre-reserved via an upper-bound count before placement.

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`, `libgadget/fof.c`

---

## 2026-06-25 — StarClusterICMFcutoff: optional pure power-law cluster mass function

Added a new parameter `StarClusterICMFcutoff` (default 1) controlling the shape of the star-cluster initial cluster mass function used for BH seeding (only relevant under star-cluster-based seeding). When 1 (default, unchanged behavior), the mass function keeps the exponential cutoff n(m) ~ m^-2 exp(-m/Mcut). When 0, the exponential cutoff is dropped and a pure power law p(M) ~ M^-2 is used over the fixed [1e2, 1e8] Msun range, independent of any cutoff mass. The flag affects both the mean-mass (Poisson rate) and the mass draws in the per-star sampler and the combined per-secFOF sampler. Because the power-law mean cluster mass is a cutoff-independent constant, it is precomputed once at init rather than recomputed per seeding event.

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`

---

## 2026-06-17 — BirthMetallicity restart fallback for older snapshots

On restart, if a snapshot lacks the 4/BirthMetallicity block (written before that feature existed), a warning is printed and each star's BirthMetallicity falls back to its current Metallicity instead of being left uninitialized. The missing block is detected from the snapshot read status.

**Files modified:** `libgadget/petaio.c`

---

## 2026-06-17 — metallicity-dependent star-cluster BH seeding factor f(Z)

Added a metallicity-dependent suppression factor f(Z) that multiplies the per-star cluster mass (Gamma*m_star) used for star-cluster BH seeding, based on the star's frozen BirthMetallicity. f(Z) = 1 below a lower threshold, 0 above an upper threshold, with a log-linear decline in between (thresholds given as log10(Z/Zsun), Zsun=0.0134). New parameters StarClusterSeedMetallicityMin/Max (default both 0 = feature disabled). The factor is applied across all star-cluster seeding paths: it scales the Poisson cluster-count rate at star formation (sampled mode), the unseeded seeding sums and seed-particle pick (primary/secondary FOF), the combined-sample per-particle and per-group draws, the bound-massive restriction, the multi-seed eligibility ranking, and the per-star BlackholeSeedSCparticle path. The stored per-star ClusterMass remains the raw Gamma*m_star.

**Files modified:** `gadget/params.c`, `libgadget/sfr_eff.c`, `libgadget/sfr_eff.h`, `libgadget/fof.c`, `libgadget/blackhole.c`

---

## 2026-06-16 14:47 (UTC-4) — analysis notebook star-cluster mass toggle for mass-radius comparison

Updated the subfind/secFOF mass-vs-radius analysis notebook cell with a new switch that can plot either stellar-mass blocks (existing behavior) or star-cluster mass blocks (bound or total) for both catalogs.

## 2026-06-15 00:08 (UTC-4) — analysis notebook bound-threshold FOF plotting helpers

Added two analysis-notebook plotting helpers to visualize only bound-mass-selected structures in one FOF: one for subfind and one for secFOF, both with configurable bound-star-mass threshold (and optional StarPot mode for subfind).

## 2026-06-14 16:49 (CDT) — secFOF bound-star restriction: forbid combining with multi-seeding

`SeedSeedFOFMassiveBoundStar=1` now aborts the run if `SeedInSecFOFMultipleSeeds=1`, since multi-seeding's extra seeds are not bound-aware (only the primary seed would be bound). Checked in both `set_secondfof_params` and the `run.c` init, mirroring the existing parameter validations; parameter help text updated. Compiles and links cleanly.

**Files modified:** `gadget/params.c`, `libgadget/secondfof.c`, `libgadget/run.c`

## 2026-06-14 16:06 (CDT) — secFOF bound-star restriction: seed at a bound star + parallelize the potential

Follow-up to the `SeedSeedFOFMassiveBoundStar` feature below. Two changes: (1) the black-hole seed location is now moved to the largest-Σ(m·Γ) **bound** unseeded star (the seed position, RNG seed, and star ID all track that one bound particle), so a massive group is never seeded at an unbound star; if no unseeded star is bound the seed is dropped. (2) The per-group O(N²) softened-potential computation is now OpenMP-parallelized over members (results unchanged/reproducible), the main performance cost for the rare >1e8 Msun groups this feature targets. Compiles and links cleanly.

**Files modified:** `libgadget/fof.c`

## 2026-06-14 16:20 (UTC-4) — secFOF: bound-star restriction for massive groups (SeedSeedFOFMassiveBoundStar)

Added a new int parameter `SeedSeedFOFMassiveBoundStar` (default 0), used with `SeedSecFOFcomSample=1`. When set to 1, any secondary-FOF group whose unseeded star-cluster mass Σ(m·Γ) exceeds the 1e8 Msun threshold is restricted to the unseeded star particles that are gravitationally bound to the secFOF (softened potential summed over all members, rest frame = deepest-potential member's velocity) before seeding; only the bound unseeded stars' Σ(m·Γ) and stellar mass feed the combined sampler, so the seed decision and the seed mass (under `BHseedMassScaleMsc=1`) use the bound subset. Groups below the threshold and runs with the feature off are unchanged. Incompatible with `SeedSecFOFcomSampleParticle=1` (hard error); requires `SeedSecFOFcomSample=1`. Compiles cleanly.

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/secondfof.c`, `libgadget/secondfof.h`, `libgadget/run.c`

## 2026-06-09 16:58 (UTC-5) — analysis notebook axis conversion warning fix

Updated the analysis notebook plotting conversion so age/redshift axis mapping is consistent and no longer triggers the astropy bracketing warning during figure rendering.

## 2026-06-09 18:56 (UTC-5) — analysis notebook SU log stitching robustness fix

Updated the analysis notebook log-stitching helper to safely handle restart files with identical starting points and avoid empty-slice reduction failures while combining SU history segments.

## 2026-06-09 — secFOF: option to link only unseeded star particles

**Branch:** SecFOFCombined

Added a new int parameter `SecFOFUnseededPart` (default 0). When set to 1, the second FOF uses only unseeded star particles as primary-linking particles; star particles that have already seeded a black hole are dropped from the primary-linking set (other configured primary types, e.g. gas, are unaffected). Seeded stars no longer anchor or join secondary-FOF groups, and are excluded from the primary-length count and the group potential-minimum. `SecFOFUnseededPart=1` requires `StarClusterOn=1`, otherwise the run aborts with an error (checked in both `set_secondfof_params` and the `run.c` init, mirroring the other secFOF parameter validations).

**Files modified:** `gadget/params.c`, `libgadget/fof.c`, `libgadget/fof.h`, `libgadget/secondfof.c`, `libgadget/run.c`

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

## 2026-06-16 — add frozen BirthMetallicity to star particles

Added a new per-star-particle property `BirthMetallicity` that records the total metallicity of the parent gas at the moment of star formation. It is set once when the star forms and is never modified afterwards (frozen), so it preserves the natal metallicity even as other quantities evolve. The field is written to the output catalogs as the `4/BirthMetallicity` block and appears in PART, PIG, and SecPIG snapshots.

**Files modified:** `libgadget/slotsmanager.h`, `libgadget/sfr_eff.c`, `libgadget/petaio.c`

---

## 2026-06-17 — add debug-only BHNgbAtSeeding black-hole property

Added a new per-black-hole property `BHNgbAtSeeding` that records the number of black-hole particles already present in the host secondary-FOF group (or FOF halo) at the moment the black hole was seeded, excluding the seed itself. It is only populated for the secondary-FOF star-cluster seeding path (`SeedInSecFOFasStarCluster=1`); all other seeding paths record 0. The value is frozen at creation (never modified by mergers). It is a debug-only, write-only field: it appears as the `5/BHNgbAtSeeding` block in PART, PIG, and SecPIG snapshots only when `OutputDebugFields=1`, and is not read back on restart.

**Files modified:** `libgadget/slotsmanager.h`, `libgadget/blackhole.h`, `libgadget/blackhole.c`, `libgadget/fof.c`, `libgadget/petaio.c`

---

## TODO

- Allow seeding in primary FOF and secondary FOF to be on in the same run.
- SecPIG particle catalog: currently forces a separate partmanager copy (no PartManager reuse) to avoid GrNr/SecGrNr corruption during the save-restore cycle in `secondfof_write`. If star fractions grow large at low redshift and the >25% threshold is hit, this will use significantly more memory. Implement proper PartManager-reuse support for SecPIG (handle domain exchange and avoid the saved-array restore) when this becomes an issue.
