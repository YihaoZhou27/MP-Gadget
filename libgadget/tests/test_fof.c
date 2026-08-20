/*Simple test for the exchange function*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#define qsort_openmp qsort

#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include "stub.h"

static struct ClockTable CT;

#define NUMPART1 8
static int
setup_particles(int NumPart, double BoxSize)
{

    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    particle_alloc_memory(PartManager, BoxSize, 1.5 * NumPart);
    PartManager->NumPart = NumPart;

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);

    int64_t newSlots[6] = {128, 0, 0, 0, 128, 128};
    slots_reserve(1, newSlots, SlotsManager);
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        P[i].ID = i + PartManager->NumPart * ThisTask;
        /* DM only*/
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
        int j;
        for(j=0; j<3; j++) {
            P[i].Pos[j] = BoxSize * (j+1) * P[i].ID / (PartManager->NumPart * NTask);
            while(P[i].Pos[j] > BoxSize)
                P[i].Pos[j] -= BoxSize;
        }
    }
    fof_init(BoxSize/cbrt(PartManager->NumPart));
    /* TODO: Here create particles in some halo-like configuration*/
    return 0;
}

static void
test_fof(void **state)
{
    int NTask;
    walltime_init(&CT);

    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    set_fof_testpar(1, 0.2, 5);
    init_forcetree_params(0.7);

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int NumPart = 512*512 / NTask;
    /* 20000 kpc*/
    double BoxSize = 20000;
    setup_particles(NumPart, BoxSize);

    /* Build a tree and domain decomposition*/
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);

    /* Example assertion: this checks that the groups were allocated. */
    assert_all_true(fof.Group);
    assert_true(fof.TotNgroups == 1);
    /* Assert some more things about the particles,
     * maybe checking the halo properties*/

    fof_finish(&fof);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(P);
    return;
}

/* ===================================================================
 * Grid primary-linker tests.
 *
 * Design note: the problem must be a GLOBAL one, identical at every rank
 * count, or "grid == treewalk at 1/2/4 ranks" says nothing about
 * decomposition independence -- it only compares two linkers on three
 * different problems.  Two things are therefore deliberate here:
 *
 *  - fof_init() is given the mean separation of the GLOBAL particle count,
 *    not PartManager->NumPart (which is the local count and would change
 *    the linking length with rank count);
 *  - positions are a pure function of the GLOBAL particle index via a
 *    splitmix64 hash, so there is no RNG stream to skip and no assumption
 *    about how many draws each particle consumes.
 *
 * The clumps are also placed on a lattice with a known membership, so the
 * expected answer is known a priori: the tests assert the ABSOLUTE result
 * (one group per clump, every particle grouped), not merely that two
 * implementations agree with each other.
 * =================================================================== */

#define NLAT 3                       /* clumps per axis */
#define NCLUMP (NLAT * NLAT * NLAT)  /* 27 well-separated clumps */
/* Kept small on purpose: the treewalk reference costs O(N * n_ngb) and every
 * test runs it two or three times. 20000 still gives ~740 particles per clump,
 * far above MinLength, so no coverage depends on the size. */
#define NPART_GLOBAL 20000
#define TESTBOX 20000.0

/* Deterministic value in [0,1) from a global index. Pure function: identical
 * on every rank and at every rank count. */
static double tfof_rand(uint64_t idx, uint64_t stream)
{
    uint64_t z = idx * 0x9E3779B97F4A7C15UL + stream * 0xBF58476D1CE4E5B9UL + 0x2545F4914F6CDD1DUL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z =  z ^ (z >> 31);
    return (double) (z >> 11) / 9007199254740992.0;
}

static double tfof_wrap(double x, double BoxSize)
{
    while(x >= BoxSize) x -= BoxSize;
    while(x < 0) x += BoxSize;
    return x;
}

/* Global particle g belongs to clump (g % NCLUMP). Clump centres sit on an
 * NLAT^3 lattice, spacing BoxSize/NLAT (>> l), and the clump at lattice site
 * (0,0,0) is centred on the origin so it straddles the periodic boundary in
 * all three dimensions. Clump radius is l/10, so every member is well within
 * l of every other and each clump must come out as exactly one group. */
static void tfof_position(int64_t g, double BoxSize, double l, double pos[3])
{
    const int c = (int) (g % NCLUMP);
    const double spacing = BoxSize / NLAT;
    const double cen[3] = { (c % NLAT) * spacing,
                            ((c / NLAT) % NLAT) * spacing,
                            (c / (NLAT * NLAT)) * spacing };
    int j;
    for(j = 0; j < 3; j++)
        pos[j] = tfof_wrap(cen[j] + (l / 10.0) * (2 * tfof_rand((uint64_t) g, j) - 1), BoxSize);
}

/* Rank of the star at global index g among ALL star particles, counting only
 * the ones in the first star_clumps clumps. Pure function of g: g/NCLUMP whole
 * cycles have passed, each contributing star_clumps stars, plus however many of
 * the current cycle precede this residue. */
static int64_t tfof_star_ordinal(int64_t g, int star_clumps)
{
    const int64_t r = g % NCLUMP;
    return (g / NCLUMP) * star_clumps + (r < star_clumps ? r : star_clumps);
}

/* star_clumps: how many of the NCLUMP clumps become type 4.
 * seed_every: flag every seed_every'th star (by global ordinal) as Seeded. */
struct tfof_opts { int star_clumps; int seed_every; int garbage_one; int swallow_one; };

static double
setup_particles_global(double BoxSize, struct tfof_opts o)
{
    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    /* Even split of a FIXED global count. */
    const int64_t base = NPART_GLOBAL / NTask;
    const int64_t rem = NPART_GLOBAL % NTask;
    const int64_t nloc = base + (ThisTask < rem ? 1 : 0);
    const int64_t gstart = ThisTask * base + (ThisTask < rem ? ThisTask : rem);

    particle_alloc_memory(PartManager, BoxSize, 1.5 * (nloc > 1 ? nloc : 1));
    PartManager->NumPart = nloc;

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    int64_t newSlots[6] = {128, 0, 0, 0, nloc + 128, 128};
    slots_reserve(1, newSlots, SlotsManager);

    /* Linking length from the GLOBAL mean separation, so it does not move
     * with rank count. */
    const double meansep = BoxSize / cbrt((double) NPART_GLOBAL);
    fof_init(meansep);
    double ll;
    {
        int pl, sl, ml, pm, mpl;
        fof_get_params(&pl, &sl, &ll, &ml, &pm, &mpl);
    }

    int64_t i;
    for(i = 0; i < nloc; i++) {
        const int64_t g = gstart + i;
        P[i].ID = (MyIDType) g;
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
        P[i].Swallowed = 0;
        P[i].PI = -1;
        double p[3];
        tfof_position(g, BoxSize, ll, p);
        int j;
        for(j = 0; j < 3; j++) P[i].Pos[j] = p[j];
    }
    /* Convert the first o.star_clumps clumps to stars. Because clumps are
     * spatially localised, most ranks end up owning ZERO stars once the
     * domain is decomposed -- which is the empty-rank case. */
    if(o.star_clumps > 0) {
        for(i = 0; i < nloc; i++) {
            const int64_t g = gstart + i;
            if((int) (g % NCLUMP) >= o.star_clumps) continue;
            slots_convert(i, 4, -1, PartManager, SlotsManager);
            /* Seeded must be a pure function of the GLOBAL index. A per-rank
             * running counter would make the seeded subset depend on how the
             * global list is split, reintroducing exactly the rank-dependence
             * the positions were fixed for. */
            STARP(i).Seeded =
                (o.seed_every > 0 &&
                 (tfof_star_ordinal(g, o.star_clumps) % o.seed_every) == 0) ? 1 : 0;
        }
    }
    if(o.garbage_one && nloc > 0) P[0].IsGarbage = 1;
    if(o.swallow_one && nloc > 0) P[0].Swallowed = 1;
    return ll;
}

static void tfof_domain_par(void)
{
    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    init_forcetree_params(0.7);
}

/* Run fof_fof under a given linker mode and return the group count. */
static int64_t tfof_run(DomainDecomp * dd, int mode, int * path)
{
    fof_set_grid_linking(mode);
    FOFGroups f = fof_fof(dd, 1, MPI_COMM_WORLD);
    const int64_t n = f.TotNgroups;
    if(path) *path = fof_get_grid_path_taken();
    fof_finish(&f);
    fof_set_grid_linking(0);
    return n;
}

/* DM primaries. Absolute answer known: one group per clump. */
static void
test_fof_grid(void **state)
{
    walltime_init(&CT);
    tfof_domain_par();
    set_fof_testpar(1, 0.2, 5);
    struct tfof_opts o = {0, 0, 0, 0};
    setup_particles_global(TESTBOX, o);

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    int path = -1;
    const int64_t n_tree = tfof_run(&ddecomp, 0, &path);
    assert_true(path == 0);
    const int64_t n_grid = tfof_run(&ddecomp, 1, &path);
    assert_true(path == 1);
    const int64_t n_ab   = tfof_run(&ddecomp, 2, &path);   /* endruns on mismatch */
    assert_true(path == 1);

    /* Absolute expectation, independent of rank count: exactly one group per
     * clump. This is what makes the test decomposition-independent -- the
     * expected value is a property of the problem, not of a reference run. */
    assert_true(n_tree == NCLUMP);
    assert_true(n_grid == NCLUMP);
    assert_true(n_ab == NCLUMP);

    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(P);
}

/* Type-4 primaries with secondary attachment, seeded-star exclusion, and
 * ranks that own no primary particles at all. */
static void
test_fof_grid_stars(void **state)
{
    walltime_init(&CT);
    tfof_domain_par();
    set_fof_testpar(1, 0.2, 5);
    /* Stars in 2 of the 27 clumps: localised, so most ranks own no star. */
    struct tfof_opts o = {2, 0, 0, 0};
    const double ll = setup_particles_global(TESTBOX, o);

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    /* primary = star (16), secondary = DM (2): exercises fof_label_secondary
     * attaching non-primary particles to the grid's components. */
    fof_set_params(16, 2, ll, 5, 0, 0);

    int path = -1;
    const int64_t n_tree = tfof_run(&ddecomp, 0, &path);
    assert_true(path == 0);
    const int64_t n_grid = tfof_run(&ddecomp, 1, &path);
    assert_true(path == 1);
    const int64_t n_ab   = tfof_run(&ddecomp, 2, &path);
    assert_true(path == 1);

    assert_true(n_tree == 2);      /* one group per star clump */
    assert_true(n_grid == n_tree);
    assert_true(n_ab == n_tree);

    /* Now drop seeded stars from the primary set and re-check. Every 3rd star
     * is seeded, so the groups keep their shape but the anchors change. */
    slots_free(SlotsManager);
    myfree(P);
    domain_free(&ddecomp);

    struct tfof_opts o2 = {2, 3, 0, 0};
    const double ll2 = setup_particles_global(TESTBOX, o2);

    /* The seeded subset must be a property of the GLOBAL problem, not of how it
     * was split across ranks. Compare the actual global counts against the pure
     * function evaluated over the whole index range: a per-rank running counter
     * would pass at 1 rank and diverge at 2 or 4. */
    {
        int nstar_l = 0, nseed_l = 0, i2;
        for(i2 = 0; i2 < PartManager->NumPart; i2++) {
            if(P[i2].Type != 4) continue;
            nstar_l++;
            if(STARP(i2).Seeded) nseed_l++;
        }
        int nstar = 0, nseed = 0;
        MPI_Allreduce(&nstar_l, &nstar, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&nseed_l, &nseed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        int e_star = 0, e_seed = 0;
        int64_t g;
        for(g = 0; g < NPART_GLOBAL; g++) {
            if((int) (g % NCLUMP) >= o2.star_clumps) continue;
            e_star++;
            if(tfof_star_ordinal(g, o2.star_clumps) % o2.seed_every == 0) e_seed++;
        }
        assert_true(nstar == e_star);
        assert_true(nseed == e_seed);
    }

    DomainDecomp dd2 = {0};
    domain_decompose_full(&dd2, MPI_COMM_WORLD);
    fof_set_params(16, 2, ll2, 5, 0, 0);
    fof_set_primary_unseeded_only(1);

    const int64_t s_tree = tfof_run(&dd2, 0, &path);
    const int64_t s_grid = tfof_run(&dd2, 1, &path);
    assert_true(path == 1);
    const int64_t s_ab   = tfof_run(&dd2, 2, &path);
    assert_true(s_grid == s_tree);
    assert_true(s_ab == s_tree);

    fof_set_primary_unseeded_only(0);
    domain_free(&dd2);
    slots_free(SlotsManager);
    myfree(P);
}

/* A garbage or swallowed primary particle must send the whole run to the
 * treewalk collectively -- the treewalk treats those asymmetrically and the
 * grid deliberately refuses to reproduce that. */
static void
test_fof_grid_exceptional(void **state)
{
    walltime_init(&CT);
    tfof_domain_par();
    set_fof_testpar(1, 0.2, 5);

    /* swallowed */
    struct tfof_opts o = {0, 0, 0, 1};
    setup_particles_global(TESTBOX, o);
    DomainDecomp dd = {0};
    domain_decompose_full(&dd, MPI_COMM_WORLD);
    int path = -1;
    const int64_t n_sw = tfof_run(&dd, 1, &path);
    assert_true(path == 0);            /* fell back, did NOT run the grid */
    assert_true(n_sw == NCLUMP);
    domain_free(&dd);
    slots_free(SlotsManager);
    myfree(P);

    /* Garbage. The flag has to be set AFTER the decomposition: the particle
     * exchange garbage-collects IsGarbage particles (exchange.c, domain.c), so
     * one marked beforehand simply does not survive to reach fof_fof -- which
     * is itself worth knowing, because it means the garbage half of the gate is
     * only reachable if something marks a particle between the decomposition
     * and the FOF. */
    struct tfof_opts o2 = {0, 0, 0, 0};
    setup_particles_global(TESTBOX, o2);
    DomainDecomp dd2 = {0};
    domain_decompose_full(&dd2, MPI_COMM_WORLD);
    if(PartManager->NumPart > 0) P[0].IsGarbage = 1;
    const int64_t n_gb = tfof_run(&dd2, 1, &path);
    assert_true(path == 0);
    assert_true(n_gb == NCLUMP);
    if(PartManager->NumPart > 0) P[0].IsGarbage = 0;
    domain_free(&dd2);
    slots_free(SlotsManager);
    myfree(P);
}

/* Hand-placed geometry: exact linking-length boundaries, the periodic wrap,
 * and a genuine (+-2,+-2,+-2) stencil-corner displacement.
 * Small enough that the expected grouping is written down by hand.
 *
 * As in the other tests the particle set is GLOBAL and split evenly across
 * ranks. Piling it all on rank 0 is not an option: domain_decompose_full
 * cannot build a top tree from one rank holding every particle in tight
 * blobs, and aborts with "unreasonably large" TopNodeAllocFactor. The
 * zero-primary-particles-on-a-rank case is covered by test_fof_grid_stars,
 * where the stars occupy only 2 of the 27 clumps. */
#define GEOBOX 100.0
#define GEOL 10.0
#define GEONPAIR 6
/* Filler: a 3x3x1 slab of blobs at z = GEOBLOBZ, well clear of every test pair
 * in z. Numerous and diffuse enough (radius 2 against L = 10) that the domain
 * top-tree can resolve them; at 40 particles of radius 0.5 the density contrast
 * exhausted TopNodeAllocFactor in domain_decompose_full. Lattice spacing 33
 * against radius 2 leaves 29 between the nearest particles of two blobs, so
 * each blob is exactly one group. */
#define GEOBLOB 9
#define GEOPERBLOB 400
#define GEOBLOBR 2.0
#define GEOBLOBZ 80.0
#define GEOFILL (GEOBLOB * GEOPERBLOB)
#define GEONTOT (2 * GEONPAIR + GEOFILL)

/* Cell size the linker will actually use. Replicates fof_grid_geometry()
 * exactly, because the corner pair below has to be placed relative to the REAL
 * cell size: for GEOBOX/GEOL the grid is 18 cells per axis and a = 5.5556, NOT
 * the conceptual l/sqrt(3) = 5.7735. Placing against the latter put the "corner"
 * pair only ONE cell apart, so the (+-2,+-2,+-2) offsets were never exercised. */
static double tfof_cell_size(double l, double BoxSize)
{
    const double dm = l * (1.0 - 1e-12);
    const double target = dm / sqrt(3.0);
    int64_t ncell = (int64_t) ceil(BoxSize / target);
    double a = BoxSize / ncell;
    while(!(sqrt(3.0) * a < dm)) { ncell++; a = BoxSize / ncell; }
    return a;
}

/* Position of global geometry-test particle g.
 *
 * Two DISTINCT guards live here and are easy to conflate:
 *   - the inclusive particle-distance bound (r2 <= L^2, not <): pair 0 only;
 *   - the (+-2,+-2,+-2) offsets being present in the stencil: pair 4 only.
 * Neither pair tests the other's property.
 *
 *  pair 0: exactly L apart along x                -> LINKED, and ONLY if the
 *          distance bound accepts equality. This is the sole inclusive-bound
 *          test; an exclusive r2 < L^2 separates it and changes the count.
 *  pair 1: just under L                           -> LINKED
 *  pair 2: just over L                            -> SEPARATE
 *  pair 3: just under L, across the periodic wrap -> LINKED
 *  pair 4: cell displacement EXACTLY (2,2,2)      -> LINKED, and reachable only
 *          via the (+-2,+-2,+-2) stencil corners. Placed at the far corner of
 *          cell (i,i,i) and the near corner of cell (i+2,i+2,i+2): their
 *          separation is just above sqrt(3)*a = 9.6225 and just below L, the
 *          only window in which such a pair can exist. Drop the corners from
 *          the stencil and this pair stops linking and the count changes.
 *          It does NOT test the inclusive bound: at 9.661 it is STRICTLY below
 *          L, so r2 < L^2 would link it just as well.
 *  pair 5: body diagonal just over L              -> SEPARATE
 * Bases are >= 25 apart in some axis so pairs cannot link into each other, and
 * all sit at z <= 56 against filler at z = 80 +- 2. Both separations are
 * asserted in the test rather than trusted. */
static void tfof_geom_position(int g, double pos[3])
{
    const double d3 = GEOL / sqrt(3.0);
    /* Corner pair, built from the REAL cell size: far corner of cell (i,i,i)
     * to near corner of cell (i+2,i+2,i+2). i = 8 keeps it clear of the others. */
    const double a = tfof_cell_size(GEOL, GEOBOX);
    const double ceps = 0.002 * a;
    const double cA = 9 * a - ceps;          /* inside cell 8  */
    const double cB = 10 * a + ceps;         /* inside cell 10 */

    const double base[GEONPAIR][3] = {
        {10, 10, 10}, {10, 10, 45}, {45, 10, 10},
        {0.5, 45, 10}, {cA, cA, cA}, {10, 45, 45}
    };
    const double off[GEONPAIR][3] = {
        {GEOL, 0, 0},
        {GEOL * (1 - 1e-9), 0, 0},
        {GEOL * (1 + 1e-6), 0, 0},
        {-GEOL * (1 - 1e-9), 0, 0},
        {cB - cA, cB - cA, cB - cA},
        {d3 * (1 + 1e-6), d3 * (1 + 1e-6), d3 * (1 + 1e-6)}
    };
    int j;
    if(g < 2 * GEONPAIR) {
        const int k = g / 2, m = g % 2;
        for(j = 0; j < 3; j++)
            pos[j] = tfof_wrap(base[k][j] + (m ? off[k][j] : 0.0), GEOBOX);
    }
    else {
        const int f = g - 2 * GEONPAIR;
        const int b = f / GEOPERBLOB;
        const double s = 33.0;
        const double bc[3] = { 5.0 + (b % 3) * s, 5.0 + ((b / 3) % 3) * s, GEOBLOBZ };
        for(j = 0; j < 3; j++)
            pos[j] = tfof_wrap(bc[j] + GEOBLOBR * (2 * tfof_rand((uint64_t) f, j) - 1), GEOBOX);
    }
}

static void
test_fof_grid_geometry(void **state)
{
    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    walltime_init(&CT);
    tfof_domain_par();
    set_fof_testpar(1, 0.2, 2);

    /* Design invariant: no filler may come within L of a test-pair particle,
     * or the expected group count silently changes. Checked rather than
     * reasoned about on paper -- it is easy to move a base and not notice. */
    {
        int k, f;
        for(k = 0; k < 2 * GEONPAIR; k++) {
            double pk[3];
            tfof_geom_position(k, pk);
            for(f = 2 * GEONPAIR; f < GEONTOT; f++) {
                double pf[3];
                tfof_geom_position(f, pf);
                double r2 = 0;
                int j;
                for(j = 0; j < 3; j++) {
                    const double dd = NEAREST(pk[j] - pf[j], GEOBOX);
                    r2 += dd * dd;
                }
                assert_true(r2 > GEOL * GEOL);
            }
        }
    }

    const int64_t nbase = GEONTOT / NTask;
    const int64_t nrem = GEONTOT % NTask;
    const int64_t nloc = nbase + (ThisTask < nrem ? 1 : 0);
    const int64_t gstart = ThisTask * nbase + (ThisTask < nrem ? ThisTask : nrem);

    particle_alloc_memory(PartManager, GEOBOX, GEONTOT + 16);
    PartManager->NumPart = nloc;
    slots_init(16, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    int64_t newSlots[6] = {16, 0, 0, 0, 16, 16};
    slots_reserve(1, newSlots, SlotsManager);

    int64_t i;
    for(i = 0; i < nloc; i++) {
        const int g = (int) (gstart + i);
        double p[3];
        tfof_geom_position(g, p);
        int j;
        for(j = 0; j < 3; j++) P[i].Pos[j] = p[j];
        P[i].ID = (MyIDType) g;
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
        P[i].Swallowed = 0;
        P[i].PI = -1;
    }

    fof_init(GEOBOX / cbrt((double) GEONTOT));
    fof_set_params(2, 0, GEOL, 2, 0, 0);

    /* The corner pair must land at a cell displacement of EXACTLY (2,2,2), or
     * it is not testing the stencil corner at all -- an earlier version used
     * the conceptual cell size l/sqrt(3) instead of the real one and only
     * asserted ">= 1", so the pair sat one cell apart and the corner offsets
     * were never exercised. Compute the real cell indices and check them. */
    {
        const double a = tfof_cell_size(GEOL, GEOBOX);
        const int64_t nc = (int64_t) (GEOBOX / a + 0.5);
        double pa[3], pb[3];
        tfof_geom_position(8, pa);      /* pair 4, first particle  */
        tfof_geom_position(9, pb);      /* pair 4, second particle */
        double r2 = 0;
        int j;
        for(j = 0; j < 3; j++) {
            int64_t ia = (int64_t) floor(pa[j] / a) % nc;
            int64_t ib = (int64_t) floor(pb[j] / a) % nc;
            if(ia < 0) ia += nc;
            if(ib < 0) ib += nc;
            assert_true(ib - ia == 2);
            const double dd = NEAREST(pa[j] - pb[j], GEOBOX);
            r2 += dd * dd;
        }
        /* It must also actually be a link: between sqrt(3)*a and L. */
        assert_true(r2 <= GEOL * GEOL);
        assert_true(r2 > 3.0 * a * a * (1 - 1e-9));
    }

    DomainDecomp dd = {0};
    domain_decompose_full(&dd, MPI_COMM_WORLD);

    int path = -1;
    const int64_t g_tree = tfof_run(&dd, 0, &path);
    const int64_t g_grid = tfof_run(&dd, 1, &path);
    assert_true(path == 1);
    const int64_t g_ab   = tfof_run(&dd, 2, &path);

    /* pairs 0,1,3,4 link (4 groups of 2); pairs 2,5 do not and their singletons
     * fall below MinLength=2, so they vanish; plus GEOBLOB filler blobs. */
    const int64_t expect = 4 + GEOBLOB;   /* 4 linked pairs + 9 filler blobs */
    assert_true(g_tree == expect);
    assert_true(g_grid == expect);
    assert_true(g_ab == expect);

    domain_free(&dd);
    slots_free(SlotsManager);
    myfree(P);
}
/* A linking length too small to grid must fall back to the treewalk, NOT
 * abort. The cell key is int64, so a box needing more than ~2.1e6 cells per
 * axis cannot be addressed; a second-FOF linking length of 0.01 code units in a
 * 12500 box needs 2165064 and is reachable from a valid configuration. The
 * geometry check lives in the collective preflight for exactly this reason --
 * if it ever migrates back into the linker, this test aborts the suite. */
static void
test_fof_grid_ungriddable(void **state)
{
    walltime_init(&CT);
    tfof_domain_par();
    set_fof_testpar(1, 0.2, 5);
    struct tfof_opts o = {0, 0, 0, 0};
    setup_particles_global(TESTBOX, o);

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    int pl, sl, ml, pm, mpl;
    double ll;
    fof_get_params(&pl, &sl, &ll, &ml, &pm, &mpl);

    /* 20000 / (1e-3/sqrt(3)) needs 3.5e7 cells per axis. */
    fof_set_params(pl, sl, 1e-3, ml, pm, mpl);
    int path = -1;
    const int64_t n_grid = tfof_run(&ddecomp, 1, &path);
    assert_true(path == 0);                 /* fell back rather than aborting */
    const int64_t n_tree = tfof_run(&ddecomp, 0, &path);
    assert_true(n_grid == n_tree);

    fof_set_params(pl, sl, ll, ml, pm, mpl);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(P);
}

/* Half-period aliasing: when 2R == ncell, periodic wrapping makes two in-range
 * stencil offsets congruent, so the canonical half stencil no longer visits
 * each unordered pair exactly once. l = 30 in a box of 100 lands exactly there
 * (ncell = 6, a = 16.667, R = 3), which no other test reaches -- the production
 * config is ncell 8874 against 2R = 6. Aliasing never changed the answer, only
 * the work, so what this guards is that the exactly-once fallback did not break
 * the answer while removing the redundant probes. */
static void
test_fof_grid_alias(void **state)
{
    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    walltime_init(&CT);
    tfof_domain_par();
    set_fof_testpar(1, 0.2, 2);

    /* Reuse the geometry-test filler: 9 well-separated blobs, but now at a
     * linking length where the whole box is only 6 cells across. At l = 30 the
     * blobs (spacing 33, radius 2) are still separate: nearest particles of two
     * blobs are 29 apart... which is NOT > 30, so at this l they merge. Use a
     * uniform lattice instead, spacing 33 > l = 30, one particle per site is
     * below MinLength, so build small tight groups at each site. */
    const int nsite = 27, nper = 150;
    const int ntot = nsite * nper;
    const int64_t nb = ntot / NTask, nr = ntot % NTask;
    const int64_t nloc = nb + (ThisTask < nr ? 1 : 0);
    const int64_t gstart = ThisTask * nb + (ThisTask < nr ? ThisTask : nr);

    particle_alloc_memory(PartManager, GEOBOX, ntot + 16);
    PartManager->NumPart = nloc;
    slots_init(16, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    int64_t rs[6] = {16, 0, 0, 0, 16, 16};
    slots_reserve(1, rs, SlotsManager);

    const double s = GEOBOX / 3.0;    /* 33.3 site spacing */
    int64_t i;
    for(i = 0; i < nloc; i++) {
        const int64_t g = gstart + i;
        const int b = (int) (g % nsite);
        const double bc[3] = { (b % 3) * s, ((b / 3) % 3) * s, (b / 9) * s };
        int j;
        for(j = 0; j < 3; j++)
            P[i].Pos[j] = tfof_wrap(bc[j] + 1.0 * (2 * tfof_rand((uint64_t) g, j) - 1), GEOBOX);
        P[i].ID = (MyIDType) g;
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].IsGarbage = 0;
        P[i].Swallowed = 0;
        P[i].PI = -1;
    }
    fof_init(GEOBOX / cbrt((double) ntot));
    /* l = 90 in a box of 100 gives ncell = 2, a = 50, R = 1 -- the smallest
     * grid the geometry can produce, and the only regime where aliasing
     * actually survives the gap filter: with ncell = 2 the offsets +1 and -1
     * are congruent, so one cell reaches the same neighbour through several
     * stencil entries. (l = 30 / ncell = 6 does NOT reach it: offsets at
     * |d| = ncell/2 have gap = 2a > l and are rejected before they can alias.)
     * l exceeds the largest periodic separation sqrt(3)*50 = 86.6, so the
     * correct answer is a single group -- degenerate, but it is the answer the
     * treewalk gives and the grid must agree. */
    fof_set_params(2, 0, 90.0, 2, 0, 0);

    DomainDecomp dd = {0};
    domain_decompose_full(&dd, MPI_COMM_WORLD);

    int path = -1;
    const int64_t n_tree = tfof_run(&dd, 0, &path);
    const int64_t n_grid = tfof_run(&dd, 1, &path);
    assert_true(path == 1);
    const int64_t n_ab = tfof_run(&dd, 2, &path);   /* endruns on any mismatch */
    assert_true(n_grid == n_tree);
    assert_true(n_ab == n_tree);
    assert_true(n_tree == 1);   /* l > max periodic separation: all one group */

    domain_free(&dd);
    slots_free(SlotsManager);
    myfree(P);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fof),
        cmocka_unit_test(test_fof_grid),
        cmocka_unit_test(test_fof_grid_stars),
        cmocka_unit_test(test_fof_grid_exceptional),
        cmocka_unit_test(test_fof_grid_geometry),
        cmocka_unit_test(test_fof_grid_alias),
        cmocka_unit_test(test_fof_grid_ungriddable),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
