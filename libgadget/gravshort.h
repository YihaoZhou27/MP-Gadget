#ifndef GRAVSHORT_H
#define GRAVSHORT_H

#include "partmanager.h"
#include "treewalk.h"
#include "gravity.h"
#include "tidalfield.h"
#include "slotsmanager.h"
#include <math.h>

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterGravShort;

typedef struct
{
    TreeWalkQueryBase base;
    MyFloat OldAcc;
    int Type;           /* Particle type: used to skip tidal for non-gas */
} TreeWalkQueryGravShort;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
    MyFloat Potential;
    MyFloat TidalTensor[6]; /* Tidal tensor: xx, yy, zz, xy, xz, yz (only used when tidal enabled) */
} TreeWalkResultGravShort;

struct GravShortPriv {
    /* Size of a PM cell, in internal units. Box / Nmesh */
    double cellsize;
    /* How many PM cells do we go
     * before we stop calculating the tree?*/
    double Rcut;
    /* Newton's constant in internal units*/
    double G;
    inttime_t Ti_Current;
    /* Matter density in internal units.
     * rho_0 = Omega0 * rho_crit
     * rho_crit = 3 H^2 /(8 pi G).
     * This is (rho_0)^(1/3) ,
     * Note: should account for
     * massive neutrinos, but doesn't. */
    double cbrtrho0;
    /* Pointer to the place to store accelerations*/
    MyFloat (*Accel)[3];
    /* Tidal tensor storage, NULL if tidal field not computed.
     * Indexed by particle index, 6 components per particle. */
    MyFloat (*TidalTensorStore)[6];
    /* Per-type tidal flags: which particle types need tidal computation */
    int TidalGas; /* Compute tidal for gas (Type 0) */
    int TidalBH;  /* Compute tidal for BH (Type 5) */
    double atime; /* scale factor of this walk: stamps the BH tidal field (TidalFieldAtime) */
};

#define GRAV_GET_PRIV(tw) ((struct GravShortPriv *) ((tw)->priv))

static void
grav_short_postprocess(int i, TreeWalk * tw)
{
    double G = GRAV_GET_PRIV(tw)->G;
    GRAV_GET_PRIV(tw)->Accel[i][0] *= G;
    GRAV_GET_PRIV(tw)->Accel[i][1] *= G;
    GRAV_GET_PRIV(tw)->Accel[i][2] *= G;

    if(tw->tree->full_particle_tree_flag) {
        /* On a PM step, update the stored full tree grav accel for the next PM step.
         * Needs to be done here so internal treewalk iterations don't get a partial acceleration.*/
        P[i].FullTreeGravAccel[0] = GRAV_GET_PRIV(tw)->Accel[i][0];
        P[i].FullTreeGravAccel[1] = GRAV_GET_PRIV(tw)->Accel[i][1];
        P[i].FullTreeGravAccel[2] = GRAV_GET_PRIV(tw)->Accel[i][2];
        /* calculate the potential */
        P[i].Potential += P[i].Mass / (FORCE_SOFTENING() / 2.8);
        /* remove self-potential */
        P[i].Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) * GRAV_GET_PRIV(tw)->cbrtrho0;
        P[i].Potential *= G;
    }

    /* Compute tidal field eigenvalues for gas particles.
     * Combine PM (long-range) and tree (short-range) contributions.
     * TidalTensorStore is in units without G (G applied in store_eigenvalues).
     * TidalTensorPM is in physical units (includes G from PM potential),
     * so divide by G to match. */
    if(GRAV_GET_PRIV(tw)->TidalGas && P[i].Type == 0) {
        int PI = P[i].PI;
        int k;
        for(k = 0; k < 6; k++)
            GRAV_GET_PRIV(tw)->TidalTensorStore[i][k] += SphP[PI].TidalTensorPM[k] / G;
        tidal_field_store_eigenvalues(i, GRAV_GET_PRIV(tw)->TidalTensorStore[i], G);
    }
    /* Compute tidal field eigenvalues and strength for BH particles.  The eigenvalues are
     * kept because the star-cluster relaxation rate (SCEvolutionRelaxation) needs the
     * E-MOSAICS strength max(lambda) + Omega^2, which the norm cannot supply; one
     * eigen-decomposition serves both.  COMOVING like the gas eigenvalues. */
    if(GRAV_GET_PRIV(tw)->TidalBH && P[i].Type == 5) {
        int PI = P[i].PI;
        int k;
        for(k = 0; k < 6; k++)
            GRAV_GET_PRIV(tw)->TidalTensorStore[i][k] += BhP[PI].TidalTensorPM[k] / G;
        double eig[3];
        tidal_field_eigenvalues(GRAV_GET_PRIV(tw)->TidalTensorStore[i], G, eig);
        for(k = 0; k < 3; k++)
            BhP[PI].TidalFieldEigenvalues[k] = eig[k];
        BhP[PI].TidalFieldStrength = sqrt(eig[0] * eig[0] + eig[1] * eig[1] + eig[2] * eig[2]);
        BhP[PI].TidalFieldAtime = GRAV_GET_PRIV(tw)->atime;
    }
}

/*Compute the absolute magnitude of the acceleration for a particle.*/
static MyFloat
grav_get_abs_accel(struct particle_data * PP, const double G)
{
    double aold=0;
    int j;
    for(j = 0; j < 3; j++) {
       double ax = PP->FullTreeGravAccel[j] + PP->GravPM[j];
       aold += ax*ax;
    }
    return sqrt(aold) / G;
}

static void
grav_short_copy(int place, TreeWalkQueryGravShort * input, TreeWalk * tw)
{
    input->OldAcc = grav_get_abs_accel(&P[place], GRAV_GET_PRIV(tw)->G);
    input->Type = P[place].Type;
}

static void
grav_short_reduce(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][0], result->Acc[0]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][1], result->Acc[1]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][2], result->Acc[2]);
    if(tw->tree->full_particle_tree_flag)
        TREEWALK_REDUCE(P[place].Potential, result->Potential);

    MyFloat (*TidalStore)[6] = GRAV_GET_PRIV(tw)->TidalTensorStore;
    if(TidalStore &&
       ((GRAV_GET_PRIV(tw)->TidalGas && P[place].Type == 0) ||
        (GRAV_GET_PRIV(tw)->TidalBH && P[place].Type == 5))) {
        int k;
        for(k = 0; k < 6; k++)
            TREEWALK_REDUCE(TidalStore[place][k], result->TidalTensor[k]);
    }
}

#endif
