#include <math.h>
#include "gasveldisp.h"
#include "treewalk.h"
#include "density.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*
 * Compute gas and stellar velocity dispersion for gas particles.
 * Two treewalks:
 *   1. Gas VDisp: reuses the existing gasTree (GASMASK | BHMASK), filters to gas in ngbiter.
 *   2. Star VDisp: builds a small STARMASK-only tree.
 */

/* ---- Shared private data for both treewalks ---- */

struct GasVDispPriv {
    struct kick_factor_data kf;
    MyFloat (*V1sum)[3];
    MyFloat *V2sum;
    MyFloat *Mtot;
    int *Ncount;
};
#define GVDISP_GET_PRIV(tw) ((struct GasVDispPriv *) (tw)->priv)

/* ---- Query/Result/Iterator (shared by both walks) ---- */

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Vel[3];
} TreeWalkQueryGasVDisp;

typedef struct {
    TreeWalkResultBase base;
    MyFloat V2sum;
    MyFloat V1sum[3];
    MyFloat Mtot;
    int Ncount;
    int _pad;  /* ensure 8-byte alignment */
} TreeWalkResultGasVDisp;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterGasVDisp;

/* ---- Callbacks for gas velocity dispersion (walk 1) ---- */

static int
gas_vdisp_haswork(int n, TreeWalk * tw)
{
    return (P[n].Type == 0) && !P[n].Swallowed && !P[n].IsGarbage;
}

/* Zero all 6 output fields before the first treewalk to avoid stale values */
static void
gas_vdisp_preprocess(int i, TreeWalk * tw)
{
    SPHP(i).VDisp_gas = 0;
    SPHP(i).VDisp_star = 0;
    SPHP(i).VDisp_mgas = 0;
    SPHP(i).VDisp_mstar = 0;
    SPHP(i).VDisp_Ngas = 0;
    SPHP(i).VDisp_Nstar = 0;
}

static void
gas_vdisp_copy(int place, TreeWalkQueryGasVDisp * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;
    SPH_VelPred(place, I->Vel, &GVDISP_GET_PRIV(tw)->kf);
}

static void
gas_vdisp_gas_ngbiter(TreeWalkQueryGasVDisp * I,
        TreeWalkResultGasVDisp * O,
        TreeWalkNgbIterGasVDisp * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.Hsml = I->Hsml;
        /* Tree has GASMASK | BHMASK; we search GASMASK only */
        iter->base.mask = GASMASK;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }

    int other = iter->base.other;
    if(P[other].Type != 0)
        return;

    MyFloat VelPred[3];
    SPH_VelPred(other, VelPred, &GVDISP_GET_PRIV(lv->tw)->kf);

    int d;
    for(d = 0; d < 3; d++) {
        double vel = VelPred[d] - I->Vel[d];
        O->V1sum[d] += vel;
        O->V2sum += vel * vel;
    }
    O->Mtot += P[other].Mass;
    O->Ncount += 1;
}

static void
gas_vdisp_reduce(int place, TreeWalkResultGasVDisp * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int PI = P[place].PI;
    int k;
    TREEWALK_REDUCE(GVDISP_GET_PRIV(tw)->V2sum[PI], remote->V2sum);
    TREEWALK_REDUCE(GVDISP_GET_PRIV(tw)->Mtot[PI], remote->Mtot);
    TREEWALK_REDUCE(GVDISP_GET_PRIV(tw)->Ncount[PI], remote->Ncount);
    for(k = 0; k < 3; k++)
        TREEWALK_REDUCE(GVDISP_GET_PRIV(tw)->V1sum[PI][k], remote->V1sum[k]);
}

static void
gas_vdisp_gas_postprocess(int i, TreeWalk * tw)
{
    int PI = P[i].PI;
    int n = GVDISP_GET_PRIV(tw)->Ncount[PI];
    if(n > 0) {
        double vdisp = GVDISP_GET_PRIV(tw)->V2sum[PI] / n;
        int d;
        for(d = 0; d < 3; d++) {
            double vmean = GVDISP_GET_PRIV(tw)->V1sum[PI][d] / n;
            vdisp -= vmean * vmean;
        }
        if(vdisp > 0)
            SPHP(i).VDisp_gas = sqrt(vdisp / 3);
        SPHP(i).VDisp_mgas = GVDISP_GET_PRIV(tw)->Mtot[PI];
        SPHP(i).VDisp_Ngas = n;
    }
}

/* ---- Callbacks for stellar velocity dispersion (walk 2) ---- */

static void
gas_vdisp_star_ngbiter(TreeWalkQueryGasVDisp * I,
        TreeWalkResultGasVDisp * O,
        TreeWalkNgbIterGasVDisp * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.Hsml = I->Hsml;
        iter->base.mask = STARMASK;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }

    int other = iter->base.other;
    if(P[other].Type != 4)
        return;

    MyFloat VelPred[3];
    DM_VelPred(other, VelPred, &GVDISP_GET_PRIV(lv->tw)->kf);

    int d;
    for(d = 0; d < 3; d++) {
        double vel = VelPred[d] - I->Vel[d];
        O->V1sum[d] += vel;
        O->V2sum += vel * vel;
    }
    O->Mtot += P[other].Mass;
    O->Ncount += 1;
}

static void
gas_vdisp_star_postprocess(int i, TreeWalk * tw)
{
    int PI = P[i].PI;
    int n = GVDISP_GET_PRIV(tw)->Ncount[PI];
    if(n > 0) {
        double vdisp = GVDISP_GET_PRIV(tw)->V2sum[PI] / n;
        int d;
        for(d = 0; d < 3; d++) {
            double vmean = GVDISP_GET_PRIV(tw)->V1sum[PI][d] / n;
            vdisp -= vmean * vmean;
        }
        if(vdisp > 0)
            SPHP(i).VDisp_star = sqrt(vdisp / 3);
        SPHP(i).VDisp_mstar = GVDISP_GET_PRIV(tw)->Mtot[PI];
        SPHP(i).VDisp_Nstar = n;
    }
}

/* ---- Helper: allocate/free temporary arrays ---- */

static void
alloc_priv_arrays(struct GasVDispPriv * priv, int64_t ngas)
{
    priv->V1sum = (MyFloat (*)[3]) mymalloc("GVDISP_V1", ngas * sizeof(priv->V1sum[0]));
    priv->V2sum = (MyFloat *) mymalloc("GVDISP_V2", ngas * sizeof(MyFloat));
    priv->Mtot  = (MyFloat *) mymalloc("GVDISP_M",  ngas * sizeof(MyFloat));
    priv->Ncount = (int *) mymalloc("GVDISP_N", ngas * sizeof(int));
}

static void
free_priv_arrays(struct GasVDispPriv * priv)
{
    myfree(priv->Ncount);
    myfree(priv->Mtot);
    myfree(priv->V2sum);
    myfree(priv->V1sum);
}

/* ---- Main entry point ---- */

void
gas_star_veldisp(const ActiveParticles * act, Cosmology * CP,
                 const DriftKickTimes * times,
                 const ForceTree * gasTree,
                 DomainDecomp * ddecomp, const char * OutputDir)
{
    struct GasVDispPriv priv[1] = {{0}};
    int64_t ngas = SlotsManager->info[0].size;

    init_kick_factor_data(&priv->kf, times, CP);

    /* ---- Walk 1: gas velocity dispersion using the existing gasTree ---- */
    {
        TreeWalk tw[1] = {{0}};
        tw->ev_label = "GAS_VDISP";
        tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
        tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterGasVDisp);
        tw->ngbiter = (TreeWalkNgbIterFunction) gas_vdisp_gas_ngbiter;
        tw->haswork = gas_vdisp_haswork;
        tw->preprocess = (TreeWalkProcessFunction) gas_vdisp_preprocess;
        tw->postprocess = (TreeWalkProcessFunction) gas_vdisp_gas_postprocess;
        tw->fill = (TreeWalkFillQueryFunction) gas_vdisp_copy;
        tw->reduce = (TreeWalkReduceResultFunction) gas_vdisp_reduce;
        tw->query_type_elsize = sizeof(TreeWalkQueryGasVDisp);
        tw->result_type_elsize = sizeof(TreeWalkResultGasVDisp);
        tw->tree = gasTree;
        tw->priv = priv;

        if(!gasTree->tree_allocated_flag)
            endrun(0, "gasTree not allocated for gas_star_veldisp\n");

        alloc_priv_arrays(priv, ngas);
        treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
        free_priv_arrays(priv);
    }
    walltime_measure("/GasStarVDisp/Gas");

    /* ---- Walk 2: stellar velocity dispersion using a small star-only tree ---- */
    {
        ForceTree starTree = {0};
        force_tree_rebuild_mask(&starTree, ddecomp, STARMASK, OutputDir);

        TreeWalk tw[1] = {{0}};
        tw->ev_label = "STAR_VDISP";
        tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
        tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterGasVDisp);
        tw->ngbiter = (TreeWalkNgbIterFunction) gas_vdisp_star_ngbiter;
        tw->haswork = gas_vdisp_haswork;
        tw->postprocess = (TreeWalkProcessFunction) gas_vdisp_star_postprocess;
        tw->fill = (TreeWalkFillQueryFunction) gas_vdisp_copy;
        tw->reduce = (TreeWalkReduceResultFunction) gas_vdisp_reduce;
        tw->query_type_elsize = sizeof(TreeWalkQueryGasVDisp);
        tw->result_type_elsize = sizeof(TreeWalkResultGasVDisp);
        tw->tree = &starTree;
        tw->priv = priv;

        alloc_priv_arrays(priv, ngas);
        treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
        free_priv_arrays(priv);

        force_tree_free(&starTree);
    }
    walltime_measure("/GasStarVDisp/Star");
}
