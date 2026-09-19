/*! \file starcluster_evolution.c
 *  \brief Mass evolution of the star clusters attached to BH particles:
 *  stellar evolution (SCEvolutionStellar) and two-body relaxation (SCEvolutionRelaxation).
 *
 *  Star clusters are SSPs (simple stellar populations) that live on BH particles.  A cluster's
 *  mass is a budget (Gamma x birth mass) drawn from the star particles of its host group, which
 *  keep their full mass and do their own stellar evolution (metal_return.c).  The stellar part
 *  here therefore only reduces the cluster mass by the SSP return fraction (same yield tables and
 *  Chabrier IMF as metal_return.c): depositing the ejecta into the gas would count them twice.
 *  The relaxation part strips mass from the clusters at a rate set by the BH tidal field; the
 *  stripped stars are not returned to the gas either.
 *
 *  Every channel records what it did to the cluster in the BH fields SC_Mloss{Stellar,Relax,TDE}
 *  (cumulative mass lost, code units) and SC_dlnReff{Stellar,Relax} (cumulative ln R_eff change),
 *  so that the total change of StarClusterMass and SC_Reff can be split by channel afterwards
 *  (see slotsmanager.h for the identities they satisfy).
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_integration.h>

#include "physconst.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "metal_return.h"
#include "starcluster_evolution.h"
#include "blackhole.h"
#include "cosmology.h"
#include "timebinmgr.h"
#include "sfr_eff.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "metal_tables.h"

/* StarClusterBHDyn=1: the cluster is part of the BH's dynamical mass, so P.Mass is
 * re-derived whenever StarClusterMass changes (same rule as blackhole_feedback_postprocess). */
static void
sc_update_dyn_mass(int place)
{
    double target = BHP(place).Mtrack + BHP(place).StarClusterMass;
    double SeedBHDynMass = get_bh_seed_dyn_mass();
    if(target < SeedBHDynMass)
        target = SeedBHDynMass;
    P[place].Mass = target;
}

/* Current effective radius (physical pc) of the cluster on BH particle place, setting it if
 * it is not set yet (a restart from a snapshot without SC_Reff): from SC_initReff, or from
 * the StarClusterReffRelation median at the current mass when that is unknown too (which
 * then also becomes the initial radius). */
static double
sc_reff_now(int place)
{
    if(BHP(place).SC_Reff <= 0) {
        if(BHP(place).SC_initReff <= 0)
            BHP(place).SC_initReff = starcluster_median_reff_pc(BHP(place).StarClusterMass);
        BHP(place).SC_Reff = BHP(place).SC_initReff;
    }
    return BHP(place).SC_Reff;
}

/* ==================================================================================
 * Stellar evolution (SCEvolutionStellar)
 * ================================================================================== */

/* Yield interpolators and IMF normalisation, built on the first call and kept for the run:
 * they depend only on the static yield tables, and setup_metal_table_interp allocates new
 * gsl_interp2d objects on every call.  Built and used only in the serial part of the step
 * (the interpolators are then only read, with no accelerators, by the threads). */
static struct interps sc_interp;
static double sc_imf_norm = -1;

void
starcluster_stellar_evolution(const ActiveParticles * act, Cosmology * CP, const double atime, const int size_evolution)
{
    const int nthread = omp_get_max_threads();
    gsl_integration_workspace ** gsl_work = ta_malloc("sc_gsl_work", gsl_integration_workspace *, nthread);
    int i;
    for(i = 0; i < nthread; i++)
        gsl_work[i] = gsl_integration_workspace_alloc(GSL_WORKSPACE);

    if(sc_imf_norm < 0) {
        setup_metal_table_interp(&sc_interp);
        sc_imf_norm = compute_imf_norm(gsl_work[0]);
    }
    const int bhdyn = get_starcluster_bhdyn_on();

    #pragma omp parallel for
    for(i = 0; i < act->NumActiveParticle; i++)
    {
        const int p = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p].Type != 5 || P[p].IsGarbage || P[p].Swallowed)
            continue;
        if(BHP(p).StarClusterMass <= 0 || BHP(p).StarClusterFormationTime <= 0)
            continue;
        const int tid = omp_get_thread_num();
        const double age = atime_to_myr(CP, BHP(p).StarClusterFormationTime, atime, gsl_work[tid]);
        const double metal = BHP(p).StarClusterMetallicity;
        /* Birth mass of the stars still in the cluster (relaxation rescales
         * StarClusterTotalMassReturned with the cluster, so this stays consistent). */
        const double initialmass = BHP(p).StarClusterMass + BHP(p).StarClusterTotalMassReturned;

        /* Cumulative SSP return since the cluster's birth, rather than a sum of per-step windows:
         * BH steps are short, so per-step windows would be far narrower than the tolerance of the
         * dying-mass root finding, and there is no threshold below which a return is skipped. */
        double mlow, mhigh;
        find_mass_bin_limits(&mlow, &mhigh, 0, age, metal, sc_interp.lifetime_interp);
        const double returned = initialmass * mass_yield(0, age, metal, CP->HubbleParam, &sc_interp,
                                                         sc_imf_norm, gsl_work[tid], mlow, mhigh);
        BHP(p).StarClusterLastEnrichmentMyr = age;
        /* Only ever lose mass (the return grows monotonically with age) */
        if(returned <= BHP(p).StarClusterTotalMassReturned || returned >= initialmass)
            continue;

        const double m_old = BHP(p).StarClusterMass;
        const double m_new = initialmass - returned;
        /* StarClusterSizeEvolution: adiabatic expansion from the stellar mass loss,
         * (dr_h/r_h)_sev = |dm_sev|/m, applied as the inverse ratio of the cluster mass after
         * and before the loss (Guerra et al. 2026 sect. 2.2.3).  With r_h = (4/3) R_eff at
         * fixed profile shape the same factor applies to R_eff. */
        if(size_evolution) {
            BHP(p).SC_Reff = sc_reff_now(p) * m_old / m_new;
            BHP(p).SC_dlnReffStellar += log(m_old / m_new);
        }
        BHP(p).StarClusterMass = m_new;
        BHP(p).StarClusterTotalMassReturned = returned;
        BHP(p).SC_MlossStellar += m_old - m_new;
        if(bhdyn)
            sc_update_dyn_mass(p);
    }

    for(i = nthread - 1; i >= 0; i--)
        gsl_integration_workspace_free(gsl_work[i]);
    ta_free(gsl_work);

    walltime_measure("/BH/SCStellar");
}

/* ==================================================================================
 * Two-body relaxation (SCEvolutionRelaxation)
 * ================================================================================== */

/* Mode 1, E-MOSAICS: Kruijssen et al. (2011); Pfeffer et al. (2018) eq. 13 (King W0 = 5):
 * dM/dt = -M / t_dis,  t_dis = t0_sun (M/Msun)^gamma (T/T_sun)^-1/2. */
#define SC_RLX_EMOS_GAMMA   0.62
#define SC_RLX_EMOS_T0SUN   21.3        /* Myr */
#define SC_RLX_EMOS_TSUN    7.01e2      /* Gyr^-2 */
/* Mode 2, GB08: Gieles & Baumgardt (2008) / Alexander & Gieles (2012) as adopted in
 * EMP-Pathfinder (Reina-Campos et al. 2022, eqs. 53-56). */
#define SC_RLX_MSTAR        0.42        /* Msun: mean stellar mass of a Chabrier IMF over 0.08-120 Msun */
#define SC_RLX_LNL_GAMMA    0.11        /* Coulomb logarithm ln(gamma N) of equal-mass clusters */
#define SC_RLX_XI0          0.0142      /* escapers per t_rh in isolation */
#define SC_RLX_ZETA         0.1         /* relaxation-driven energy change per t_rh */
#define SC_RLX_RHRT1        0.145       /* r_h/r_t of a tidally filling equal-mass cluster */
#define SC_RLX_PZ           1.61        /* z of eq. 56 */
#define SC_RLX_N1           1.5e4       /* m_1 = 1.5e4 <m> of eq. 56, i.e. N_1 = 1.5e4 stars */
#define SC_RLX_PX           0.75        /* x of eq. 56 */
/* Both modes: a cluster below this mass is dissolved (EMP-Pathfinder follows clusters down to 100 Msun). */
#define SC_RLX_MDISSOLVE    100.0       /* Msun */
/* GB08 sub-cycling within one BH step: at most this fraction of the mass per sub-step (the
 * isolated-limit expansion then changes R_eff by ~1% per sub-step), and a safety cap on the
 * number of sub-steps (0.2% per sub-step reaches 100 Msun from 1e8 in ~7000). */
#define SC_RLX_SUBLOSS      0.002
#define SC_RLX_MAXSUB       20000

/* Newton's constant in pc^3 Msun^-1 Myr^-2 */
static double
sc_rlx_G(void)
{
    const double pc = CM_PER_MPC / 1e6;
    return GRAVITY * SOLAR_MASS * SEC_PER_MEGAYEAR * SEC_PER_MEGAYEAR / (pc * pc * pc);
}

double
sc_relax_mass_emosaics(double m_msun, double T_gyr2, double dt_myr)
{
    if(m_msun <= 0 || T_gyr2 <= 0 || dt_myr <= 0)
        return m_msun;
    /* d(M^gamma)/dt = -gamma/t0_sun (T/T_sun)^1/2 is independent of M, so the update is
     * exact over a step at fixed T. */
    const double K = SC_RLX_EMOS_GAMMA / (SC_RLX_EMOS_T0SUN * sqrt(SC_RLX_EMOS_TSUN));
    const double mg = pow(m_msun, SC_RLX_EMOS_GAMMA) - K * sqrt(T_gyr2) * dt_myr;
    return mg > 0 ? pow(mg, 1. / SC_RLX_EMOS_GAMMA) : 0;
}

double
sc_relax_trh_myr(double m_msun, double rh_pc)
{
    /* Eq. 54 (Spitzer & Hart 1971): t_rh = 0.138 N^1/2 r_h^3/2 / (<m>^1/2 G^1/2 ln(gamma N)) */
    const double N = m_msun / SC_RLX_MSTAR;
    const double lnL = log(SC_RLX_LNL_GAMMA * N);
    if(lnL <= 0 || rh_pc <= 0)
        return 0;
    return 0.138 * sqrt(N) * pow(rh_pc, 1.5) / (sqrt(SC_RLX_MSTAR * sc_rlx_G()) * lnL);
}

double
sc_relax_xi(double m_msun, double rh_pc, double T_myr2)
{
    /* Eq. 56: P = ((r_h/r_t) / [r_h/r_t]_1)^z (m log(0.11 m_1) / (m_1 log(0.11 m)))^(1-x),
     * with m_1 = 1.5e4 <m>, i.e. Alexander & Gieles (2012) eq. 25 in N = m/<m>, whose logs
     * are the Coulomb logarithms ln(0.11 N) of eq. 54.  Tidal radius r_t = (G m / T)^1/3
     * (eq. 50); T <= 0 has no finite tidal radius: P = 0, the isolated limit.
     * (Pev, since P is the particle array.) */
    const double N = m_msun / SC_RLX_MSTAR;
    double Pev = 0;
    if(T_myr2 > 0 && rh_pc > 0 && SC_RLX_LNL_GAMMA * N > 1) {
        const double rt = cbrt(sc_rlx_G() * m_msun / T_myr2);
        Pev = pow(rh_pc / rt / SC_RLX_RHRT1, SC_RLX_PZ)
            * pow(N * log(SC_RLX_LNL_GAMMA * SC_RLX_N1) / (SC_RLX_N1 * log(SC_RLX_LNL_GAMMA * N)), 1 - SC_RLX_PX);
    }
    /* Eq. 55 */
    return SC_RLX_XI0 * (1 - Pev) + 0.6 * SC_RLX_ZETA * Pev;
}

double
sc_relax_mass_gb08(double m_msun, double rh_pc, double T_myr2, double dt_myr)
{
    if(m_msun <= 0 || dt_myr <= 0)
        return m_msun;
    const double trh = sc_relax_trh_myr(m_msun, rh_pc);
    if(trh <= 0)
        return 0;
    const double xi = sc_relax_xi(m_msun, rh_pc, T_myr2);
    /* Eq. 53, dm/dt = -xi m / t_rh.  At fixed r_h, m^1/2 / t_rh depends on m only through
     * ln(gamma N), so d(m^1/2)/dt = -(xi/2) m^1/2 / t_rh is nearly constant: integrate
     * m^1/2 linearly over the step, which also reaches zero cleanly. */
    const double s = sqrt(m_msun) * (1 - 0.5 * xi * dt_myr / trh);
    return s > 0 ? s * s : 0;
}

double
sc_relax_gb08_step(double m_msun, double reff_pc, double T_myr2, double dt_myr, int size_evolution, double * reff_out)
{
    /* GB08 relaxation (and, with size_evolution, its size change) over dt, sub-cycled so that no
     * sub-step removes more than SC_RLX_SUBLOSS of the mass.  The m^1/2 update of
     * sc_relax_mass_gb08 holds xi and ln(gamma N) fixed, and the expansion has to feed back on
     * t_rh within the interval.  An ordinary BH step needs a single sub-step; a deferred interval
     * (up to a PM step, see starcluster_relaxation) can span many t_rh of a compact cluster.
     * Stops once the cluster is below SC_RLX_MDISSOLVE (the caller dissolves it). */
    double m = m_msun, reff = reff_pc, left = dt_myr;
    int nsub = 0;
    while(left > 0 && m >= SC_RLX_MDISSOLVE) {
        const double rh = 4. / 3. * reff;
        const double trh = sc_relax_trh_myr(m, rh);
        const double xi = sc_relax_xi(m, rh, T_myr2);
        double h = left;
        if(trh > 0 && xi > 0 && ++nsub < SC_RLX_MAXSUB)
            h = fmin(left, SC_RLX_SUBLOSS * trh / xi);
        const double mn = sc_relax_mass_gb08(m, rh, T_myr2, h);
        if(size_evolution)
            reff *= sc_size_factor_rlx(m, mn, xi);
        m = mn;
        left -= h;
    }
    if(reff_out)
        *reff_out = reff;
    return m;
}

double
sc_size_factor_rlx(double m_old, double m_new, double xi)
{
    /* Dynamical size evolution from two-body relaxation, Guerra et al. (2026) eq. 22 without
     * the tidal-shock term (Gieles & Renaud 2016; Reina-Campos et al. 2023 App. E):
     * dr_h/r_h = (2 - zeta/xi) dm_rlx/m.  With xi fixed over the step this integrates to
     * r_new/r_old = (m_new/m_old)^(2 - zeta/xi): expansion in isolation (xi = xi0, exponent
     * -5.0), contraction when tidally limited (xi -> 3 zeta/5, exponent +1/3).  The same
     * factor applies to R_eff = (3/4) r_h. */
    if(m_old <= 0 || m_new <= 0 || xi <= 0)
        return 1;
    return pow(m_new / m_old, 2 - SC_RLX_ZETA / xi);
}

/* Remove the whole cluster from the BH: the same reset as a GW-recoil ejection
 * (blackhole_feedback_postprocess), so a cluster reacquired in a later merger restarts
 * its stellar evolution cleanly.  The radii describe the cluster and go with it. */
static void
sc_dissolve(int place)
{
    BHP(place).StarClusterMass = 0;
    BHP(place).StarClusterMetallicity = 0;
    memset(BHP(place).StarClusterMetals, 0, sizeof(BHP(place).StarClusterMetals));
    BHP(place).StarClusterTotalMassReturned = 0;
    BHP(place).StarClusterLastEnrichmentMyr = -1;
    BHP(place).StarClusterFormationTime = 0;
    BHP(place).SC_initReff = 0;
    BHP(place).SC_Reff = 0;
    BHP(place).SC_RlxPendingMyr = 0;
    /* the per-channel records describe the cluster too; its history stays in the detail files */
    BHP(place).SC_MlossStellar = 0;
    BHP(place).SC_MlossRelax = 0;
    BHP(place).SC_MlossTDE = 0;
    BHP(place).SC_dlnReffStellar = 0;
    BHP(place).SC_dlnReffRelax = 0;
}

void
starcluster_relaxation(const ActiveParticles * act, const Cosmology * CP, const double atime, const int mode, const int size_evolution, const struct UnitSystem units)
{
    /* Do nothing if no BHs yet */
    int64_t totbh;
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totbh == 0)
        return;

    const double h = CP->HubbleParam;
    /* code mass is 1e10 Msun/h and code time UnitTime/h */
    const double mass_to_msun = units.UnitMass_in_g / (SOLAR_MASS * h);
    const double time_to_myr = units.UnitTime_in_s / h / SEC_PER_MEGAYEAR;
    /* code time^-2 -> Gyr^-2 */
    const double invt2_to_gyr2 = 1. / pow(time_to_myr / 1e3, 2);
    const double hubble = hubble_function(CP, atime);
    const int bhdyn = get_starcluster_bhdyn_on();

    int64_t nevolved = 0, ndissolved = 0, ndeferred = 0;
    double mlost = 0;
    int i;
    #pragma omp parallel for reduction(+: nevolved, ndissolved, ndeferred, mlost)
    for(i = 0; i < act->NumActiveParticle; i++)
    {
        const int p = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p].Type != 5 || P[p].IsGarbage || P[p].Swallowed)
            continue;
        if(BHP(p).StarClusterMass <= 0)
            continue;
        /* the BH's own step, as in the accretion */
        double dt_myr = get_dloga_for_bin(P[p].TimeBinHydro, P[p].Ti_drift) / hubble * time_to_myr;
        /* E-MOSAICS tidal strength T = max(lambda) + Omega^2, Omega^2 = -(1/3) sum lambda
         * (Pfeffer et al. 2018, App. C), whose lambda = -mu for the stored descending
         * Hessian eigenvalues mu: T = -mu_3 + (mu_1 + mu_2 + mu_3)/3.  Refreshed at every
         * BH gravity step, or held from the last PM step under hierarchical gravity. */
        const MyFloat * mu = BHP(p).TidalFieldEigenvalues;
        const double a_field = BHP(p).TidalFieldAtime;
        /* No tidal field yet (TidalFieldAtime still the 0 set at creation): a BH seeded after the
         * gravity of a PM step, which with SplitGravityTimestepsOn=1 gets its first field at the
         * next PM step.  Do not relax it with T = 0 (no loss for E-MOSAICS, the isolated limit for
         * GB08): defer, and relax the whole interval with the first field. */
        if(a_field <= 0) {
            BHP(p).SC_RlxPendingMyr += dt_myr;
            ndeferred++;
            continue;
        }
        dt_myr += BHP(p).SC_RlxPendingMyr;
        BHP(p).SC_RlxPendingMyr = 0;
        /* The eigenvalues are comoving (trace = 4 pi G rho_comoving) and cached from the scale factor
         * a_field of their measurement: convert with a_field^-3, not the current a^-3, so the PHYSICAL
         * strength is held between evaluations (the PM-hold validated in TidalField_PMcadence.ipynb)
         * instead of decaying as (a_field/a)^3. */
        const double T_gyr2 = (-mu[2] + (mu[0] + mu[1] + mu[2]) / 3.) * invt2_to_gyr2 / (a_field * a_field * a_field);
        const double m_old = BHP(p).StarClusterMass * mass_to_msun;
        double m_new, reff_old = 0, reff_new = 0;
        if(mode == 1)
            m_new = sc_relax_mass_emosaics(m_old, T_gyr2, dt_myr);     /* exact at fixed T */
        else {
            /* r_h = (4/3) R_eff of the cluster's current radius (its initial one unless
             * StarClusterSizeEvolution=1), sub-cycled with its size change */
            reff_old = sc_reff_now(p);
            m_new = sc_relax_gb08_step(m_old, reff_old, T_gyr2 * 1e-6, dt_myr, size_evolution, &reff_new);
        }
        nevolved++;
        if(m_new < SC_RLX_MDISSOLVE) {
            sc_dissolve(p);
            mlost += m_old;
            ndissolved++;
        }
        else {
            /* The escapers are a representative sample of the cluster's stars: they take their
             * share of the mass already returned by stellar evolution with them, so the stellar
             * return keeps acting on the initial mass of the stars that are still bound. */
            const double frac = m_new / m_old;
            BHP(p).StarClusterMass *= frac;
            BHP(p).StarClusterTotalMassReturned *= frac;
            BHP(p).SC_MlossRelax += (m_old - m_new) / mass_to_msun;
            mlost += m_old - m_new;
            /* StarClusterSizeEvolution with GB08: the size responds to the relaxation mass loss */
            if(size_evolution && mode == 2) {
                BHP(p).SC_Reff = reff_new;
                if(reff_old > 0 && reff_new > 0)
                    BHP(p).SC_dlnReffRelax += log(reff_new / reff_old);
            }
        }
        if(bhdyn)
            sc_update_dyn_mass(p);
    }

    int64_t counts[3] = {nevolved, ndissolved, ndeferred}, totcounts[3];
    double totmlost;
    MPI_Reduce(counts, totcounts, 3, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mlost, &totmlost, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* The reduced totals exist on rank 0 only: evaluate them there only. */
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0 && (totcounts[0] > 0 || totcounts[2] > 0))
        message(0, "SC relaxation: %ld clusters evolved, %ld dissolved, %ld deferred (no tidal field yet), %g Msun stripped.\n",
                (long) totcounts[0], (long) totcounts[1], (long) totcounts[2], totmlost);

    walltime_measure("/BH/SCRelax");
}

void
starcluster_relaxation_message(const int mode, const int hierarchical)
{
    const char * cadence = hierarchical
        ? "held from the last PM step (SplitGravityTimestepsOn=1; a new seed defers its relaxation to its first field, at the next PM step)"
        : "recomputed at every BH gravity step (SplitGravityTimestepsOn=0)";
    if(mode == 1)
        message(0, "SCEvolutionRelaxation=1: E-MOSAICS two-body relaxation of the star clusters on BHs, "
                   "t_dis = %g Myr (M/Msun)^%g (T/%g Gyr^-2)^-1/2 with T = max(lambda)+Omega^2 of the BH tidal tensor, %s; "
                   "clusters below %g Msun are dissolved.\n",
                SC_RLX_EMOS_T0SUN, SC_RLX_EMOS_GAMMA, SC_RLX_EMOS_TSUN, cadence, SC_RLX_MDISSOLVE);
    else
        message(0, "SCEvolutionRelaxation=2: GB08 two-body relaxation of the star clusters on BHs "
                   "(Reina-Campos et al. 2022 eqs. 53-56: <m> = %g Msun, ln(%g N), xi0 = %g, zeta = %g, "
                   "[r_h/r_t]_1 = %g, z = %g, m_1 = %g <m>, x = %g), r_h = 4/3 of the cluster's current R_eff SC_Reff "
                   "(starting from the drawn R_eff, or the StarClusterReffRelation median when none was drawn), "
                   "r_t = (G M/T)^1/3 with T = max(lambda)+Omega^2 of the BH tidal tensor, %s; "
                   "clusters below %g Msun are dissolved.\n",
                SC_RLX_MSTAR, SC_RLX_LNL_GAMMA, SC_RLX_XI0, SC_RLX_ZETA, SC_RLX_RHRT1, SC_RLX_PZ,
                SC_RLX_N1, SC_RLX_PX, cadence, SC_RLX_MDISSOLVE);
}

/* ==================================================================================
 * Growth by tidal disruption events (StarClusterTDEtoBH)
 * ================================================================================== */

/* Rizzuto et al. (2023, MNRAS 521, 2930) eq. 9, the TDE rate of a BH in a dense star cluster:
 *   Ndot = 1.1 F f_b ln(0.22 M_BH/m_*) (M_BH/1e3 Msun) (rho/1e7 Msun pc^-3) (100 km/s / sigma)^3 Myr^-1,
 * i.e. their eq. 7, Ndot = F N_b/t_rel, with N_b = f_b N the stars bound to the BH among the
 * N = 2 M_BH/m_* inside its influence radius R_inf (the sphere holding a stellar mass 2 M_BH,
 * their footnote 9) and t_rel Spitzer's relaxation time at R_inf with Coulomb log ln(0.11 N).
 * rho and sigma there come from the Williams et al. (2026, arXiv:2603.26872) "cluster with black
 * hole" structure, the same profile family as the CW seed model (cwmodel.c): a Bahcall-Wolf power
 * law rho ~ r^-alpha with alpha = 7/4 inside r_max = 1.4 R_eff, M(<r) = c_M r^(3-alpha) with
 * c_M = M_SC/r_max^(3-alpha), rho = (3-alpha)/(4 pi) c_M r^-alpha, and the isotropic Jeans
 * dispersion of their eq. 12, sigma^2(r) = c_v G [M(<r) + M_BH]/((1+alpha) r).
 * Conventions, from the post-processing study code_v3/BHgrowth_TDE_Rizzuto23.ipynb: rho is the
 * LOCAL density at R_inf (Rizzuto's fig. B1); sigma is the 3-D, mass-weighted mean of the Jeans
 * dispersion of the stars inside R_inf,
 *   <sigma^2> = 3 c_v G/((1+alpha) R_inf) [ (3-alpha)/(2-alpha) M_BH + (3-alpha)/(5-2 alpha) M(<R_inf) ],
 * which reproduces the sigma of Rizzuto's virial fit (their eq. B2) within 6% and is therefore the
 * convention their F was calibrated with (the 1-D local value is 2.6x smaller and would need
 * F ~ 0.05).  When 2 M_BH > M_SC the whole cluster is inside the sphere of influence: R_inf = r_max
 * and eq. 9 is used in its eq. 7 form with N = M_SC/m_*, i.e. M_BH -> M_SC/2 in its two explicit
 * factors.  At fixed cluster the rate falls as M_BH^-0.7 ln(0.22 M_BH/m_*): a heavier BH collects
 * its bound stars from a sparser part of the cusp. */
#define SC_TDE_ALPHA        1.75    /* rho ~ r^-alpha (Bahcall & Wolf 1976; Williams+26 BH systems) */
#define SC_TDE_RMAX_REFF    1.4     /* r_max / R_eff, as the CW seed model */
#define SC_TDE_CV           1.0     /* c_v of Williams+26 eq. 12 (cwmodel.c CW_CV) */
#define SC_TDE_F            0.8     /* Rizzuto+23 prefactor F (their per-model fits: 0.6-1.0) */
#define SC_TDE_FB           0.2     /* fraction of the stars inside R_inf bound to the BH (Rizzuto+23 sec. 4.6) */
#define SC_TDE_MSTAR        1.0     /* Msun: mass of a disrupted star (also in the Coulomb log) */
#define SC_TDE_FACC         0.5     /* fraction of the star's mass the BH keeps per TDE (Rizzuto+23 eq. 11 f_c; Rees 1988);
                                       the rest is unbound debris ejected at thousands of km/s.  Either way the whole
                                       star (m_*) leaves the cluster: this is the TDE term of the cluster mass evolution. */

/* StarClusterEnhancedTDE4Merger: the eccentric Kozai-Lidov burst of Mockler et al. (2023) around
 * the lighter BH of a merging pair.  Their fiducial binary sits at a_bin = SC_MTDE_ABIN_RH r_h,1
 * with eccentricity SC_MTDE_EBIN, the hierarchical radius inside which the stars orbiting the
 * secondary are perturbed coherently is a_hier = SC_MTDE_EPS_HIER a_bin (1-e^2)/e (their eq. 2),
 * and a fraction SC_MTDE_EPS_DIS of the stars inside it end up disrupted (their Table 1: lower
 * limits 0.05-0.11 at q = 10, upper limits 0.27-0.45 with partial disruptions included; for
 * alpha = 7/4 and q ~ 5 the supported range is ~0.15-0.4, see code_TDE/TDE_rates.ipynb).  With
 * r_h,1 defined by the enclosed mass the cluster cancels: N_hier = 2 (0.075)^(3-alpha) m1/m_* =
 * 0.079 m1/m_* whenever 2 m1 <= M_SC, and only when the secondary outweighs half its cluster does
 * the cluster mass set the count, N_hier = 0.075^(3-alpha) M_SC/m_*.  R_eff never enters. */
#define SC_MTDE_EPS_DIS     0.25    /* fraction of the stars inside a_hier that are disrupted */
#define SC_MTDE_ABIN_RH     0.5     /* a_bin / r_h,1 */
#define SC_MTDE_EBIN        0.5     /* binary eccentricity */
#define SC_MTDE_EPS_HIER    0.1     /* hierarchical limit, Mockler+23 eq. 1-2 */

double
sc_merger_tde_ntde(double m1_msun, double msc_msun, double reff_pc)
{
    if(m1_msun <= 0 || msc_msun <= 0 || reff_pc <= 0)
        return 0;
    const double a = SC_TDE_ALPHA;
    const double r_max = SC_TDE_RMAX_REFF * reff_pc;
    const double c_M = msc_msun / pow(r_max, 3 - a);
    /* sphere holding 2 m1 of cluster stars, at most the cluster's edge */
    double r_h = pow(2 * m1_msun / c_M, 1. / (3 - a));
    if(r_h > r_max)
        r_h = r_max;
    const double a_hier = SC_MTDE_EPS_HIER * SC_MTDE_ABIN_RH * r_h * (1 - SC_MTDE_EBIN * SC_MTDE_EBIN) / SC_MTDE_EBIN;
    return SC_MTDE_EPS_DIS * c_M * pow(a_hier, 3 - a) / SC_TDE_MSTAR;
}

double
sc_merger_tde_mass_msun(double m1_msun, double msc_msun, double reff_pc)
{
    return SC_TDE_FACC * SC_TDE_MSTAR * sc_merger_tde_ntde(m1_msun, msc_msun, reff_pc);
}

double
sc_tde_rate_per_myr(double mbh_msun, double msc_msun, double reff_pc)
{
    if(mbh_msun <= 0 || msc_msun <= 0 || reff_pc <= 0)
        return 0;
    const double a = SC_TDE_ALPHA;
    /* Newton's constant in pc (km/s)^2 / Msun */
    const double G = GRAVITY * SOLAR_MASS / (CM_PER_MPC / 1e6) / 1e10;
    const double r_max = SC_TDE_RMAX_REFF * reff_pc;
    const double c_M = msc_msun / pow(r_max, 3 - a);
    const double c_rho = (3 - a) * c_M / (4 * M_PI);
    /* influence radius M(<R_inf) = 2 M_BH, at most the cluster's edge */
    double r_inf = pow(2 * mbh_msun / c_M, 1. / (3 - a));
    if(r_inf > r_max)
        r_inf = r_max;
    const double M_enc = c_M * pow(r_inf, 3 - a);   /* 2 M_BH, or M_SC when the whole cluster is inside */
    const double M_eff = 0.5 * M_enc;                /* eq. 9's M_BH: N = M_enc/m_* stars inside R_inf */
    const double rho = c_rho * pow(r_inf, -a);
    const double sigma2 = 3 * SC_TDE_CV * G / ((1 + a) * r_inf)
        * (mbh_msun * (3 - a) / (2 - a) + M_enc * (3 - a) / (5 - 2 * a));
    const double lnL = log(0.22 * M_eff / SC_TDE_MSTAR);
    if(lnL <= 0 || sigma2 <= 0)
        return 0;
    return 1.1 * SC_TDE_F * SC_TDE_FB * lnL * (M_eff / 1e3) * (rho / 1e7) * pow(100. / sqrt(sigma2), 3);
}

void
starcluster_tde_growth(const ActiveParticles * act, const Cosmology * CP, const double atime, const struct UnitSystem units, const int single_on, const int merger_on)
{
    /* Do nothing if no BHs yet */
    int64_t totbh;
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totbh == 0)
        return;

    const double h = CP->HubbleParam;
    /* code mass is 1e10 Msun/h and code time UnitTime/h */
    const double mass_to_msun = units.UnitMass_in_g / (SOLAR_MASS * h);
    const double time_to_myr = units.UnitTime_in_s / h / SEC_PER_MEGAYEAR;
    const double hubble = hubble_function(CP, atime);
    const int bhdyn = get_starcluster_bhdyn_on();

    int64_t nbh = 0, neaten = 0, nburst = 0, ndone = 0;
    double ntde = 0, mgain = 0, mburst = 0;
    int i;
    #pragma omp parallel for reduction(+: nbh, neaten, nburst, ndone, ntde, mgain, mburst)
    for(i = 0; i < act->NumActiveParticle; i++)
    {
        const int p = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p].Type != 5 || P[p].IsGarbage || P[p].Swallowed)
            continue;
        /* the rates describe this step: 0 unless something is disrupted */
        BHP(p).NdotTDE = 0;
        BHP(p).MdotTDE = 0;
        BHP(p).MdotTDEMerger = 0;
        if(BHP(p).Mass <= 0)
            continue;
        /* the BH's own step, as in the accretion */
        const double dt_code = get_dloga_for_bin(P[p].TimeBinHydro, P[p].Ti_drift) / hubble;
        const double dt_myr = dt_code * time_to_myr;
        int changed = 0;

        /* --- StarClusterEnhancedTDE4Merger: the burst reservoir filled at the BH's mergers drains
         * at a constant rate over the window left (every merger restarts the window for the whole
         * reservoir, see blackhole_feedback_postprocess).  A property of the remnant, so it runs
         * whether or not the BH still carries a cluster.  The window closing inside this step
         * finishes the burst. --- */
        if(merger_on && BHP(p).MergerTDEMassLeft > 0 && dt_myr > 0) {
            double dm;
            if(BHP(p).MergerTDETimeLeftMyr > dt_myr) {
                dm = BHP(p).MergerTDEMassLeft * dt_myr / BHP(p).MergerTDETimeLeftMyr;
                BHP(p).MergerTDETimeLeftMyr -= dt_myr;
                BHP(p).MergerTDEMassLeft -= dm;
            }
            else {
                dm = BHP(p).MergerTDEMassLeft;
                BHP(p).MergerTDEMassLeft = 0;
                BHP(p).MergerTDETimeLeftMyr = 0;
                ndone++;
            }
            if(dm > 0) {
                /* stellar debris, not gas: outside the Eddington cap and credited to Mtrack too
                 * (the same reasoning as for the single-BH channel below) */
                BHP(p).Mass += dm;
                BHP(p).Mtrack += dm;
                BHP(p).MdotTDEMerger = dm / dt_code;
                nburst++;
                mburst += dm * mass_to_msun;
                changed = 1;
            }
        }

        /* --- StarClusterTDEtoBH: the BH disrupting the stars of its own cluster --- */
        if(single_on && BHP(p).StarClusterMass > 0) {
            const double m_sc = BHP(p).StarClusterMass * mass_to_msun;
            /* on the cluster as evolved by the relaxation and stellar evolution of this step, and
             * on the BH mass including this step's merger-burst gain */
            const double ndot = sc_tde_rate_per_myr(BHP(p).Mass * mass_to_msun, m_sc, sc_reff_now(p));
            if(ndot > 0) {
                /* stellar mass disrupted over the step, at most the whole cluster */
                double m_dis = SC_TDE_MSTAR * ndot * dt_myr;
                if(m_dis > m_sc)
                    m_dis = m_sc;
                const double m_gain = SC_TDE_FACC * m_dis;
                const double m_gain_code = m_gain / mass_to_msun;
                /* The realised rate [Myr^-1]: the model rate, unless the step ran out of stars, in which
                 * case the cluster's stars over the step.  Both stored rates and the step totals describe
                 * what was applied, so MdotTDE = f_acc m_* NdotTDE and Mass grows by MdotTDE x dt exactly. */
                const double ndot_real = (dt_myr > 0) ? m_dis / (SC_TDE_MSTAR * dt_myr) : ndot;
                BHP(p).NdotTDE = ndot_real * time_to_myr;
                BHP(p).MdotTDE = SC_TDE_FACC * SC_TDE_MSTAR * ndot_real * time_to_myr / mass_to_msun;
                /* Stellar debris, not gas: added to the BH mass directly, outside the Eddington cap of the
                 * Bondi accretion (which then sees the heavier BH) and without feedback or luminosity. */
                BHP(p).Mass += m_gain_code;
                /* Mtrack is the mass the BH particle has physically taken in, and the gas swallowing of
                 * blackhole() makes it follow Mass (a gas particle is swallowed with a probability set by
                 * Mass - Mtrack).  This mass came from the cluster's stars, not from the gas, so credit it
                 * to Mtrack as well (as blackhole_make_one does for a seed mass above the parent mass):
                 * otherwise stellar-fed growth would draw the same mass out of the gas again, and under
                 * StarClusterBHDyn=1 the dynamical mass would lose the retained half along with the unbound
                 * one.  P.Mass is rederived from Mtrack (+ StarClusterMass) in blackhole_feedback_postprocess
                 * this same step, and below for StarClusterBHDyn=1. */
                BHP(p).Mtrack += m_gain_code;
                nbh++;
                ntde += m_dis / SC_TDE_MSTAR;       /* disruptions actually realised (at most the cluster's stars) */
                mgain += m_gain;
                changed = 1;
                /* The TDE term of the cluster mass evolution: the disrupted stars leave the cluster (the
                 * BH keeps SC_TDE_FACC of each, the rest is ejected).  Like the relaxation escapers they are
                 * a representative sample of its stars and take their share of the stellar-evolution return
                 * with them; a cluster eaten down to the dissolution mass is removed. */
                const double m_new = m_sc - m_dis;
                if(m_new < SC_RLX_MDISSOLVE) {
                    sc_dissolve(p);
                    neaten++;
                }
                else {
                    const double frac = m_new / m_sc;
                    BHP(p).StarClusterMass *= frac;
                    BHP(p).StarClusterTotalMassReturned *= frac;
                    BHP(p).SC_MlossTDE += m_dis / mass_to_msun;
                }
            }
        }
        /* StarClusterBHDyn=1: P.Mass = Mtrack + StarClusterMass, so the dynamical mass loses
         * only the unbound part of each disrupted star and gains the burst mass. */
        if(bhdyn && changed)
            sc_update_dyn_mass(p);
    }

    int64_t counts[4] = {nbh, neaten, nburst, ndone}, totcounts[4];
    double sums[3] = {ntde, mgain, mburst}, totsums[3];
    MPI_Reduce(counts, totcounts, 4, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(sums, totsums, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* The reduced totals exist on rank 0 only: evaluate them there only. */
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0 && totcounts[0] > 0)
        message(0, "SC TDE: %ld BHs disrupting cluster stars, %g TDEs this step, %g Msun added to the BHs, %ld clusters eaten up.\n",
                (long) totcounts[0], totsums[0], totsums[1], (long) totcounts[1]);
    if(ThisTask == 0 && totcounts[2] > 0)
        message(0, "SC merger TDE: %ld BHs adding merger-burst mass, %g Msun this step, %ld bursts completed.\n",
                (long) totcounts[2], totsums[2], (long) totcounts[3]);

    walltime_measure("/BH/SCTDE");
}

void
starcluster_merger_tde_message(void)
{
    const double x_hier = SC_MTDE_EPS_HIER * SC_MTDE_ABIN_RH * (1 - SC_MTDE_EBIN * SC_MTDE_EBIN) / SC_MTDE_EBIN;
    message(0, "StarClusterEnhancedTDE4Merger=1: at every BH-BH merger the stars orbiting the lighter BH are disrupted in the "
               "eccentric Kozai-Lidov burst of Mockler et al. (2023): N_TDE = %g x the stars inside the hierarchical radius "
               "a_hier = %g a_bin (1-e^2)/e of a binary at a_bin = %g r_h,1 with e = %g, r_h,1 the sphere holding 2 m1 of the "
               "secondary's cluster stars (at most r_max = %g R_eff) on the same rho ~ r^-%g cusp as StarClusterTDEtoBH, so "
               "N_TDE = %.4f m1/m_* while 2 m1 <= M_SC and %.4f M_SC/m_* when the secondary outweighs half its cluster "
               "(0 without a cluster). The remnant gains %g m_* per event (m_* = %g Msun), added at a constant rate over the "
               "%g Myr after the merger (MdotTDEMerger; the reservoir MergerTDEMassLeft and its window MergerTDETimeLeftMyr "
               "are in the snapshots), on top of and not limited by the Eddington-capped gas accretion and credited to Mtrack; "
               "a swallowed BH's unfinished burst passes to its remnant.\n",
            SC_MTDE_EPS_DIS, SC_MTDE_EPS_HIER, SC_MTDE_ABIN_RH, SC_MTDE_EBIN, SC_TDE_RMAX_REFF, SC_TDE_ALPHA,
            SC_MTDE_EPS_DIS * 2 * pow(x_hier, 3 - SC_TDE_ALPHA), SC_MTDE_EPS_DIS * pow(x_hier, 3 - SC_TDE_ALPHA),
            SC_TDE_FACC, SC_TDE_MSTAR, SC_MTDE_TBIN_MYR);
}

void
starcluster_tde_message(void)
{
    message(0, "StarClusterTDEtoBH=1: BHs grow by tidally disrupting the stars of their star cluster at the "
               "Rizzuto et al. (2023) eq. 9 rate (F = %g, f_b = %g, m_* = %g Msun) on the Williams et al. (2026) "
               "cluster-with-BH structure (rho ~ r^-%g inside r_max = %g R_eff of the cluster's current SC_Reff, c_v = %g; "
               "rho at the influence radius M(<R_inf) = 2 M_BH, sigma the 3-D mean inside it). Each disruption adds "
               "%g m_* to the BH mass (MdotTDE, in NdotTDE and MdotTDE of the BH), on top of and not limited by the "
               "Eddington-capped gas accretion, and removes the whole star (m_*) from the cluster mass.\n",
            SC_TDE_F, SC_TDE_FB, SC_TDE_MSTAR, SC_TDE_ALPHA, SC_TDE_RMAX_REFF, SC_TDE_CV, SC_TDE_FACC);
}

void
starcluster_size_evolution_message(const int size_stellar, const int size_relaxation)
{
    if(!size_stellar && !size_relaxation) {
        message(0, "StarClusterSizeEvolution: off; star-cluster radii SC_Reff stay at their initial SC_initReff.\n");
        return;
    }
    message(0, "StarClusterSizeEvolution=1: the effective radius SC_Reff of the star clusters on BHs evolves "
               "(Guerra et al. 2026 sect. 2.2.3, applied to R_eff = (3/4) r_h at fixed profile shape; no tidal-shock term) with:%s%s\n",
            size_stellar ? " [stellar evolution] adiabatic expansion R_eff -> R_eff m_old/m_new from the stellar-evolution mass loss;" : "",
            size_relaxation ? " [two-body relaxation, GB08] dR_eff/R_eff = (2 - zeta/xi) dm_rlx/m (Guerra et al. 2026 eq. 22)." : "");
}
