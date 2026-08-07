/* Williams et al. 2026 VMS-in-a-dense-star-cluster model (no BH track).
 *
 * C port of the AUTHOR'S ORIGINAL code (code/CWmodel_oricode,
 * src/timescales/analysis/modelv2.py, the STAR-ONLY branch of
 * create_dynamical_model_integral), tracking upstream commit 0f2d04a
 * (2026-08-06, "Fixed bugs following Yihao's comments").  NOTE: that commit
 * postdates the PUBLISHED Figure 4 and raises the inflow by 2*(1+alpha)=4.4x,
 * i.e. M_VMS by ~+0.31 dex, so this model no longer reproduces the figure as
 * printed in the paper.  Every rate here is a
 * closed-form antiderivative lifted directly from her integrals.py / physics
 * modules (Mdot_pl_no_bh_limits, Mdot_deplete_noBH_limits,
 * Mdot_binaries_pl_limits, r_no_relax, stellar_df_radius), so there are no
 * numerical integrals or root-finds.
 *
 * Model choices FROZEN at her modelv2.py defaults:
 *   e=0.5, cv=1, lnLambda=ln(0.1 N) (coulomb_log, Hamilton+18/B&T),
 *   Mstar=Mc=1 Msun, no mass_accretion_ratio on Mdot_df (deleted in 0f2d04a), f_vms=2e-2,
 *   binary heating ON with fixed sigma=20 km/s and mubs=0.153619, mubb=0.17507,
 *   r_min = relaxation radius r_no_relax (t_relax=P_orb), r_df from
 *   stellar_df_radius with q=Mc/Mstar=1 (t_df=t_relax), disruption time
 *   t_d = min(t_ms(1Msun), t_universe, 0.2*t_relax(r_max)) then capped by
 *   t_ms(2Msun) for the inflow integrals (her `newts`).  M_VMS is her DIRECT
 *   equilibrium (1-f_vms)Mdot_in = C M^2.1, no fml_vms fixed-point iteration.
 * The density power-law index alpha (rho ~ r^-alpha) is a caller input
 * (CWmodelAlpha; default 1.2).  The Rose et al. 2020 eccentricity functions
 * f1(e,alpha), f2(e,alpha) are recomputed at runtime from the Gauss
 * hypergeometric 2F1 (cw_f1_ecc/cw_f2_ecc) exactly as her get_ecc_functions.
 *
 * Simulation-integration adaptations (see cwmodel.h and the CW_* notes below):
 *   - the halo interaction/merger time (t_merger) is OMITTED (the C seeding
 *     path has no neighbour-halo/cosmology context; dropping it only lengthens
 *     t_d);
 *   - the mean-density cap (rho_mean >= CW_RHO_MEAN_CAP) short-circuits to
 *     CW_HIGHRHO_MBH_FRAC*M_cl (an MP-Gadget safety net, not in her code);
 *   - the metallicity is the per-cluster simulation Z (CWmodelMetallicity),
 *     not her hard-coded Z=0.1 solar, and is used with NO lower clamp (the
 *     former CW_ZRATIO_FLOOR = 1e-4 has been removed); Z <= 0 returns M_cl,
 *     the model's own zero-wind limit;
 *   - f_IMF is MP-Gadget's Chabrier value (0.0969) rather than modelv2.py's
 *     Salpeter 0.0649, so the seed model uses the same IMF as the metal return.
 *
 * All internal calculations are in CGS with the same physical constants as the
 * Python model.  Self-contained: only math.h, no simulation state. */

#include <math.h>
#include "cwmodel.h"

/* --- physical constants (CGS), identical to the Python model --- */
#define CW_G     6.67430e-8            /* cm^3 g^-1 s^-2 */
#define CW_MSUN  1.98892e33            /* g */
#define CW_RSUN  6.957e10              /* cm */
#define CW_PC    3.0856775814913673e18 /* cm */
#define CW_YR    3.1557600e7           /* s (Julian year) */
#define CW_KMS   1.0e5                 /* cm/s */

/* --- frozen model parameters (modelv2.py defaults) --- */
#define CW_E            0.5            /* orbital eccentricity (Rose et al. 2020) */
#define CW_CV           1.0            /* Eq. 12 sigma coefficient (get_veldisp_constant) */
#define CW_LAM          0.1            /* Coulomb log: ln(lam*N) = coulomb_log(...) */
/* IMF mass fraction f^IMF_{M*} = imf.mass_fraction(Mstar, Mstar+0.5 Msun), i.e. the
 * fraction of the IMF's MASS in [1, 1.5] Msun.
 *
 * modelv2.py assumes a Salpeter IMF (alpha=2.35 over 0.1-100 Msun), giving 0.0649,
 * and that is what this port used to carry.  MP-Gadget's own IMF is the Chabrier
 * (2003) of libgadget/metal_return.c:chabrier_imf, normalised over
 * [MINMASS, MAXMASS] = [0.1, 40] Msun (metal_tables.h), for which the same fraction
 * is 0.0969 -- 1.494x the Salpeter value.  Using the simulation's own IMF here keeps
 * the seed model consistent with the metal return and star formation it runs
 * alongside.
 *
 * CAUTION: this number is tied to metal_return.c's chabrier_imf AND to
 * [MINMASS, MAXMASS].  If either changes, recompute it as
 *     int_1^1.5 m xi(m) dm / int_MINMASS^MAXMASS m xi(m) dm.
 * (It is frozen rather than integrated at runtime to keep this file self-contained:
 * only math.h, no simulation state.)
 *
 * NOTE f_IMF is NOT a simple rescaling of M_VMS.  It multiplies cw_mdot_df_anti and
 * cw_mdot_dep_anti but NOT cw_mdot_bin_anti, so Mdot_in is affine in f_IMF:
 *   Mdot_in = (1-f_vms) [ f_IMF (A_df - A_dep) - A_bin ].
 * Clusters that only marginally beat the binary-heating term gain much more than the
 * naive (0.0969/0.0649)^(1/2.1) = 1.21x. */
#define CW_FIMF         0.0969
/* (CW_ACCR_RATIO: the mass_accretion_ratio=0.5 factor that used to multiply
 * Mdot_df was removed with CWmodel_oricode commit 0f2d04a -- see the header.) */
#define CW_FVMS         2.0e-2         /* constant VMS mass-loss fraction f_vms */
/* Binary-heating magnitude (Mdot_binaries_pl_limits active-version defaults). */
#define CW_MU_BS        0.153619       /* mu_bs (binary-single heating) */
#define CW_MU_BB        0.17507        /* mu_bb (binary-binary heating) */
#define CW_SIGMA_BIN    (20.0*CW_KMS)  /* fixed velocity dispersion in the (3 sigma)^2 term */
/* MS lifetimes: main_sequence_lifetime_approximation = 1e10 yr (Msun/M)^2.5. */
#define CW_TMS1         (1.0e10 * CW_YR)                 /* t_ms(1 Msun) */
#define CW_TMS2         (1.0e10 * 0.176776695296637 * CW_YR)  /* t_ms(2 Msun) = 1e10*(1/2)^2.5 yr */
/* Mean-density system exclusion (MP-Gadget safety net, NOT in modelv2.py):
 * at mean densities inside r_max at or above the cap the collision model is not
 * applied; the seed is CW_HIGHRHO_MBH_FRAC of the cluster mass instead. */
#define CW_RHO_MEAN_CAP     6.0e7      /* Msun/pc^3 */
#define CW_HIGHRHO_MBH_FRAC 0.01       /* seed mass fraction of M_cl above the cap */
/* Metallicity for the Vink 2018 wind: Z/Zsun with Zsun=0.0134 (the code's
 * convention, cf. get_seed_metallicity_factor).  Used as given -- the former
 * CW_ZRATIO_FLOOR = 1e-4 lower clamp has been removed; see the Z <= 0 note in
 * cw_final_vms_mass_msun. */
#define CW_ZSUN         0.0134

/* Derived cluster structure (Cluster with Mstar=Mc=1 Msun). */
struct cw_cluster {
    double alpha;   /* density power-law index (CWmodelAlpha input) */
    double M;       /* total stellar mass [g] */
    double r_max;   /* outer (virial) radius [cm] */
    double Mstar;   /* typical stellar mass = collider mass Mc [g] */
    double Rstar;   /* stellar radius at Mstar [cm] */
    double rc;      /* contact radius = R* + Rc (Mc = M*) [cm] */
    double c_M;     /* M(r) = c_M r^(3-alpha)     [= her cm] */
    double c_rho;   /* rho(r) = c_rho r^-alpha    [= her crho] */
    double lnL;     /* Coulomb log ln(0.1 N) */
    double F1;      /* f1(e) rc^2 (Eq. 8) */
    double F2;      /* 2 G f2(e) rc (M*+Mc) */
};

/* MS mass-radius: R = Rsun * m^0.8 (m<1) or m^0.57 (m>=1) Msun
 * (stellar_radius_approximation).  Only ever called at m=1 here (R = Rsun). */
static double cw_stellar_radius(double m_msun)
{
    double p = (m_msun < 1.0) ? 0.8 : 0.57;
    return CW_RSUN * pow(m_msun, p);
}

/* Gauss hypergeometric 2F1(a,b;c;z) for the Rose et al. 2020 eccentricity
 * functions (scipy.special.hyp2f1).  Only ever called with c=1, a=0.5,
 * b=alpha-0.5 or alpha-1.5, and z = 2e/(e-1) < 0 or z = 2e/(e+1) in (0,1).
 * A negative z is mapped into (0,1) by the Pfaff transformation
 *   2F1(a,b;c;z) = (1-z)^-a 2F1(a, c-b; c; z/(z-1)),
 * so the power series always converges geometrically. */
static double cw_hyp2f1(double a, double b, double c, double z)
{
    double pref = 1.0;
    if(z < 0.0) {                       /* Pfaff -> argument in (0,1) */
        pref = pow(1.0 - z, -a);
        b = c - b;
        z = z / (z - 1.0);
    }
    double term = 1.0, sum = 1.0;
    int n;
    for(n = 0; n < 1000; n++) {
        term *= (a + n) * (b + n) / ((c + n) * (n + 1)) * z;
        sum += term;
        if(fabs(term) <= 1e-15 * fabs(sum))
            break;
    }
    return pref * sum;
}

/* Rose et al. 2020 Eqs. 20-21 (get_ecc_functions): collision-rate eccentricity
 * factors as functions of e and the density index alpha. */
static double cw_f1_ecc(double e, double alpha)
{
    return 0.5 * pow(1.0 - e, 0.5 - alpha) * cw_hyp2f1(0.5, alpha - 0.5, 1.0, 2.0 * e / (e - 1.0))
         + 0.5 * pow(1.0 + e, 0.5 - alpha) * cw_hyp2f1(0.5, alpha - 0.5, 1.0, 2.0 * e / (e + 1.0));
}

static double cw_f2_ecc(double e, double alpha)
{
    return 0.5 * pow(1.0 - e, 1.5 - alpha) * cw_hyp2f1(0.5, alpha - 1.5, 1.0, 2.0 * e / (e - 1.0))
         + 0.5 * pow(1.0 + e, 1.5 - alpha) * cw_hyp2f1(0.5, alpha - 1.5, 1.0, 2.0 * e / (e + 1.0));
}

static void cw_cluster_init(struct cw_cluster * cl, double M_msun, double r_max_pc, double alpha)
{
    cl->alpha = alpha;
    cl->M = M_msun * CW_MSUN;
    cl->r_max = r_max_pc * CW_PC;
    cl->Mstar = 1.0 * CW_MSUN;
    cl->Rstar = cw_stellar_radius(1.0);
    cl->rc = 2.0 * cl->Rstar;                    /* sum of radii (Mc = M*) */
    cl->c_M = cl->M / pow(cl->r_max, 3.0 - alpha);
    cl->c_rho = (3.0 - alpha) * cl->c_M / (4.0 * M_PI);
    cl->lnL = log(CW_LAM * cl->M / cl->Mstar);
    cl->F1 = cw_f1_ecc(CW_E, alpha) * cl->rc * cl->rc;
    cl->F2 = 2.0 * CW_G * cw_f2_ecc(CW_E, alpha) * cl->rc * (cl->Mstar + cl->Mstar);
}

/* ---- structural profiles (r in cm) ---- */
static double cw_Menc(const struct cw_cluster * cl, double r)
{
    return cl->c_M * pow(r, 3.0 - cl->alpha);
}

static double cw_rho(const struct cw_cluster * cl, double r)
{
    return cl->c_rho * pow(r, -cl->alpha);
}

static double cw_sigma(const struct cw_cluster * cl, double r)
{
    return sqrt(CW_CV * CW_G * cw_Menc(cl, r) / ((1.0 + cl->alpha) * r));
}

/* Relaxation time (relaxation_timescale): 0.34 sigma^3 / (G^2 rho Mstar lnL). */
static double cw_trelax(const struct cw_cluster * cl, double r)
{
    double s = cw_sigma(cl, r);
    return 0.34 * s * s * s / (CW_G * CW_G * cw_rho(cl, r) * cl->Mstar * cl->lnL);
}

/* ---- radii (closed form; both in cm) ---- */

/* r_no_relax: radius where t_relax = P_orb (inner integration limit rmin).
 * rho0*r0^alpha == c_rho, mass_spectrum_psi=1. */
static double cw_r_no_relax(const struct cw_cluster * cl)
{
    double a = cl->alpha;
    double num = 8.0 * M_PI * 0.34 * pow(CW_CV, 1.5) * cl->c_rho;
    double denom = pow(1.0 + a, 1.5) * (3.0 - a) * (3.0 - a) * cl->Mstar * cl->lnL;
    return pow(num / denom, 1.0 / (a - 3.0));
}

/* stellar_df_radius: solve t_df(r) = td with q = Mc/Mstar = 1 (t_df = t_relax),
 * closed form r = RHS^(1/(3-alpha/2)).  td in seconds. */
static double cw_r_df_stellar(const struct cw_cluster * cl, double td_s)
{
    double a = cl->alpha, q = 1.0;
    double sfac = CW_CV * CW_G / (1.0 + a);
    double RHS = (CW_G * CW_G * cl->Mstar * td_s * cl->lnL) / (0.34 * q)
        * cl->c_rho / (pow(sfac, 1.5) * pow(cl->c_M, 1.5));
    return pow(RHS, 1.0 / (3.0 - a / 2.0));
}

/* ---- mass-rate antiderivatives (evaluate at r_df minus at rmin; g/s) ---- */

/* Mdot_pl_no_bh_limits integrand antiderivative (DF inflow), q=Mc/Mstar=1,
 * reduced_mass=Mstar/2.  func1+func2-func3-func4.  ts in seconds. */
static double cw_mdot_df_anti(const struct cw_cluster * cl, double ts_s, double r)
{
    double a = cl->alpha, cv = CW_CV, G = CW_G;
    double Ms = cl->Mstar, Mc = cl->Mstar;
    double rstar = cl->Rstar, rcoll = cl->Rstar;
    double crho = cl->c_rho, cm = cl->c_M, F1 = cl->F1, F2 = cl->F2, lnL = cl->lnL;
    double Massratio = Mc / Ms;                       /* = 1 */
    double reduced_mass = Ms * Mc / (Ms + Mc);        /* = Mstar/2 */
    double pref_num1 = M_PI * G * G * ts_s * CW_FIMF * (Ms + Mc) * Massratio * lnL;
    double pref_den1 = 0.34 * Ms;
    double pref_num2 = M_PI * ts_s * CW_FIMF * reduced_mass * Massratio * G * lnL * (3.0 - a) * (Ms + Mc);
    double pref_den2 = 0.34 * ((Ms * Ms / rstar) + (Mc * Mc / rcoll)) * Ms;
    double f1 = pref_num1 / pref_den1 * F1 * (3.0 - a) * (1.0 + a) * (1.0 + a) / cv / G
        * crho * crho / (1.0 - 2.0 * a) * pow(r, 1.0 - 2.0 * a);
    /* (1+a)^2: raised from (1+a) upstream in CWmodel_oricode commit 0f2d04a
     * (2026-08-06) so integrate_func2 matches the printed Eq. A11 term 2. */
    double f2 = pref_num1 / pref_den1 * F2 * (3.0 - a) * (1.0 + a) * (1.0 + a) / cv / cv / G / G
        * crho * crho / cm / (-a - 1.0) * pow(r, -1.0 - a);
    double f3 = pref_num2 / pref_den2 * F1 * (3.0 - a) * crho * crho * cm
        / (3.0 - 3.0 * a) * pow(r, 3.0 - 3.0 * a);
    double f4 = pref_num2 / pref_den2 * F2 * (3.0 - a) * (1.0 + a) / cv / G
        * crho * crho / (1.0 - 2.0 * a) * pow(r, 1.0 - 2.0 * a);
    return f1 + f2 - f3 - f4;
}

/* Mdot_deplete_noBH_limits antiderivative (depletion rate). */
static double cw_mdot_dep_anti(const struct cw_cluster * cl, double r)
{
    double a = cl->alpha, cv = CW_CV, G = CW_G;
    double Ms = cl->Mstar, Mc = cl->Mstar;
    double rstar = cl->Rstar, rcoll = cl->Rstar;
    double crho = cl->c_rho, cm = cl->c_M, F1 = cl->F1, F2 = cl->F2;
    double reduced_mass = Ms * Mc / (Ms + Mc);
    double D = G * Ms * Ms / rstar + G * Mc * Mc / rcoll;
    double pref = 4.0 * M_PI * M_PI * (Ms + Mc) * CW_FIMF * crho * crho / (Ms * Ms);
    double sfac = cv * G / (1.0 + a);
    double f1 = (F1 - F2 * reduced_mass / D) * sqrt(sfac) * sqrt(cm)
        / (4.0 - 2.5 * a) * pow(r, 4.0 - 2.5 * a);
    double f2 = F1 * reduced_mass / D * pow(sfac, 1.5) * pow(cm, 1.5)
        / (6.0 - 3.5 * a) * pow(r, 6.0 - 3.5 * a);
    double f3 = F2 / sqrt(sfac) / sqrt(cm)
        / (2.0 - 1.5 * a) * pow(r, 2.0 - 1.5 * a);
    return pref * (f1 - f2 + f3);
}

/* Mdot_binaries_pl_limits antiderivative (binary heating, fixed sigma). */
static double cw_mdot_bin_anti(const struct cw_cluster * cl, double r)
{
    double a = cl->alpha, cv = CW_CV, G = CW_G, Ms = cl->Mstar;
    double crho = cl->c_rho, cm = cl->c_M;
    double pref = 4.0 * M_PI * Ms * G * G / (9.0 * CW_SIGMA_BIN * CW_SIGMA_BIN)
        * (CW_MU_BS + CW_MU_BB) * sqrt((1.0 + a) / cv / G) / sqrt(cm)
        * crho * crho / (2.0 - 1.5 * a);
    return pref * pow(r, 2.0 - 1.5 * a);
}

/* Full pipeline for one dense star cluster (modelv2.py STAR-ONLY branch).
 * See cwmodel.h for the input/output conventions. */
double cw_final_vms_mass_msun(double M_msun, double r_max_pc, double Z_massfrac,
                              double t_universe_sec, double alpha)
{
    struct cw_cluster cl;
    if(M_msun <= 0 || r_max_pc <= 0 || t_universe_sec <= 0)
        return 0.0;
    /* rho_mean >= CW_RHO_MEAN_CAP: MP-Gadget safety net (not in modelv2.py);
     * seed a fixed fraction of the cluster mass instead of the collision M_VMS. */
    double rho_mean = M_msun / (4.0 * M_PI / 3.0 * r_max_pc * r_max_pc * r_max_pc);
    if(rho_mean >= CW_RHO_MEAN_CAP)
        return CW_HIGHRHO_MBH_FRAC * M_msun;
    cw_cluster_init(&cl, M_msun, r_max_pc, alpha);

    /* Disruption time.  ts = min(t_ms(1Msun), t_universe, 0.2 t_relax(r_max));
     * t_merger OFF.  The inflow integrals use newts = min(ts, t_ms(2Msun)),
     * while r_df is set from ts (matching modelv2.py's ts vs newts split). */
    double t_cc = 0.2 * cw_trelax(&cl, cl.r_max);
    double ts = CW_TMS1;
    if(t_universe_sec < ts) ts = t_universe_sec;
    if(t_cc < ts) ts = t_cc;
    double newts = (ts < CW_TMS2) ? ts : CW_TMS2;

    double rmin = cw_r_no_relax(&cl);
    double r_df = cw_r_df_stellar(&cl, ts);
    if(r_df > cl.r_max)                 /* keep the inflow inside the cluster */
        r_df = cl.r_max;
    if(rmin >= r_df)                    /* no migration region -> no VMS */
        return 0.0;

    /* Net inflow: (1-f_vms) (Mdot_df - Mdot_dep - Mdot_bin).  The former
     * mass_accretion_ratio=0.5 factor on Mdot_df was deleted upstream in
     * CWmodel_oricode commit 0f2d04a (2026-08-06); Eq. 23 never had it. */
    double Mdot_df =
        cw_mdot_df_anti(&cl, newts, r_df) - cw_mdot_df_anti(&cl, newts, rmin);
    double Mdot_dep = cw_mdot_dep_anti(&cl, r_df) - cw_mdot_dep_anti(&cl, rmin);
    double Mdot_bin = cw_mdot_bin_anti(&cl, r_df) - cw_mdot_bin_anti(&cl, rmin);
    double Mdot_in = (1.0 - CW_FVMS) * (Mdot_df - Mdot_dep - Mdot_bin);   /* g/s */
    if(Mdot_in <= 0.0)
        return 0.0;

    /* Equilibrium VMS mass (direct): (1-f_vms)Mdot_in = C M^2.1, C = 1e-9.13 Z^0.74,
     * Z = Z/Zsun.  No fml_vms iteration (f_vms is the constant CW_FVMS above).
     *
     * The metallicity is used as given: there is no lower clamp.  (A Z/Zsun >= 1e-4
     * floor used to sit here purely to stop C -> 0 diverging; it was not part of
     * modelv2.py, which runs at a single fixed Z = 0.1 solar.)  The only case that
     * still needs handling is Z <= 0 exactly: the wind vanishes, so the equilibrium
     * VMS mass is unbounded and the model's own limit is that the VMS consumes the
     * whole cluster.  Return that limit directly rather than letting an infinity
     * propagate -- cw_seed_mass_code caps at M_msun anyway, so this is the same
     * answer, just finite. */
    double Mdot_in_msun_yr = Mdot_in / CW_MSUN * CW_YR;
    double Z_over_Zsun = Z_massfrac / CW_ZSUN;
    if(Z_over_Zsun <= 0.0)
        return M_msun;
    double C = pow(10.0, -9.13) * pow(Z_over_Zsun, 0.74);
    return pow(Mdot_in_msun_yr / C, 1.0 / 2.1);
}
