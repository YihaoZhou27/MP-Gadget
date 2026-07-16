/* Williams et al. 2026 VMS-in-a-dense-star-cluster model (no BH track).
 *
 * Line-for-line C port of code_v2/WilliamModel/scmodel.py (final_vms_mass and
 * everything it calls) with the model choices FROZEN at the scmodel.py Params
 * defaults except inflow_rmin_factor = 5:
 *   c_v=1, lambda=0.1 (lnLambda=ln(0.1 N)), f_IMF=0.3, e=0.5,
 *   Mstar=Mc=1 Msun, fml_vms_cap=2e-2, t_ms=1e10 yr, inflow_mode='literal'
 *   (Mdot_in = Mdot_df(A9/A11) - Mdot_dep), t_cc=0.2*t_relax(r_h),
 *   binary heating OFF (Mdot_bin=0), halo-merger time OFF (t_mrg=inf),
 *   inflow_rmin_factor=5 (kappa=5; r_min_inflow = 5*r_min_df).
 * The density power-law index alpha (rho ~ r^-alpha) is a caller input
 * (CWmodelAlpha; default 1.2).  Because the Rose et al. 2020 eccentricity
 * functions f1(e,alpha), f2(e,alpha) depend on alpha, they are recomputed at
 * runtime from the Gauss hypergeometric 2F1 (cw_f1_ecc/cw_f2_ecc), matching
 * scmodel.py f1_ecc/f2_ecc exactly for any alpha; e stays fixed at 0.5.
 *
 * All internal calculations are in CGS with the same physical constants as
 * scmodel.py (NOT physconst.h, so the C and Python models agree exactly).
 * Self-contained: only math.h, no simulation state. */

#include <math.h>
#include "cwmodel.h"

/* --- physical constants (CGS), identical to scmodel.py --- */
#define CW_G     6.67430e-8            /* cm^3 g^-1 s^-2 */
#define CW_MSUN  1.98892e33            /* g */
#define CW_RSUN  6.957e10              /* cm */
#define CW_PC    3.0856775814913673e18 /* cm */
#define CW_YR    3.1557600e7           /* s (Julian year) */

/* --- frozen model parameters (scmodel.py Params defaults, kappa=5) --- */
#define CW_E            0.5            /* orbital eccentricity (Rose et al. 2020) */
#define CW_CV           1.0            /* Eq. 12 sigma coefficient */
#define CW_LAM          0.1            /* Coulomb log: ln(lam*N) */
#define CW_FIMF         0.3            /* IMF fraction at the typical mass */
#define CW_FML_VMS_CAP  2.0e-2         /* cap on the VMS mass-loss fraction */
#define CW_TMS          (1.0e10 * CW_YR)  /* MS lifetime at 1 Msun */
#define CW_RMIN_FACTOR  5.0            /* kappa: r_min_inflow = kappa*r_min_df */
#define CW_NPTS         2000           /* log-grid points for the integrals */
/* Mean-density system exclusion (scmodel.py rho_mean_cap, paper Table 1):
 * at mean densities inside r_max at or above the cap the collision model is
 * not applied; the seed is CW_HIGHRHO_MBH_FRAC of the cluster mass instead. */
#define CW_RHO_MEAN_CAP     6.0e7      /* Msun/pc^3 */
#define CW_HIGHRHO_MBH_FRAC 0.01       /* seed mass fraction of M_cl above the cap */
/* Metallicity for the Vink 2018 wind: Z/Zsun with Zsun=0.0134 (the code's
 * convention, cf. get_seed_metallicity_factor), floored so pristine (Z=0)
 * clusters keep a finite wind (C -> 0 would make M_VMS diverge). */
#define CW_ZSUN         0.0134
#define CW_ZRATIO_FLOOR 1.0e-4

/* Derived cluster structure (scmodel.py class Cluster with Mstar=Mc=1 Msun). */
struct cw_cluster {
    double alpha;   /* density power-law index (CWmodelAlpha input) */
    double M;       /* total stellar mass [g] */
    double r_max;   /* outer (virial) radius [cm] */
    double Mstar;   /* typical stellar mass = collider mass Mc [g] */
    double Rstar;   /* stellar radius at Mstar [cm] */
    double rc;      /* contact radius = 2 R* (Mc = M*) [cm] */
    double c_M;     /* M(r) = c_M r^(3-alpha) */
    double c_rho;   /* rho(r) = c_rho r^-alpha */
    double r_h;     /* half-mass radius [cm] */
    double lnL;     /* Coulomb log ln(0.1 N) */
    double F1;      /* f1(e) rc^2 (Eq. 8) */
    double F2;      /* 2 G f2(e) rc (M*+Mc) */
    double bindE;   /* G Mi^2/Ri + G M*^2/R*  (f_ml denominator, Eq. 16) */
};

/* MS mass-radius (rough): R = Rsun * m^0.8, m in Msun (scmodel.py stellar_radius).
 * Only ever called at m=1 here, but kept for fidelity. */
static double cw_stellar_radius(double m_msun)
{
    return CW_RSUN * pow(m_msun, 0.8);
}

/* Gauss hypergeometric 2F1(a,b;c;z) for the Rose et al. 2020 eccentricity
 * functions (scipy.special.hyp2f1 in scmodel.py).  Only ever called with c=1,
 * a=0.5, b=alpha-0.5 or alpha-1.5, and z = 2e/(e-1) < 0 or z = 2e/(e+1) in
 * (0,1).  A negative z is mapped into (0,1) by the Pfaff transformation
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

/* Rose et al. 2020 Eqs. 20-21 (scmodel.py f1_ecc/f2_ecc): collision-rate
 * eccentricity factors as functions of e and the density index alpha. */
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
    cl->r_h = cl->r_max * pow(0.5, 1.0 / (3.0 - alpha));
    cl->lnL = log(CW_LAM * cl->M / cl->Mstar);
    cl->F1 = cw_f1_ecc(CW_E, alpha) * cl->rc * cl->rc;
    cl->F2 = 2.0 * CW_G * cw_f2_ecc(CW_E, alpha) * cl->rc * (cl->Mstar + cl->Mstar);
    cl->bindE = CW_G * cl->Mstar * cl->Mstar / cl->Rstar * 2.0;  /* Mi=M*, Ri=R* */
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

static double cw_Porb(const struct cw_cluster * cl, double r)
{
    return 2.0 * M_PI * sqrt(r * r * r / (CW_G * cw_Menc(cl, r)));
}

/* ---- timescales ---- */
static double cw_trelax(const struct cw_cluster * cl, double r)
{
    double s = cw_sigma(cl, r);
    return 0.34 * s * s * s / (CW_G * CW_G * cw_rho(cl, r) * cl->Mstar * cl->lnL);
}

static double cw_tcoll(const struct cw_cluster * cl, double r)
{
    double s = cw_sigma(cl, r);
    double n = cw_rho(cl, r) / cl->Mstar;
    double rate = M_PI * n * s * (cl->F1 + cl->F2 / (s * s));
    return 1.0 / rate;
}

/* ---- collision mass loss (Eq. 16) and dynamical friction ---- */
static double cw_fml(const struct cw_cluster * cl, double r)
{
    double mu = cl->Mstar * 0.5;                 /* Mi = Mstar: mu = Mstar/2 */
    double s = cw_sigma(cl, r);
    double val = mu * s * s / cl->bindE;
    /* cap just below 1: fml=1 is fully destructive; avoids 0/0 in t_df */
    return (val < 1.0 - 1.0e-9) ? val : 1.0 - 1.0e-9;
}

/* Migrating collision-product mass M_p = (1-fml)(Mc+M*). */
static double cw_Mproduct(const struct cw_cluster * cl, double r)
{
    return (1.0 - cw_fml(cl, r)) * 2.0 * cl->Mstar;
}

static double cw_tdf(const struct cw_cluster * cl, double r)
{
    return (cl->Mstar / cw_Mproduct(cl, r)) * cw_trelax(cl, r);
}

/* Relaxation floor: the radius where t_relax(r) = P_orb(r) (paper's inner limit
 * for the df/inflow integrals).  Geometric bisection, as scmodel.py r_min_df. */
static double cw_rmin_df(const struct cw_cluster * cl)
{
    double lo = cl->rc, hi = cl->r_max;
    int it;
    if(cw_trelax(cl, hi) - cw_Porb(cl, hi) <= 0.0)
        return hi;      /* relaxation faster than orbit everywhere -> r_max */
    if(cw_trelax(cl, lo) - cw_Porb(cl, lo) >= 0.0)
        return lo;
    for(it = 0; it < 80; it++) {
        double mid = sqrt(lo * hi);
        if(cw_trelax(cl, mid) - cw_Porb(cl, mid) < 0.0)
            lo = mid;
        else
            hi = mid;
    }
    return (lo > cl->rc) ? lo : cl->rc;
}

/* Dynamical-friction radius: largest r in [rmin_df, rmax] with t_df(r) <= t_d
 * (t_df increasing in r).  Geometric bisection, as scmodel.py r_df. */
static double cw_rdf(const struct cw_cluster * cl, double t_d, double rmin_df)
{
    double lo = rmin_df, hi = cl->r_max;
    int it;
    if(lo >= hi)
        return lo;
    if(cw_tdf(cl, lo) > t_d)
        return lo;                    /* no migration region */
    if(cw_tdf(cl, hi) <= t_d)
        return hi;
    for(it = 0; it < 80; it++) {
        double mid = sqrt(lo * hi);
        if(cw_tdf(cl, mid) <= t_d)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

/* Trapezoid integral of f over a CW_NPTS-point log grid on [r_lo, r_hi],
 * mirroring np.trapz on np.logspace in scmodel.py.  which selects the
 * integrand (0: Mdot_df literal A9/A11; 1: Mdot_dep, Eq. A14). */
static double cw_inflow_integral(const struct cw_cluster * cl, double t_d,
                                 double r_lo, double r_hi, int which)
{
    double dlog = (log(r_hi) - log(r_lo)) / (CW_NPTS - 1);
    double sum = 0, f_prev = 0, r_prev = 0;
    int i;
    for(i = 0; i < CW_NPTS; i++) {
        double r = exp(log(r_lo) + i * dlog);
        /* stars per unit radius at the typical mass (Eq. 10) */
        double dN_dr = (CW_FIMF / cl->Mstar) * (3.0 - cl->alpha)
            * cl->c_M * pow(r, 2.0 - cl->alpha);
        double Mcoll = (1.0 - cw_fml(cl, r)) * 2.0 * cl->Mstar;
        double f;
        if(which == 0)      /* literal Eq. A9/A11: (t_d/t_coll) dN/dr Mcoll / t_df */
            f = (t_d / cw_tcoll(cl, r)) * dN_dr * Mcoll / cw_tdf(cl, r);
        else                /* depletion Eq. A14: Gamma Mcoll 4 pi r^2 f_IMF n(r) */
            f = (1.0 / cw_tcoll(cl, r)) * Mcoll * 4.0 * M_PI * r * r
                * CW_FIMF * cw_rho(cl, r) / cl->Mstar;
        if(i > 0)
            sum += 0.5 * (f + f_prev) * (r - r_prev);
        f_prev = f;
        r_prev = r;
    }
    return sum;             /* g/s */
}

/* ---- VMS growth: wind, mass-loss fraction, equilibrium (scmodel.py) ---- */
static double cw_vms_radius(double M_vms_msun)
{
    /* Hosokawa+13, Eq. 26 [cm] */
    return 2600.0 * CW_RSUN * sqrt(M_vms_msun / 100.0);
}

static double cw_fml_vms(double M_vms_msun, const struct cw_cluster * cl)
{
    double R_vms = cw_vms_radius(M_vms_msun);
    double M_vms = M_vms_msun * CW_MSUN;
    double sigma2 = 2.0 * CW_G * M_vms / R_vms;              /* Eq. 27, R_Roche ~ R_VMS */
    double mu = M_vms * cl->Mstar / (M_vms + cl->Mstar);
    double denom = CW_G * M_vms * M_vms / R_vms
        + CW_G * cl->Mstar * cl->Mstar / cl->Rstar;
    double val = mu * sigma2 / denom;
    return (val < CW_FML_VMS_CAP) ? val : CW_FML_VMS_CAP;
}

/* Solve (1-fml_vms) Mdot_in = Mdot_wind(M_vms) for M_vms [Msun] (Eqs. 23-27,
 * Vink 2018 wind).  Fixed-point iteration as scmodel.py solve_vms_mass. */
static double cw_solve_vms_mass(double Mdot_in_g_s, double Z_over_Zsun,
                                const struct cw_cluster * cl)
{
    if(Mdot_in_g_s <= 0.0)
        return 0.0;
    double Mdot_in = Mdot_in_g_s / CW_MSUN * CW_YR;          /* Msun/yr */
    double C = pow(10.0, -9.13) * pow(Z_over_Zsun, 0.74);    /* wind prefactor */
    double fml = 0.0, M_vms = 1.0;
    int it;
    for(it = 0; it < 60; it++) {
        double M_new = pow((1.0 - fml) * Mdot_in / C, 1.0 / 2.1);
        if(M_new < 1e-3)
            M_new = 1e-3;
        fml = cw_fml_vms(M_new, cl);
        if(fabs(M_new - M_vms) / (M_new > 1e-30 ? M_new : 1e-30) < 1e-8) {
            M_vms = M_new;
            break;
        }
        M_vms = M_new;
    }
    return M_vms;
}

/* Full pipeline for one dense star cluster (scmodel.py final_vms_mass).
 * See cwmodel.h for the input/output conventions. */
double cw_final_vms_mass_msun(double M_msun, double r_max_pc, double Z_massfrac,
                              double t_universe_sec, double alpha)
{
    struct cw_cluster cl;
    if(M_msun <= 0 || r_max_pc <= 0 || t_universe_sec <= 0)
        return 0.0;
    /* rho_mean >= CW_RHO_MEAN_CAP (scmodel.py mean_density_msun_pc3 vs
     * rho_mean_cap): outside the model's validity; seed a fixed fraction of
     * the cluster mass instead of the collision-inflow M_VMS. */
    double rho_mean = M_msun / (4.0 * M_PI / 3.0 * r_max_pc * r_max_pc * r_max_pc);
    if(rho_mean >= CW_RHO_MEAN_CAP)
        return CW_HIGHRHO_MBH_FRAC * M_msun;
    cw_cluster_init(&cl, M_msun, r_max_pc, alpha);

    /* Disruption time t_d = min(t_ms, t_universe, t_cc); merger time OFF (inf).
     * t_cc = 0.2 t_relax(r_h) (core collapse, half-mass radius). */
    double t_d = CW_TMS;
    if(t_universe_sec < t_d)
        t_d = t_universe_sec;
    double t_cc = 0.2 * cw_trelax(&cl, cl.r_h);
    if(t_cc < t_d)
        t_d = t_cc;

    double rmin_df = cw_rmin_df(&cl);
    double rmin_inflow = CW_RMIN_FACTOR * rmin_df;   /* kappa=5 normalization */
    double r_df = cw_rdf(&cl, t_d, rmin_df);
    if(rmin_inflow >= r_df)                          /* cutoff above migration region */
        return 0.0;

    /* Net inflow (literal mode): Mdot_df - Mdot_dep; binary heating OFF. */
    double Mdot_df = cw_inflow_integral(&cl, t_d, rmin_inflow, r_df, 0);
    double Mdot_dep = cw_inflow_integral(&cl, t_d, rmin_inflow, r_df, 1);
    double Mdot_in = Mdot_df - Mdot_dep;

    double Z_over_Zsun = Z_massfrac / CW_ZSUN;
    if(Z_over_Zsun < CW_ZRATIO_FLOOR)
        Z_over_Zsun = CW_ZRATIO_FLOOR;

    return cw_solve_vms_mass(Mdot_in, Z_over_Zsun, &cl);
}
