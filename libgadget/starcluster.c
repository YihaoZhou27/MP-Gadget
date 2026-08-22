#include <math.h>
#include "starcluster.h"

/* Tabulated P/k_B [K/cm^3] vs Cluster Formation Efficiency (Kruijssen 2012) */
#define N_CFE_TABLE 67

static const double cfe_table_Poverk[N_CFE_TABLE] = {
    75.888091, 91.484527, 110.735878, 131.467179, 155.943689,
    186.488064, 224.032756, 282.488829, 338.376028, 422.064647,
    514.355473, 623.015161, 750.620938, 899.819753, 1085.065898,
    1286.583593, 1555.665401, 1892.352829, 2382.652936, 2968.777538,
    3626.366238, 4484.868455, 5518.214167, 6831.201956, 8341.096740,
    10142.394847, 12668.010162, 15939.442273, 19636.698438, 25117.899642,
    31831.733444, 40304.926504, 52193.261728, 67287.670496, 88110.717732,
    115399.886091, 152051.374284, 196462.212045, 253377.852017, 351128.389429,
    470419.193665, 651649.930829, 908223.662905, 1289326.914421, 1906333.776160,
    2631561.596253, 3731824.878999, 5191600.279572, 7230780.207271, 10116887.217027,
    14518692.705531, 20441980.926123, 27690858.698397, 40098574.105238, 55794695.010577,
    77439737.574962, 112726596.098589, 174062391.682909, 259511742.091851, 411649414.644125,
    598820500.350550, 823434675.036126, 1054401293.326127, 1879928071.897331,
    2609477759.152397, 3822531039.908538, 6292604717.832682
};

static const double cfe_table_CFE[N_CFE_TABLE] = {
    0.005514049, 0.006046433, 0.006683609, 0.007409001, 0.008154323,
    0.008952615, 0.009975734, 0.011231373, 0.012543065, 0.014076916,
    0.015599012, 0.017298342, 0.018983374, 0.020977276, 0.023109256,
    0.025537182, 0.028141417, 0.031394351, 0.035241752, 0.039171890,
    0.043549434, 0.048701613, 0.054008474, 0.060259324, 0.065853360,
    0.072597388, 0.080640742, 0.089692349, 0.099439882, 0.110838785,
    0.123057962, 0.136178772, 0.149834902, 0.166635835, 0.185049704,
    0.204865894, 0.225981950, 0.247341368, 0.268613203, 0.293733389,
    0.318695094, 0.347426172, 0.375839574, 0.406417462, 0.442397604,
    0.473380274, 0.502709684, 0.529826907, 0.560423707, 0.588666877,
    0.619465258, 0.647783520, 0.672195835, 0.698129726, 0.719178155,
    0.736153608, 0.761007925, 0.783092441, 0.810441439, 0.826671981,
    0.844528326, 0.863334047, 0.875394028, 0.898447765, 0.915842445,
    0.933183799, 0.951698429
};

/*
 * Log-linear interpolation for Cluster Formation Efficiency.
 * Matches the Python: interp1d(log10(Poverk), log10(CFE), kind="linear", fill_value="extrapolate"),
 * except that the result is capped at 1: Gamma is a mass fraction, and the upward extrapolation
 * of the last table segment (slope d logCFE/d logP ~ 0.04) would otherwise exceed 1 for
 * P/k_B > 2.2e10 K cm^-3 (reached by the densest star-forming gas at high z, giving Gamma up to ~1.1).
 */
double get_cluster_formation_efficiency(double Pressure_over_kB)
{
    if(Pressure_over_kB <= 0)
        return cfe_table_CFE[0];

    double logP = log10(Pressure_over_kB);

    int lo = 0, hi = N_CFE_TABLE - 1;

    if(Pressure_over_kB <= cfe_table_Poverk[0]) {
        lo = 0; hi = 1;
    } else if(Pressure_over_kB >= cfe_table_Poverk[N_CFE_TABLE - 1]) {
        lo = N_CFE_TABLE - 2; hi = N_CFE_TABLE - 1;
    } else {
        while(hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if(cfe_table_Poverk[mid] <= Pressure_over_kB)
                lo = mid;
            else
                hi = mid;
        }
    }

    double logP0 = log10(cfe_table_Poverk[lo]);
    double logP1 = log10(cfe_table_Poverk[hi]);
    double logC0 = log10(cfe_table_CFE[lo]);
    double logC1 = log10(cfe_table_CFE[hi]);

    double t = (logP - logP0) / (logP1 - logP0);
    double cfe = pow(10.0, logC0 + t * (logC1 - logC0));
    /* Cluster formation efficiency is a mass fraction: cap the extrapolation at 1. */
    if(cfe > 1.0)
        cfe = 1.0;
    return cfe;
}

/* ---- Kruijssen (2012) local model, eq. 26, evaluated directly (StarClusterCFEsigma > 0) ---------------------- */
#define CFE_K12_SSFR_FF   0.012   /* star formation rate per free-fall time */
#define CFE_K12_EPS_CORE  0.5     /* maximum (core) star formation efficiency */
#define CFE_K12_B         0.5     /* turbulence forcing parameter in sigma_rho^2 = ln(1 + 3 b^2 Mach^2) (eq. 4) */
#define CFE_K12_NX        201     /* quadrature points in ln x (0.02% accuracy against a 1001-point reference) */
#define CFE_K12_LNXMAX    25.0    /* integration range ln x in [-LNXMAX, LNXMAX] */

/* Star formation efficiency at overdensity x = rho_x / rho: the minimum of the core efficiency,
 * the feedback-limited efficiency (eq. 46) and the incomplete-star-formation efficiency (eq. 22). */
static double
cfe_k12_eps(const double x, const double rho, const double sigma, const double G, const double t_sn, const double phi_fb, const double t_inc)
{
    const double t_ff = sqrt(3.0 * M_PI / (32.0 * G * rho * x));                                   /* eq. 15 */
    const double eps_fb = CFE_K12_SSFR_FF * t_sn / (2.0 * t_ff)
                        * (1.0 + sqrt(1.0 + 4.0 * t_ff * sigma * sigma / (phi_fb * CFE_K12_SSFR_FF * t_sn * t_sn * x)));
    const double eps_inc = CFE_K12_SSFR_FF * t_inc / t_ff;
    double eps = CFE_K12_EPS_CORE;
    if(eps_fb < eps)
        eps = eps_fb;
    if(eps_inc < eps)
        eps = eps_inc;
    return eps;
}

double
get_cluster_formation_efficiency_k12(double rho, double sigma, double cs, double G, double t_sn, double phi_fb, double t_inc)
{
    if(rho <= 0)
        return 0;
    if(sigma < 0)
        sigma = 0;
    const double mach = sigma / cs;
    const double s2 = log(1.0 + 3.0 * CFE_K12_B * CFE_K12_B * mach * mach);
    /* No turbulence: the density PDF is a delta function at x = 1 and the bound fraction is eps(1) / eps_core. */
    if(s2 < 1e-10)
        return cfe_k12_eps(1.0, rho, sigma, G, t_sn, phi_fb, t_inc) / CFE_K12_EPS_CORE;
    const double dlnx = 2.0 * CFE_K12_LNXMAX / (CFE_K12_NX - 1);
    const double norm = 1.0 / sqrt(2.0 * M_PI * s2);
    double num = 0, den = 0;
    int k;
    for(k = 0; k < CFE_K12_NX; k++) {
        const double lnx = -CFE_K12_LNXMAX + k * dlnx;
        const double x = exp(lnx);
        const double pdf = norm * exp(-(lnx + 0.5 * s2) * (lnx + 0.5 * s2) / (2.0 * s2));          /* eqs 2-3 */
        const double w = x * pdf;
        if(w <= 0)
            continue;
        const double eps = cfe_k12_eps(x, rho, sigma, G, t_sn, phi_fb, t_inc);
        num += eps * eps * w;             /* gamma eps x p with gamma = eps / eps_core (eqs 25-26) */
        den += eps * w;
    }
    if(den <= 0)
        return 0;
    double cfe = num / (den * CFE_K12_EPS_CORE);
    if(cfe > 1.0)
        cfe = 1.0;
    return cfe;
}
