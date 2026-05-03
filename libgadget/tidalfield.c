#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils/endrun.h"

#include "tidalfield.h"
#include "slotsmanager.h"
#include "partmanager.h"

/*
 * Tidal field module: computes eigenvalues of the gravitational tidal tensor
 * T_ij = d^2 Phi / (dx_i dx_j) for gas particles.
 *
 * The tidal tensor is accumulated during the gravity short-range tree walk
 * (in gravshort-tree.c). This file provides:
 *   - Parameter handling (GasTidalField on/off)
 *   - Eigenvalue computation from the accumulated tensor
 *   - Trace diagnostics (Poisson equation check)
 */

static struct tidalfield_params TidalParams;

void
set_tidalfield_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        TidalParams.GasTidalField = param_get_int(ps, "GasTidalField");
    }
    MPI_Bcast(&TidalParams, sizeof(struct tidalfield_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int
get_tidalfield_on(void)
{
    return TidalParams.GasTidalField;
}

/* --- Eigenvalue computation for 3x3 real symmetric matrix ---
 *
 * Uses the analytic solution (Cardano/trigonometric method).
 * Input: a=T_xx, b=T_yy, c=T_zz, d=T_xy, e=T_xz, f=T_yz
 * Output: eigenvalues sorted descending: eig[0] >= eig[1] >= eig[2]
 */
static void
eigen_symmetric_3x3(double a, double b, double c, double d, double e, double f, double eig[3])
{
    double p1 = d * d + e * e + f * f;
    if(p1 == 0) {
        /* Matrix is diagonal */
        eig[0] = a; eig[1] = b; eig[2] = c;
    }
    else {
        double q = (a + b + c) / 3.0;
        double p2 = (a - q) * (a - q) + (b - q) * (b - q) + (c - q) * (c - q) + 2.0 * p1;
        double p = sqrt(p2 / 6.0);
        double p_inv = 1.0 / p;

        /* B = (1/p) * (A - q*I) */
        double b11 = (a - q) * p_inv;
        double b22 = (b - q) * p_inv;
        double b33 = (c - q) * p_inv;
        double b12 = d * p_inv;
        double b13 = e * p_inv;
        double b23 = f * p_inv;

        /* det(B) for symmetric matrix */
        double detB = b11 * (b22 * b33 - b23 * b23)
                    - b12 * (b12 * b33 - b23 * b13)
                    + b13 * (b12 * b23 - b22 * b13);

        double r = detB / 2.0;
        /* Clamp r to [-1, 1] for numerical safety */
        if(r <= -1.0)
            r = -1.0;
        else if(r >= 1.0)
            r = 1.0;

        double phi = acos(r) / 3.0;

        eig[0] = q + 2.0 * p * cos(phi);
        eig[2] = q + 2.0 * p * cos(phi + 2.0 * M_PI / 3.0);
        eig[1] = 3.0 * q - eig[0] - eig[2]; /* trace identity */
    }

    /* Sort descending: eig[0] >= eig[1] >= eig[2] */
    if(eig[0] < eig[1]) { double t = eig[0]; eig[0] = eig[1]; eig[1] = t; }
    if(eig[1] < eig[2]) { double t = eig[1]; eig[1] = eig[2]; eig[2] = t; }
    if(eig[0] < eig[1]) { double t = eig[0]; eig[0] = eig[1]; eig[1] = t; }
}

/* Compute eigenvalues from a tidal tensor and store in SphP */
void
tidal_field_store_eigenvalues(int i, const MyFloat tensor[6], double G)
{
    int PI = P[i].PI;
    int k;

    double T[6];
    for(k = 0; k < 6; k++)
        T[k] = tensor[k] * G;

    double eig[3];
    eigen_symmetric_3x3(T[0], T[1], T[2], T[3], T[4], T[5], eig);

    SphP[PI].TidalFieldEigenvalues[0] = eig[0];
    SphP[PI].TidalFieldEigenvalues[1] = eig[1];
    SphP[PI].TidalFieldEigenvalues[2] = eig[2];
}

/* Print trace diagnostics: Tr(T_total) should equal 4*pi*G*rho (Poisson equation).
 * The total trace is the sum of tree (short-range) and PM (long-range) contributions. */
void
tidal_field_diagnostics(MyFloat (*TensorStore)[6], double G)
{
    double local_max_err = 0;
    double local_sum_err = 0;
    int64_t local_count = 0;
    int local_warn_count = 0;
    int i;
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].Type != 0 || P[i].IsGarbage || P[i].Swallowed)
            continue;
        int PI = P[i].PI;
        /* Total trace = tree (short-range, without G) * G + PM (long-range, already includes G) */
        double trace_tree = (TensorStore[i][0] + TensorStore[i][1] + TensorStore[i][2]) * G;
        double trace_pm = SphP[PI].TidalTensorPM[0] + SphP[PI].TidalTensorPM[1] + SphP[PI].TidalTensorPM[2];
        double trace = trace_tree + trace_pm;
        double trace_expected = 4.0 * M_PI * G * SphP[PI].Density;
        double err = 0;
        if(trace_expected != 0)
            err = fabs(trace - trace_expected) / fabs(trace_expected);
        if(err > local_max_err)
            local_max_err = err;
        local_sum_err += err;
        local_count++;
        if(err > 0.5 && local_warn_count < 10) {
            message(1, "TidalField trace check: particle %ld trace=%g expected=%g err=%.1f%%\n",
                    (long)P[i].ID, trace, trace_expected, err * 100);
            local_warn_count++;
        }
    }

    double global_max_err;
    double global_sum_err;
    int64_t global_count;
    MPI_Reduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_err, &global_sum_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if(global_count > 0)
        message(0, "TidalField trace diagnostic: mean fractional error = %.4f, max = %.4f (over %ld gas particles)\n",
                global_sum_err / global_count, global_max_err, (long)global_count);
}
