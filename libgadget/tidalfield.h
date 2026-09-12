#ifndef TIDALFIELD_H
#define TIDALFIELD_H

#include "forcetree.h"
#include "petapm.h"
#include "utils/paramset.h"
#include "types.h"

struct tidalfield_params
{
    int GasTidalField;              /* Master switch: 1 to compute tidal field for gas particles */
};

/* Set up the module parameters */
void set_tidalfield_params(ParameterSet * ps);

/* Returns 1 if tidal field computation is enabled */
int get_tidalfield_on(void);

/* Compute eigenvalues from a tidal tensor and store in SphP[PI].TidalFieldEigenvalues.
 * tensor: 6-component symmetric tensor (xx, yy, zz, xy, xz, yz) without the G factor.
 * G: Newton's constant in internal units. */
void tidal_field_store_eigenvalues(int i, const MyFloat tensor[6], double G);

/* Compute tidal field strength (Frobenius norm of eigenvalues) from a tidal tensor.
 * tensor: 6-component symmetric tensor (xx, yy, zz, xy, xz, yz) without the G factor.
 * G: Newton's constant in internal units.
 * Returns sqrt(sum of eigenvalues^2). */
double tidal_field_norm(const MyFloat tensor[6], double G);

/* Compute the eigenvalues of a tidal tensor into eig[3], sorted descending.
 * Same convention as tidal_field_store_eigenvalues, but hands the eigenvalues back to
 * the caller instead of writing them into SphP, so a BH can store them and derive the
 * Frobenius norm without a second eigen-decomposition.
 * tensor: 6-component symmetric tensor (xx, yy, zz, xy, xz, yz) without the G factor.
 * G: Newton's constant in internal units. */
void tidal_field_eigenvalues(const MyFloat tensor[6], double G, double eig[3]);

/* Print trace diagnostics after the combined gravity+tidal treewalk.
 * TensorStore: per-particle 6-component tidal tensor (without G factor).
 * G: Newton's constant in internal units. */
void tidal_field_diagnostics(MyFloat (*TensorStore)[6], double G);

#endif
