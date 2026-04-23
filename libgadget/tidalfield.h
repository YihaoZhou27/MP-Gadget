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

/* Print trace diagnostics after the combined gravity+tidal treewalk.
 * TensorStore: per-particle 6-component tidal tensor (without G factor).
 * G: Newton's constant in internal units. */
void tidal_field_diagnostics(MyFloat (*TensorStore)[6], double G);

#endif
