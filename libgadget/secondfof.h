#ifndef SECONDFOF_H
#define SECONDFOF_H

#include "utils/paramset.h"
#include "domain.h"
#include "cosmology.h"
#include "fof.h"

void set_secondfof_params(ParameterSet * ps);
int get_secondfof_on(void);
const char * get_secondfof_filebase(void);
int get_secondfof_only(void);
int get_seed_in_secfof(void);

/* Run a secondary FOF specifically for BH seeding: swaps FOF params,
 * runs fof_fof, calls fof_seed, then restores params and frees. */
void secondfof_seed(DomainDecomp * ddecomp, ActiveParticles * act, ForceTree * tree,
                    double atime, const RandTable * rnd, Cosmology * CP, MPI_Comm Comm);

/* Opaque handle holding second FOF results between run and write phases */
typedef struct SecondFOFResult SecondFOFResult;

/* Run the second FOF: compute groups, set P[i].SecGrNr, compute extra properties.
 * Preserves the existing P[i].GrNr from the halo FOF.
 * Returns a handle that must be passed to secondfof_write() and then secondfof_finish(). */
/* atime / CP are needed only by the BHseedSecFOFbound pass that fills the
 * catalogue's SecBound* blocks; they are otherwise unused. */
SecondFOFResult * secondfof_run(DomainDecomp * ddecomp, int OutputPotential,
                                double atime, Cosmology * CP, MPI_Comm Comm);

/* Write the SecPIG catalog to disk. Call after checkpoint for I/O safety. */
void secondfof_write(SecondFOFResult * result, const char * OutputDir, int snapnum,
                     double atime, Cosmology * CP, const double * MassTable,
                     int MetalReturnOn, int OutputDebugFields, MPI_Comm Comm);

/* Free all second FOF data. */
void secondfof_finish(SecondFOFResult * result);

#endif
