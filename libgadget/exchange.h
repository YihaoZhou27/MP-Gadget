#ifndef __EXCHANGE_H
#define __EXCHANGE_H

#include "partmanager.h"
#include "slotsmanager.h"
#include "drift.h"

typedef int (*ExchangeLayoutFunc) (int p, const void * userdata);

typedef struct PreExchangeList{
    /*List of particles to exchange and garbage particles (to receive incoming data).*/
    int * ExchangeList;
    /*Total number of exchanged particles*/
    size_t nexchange;
    /*Number of garbage particles*/
    int64_t ngarbage;
} PreExchangeList;

/* Do the domain exchange with a pre-computed domain.
 * ExchangeLayoutFunc returns the MPI task the particle should move to.
 * layout_userdata is a pointer to private memory for the layout func.
 * preexch is a list of particles generated by running the layoutfunc somewhere before the domain exchange and may be NULL: it is a pure accelerator.
 * pman is the particle data struct
 * sman is the slots data struct.
 * MPI_Comm is a communicator for the exchange*/
int domain_exchange(ExchangeLayoutFunc, const void * layout_userdata, PreExchangeList * preexch, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);
/* Uses an MPI aware sort to check that all particle IDs are unique. Used in debugging to catch exchange errors. */
void domain_test_id_uniqueness(struct part_manager_type * pman);
/* Set the maximum bytes sent across in a single exchange iteration, so the unit test can exercise the path with multiple exchange iterations.*/
void domain_set_max_exchange(const size_t maxexch);

#endif
