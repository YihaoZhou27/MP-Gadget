#include <string.h>
#include "partmanager.h"
#include "scinfo.h"

/* Per-rank StarClusterDetails file handle.  NULL disables recording (either the
 * feature is off or this rank does not write details). */
static FILE * FdSC = NULL;

void
scinfo_set_file(FILE * fd)
{
    FdSC = fd;
    /* MinMscForSCdetail can turn this into a bulk stream (a single group draw may
     * contribute O(1e5) records), so give stdio a large buffer instead of letting it
     * issue a write syscall every ~25 records.  Safe here: open_outputfiles calls us
     * immediately after fopen, before any I/O on the stream. */
    if(FdSC)
        setvbuf(FdSC, NULL, _IOFBF, 1 << 20);
}

int
scinfo_enabled(void)
{
    return FdSC != NULL;
}

void
scinfo_flush(void)
{
    if(FdSC)
        fflush(FdSC);
}

/* Assemble and write one record.  pos is in the translated frame (P[].Pos); the
 * particle offset is subtracted here so every caller stores the same convention. */
static void
scinfo_write(MyIDType id, const double * pos, double atime, double SCmass,
             double SCMassTotal, double StellarMassTotal, double SCMassSeeded,
             double metallicity, double Reff, double Mbh_seed, int NBHInGroup,
             int64_t GrNr, const struct SCmetdist * metdist,
             const struct SCboundinfo * bound, int flag)
{
    struct SCseedinfo info;
    memset(&info, 0, sizeof(info));
    const int size = sizeof(struct SCseedinfo) - sizeof(info.size1) - sizeof(info.size2);
    info.size1 = size;
    info.size2 = size;

    info.ID = id;
    info.GrNr = GrNr;
    info.a = atime;
    int k;
    for(k = 0; k < 3; k++)
        info.Pos[k] = pos[k] - PartManager->CurrentParticleOffset[k];

    info.StarClusterMass = SCmass;
    info.StarClusterMassTotal = SCMassTotal;
    info.StellarMassTotal = StellarMassTotal;
    info.SCMass_seeded = SCMassSeeded;
    info.Metallicity = metallicity;
    info.Reff = Reff;
    info.Mbh_seed = Mbh_seed;
    info.NBHInGroup = NBHInGroup;
    info.Flag = flag;
    if(metdist) {
        info.MetUnseededMin    = metdist->min;
        info.MetUnseededMax    = metdist->max;
        info.MetUnseededMedian = metdist->median;
        info.MetUnseededP25    = metdist->p25;
        info.MetUnseededP75    = metdist->p75;
        info.MetUnseededStd    = metdist->std;
    }
    if(bound) {
        info.BoundStarMass         = bound->mass;
        info.BoundStarMassUnseeded = bound->mass_unseeded;
        info.BoundSCMass           = bound->scmass;
        info.BoundRdm              = bound->rdm;
        info.BoundDMMass           = bound->mdm;
        info.BoundStarNum          = bound->num;
        info.BoundMode             = bound->mode;
    }

    fwrite(&info, sizeof(struct SCseedinfo), 1, FdSC);
}

void
scinfo_record_seed(int index, double atime, double SCmass, double SCMassTotal,
                   double StellarMassTotal, double SCMassSeeded, double metallicity,
                   double Reff, double Mbh_seed, int NBHInGroup, int64_t GrNr,
                   const struct SCmetdist * metdist, const struct SCboundinfo * bound, int flag)
{
    if(!FdSC)
        return;

    scinfo_write(P[index].ID, P[index].Pos, atime, SCmass, SCMassTotal, StellarMassTotal,
                 SCMassSeeded, metallicity, Reff, Mbh_seed, NBHInGroup, GrNr, metdist,
                 bound, flag);
    /* Seed events are rare (at most once per PM step), so flushing each record is
     * negligible and keeps the file current / crash-durable. */
    fflush(FdSC);
}

void
scinfo_record_cluster(MyIDType id, const double * pos, double atime, double SCmass,
                      double SCMassTotal, double StellarMassTotal, double SCMassSeeded,
                      double metallicity, double Reff, double Mbh_seed, int NBHInGroup,
                      int64_t GrNr, const struct SCmetdist * metdist,
                      const struct SCboundinfo * bound, int flag)
{
    if(!FdSC)
        return;

    /* Mbh_seed here is a model mass only: no BH particle exists for these clusters.
     * Deliberately unflushed -- the caller flushes once per batch. */
    scinfo_write(id, pos, atime, SCmass, SCMassTotal, StellarMassTotal, SCMassSeeded,
                 metallicity, Reff, Mbh_seed, NBHInGroup, GrNr, metdist, bound, flag);
}
