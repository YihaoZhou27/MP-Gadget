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
}

void
scinfo_record_seed(int index, double atime, double SCmass, double SCMassTotal,
                   double SCMassSeeded, double metallicity, double Reff,
                   double Mbh_seed, int NBHInGroup, int64_t GrNr,
                   const struct SCmetdist * metdist)
{
    if(!FdSC)
        return;

    struct SCseedinfo info;
    memset(&info, 0, sizeof(info));
    const int size = sizeof(struct SCseedinfo) - sizeof(info.size1) - sizeof(info.size2);
    info.size1 = size;
    info.size2 = size;

    info.ID = P[index].ID;
    info.GrNr = GrNr;
    info.a = atime;
    int k;
    for(k = 0; k < 3; k++)
        info.Pos[k] = P[index].Pos[k] - PartManager->CurrentParticleOffset[k];

    info.StarClusterMass = SCmass;
    info.StarClusterMassTotal = SCMassTotal;
    info.SCMass_seeded = SCMassSeeded;
    info.Metallicity = metallicity;
    info.Reff = Reff;
    info.Mbh_seed = Mbh_seed;
    info.NBHInGroup = NBHInGroup;
    if(metdist) {
        info.MetUnseededMin    = metdist->min;
        info.MetUnseededMax    = metdist->max;
        info.MetUnseededMedian = metdist->median;
        info.MetUnseededP25    = metdist->p25;
        info.MetUnseededP75    = metdist->p75;
        info.MetUnseededStd    = metdist->std;
    }

    fwrite(&info, sizeof(struct SCseedinfo), 1, FdSC);
    /* Seed events are rare (at most once per PM step), so flushing each record is
     * negligible and keeps the file current / crash-durable. */
    fflush(FdSC);
}
