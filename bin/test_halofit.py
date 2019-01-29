import hmvec
import numpy as np
from orphics import io

halofits = ['takahashi','original','bird','peacock','mead','casarini','mead2015']
zs = np.array([0.,1.,2.,3.])
ks = np.geomspace(1e-4,20.,1000)
pks = {}
for halofit in halofits:
    print(halofit)
    hc = hmvec.HaloCosmology(zs,ks,halofit=halofit)
    pks[halofit] = hc.nPzk



for i,z in enumerate(zs):
    pl = io.Plotter(xyscale='loglin',xlabel='$k$',ylabel='$P/P_0$')
    for halofit in halofits[1:]:
        pl.add(ks,pks[halofit][i]/pks[halofits[0]][i],label=halofit)
    pl.hline(y=1)
    pl.legend(loc='upper left')
    pl._ax.set_ylim(0.7,1.3)
    pl.done("halofit_comp_z_%d.png" % i)
