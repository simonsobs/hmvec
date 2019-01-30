import hmvec as hm
import numpy as np
from orphics import io
from enlib import bench

zs = np.linspace(0.1,3.,4)[-1:]
ms = np.geomspace(2e10,1e17,200)
ks = np.geomspace(1e-4,100,1001)
with bench.show("num"):
    hcos = hm.HaloCosmology(zs,ks,ms=ms,nfw_numeric=True)
opmm_1h = hcos.get_power_1halo_auto(name="nfw")
opmm_2h = hcos.get_power_2halo_auto(name="nfw")
hcos = hm.HaloCosmology(zs,ks,ms=ms,nfw_numeric=False)
apmm_1h = hcos.get_power_1halo_auto(name="nfw")
apmm_2h = hcos.get_power_2halo_auto(name="nfw")

pl = io.Plotter(xyscale='loglin')
for i,z in enumerate(zs):
    pl.add(ks,(apmm_1h[i]-opmm_1h[i])/opmm_1h[i],ls='--')
    pl.add(ks,(apmm_2h[i]-opmm_2h[i])/opmm_2h[i],ls='--')
pl.hline(y=0)
pl.done()
