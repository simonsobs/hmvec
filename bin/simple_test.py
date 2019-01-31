import hmvec as hm
import numpy as np
import matplotlib.pyplot as plt

zs = np.linspace(0.1,3.,4)
ms = np.geomspace(2e10,1e17,200)
ks = np.geomspace(1e-4,100,1001)
hcos = hm.HaloCosmology(zs,ks,ms=ms,nfw_numeric=True)
pmm_1h = hcos.get_power_1halo(name="nfw")
pmm_2h = hcos.get_power_2halo(name="nfw")

for i,z in enumerate(zs):
    plt.loglog(ks,pmm_2h[i]+pmm_1h[i],ls="-",color='C%d' % i,lw=3)
    plt.loglog(ks,pmm_2h[i],ls="--",color='C%d' % i)
    plt.loglog(ks,pmm_1h[i],ls="-.",color='C%d' % i)
plt.xlabel('$k \\mathrm{Mpc}^{-1}$ ')
plt.ylabel('$P(k)  \\mathrm{Mpc}^{3}$')
plt.show()
