from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os,sys
import hmvec as hm # Git clone and pip install as in readme from github.com/msyriac/hmvec
import numpy as np

zgalaxy = 0.6 # delta-function lens population
zsource = 1.0 # delta-function source population
ngal = 1e-4 # number density of lenses per mpc3 (to solve for stellar mass threshold in HOD)

# We set up the grid to integrate on
zmax = zsource + 1
numzs = 20
zs = np.linspace(0.01,zmax,numzs) # redshifts
ms = np.geomspace(2e10,1e17,100) # masses
ks = np.geomspace(1e-4,100,1001) # wavenumbers

# Initialize halo model
hcos = hm.HaloModel(zs,ks,ms=ms)
# We add an HOD which should be CMASS-like at z=zgalaxy
hcos.add_hod("g",ngal=ngal+zs*0.,corr="max")
print("Galaxy bias at zgalaxy=",zgalaxy, " is ",interp1d(zs,hcos.hods['g']['bg'])(zgalaxy))
# We add a gas profile
hcos.add_battaglia_profile("electron",family="AGN",xmax=50,nxs=30000)

"""
Galaxy-galaxy lensing
"""
Pgn = hcos.get_power("g","nfw",verbose=False)
Pge = hcos.get_power("g","electron",verbose=False )
Pgm = hcos.total_matter_galaxy_power_spectrum(Pgn,Pge)

# Compare P(k)
for i in [0,-1]:
    plt.plot(ks,Pgm[i]/Pgn[i],color="C%d" % (i+2))
plt.axhline(y=1,ls='--')
plt.xlabel('$k \\mathrm{Mpc}^{-1}$')
plt.ylabel('$P_{gm}^{\\rm{fb}}/P_{gm}^{\\rm{no-fb}}$')
plt.xscale('log')
plt.yscale('linear')
plt.savefig("pkgm.png",bbox_inches='tight')

# Do Limber integrals
ells = np.linspace(80,6000,100)
Ckg0 = hcos.C_kg(ells,zs,ks,Pgn,gzs=zgalaxy,lzs=zsource)
Ckg = hcos.C_kg(ells,zs,ks,Pgm,gzs=zgalaxy,lzs=zsource)

# Compare Ckg
plt.clf()
plt.plot(ells,Ckg/Ckg0)
plt.axhline(y=1,ls='--')
plt.xlabel('$\\ell$')
plt.ylabel('$C_{kg}^{\\rm{fb}}/C_{kg}^{\\rm{no-fb}}$')
plt.xscale('linear')
plt.yscale('linear')
plt.savefig("clkg.png",bbox_inches='tight')

"""
Cosmic Shear
"""

Pnn = hcos.get_power("nfw",verbose=False)
Pne = hcos.get_power("nfw","electron",verbose=False )
Pee = hcos.get_power("electron","electron",verbose=False )
Pmm = hcos.total_matter_power_spectrum(Pnn,Pne,Pee)

# Do Limber integral
Ckk0 = hcos.C_kk(ells,zs,ks,Pnn,lzs1=zsource,lzs2=zsource)
Ckk = hcos.C_kk(ells,zs,ks,Pmm,lzs1=zsource,lzs2=zsource)

# Compare Ckk
plt.clf()
plt.plot(ells,Ckk/Ckk0)
plt.axhline(y=1,ls='--')
plt.xlabel('$\\ell$')
plt.ylabel('$C_{kk}^{\\rm{fb}}/C_{kk}^{\\rm{no-fb}}$')
plt.xscale('linear')
plt.yscale('linear')
plt.savefig("clkk.png",bbox_inches='tight')
