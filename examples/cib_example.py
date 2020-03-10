import numpy as np
import hmvec as hm

#Grid for Integration
Nz = 100                                 # num of redshifts
Nm = 100                                 # num of masses
Nk = 10001                                # num of wavenumbers
redshifts = np.linspace(0.01, 3, Nz)             # redshifts
masses = np.geomspace(1e11, 1e15, Nm)           # masses
ks = np.geomspace(1e-3, 100, Nk)               # wavenumbers
frequencies = 271.0

#Initialize Halo Model
hcos = hm.HaloModel(redshifts, ks, ms=masses, v_obs=frequencies)

#Get Power Spectra
Pjj_2h = hcos.get_power_2halo("cib", "cib")  # P(z,k)

#Limber Integrals
eplotlls = np.linspace(100, 1000)
Ccc = hm.C_cc()

#Plot
plt.loglog(ells,Ccc)
plt.show()
