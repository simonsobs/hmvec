import numpy as np
import hmvec as hm
import matplotlib.pyplot as plt

#Grid for Integration
Nz = 100                                 # num of redshifts
Nm = 100                                 # num of masses
Nk = 10001                                # num of wavenumbers
redshifts = np.linspace(0.01, 3, Nz)             # redshifts
masses = np.geomspace(1e11, 1e15, Nm)           # masses
ks = np.geomspace(1e-3, 100, Nk)               # wavenumbers
frequencies = np.array([271.0])

#Initialize Halo Model
hcos = hm.HaloModel(redshifts, ks, ms=masses)

#Get Power Spectra
for i in frequencies:
    Pjj_2h = hcos.get_power_2halo("cib", "cib", nu_obs=frequencies[i])  # P(z,k)

