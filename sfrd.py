import numpy as np
from hmvec import hmvec as hm
import matplotlib.pyplot as plt
import astropy.constants as const

#Grid for Integration
Nz = 600                                 # num of redshifts
Nm = 500                                 # num of masses
Nk = 100                                # num of wavenumbers
redshifts = np.linspace(0.01, 3, Nz)             # redshifts
masses = np.geomspace(1e10, 1e16, Nm)           # masses
ks = np.geomspace(1e-3, 100, Nk)               # wavenumbers

#Calculate Frequency Range
lamdarange = np.array([8.0e-6, 1000.0e-6])
nurange = 3.0e8 / lamdarange

#Initialize Halo Model
hcos = hm.HaloModel(redshifts, ks, ms=masses)

#Get SFRD
sfrd = hcos.get_sfrd(nurange, model='Planck')

#Plot
plt.plot(redshifts, sfrd, label='Planck theory')
plt.ylabel('SFRD')
plt.xlabel('z')
plt.legend()
plt.savefig('sfrd.pdf', dpi=500, bbox_inches='tight')