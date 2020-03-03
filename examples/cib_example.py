import numpy as np
import hmvec as hm
import matplotlib.pyplot as plt

#Grid for Integration
Nz = 100                   # number of redshifts
Nm = 100
redshifts = np.linspace(0.01,3,Nz)             # redshifts
masses = np.geomspace(1e11,1e15,100)        # masses

#Initialize Halo Model
hcos = hm.HaloModel(redshifts, ks, ms=masses, mass_function='tinker')
