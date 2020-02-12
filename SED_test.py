#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import sympy as sym
import astropy.constants as const
from astropy import units as u
from astropy.modeling.blackbody import blackbody_nu
from scipy.optimize import fsolve
import pdb
get_ipython().run_line_magic('matplotlib', 'inline')


# # Code without Full SED

# In[36]:


def blackbody(v, T):
    #Gives a blackbody spectrum as a function of frequency and temperature

    # Defining physical constants
    c = const.c.cgs.value
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value

    return (2.0*h/c**2) * v**3 / (np.exp((v/T) * (h/k_B)) - 1.0)

def boltzmann(x, T):
    # Returns the Boltzmann factor for a given temperature and x

    # Defining physical constants
    c = const.c.cgs.value
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value

    return np.exp(h*x / (k_B*T))

def sed(nu_obs, z, beta=1.6, alpha=0.2):
    #Undoing Redshift: from Observing to Original
    nu = nu_obs * (1+z)
    temp_obs = 20.7                        # effective dust temperature at z=0
    temp_array = temp_obs * (1+z)**alpha

    return nu * blackbody(nu, temp_array)

def phi(z, delta=2.4):
    # Redshift dependent global normalization
    return (1+z)**delta

def sigma(M,z):
    #Halo mass dependance of glaxy luminosity

    logM_eff = 12.3  # M_eff = log10(mass peak of specific IR emissivity) in solar masses
    sigma = 0.3      # standard deviation of the Gaussian

    return M/np.sqrt(2*np.pi*sigma**2) * np.exp(- (np.log10(M)-logM_eff)**2 / (2*sigma**2))

def luminosity(M, z, v_obs):
    return L_o * sigma(M,z) * phi(z) * sed(v_obs, z)

#Input data
nu_obs = np.array([353.0e9])
redshifts = np.array([0, 1, 2])

#Run the code
luminosity(M, redshifts, nu_obs)
