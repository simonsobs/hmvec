import numpy as np
from matplotlib import pyplot as plt
import sympy as sym
import astropy.constants as const
from astropy import units as u
from astropy.modeling.blackbody import blackbody_nu
from scipy.optimize import fsolve
import pdb

def blackbody(v, T):
    # Gives a blackbody spectrum as a function of frequency and temperature

    #Defining physical constants
    c = const.c.cgs.value
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value

    return (2.0*h/c**2) * v**3 / (np.exp((v/T) * (h/k_B)) - 1.0)

def boltzmann(x, T):
    # Returns the Boltzmann factor for a given temperature and x

    #Defining physical constants
    c = const.c.cgs.value
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value

    return np.exp(h*x / (k_B*T))

def capitalTheta(nu_obs, z, beta, alpha):
    # Rest frame SED

    #Undoing Redshift: from Observing to Original
    nu = nu_obs * (1+z)
    temp_obs = 20.7                        # effective dust temperature at z=0
    temp_array = temp_obs * (1+z)**alpha

    return nu * blackbody(nu, temp_array)

def capitalPhi(z, delta):
    # Redshift dependent global normalization
    return (1+z)**delta

def capitalSigma(M):
    # Halo mass dependance of galaxy luminosity

    logM_eff = 12.3  # logM_eff = log10(mass peak of specific IR emissivity) in solar masses
    sigma2 = 0.3      # (standard deviation)^2 of the Gaussian

    return M/np.sqrt(2*np.pi*sigma2) * np.exp(- (np.log10(M)-logM_eff)**2 / (2*sigma2))

def luminosity(M, z, v_obs, a=0.2, b=1.6, d=2.4, L_o=1):
    #Data Dictionary
    #   a = alpha
    #   b = beta
    #   d = delta

    return L_o * capitalSigma(M) * capitalPhi(z, d) * capitalTheta(v_obs, z, b, a)

# #Input data
# nu_obs = np.array([353.0e9])
# redshifts = np.array([0, 1, 2])
#
# #Run the code
# luminosity(M, redshifts, nu_obs)
