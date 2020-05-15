import numpy as np
from matplotlib import pyplot as plt
import sympy as sym
import astropy.constants as const
from astropy import units as u
from astropy.modeling.blackbody import blackbody_nu
from scipy.optimize import fsolve

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
    """ Rest frame SED """

    #Undoing Redshift: from Observing to Original
    nu = nu_obs * (1+z)
    temp_obs = 20.7                        # effective dust temperature at z=0
    temp_array = temp_obs * (1+z)**alpha

    return nu * blackbody(nu, temp_array)

def capitalPhi(z, delta):
    """ Redshift dependent global normalization """
    return (1+z)**delta

def capitalSigma(M):
    """ Halo mass dependance of galaxy luminosity """

    logM_eff = 12.3  # logM_eff = log10(mass peak of specific IR emissivity) in solar masses
    sigma2 = 0.3      # (standard deviation)^2 of the Gaussian

    return M/np.sqrt(2*np.pi*sigma2) * np.exp(- (np.log10(M)-logM_eff)**2 / (2*sigma2))

def luminosity(z, M, Nks, v_obs, a=0.2, b=1.6, d=2.4, L_o=1):  
    """Luminosity of CIB galaxies. It depends only on mass and redshift, but the luminosity is on a grid of [z, M, k/r].
    
    Arguments:
        M [1darray]: galaxy's masses
        z [1darray]: redshifts
        Nks [int]: number of k's
        v_obs [float]: observing frequency
    
    Keyword Arguments:
        a [float]: fit parameter - alpha (default: 0.2)
        b [float]: fit parameter - beta  (default: 1.6)
        d [float]: fit parameter - delta (default: 2.4)
        L_o [float]: fit parameter - normalization constant (default: 1)

    Returns:
        [3darray, float] -- luminosity[z, M, k/r]
    """     
    
    #Calculate the z and M Dependence
    Lz = capitalPhi(z, d) * capitalTheta(v_obs, z, b, a)
    Lm = capitalSigma(M)
    
    #Put Luminosity on Grid
    Lk = np.ones(Nks)
    Lzz, Lmm, _ = np.meshgrid(Lz,Lm,Lk, indexing='ij')
    L = Lzz * Lmm

    return L_o * L