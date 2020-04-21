import numpy as np
from matplotlib import pyplot as plt
import sympy as sym
import astropy.constants as const
from astropy import units as u
from astropy.modeling.blackbody import blackbody_nu
from scipy.optimize import fsolve

def blackbody(v, T):
    # Defining physical constants
    c = const.c.cgs.value
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value
    
    return (2.0*h/c**2) * v**3 / (np.exp((v/T) * (h/k_B)) - 1.0)

def boltzmann(x, T):
    # Defining physical constants
    c = const.c.cgs.value
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value
    
    return np.exp(h*x / (k_B*T))

def deriv_cond(v, T, b, g):
    # Defining physical constants
    c = const.c.cgs.value
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value
    
    first_term = (b+3)
    second_term = - ((h * v)/(k_B * T)) * boltzmann(v,T) / (boltzmann(v,T) - 1)
    third_term = g
    
    return first_term + second_term + third_term

def sysEquations(var, *constants):
    temps, b, g = constants  
    vo_g = var[:temps.size]
    A_g = var[temps.size:]
    
    eq1 = (vo_g)**b * blackbody(vo_g, temps) - A_g * (vo_g)**(-g)
    eq2 = deriv_cond(vo_g, temps, b, g)

    return np.concatenate((eq1, eq2))
    
def capitalTheta(nu_obs, z, alpha, beta, gamma, plot=False, bandpass=False):
    """ Rest frame SED """

    #Define the 2 parts of the SED as two functions
    def low_sed(nu, T):
        return (nu**beta)*blackbody(nu, T)
    def high_sed(nu, A):
        return A * nu**(-gamma)

    #Undoing Redshift: from Observing to Original
    freq_array = nu_obs * (1+z)
    temp_obs = 24.4
    temp_array = temp_obs * (1+z)**alpha
    
    #Get nu_o and proportionality constant A
    nu_o_guess = np.ones(temp_array.shape, dtype=np.float64) * 7.0e12   #initial guess
    A_guess = np.ones(temp_array.shape, dtype=np.float64) * 1.0e32      #initial guess
    sol = fsolve(sysEquations, np.concatenate((nu_o_guess, A_guess)), args=(temp_array, beta, gamma))
    nu_o_array = sol[:temp_array.size]
    A_array = sol[temp_array.size:]

    #Calculate SED values
    if bandpass:
        #Create the sed output over the whole range of frequencies here
        print("This feature is not currently available. Now killing your luminosity.")
        sed = np.zeros(nu_obs.shape)
    else:
        sed = np.where(freq_array < nu_o_array, low_sed(freq_array, temp_array), high_sed(freq_array, A_array))
    
    #Plot the whole spectrum
    if plot:
        #Setup
        nu_range = np.logspace(9, 14, 2000)
        plt.figure()

        #Spectra
        for i, nu_o in enumerate(nu_o_array):
            Nz = len(z)
            samplei = np.array([1, Nz*0.25, Nz*0.5, Nz*0.75, Nz])   #take quarters of the redshift range
            if not (i in (samplei.astype(int) - 1) ):
                continue

            #Calculation
            spectrum = np.where(nu_range < nu_o, low_sed(nu_range, temp_array[i]), high_sed(nu_range, A_array[i]))
            
            #Plot curves
            plt.plot(nu_range, spectrum, label='{} {}'.format("z =", z[i]))
            
            #Marking v_o on the graph
            plt.axvline(x = nu_o, ls=':', lw=0.2)
            
        #Plot Properties
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim([1e9, 1e12])
        #plt.xlim([3e11, 1e15])
        plt.ylabel(r'$\Theta (\nu, z)$')
        plt.xlabel(r'$\nu$ (Hz)')
        plt.legend()
        plt.savefig('sed.png', dpi=900, bbox_inches='tight');

    return sed

def capitalPhi(z, delta):
    """ Redshift dependent global normalization """
    return (1+z)**delta

def capitalSigma(M):
    """ Halo mass dependance of galaxy luminosity """

    logM_eff = 12.3  # logM_eff = log10(mass peak of specific IR emissivity) in solar masses
    sigma2 = 0.3      # (standard deviation)^2 of the Gaussian

    return M/np.sqrt(2*np.pi*sigma2) * np.exp(- (np.log10(M)-logM_eff)**2 / (2*sigma2))

def luminosity(z, M, Nks, v_obs, a=0.2, b=1.6, g=1.7, d=2.4, L_o=1):  
    """Luminosity of CIB galaxies. It depends only on mass and redshift, but the luminosity is on a grid of [z, M, k/r].
    
    Arguments:
        M [1darray]: galaxy's masses
        z [1darray]: redshifts
        Nks [int]: number of k's
        v_obs [float]: single observing frequency
    
    Keyword Arguments:
        a [float]: fit parameter - alpha (default = 0.2)
        b [float]: fit parameter - beta  (default = 1.6)
        g [float]: fit parameter - gamma (default = 1.7)
        d [float]: fit parameter - delta (default = 2.4)
        L_o [float]: fit parameter - normalization constant (default: 1)

    Returns:
        [3darray, float] : luminosity[z, M, k/r]
    """     
    
    #Calculate the z and M Dependence
    Lz = capitalPhi(z, d) * capitalTheta(v_obs, z, a, b, g)
    Lm = capitalSigma(M)
    
    #Put Luminosity on Grid
    Lk = np.ones(Nks)
    Lzz, Lmm, _ = np.meshgrid(Lz,Lm,Lk, indexing='ij')
    L = Lzz * Lmm

    return L_o * L

nu_obs = np.array([1000.0e9])
redshifts = np.linspace(0.01, 5, 500)
capitalTheta(nu_obs, redshifts, alpha=0.2, beta=1.6, gamma=1.7, plot=True)