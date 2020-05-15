import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as c
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
    
def capitalTheta(nu_obs, z, alpha, beta, gamma, T_o, plot=False):
    """ Rest frame SED """

    # Everything with the "_array" suffix is a Nz x M array, where Nz is the number of redshifts

    #Define some functions for making the SED
    def lowSED(nu, T):
        return (nu**beta)*blackbody(nu, T)
    def highSED(nu, A):
        return A * nu**(-gamma)
    def wholeSED(x, splitpoint, T, A):
        return np.where(x < splitpoint, lowSED(x, T), highSED(x, A))
        
    #Is there a bandpass or just 1 frequency?
    bandpassflag = False if len(nu_obs) == 1 else True

    #Undoing Redshift: from Observing to Original
    if bandpassflag:
        freq_array = np.outer((1+z), nu_obs)
    else:
        freq_array = (1+z) * nu_obs
    temp_array = T_o * (1+z)**alpha
    
    #Get nu_o and proportionality constant A
    nu_o_guess = np.ones(temp_array.shape, dtype=np.float64) * 5.0e12   #initial guess
    A_guess = np.ones(temp_array.shape, dtype=np.float64) * 1.0e32      #initial guess
    sol = fsolve(sysEquations, np.concatenate((nu_o_guess, A_guess)), args=(temp_array, beta, gamma))
    nu_o_array = sol[:temp_array.size]
    A_array = sol[temp_array.size:]

    #Range of Frequencies
    if bandpassflag:      
        #Initialize SED Arrays
        lowsed = np.array([])
        highsed = np.array([])
        splitsed = np.array([])

        #Expand T's and A's over Frequency Range
        Nf = 50     # num of frequencies in integrating range
        tempf_array = np.outer(temp_array, np.ones(Nf))
        Af_array = np.outer(A_array, np.ones(Nf))

        #Find Low SEDs
        ilower = np.where(freq_array[:,1] < nu_o_array)[0]
        #Find High SEDs
        iupper = np.where(freq_array[:,0] > nu_o_array)[0]
        
        #Low SEDs
        if ilower.size != 0:        
            isplit = ilower[-1] + 1      # the '+1' is because exclusivity of endpoint when slicing

            #Calculation
            lowranges = np.logspace(np.log10(freq_array[:isplit, 0]), np.log10(freq_array[:isplit, 1]), num=Nf, axis=-1)
            lowsed = np.trapz(lowSED(lowranges, tempf_array[:isplit, :]), x=lowranges, axis=-1)
        else:
            isplit = 0

        #High SEDs
        if iupper.size != 0:
            ihigh = iupper[0]

            #Calculation
            highranges = np.logspace(np.log10(freq_array[ihigh:,0]), np.log10(freq_array[ihigh:,1]), num=Nf, axis=-1)
            highsed = np.trapz(highSED(highranges, Af_array[ihigh:, :]), x=highranges, axis=-1)
        else:
            ihigh = len(z)

        #Split SEDs  
        #Range
        splitranges_low = np.logspace(np.log10(freq_array[isplit:ihigh, 0]), np.log10(nu_o_array[isplit:ihigh]), num=Nf, axis=-1)
        splitranges_high = np.logspace(np.log10(nu_o_array[isplit:ihigh]), np.log10(freq_array[isplit:ihigh, 1]), num=Nf, axis=-1)
        #Calculation
        splitsed_low = np.trapz(lowSED(splitranges_low, tempf_array[isplit:ihigh, :]), x=splitranges_low, axis=-1)
        splitsed_high = np.trapz(highSED(splitranges_high, Af_array[isplit:ihigh, :]), x=splitranges_high, axis=-1)
        splitsed = splitsed_low + splitsed_high

        #Combine to get total SED
        sed = np.concatenate((lowsed, splitsed, highsed))
        if len(sed) != len(z):
            raise ValueError(f'{len(sed)} SEDs were calculated, but there are only {len(z)} redshifts! Check logic.')
            

    #Single Frequency
    else:            
        sed = np.where(freq_array < nu_o_array, lowSED(freq_array, temp_array), highSED(freq_array, A_array))
    

    #Plot the whole spectrum
    if plot:
        #Sample Redshifts
        Nz = len(z)
        quarteri = np.array([1, Nz*0.25, Nz*0.5, Nz*0.75, Nz])   #take quarters of the redshift range
        sampleindices = quarteri.astype(int) - 1

        #Setup
        nu_range = np.logspace(11, 14, 2000)
        iplt = 0
        fig, ax = plt.subplots(len(sampleindices), 1, sharey=True, figsize=(10, 20))
        alphavalue = 0.2
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colorsalpha = c.to_rgba_array(colors, alphavalue)

        #Plot
        for zi, nu_o in enumerate(nu_o_array):
            if not (zi in sampleindices):
                continue

            #Calculation
            spectrum = wholeSED(nu_range, nu_o, temp_array[zi], A_array[zi])
            
            #Plot curves
            ax[iplt].plot(nu_range, spectrum, color=colors[iplt], label=f'z = {z[zi]:0.2f}')
 
            #Marking v_o on the graph
            ax[iplt].axvline(x = nu_o, ls=':', color='k', lw=0.4, label=rf'$\nu_o$ = {nu_o:0.2e}')
            
            #Marking bandpass on graph
            if bandpassflag:
                lowend = freq_array[zi, 0]
                highend = freq_array[zi, 1]
                bandpass_range = np.logspace(np.log10(lowend), np.log10(highend))
                ax[iplt].fill_between(bandpass_range, wholeSED(bandpass_range, nu_o, temp_array[zi], A_array[zi]), color=colorsalpha[iplt], label=f'Bandpass: {lowend:.2e} to {highend:.2e} Hz')


            #Marking nu_obs at sed frame on the graph
            else:
                ax[iplt].axvline(x = freq_array[zi], color=colors[iplt], lw=0.4, label=rf'$\nu_{{obs}} = {freq_array[zi]:0.2e}$ Hz in rest frame')

            #Plot Properties
            ax[iplt].set_xscale('log')
            ax[iplt].set_yscale('log')
            ax[iplt].set_ylabel(r'$\Theta (\nu, z)$')
            ax[iplt].set_xlabel(r'$\nu$ (Hz)')
            ax[iplt].legend()

            iplt += 1
        
        #Save file
        if bandpassflag:
            plt.savefig('sed_range.pdf', dpi=900, bbox_inches='tight');
        else:
            plt.savefig('sed_1freq.pdf', dpi=900, bbox_inches='tight');

    return sed

def capitalPhi(z, delta):
    """ Redshift dependent global normalization """
    return (1+z)**delta

def capitalSigma(M, logM_eff, sigma2):
    """Halo mass dependance of galaxy luminosity 
    
    Data Dictionary
    logM_eff : log10(mass peak of specific IR emissivity) in solar masses
    sigma2   : (standard deviation)^2 of the Gaussian
    """

    return M/np.sqrt(2*np.pi*sigma2) * np.exp(- (np.log10(M)-logM_eff)**2 / (2*sigma2))

def luminosity(z, M, Nks, v_obs, a=0.2, b=1.6, g=1.7, d=2.4, Td_o=20.7, logM_eff=12.3, var = 0.3, L_o=1):  
    """Luminosity of CIB galaxies. It depends only on mass and redshift, but the luminosity is on a grid of [z, M, k/r].
    
    Arguments:
        M [1darray]: galaxy's masses
        z [1darray]: redshifts
        Nks [int]: number of k's
        v_obs [1darray]: either single observing frequency or the endpoints of a bandpass
    
    Keyword Arguments:
        a [float]: fit parameter - alpha (default = 0.2)
        b [float]: fit parameter - beta  (default = 1.6)
        g [float]: fit parameter - gamma (default = 1.7)
        d [float]: fit parameter - delta (default = 2.4)
        Td_o [float]: fit parameter - dust temp at z=0 (default = 20.7)
        logM_eff [float]: fit parameter - log(M_eff) in L-M relation (default = 12.3)
        var [float]: model parameter - variance of Gaussian part of L-M relation (default = 0.3)
        L_o [float]: fit parameter - normalization constant (default: 1)

    Returns:
        [3darray, float] : luminosity[z, M, k/r]
    """     
    
    #Calculate the z and M Dependence
    Lz = capitalPhi(z, d) * capitalTheta(v_obs, z, a, b, g, Td_o)
    Lm = capitalSigma(M, logM_eff, var)
    
    #Put Luminosity on Grid
    Lk = np.ones(Nks)
    Lzz, Lmm, _ = np.meshgrid(Lz,Lm,Lk, indexing='ij')
    L = Lzz * Lmm

    return L_o * L

#Testing
nu_obs = np.array([857.0e9])
redshifts = np.linspace(0.01, 5, 500)
sed = capitalTheta(nu_obs, redshifts, alpha=0.2, beta=1.6, gamma=1.7, T_o=20.7, plot=True)