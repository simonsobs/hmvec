import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import colors as c

import astropy.constants as const
from astropy import units as u
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
    
def capitalTheta(nu_sample, nuframe, z, alpha, beta, gamma, T_o, plot=False):
    """ Rest frame SED """

    # Everything with the "_array" suffix is a Nz x M array, where Nz is the number of redshifts

    #Define some functions for making the SED
    def lowSED(nu, T):
        return (nu**beta)*blackbody(nu, T)
    def highSED(nu, A):
        return A * nu**(-gamma)
    def wholeSED(x, splitpoint, T, A):
        return np.where(x < splitpoint, lowSED(x, T)/lowSED(splitpoint,T), highSED(x, A)/highSED(splitpoint,A))
        
    #Is there a bandpass or just 1 frequency?
    bandpassflag = False if len(nu_sample) == 1 else True

    #Put Bandpasss Ends in Order
    if bandpassflag:
        if nu_sample[0] > nu_sample[1]:
            nu_sample = nu_sample[::-1]

    #Undoing Redshift: from Observing to Original
    if nuframe.lower() == 'obs':
        if bandpassflag:
            freq_array = np.outer((1+z), nu_sample)
        else:
            freq_array = (1+z) * nu_sample
    elif nuframe.lower() == 'rest':           # already in Rest Frame
        if bandpassflag:
            freq_array = np.outer(np.ones(z.shape), nu_sample)
        else:
            freq_array = np.ones(z.shape) * nu_sample
    else:
        raise ValueError('Need a valid reference frame to view the SEDs in.')
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
        Nf = 100     # num of frequencies in integrating range
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

            #Normalization
            lowsed /= lowSED(nu_o_array[:isplit], temp_array[:isplit])
        else:
            isplit = 0

        #High SEDs
        if iupper.size != 0:
            ihigh = iupper[0]

            #Calculation
            highranges = np.logspace(np.log10(freq_array[ihigh:,0]), np.log10(freq_array[ihigh:,1]), num=Nf, axis=-1)
            highsed = np.trapz(highSED(highranges, Af_array[ihigh:, :]), x=highranges, axis=-1)

            #Normalization
            highsed /= highSED(nu_o_array[ihigh:], A_array[ihigh:])
        else:
            ihigh = len(z)

        #Split SEDs  
        #Range
        splitranges_low = np.logspace(np.log10(freq_array[isplit:ihigh, 0]), np.log10(nu_o_array[isplit:ihigh]), num=Nf, axis=-1)
        splitranges_high = np.logspace(np.log10(nu_o_array[isplit:ihigh]), np.log10(freq_array[isplit:ihigh, 1]), num=Nf, axis=-1)
        #Calculation
        splitsed_low = np.trapz(lowSED(splitranges_low, tempf_array[isplit:ihigh, :]), x=splitranges_low, axis=-1)
        splitsed_high = np.trapz(highSED(splitranges_high, Af_array[isplit:ihigh, :]), x=splitranges_high, axis=-1)
        splitsed_low /= lowSED(nu_o_array[isplit:ihigh], temp_array[isplit:ihigh])
        splitsed_high /= highSED(nu_o_array[isplit:ihigh], A_array[isplit:ihigh])
        splitsed = splitsed_low + splitsed_high

        #Combine to get total SED
        sed = np.concatenate((lowsed, splitsed, highsed))
        if len(sed) != len(z):
            raise ValueError(f'{len(sed)} SEDs were calculated, but there are only {len(z)} redshifts! Check logic.')
            

    #Single Frequency
    else:            
        sed = np.where(freq_array < nu_o_array, lowSED(freq_array, temp_array)/lowSED(nu_o_array, temp_array), highSED(freq_array, A_array)/highSED(nu_o_array, A_array) )
         

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
        for zi in sampleindices:
            nu_o = nu_o_array[zi]
            
            #Calculation
            spectrum = wholeSED(nu_range, nu_o_array[zi], temp_array[zi], A_array[zi])
            
            # plt.plot(nu_range, spectrum, color=colors[iplt], label=fr'z = {z[zi]:0.2f}, $\nu_o$ = {nu_o/1e9:,.0f} Ghz')
            
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel(r'$\nu$ (Hz)')
            # plt.ylabel(r'$\Theta (\nu, z)$')
            # plt.legend()
            # plt.ylim(bottom=1e-4)

            #Plot curves
            ax[iplt].plot(nu_range, spectrum, color=colors[iplt], label=f'z = {z[zi]:0.2f}')
 
            #Marking v_o on the graph
            ax[iplt].axvline(x = nu_o, ls=':', color='k', lw=0.4, label=rf'$\nu_o$ = {nu_o/1e9:,.2e} GHz')
            

            #Marking bandpass on graph
            if bandpassflag:
                lowend = freq_array[zi, 0]
                highend = freq_array[zi, 1]
                bandpass_range = np.logspace(np.log10(lowend), np.log10(highend))
                ax[iplt].fill_between(bandpass_range, wholeSED(bandpass_range, nu_o, temp_array[zi], A_array[zi]), 
                                        color=colorsalpha[iplt], edgecolors='none', label=f'Bandpass: {lowend/1e9:,.2e} to {highend/1e9:,.2e} GHz')


            #Marking nu_sample at sed frame on the graph
            else:
                ax[iplt].axvline(x = freq_array[zi], color=colors[iplt], lw=0.4, label=rf'$\nu_{{obs}}$ = {freq_array[zi]/1e9:,.2e} GHz in rest frame')

            #Plot Properties
            ax[iplt].set_xscale('log')
            ax[iplt].set_yscale('log')
            ax[iplt].legend()

            iplt += 1
        #Super Labels
        fig.text(0.5, 0.09, r'$\nu$ (Hz)', ha='center', fontsize='xx-large')
        fig.text(0.04, 0.5, r'$\Theta (\nu, z)$', va='center', rotation='vertical', fontsize='xx-large')

        # plt.savefig('sed_norm_planck.pdf', dpi=900, bbox_inches='tight');

        #Save file
        if bandpassflag:
            plt.savefig('sed_range.pdf', dpi=400, bbox_inches='tight');
        else:
            plt.savefig('/scratch/r/rbond/ymehta3/sed_norm_1freq.png', dpi=900, bbox_inches='tight');

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

def luminosity(z, M, Nks, nu, params, nuframe='obs'):  
    """Luminosity of CIB galaxies. It depends only on mass and redshift, but is broadcasted onto a grid of [z, M, k/r]. The fit parameters are in the "params" dictionary.

    Arguments:
        M [1darray]: galaxy's masses
        z [1darray]: redshifts
        Nks [int]: number of k's
        nu [1darray]: either single frequency or the endpoints of a bandpass
        nuframe [str:'obs'|'rest']: frame that the nu is given in
    
    Model parameters:
        alpha [float]: fit parameter - alpha 
        beta [float]: fit parameter - beta  
        gamma [float]: fit parameter - gamma 
        delta [float]: fit parameter - delta 
        Td_o [float]: fit parameter - dust temp at z=0 
        logM_eff [float]: fit parameter - log(M_eff) in L-M relation 
        var [float]: model parameter - variance of Gaussian part of L-M relation 
        L_o [float]: fit parameter - normalization constant 

    Returns:
        [3darray, float] : luminosity[z, M, k/r]
    """     
    #Unpack Parameters
    a = params['alpha']
    b = params['beta']
    d = params['delta']
    g = params['gamma']
    Td_o =params['Td_o']
    logM_eff =params['logM_eff']
    var =params['var']
    L_o =params['L_o']

    #Calculate the z and M Dependence
    Lz = capitalPhi(z, d) * capitalTheta(nu, nuframe, z, a, b, g, Td_o)
    Lm = capitalSigma(M, logM_eff, var)
    
    #Put Luminosity on Grid
    Lk = np.ones(Nks)
    Lzz, Lmm, _ = np.meshgrid(Lz,Lm,Lk, indexing='ij')
    L = Lzz * Lmm

    return L_o * L

if __name__ == "__main__":
    #Testing
    # lamdarange = np.array([8.0e-6, 1000.0e-6])
    # nurange = 3.0e8 / lamdarange
    # nurange = np.array([545.])*1e9
    
    #Setup Grid
    Nz = 100                                 # num of redshifts
    Nm = 500                                 # num of masses
    Nk = 1000                                # num of wavenumbers
    redshifts = np.linspace(0.01, 6, Nz)             
    masses = np.geomspace(1.0e10, 1.0e16, Nm)          
    ks = np.geomspace(1.0e-3, 100.0, Nk)              # wavenumbers
    
    cib_params = {}
    cib_params['alpha'] = 0.36
    cib_params['beta'] = 1.75
    cib_params['gamma'] = 1.7
    cib_params['delta'] = 3.6
    cib_params['Td_o'] = 24.4
    cib_params['logM_eff'] = 12.6
    cib_params['var'] = 0.5
    cib_params['L_o'] = 6.4e-8
    
    # redshifts = np.array([2.0])
    # sed = capitalTheta(nurange, 'obs', redshifts, alpha=0.36, beta=1.75, gamma=1.7, T_o=24.4, plot=True)
    # print(sed)

    L = luminosity(redshifts, masses, Nk, [545e9], cib_params)
    np.save('lum545', L)
    