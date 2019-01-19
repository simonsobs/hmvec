import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import camb
from camb import model
import numpy as np

"""

General vectorized FFT-based halo model implementation
Author(s): Mathew Madhavacheril

Array indexing is as follows:
[z,M,k/r]

r is always in Mpc
k is always in Mpc-1
All rho densities are in Msolar/Mpc^3
All masses m are in Msolar
No h units anywhere


"""

default_params = {
    'st_A': 0.3222,
    'st_a': 0.75,
    'st_p': 0.3,
    'st_deltac': 1.686,
    
    'omch2': 0.1198,
    'ombh2': 0.02225,
    'H0': 67.3,
    'ns': 0.9645,
    'As': 2.2e-9,
    'mnu': 0.06,
    'w0': -1.0,
    'tau':0.06,
    'nnu':3.046,
    'wa': 0.,
    'num_massive_neutrinos':3,
                        
    }


def Wkr(k,R):
    kR = k*R
    return 3.*(np.sin(kR)-kR*np.cos(kR))/(kR**3.)

    
class HaloCosmology(object):
    def __init__(self,zs,ks,params={},ms=None,mass_function="sheth-torman"):
        self.p = params
        self.mode = mass_function
        for param in default_params.keys():
            if param not in self.p.keys(): self.p[param] = default_params[param]
        self._init_cosmology(self.p)
        self.fineks = np.geomspace(1e-4,20.,1000) # ks for sigma2 integral FIXME: hard coded
        self.sPzk = self._get_linear_matter_power(zs,self.fineks) # power for sigma2 integral
        self.Pzk = self._get_linear_matter_power(zs,ks)
        self.zs = zs
        self.ks = ks
        self.rhom0 = self._rho_matter_z0()
        print(self.rhom0/1e10)
        if ms is not None: self.initialize_mass_function(ms)
    
    def _init_cosmology(self,params):
        try:
            theta = params['theta100']/100.
            H0 = None
            print("WARNING: Using theta100 parameterization. H0 ignored.")
        except:
            H0 = params['H0']
            theta = None
        
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=H0, cosmomc_theta=theta,ombh2=params['ombh2'], omch2=params['omch2'], mnu=params['mnu'], tau=params['tau'],nnu=params['nnu'],num_massive_neutrinos=params['num_massive_neutrinos'])
        try:
            self.pars.set_dark_energy(w=params['w0'],wa=params['wa'],dark_energy_model='ppf')
        except:
            assert np.abs(params['wa'])<1e-3, "Non-zero wa requires PPF, which requires devel version of pycamb to be installed."
            print("WARNING: Could not use PPF dark energy model with pycamb. Falling back to non-PPF. Please install the devel branch of pycamb.")
            self.pars.set_dark_energy(w=params['w0'])
        self.pars.NonLinear = model.NonLinear_none # always use linear matter
        self.results = camb.get_background(self.pars)
        self.params = params

        
    def _get_linear_matter_power(self,zs,ks):
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=False, 
                                                     hubble_units=False, k_hunit=False, kmax=ks.max()+1.,
                                                     zmax=zs.max()+1.) # FIXME: neutrinos
        return PK.P(zs, ks, grid=True)

    def _rho_matter_z0(self):
        omh2 = self.params['omch2']+self.params['ombh2'] # FIXME: neutrinos
        om = omh2 / (self.params['H0']/100.)**2.
        H0 = self.params['H0'] * 3.241e-20 # SI # FIXME: constants need checking
        G = 6.67259e-11 # SI
        rho_critical_z0 = 3.*(H0**2.)/8./np.pi/G # SI
        return rho_critical_z0 * om * 1.477543e37 # in msolar / megaparsec3 
    
    def R_of_m(self,ms): return (3.*ms/4./np.pi/self.rhom0)**(1./3.)
    
    def get_sigma2(self,ms):
        ks = self.fineks[None,:]
        R = self.R_of_m(ms)[:,None]
        W2 = Wkr(ks,R)**2.
        Ps = self.sPzk[:,None,:]
        integrand = Ps*W2*ks**2./2./np.pi**2.
        return np.trapz(integrand,ks,axis=-1)
        

    def initialize_mass_function(self,ms):
        sigma2 = self.get_sigma2(ms)
        self.nzm = self.get_nzm(ms,sigma2)
        self.bh = self.get_bh(ms,sigma2)

    def get_fsigmaz(self,ms=None,sigma2=None):
        if sigma2 is None: sigma2 = self.get_sigma2(ms)
        if self.mode=="sheth-torman":
            sigma = np.sqrt(sigma2)
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            deltac = self.p['st_deltac']
            return A*np.sqrt(2.*a/np.pi)*(1+((sigma2/a/deltac**2.)**p))*(deltac/sigma)*np.exp(-a*deltac**2./2./sigma2)
        else:
            raise NotImplementedError
    
    def get_bh(self,ms=None,sigma2=None):
        if sigma2 is None: sigma2 = self.get_sigma2(ms)
        if self.mode=="sheth-torman":
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            deltac = self.p['st_deltac']
            return 1. + (1./deltac)*((a*deltac**2./sigma2)-1.) + (2.*p/deltac)/(1.+(a*deltac**2./sigma2)**p)
        else:
            raise NotImplementedError

    def get_nzm(self,ms,sigma2=None):
        if sigma2 is None: sigma2 = self.get_sigma2(ms)
        ln_sigma_inv = -0.5*np.log(sigma2)
        fsigmaz = self.get_fsigmaz(ms,sigma2)
        dln_sigma_dlnm = np.gradient(ln_sigma_inv,np.log(ms),axis=-1)  # FIXME: this is probably wrong
        ms = ms[None,:]
        return self.rhom0 * fsigmaz * dln_sigma_dlnm / ms**2.

    def add_profile(self,name,rs,rhos):
        iks,iuk = fft_integral(rs,rhos)
        ks = self.ks
        uk = interp1d(iks,iuk,bounds_error=False,fill_value=(iuk[0],0.))(self.ks)

        #TODO: Add compulsory debug plot here
        

    def get_power_1halo_cross_galaxies(self,name="matter"):
        pass
    def get_power_2halo_cross_galaxies(self,name="matter"):
        pass

    def get_power_1halo_auto(self,name="matter"):
        pass
    def get_power_2halo_auto(self,name="matter"):
        pass

    def get_power_1halo_galaxy_auto(self):
        pass
    def get_power_2halo_galaxy_auto(self):
        pass
    

"""
HOD
"""
    
def Mstellar_halo(z,log10mhalo):
    # Function to compute the stellar mass Mstellar from a halo mass mv at redshift z.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    log10mstar = np.linspace(-18,18,1000)
    mh = Mhalo_stellar(z,log10mstar)
    mstar = np.interp(log10mhalo,mh,log10mstar)
    return mstar


def Mhalo_stellar(z,log10mstellar):
    # Function to compute halo mass as a function of the stellar mass.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    a = 1./(1+z) 
    Mstar00=10.72 ; Mstara=0.55 ; M1=12.35 ; M1a=0.28
    beta0=0.44 ; beta_a=0.18 ; gamma0=1.56 ; gamma_a=2.51
    delta0=0.57 ; delta_a=0.17
    log10M1 = M1 + M1a*(a-1)
    log10Mstar0 = Mstar00 + Mstara*(a-1)
    beta = beta0 + beta_a*(a-1)
    gamma = gamma0 + gamma_a*(a-1)
    delta = delta0 + delta_a*(a-1)
    log10mstar = log10mstellar
    log10mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
    return log10mh


def avg_Nc(log10mhalo,z,log10mstellar_thresh):
    """<Nc(m)>"""
    sig_log_mstellar = 0.2
    log10mstar = Mstellar_halo(z,log10mhalo)
    num = log10mstellar_thresh - log10mstar
    denom = np.sqrt(2.) * sig_log_mstellar
    return 0.5*(1. - erf(num/denom))

def avg_Ns(log10mhalo,z,log10mstellar_thresh,Nc=None):
    Bsat=9.04
    betasat=0.74
    alphasat=1.
    Bcut=1.65
    betacut=0.59
    mthresh = Mhalo_stellar(z,log10mstellar_thresh)
    Msat=(10.**(12.))*Bsat*10**((mthresh-12)*betasat)
    Mcut=(10.**(12.))*Bcut*10**((mthresh-12)*betacut)
    Nc = avg_Nc(log10mhalo,z,log10mstellar_thresh,sig_log_mstellar=0.2) if Nc is None else Nc
    masses = 10**log10mhalo
    return Nc*((masses/Msat)**alphasat)*np.exp(-Mcut/(masses))    

"""
Profiles
"""


def rho_nfw(r,rhos,rs):
    rrs = r/rs
    return rhos/rrs/(1.+rrs)**2. 


"""
FFT routines
"""

def uk_fft(rhofunc,rvir,dr=0.001,rmax=100):
    rps = np.arange(dr,rmax,dr)
    rs = rps
    theta = np.ones(rs.shape)
    theta[np.abs(rs)>rvir] = 0
    rhos = rhofunc(np.abs(rs))
    integrand = rhos * theta
    m = np.trapz(integrand*rs**2.,rs)*4.*np.pi
    ks,ukt = fft_integral(rs,integrand)
    uk = 4.*np.pi*ukt/ks/m
    return ks,uk
    

def uk_brute_force(r,rho,rvir,ks):
    sel = np.where(r<rvir)
    rs = r[sel]
    rhos = rho[sel]
    m = np.trapz(rhos*rs**2.,rs)*4.*np.pi
    # rs in dim 0, ks in dim 1
    rs2d = rs[...,None]
    rhos2d = rhos[...,None]
    ks2d = ks[None,...]
    sinkr = np.sin(rs2d*ks2d)
    integrand = 4.*np.pi*rs2d*sinkr*rhos2d/ks2d
    return np.trapz(integrand,rs,axis=0)/m

def fft_integral(x,y,axis=-1):
    """
    Calculates
    \int dx x sin(kx) y(|x|) from 0 to infinity using an FFT,
    which appears often in fourier transforms of 1-d profiles.
    For y(x) = exp(-x**2/2), this has the analytic solution
    sqrt(pi/2) exp(-k**2/2) k
    which this function can be checked against.
    """
    extent = x[-1]-x[0]
    N = x.size
    step = extent/N
    integrand = x*y
    uk = -np.fft.rfft(integrand,axis=axis).imag*step
    ks = np.fft.rfftfreq(N, step) *2*np.pi
    return ks,uk
    
def analytic_fft_integral(ks): return np.sqrt(np.pi/2.)*np.exp(-ks**2./2.)*ks
