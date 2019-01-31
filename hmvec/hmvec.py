import sys,os
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import camb
from camb import model
import numpy as np
from . import tinker
import scipy
from scipy.integrate import simps

"""

General vectorized FFT-based halo model implementation
Author(s): Mathew Madhavacheril
Credits: Follows approach in Matt Johnson and Moritz
Munchmeyer's implementation in the appendix of 1810.13423.
Some of the HOD functions are copied from there.

Array indexing is as follows:
[z,M,k/r]

r is always in Mpc
k is always in Mpc-1
All rho densities are in Msolar/Mpc^3
All masses m are in Msolar
No h units anywhere

TODO: copy member functions like Duffy concentration to independent
barebones functions in separate script/library

Known issues:
1. The 1-halo term will add some power at the largest scales. A softening term
has been added that effectively zeros the 1-halo term below k<0.01.
2. sigma2 becomes very innacurate for low masses, because lower masses require higher
k in the linear matter power. For this reason, low-mass halos
are never accounted for correctly, and thus consistency relations do not hold.
Currently the consistency relation is subtracted out to get the 2-halo
power to agree with Plin at large scales.
3. Higher redshifts have less than expected 1-halo power compared to halofit. 

Limitations:
1. Tinker 2010 option and Sheth-Torman have only been coded up for M200m and mvir
respectively.

"""
battaglia_defaults = {}
battaglia_defaults['AGN'] = {
    'rho0_A0':4000.,
    'rho0_alpham':0.29,
    'rho0_alphaz':-0.66,
    'alpha_A0':0.88,
    'alpha_alpham':-0.03,
    'alpha_alphaz':0.19,
    'beta_A0':3.83,
    'beta_alpham':0.04,
    'beta_alphaz':-0.025
}
battaglia_defaults['SH'] = {
    'rho0_A0':19000.,
    'rho0_alpham':0.09,
    'rho0_alphaz':-0.95,
    'alpha_A0':0.70,
    'alpha_alpham':-0.017,
    'alpha_alphaz':0.27,
    'beta_A0':4.43,
    'beta_alpham':0.005,
    'beta_alphaz':0.037
}
    

default_params = {
    
    # Mass function
    'st_A': 0.3222,
    'st_a': 0.707,
    'st_p': 0.3,
    'st_deltac': 1.686,
    'sigma2_kmin':1e-4,
    'sigma2_kmax':2000,
    'sigma2_numks':10000,
    'Wkr_taylor_switch':0.01,

    # Profiles
    'duffy_A_vir':7.85, # for Mvir
    'duffy_alpha_vir':-0.081,
    'duffy_beta_vir':-0.71,
    'duffy_A_mean':10.14, # for M200rhomeanz
    'duffy_alpha_mean':-0.081,
    'duffy_beta_mean':-1.01,
    'nfw_integral_numxs':40000, # not sufficient
    'nfw_integral_xmax':200,
    'battaglia_gas_gamma':-0.2,
    'battaglia_gas_family': 'AGN',

    # Power spectra
    'kstar_damping':0.01,
    'default_halofit':'mead',
    
    # Cosmology
    'omch2': 0.1198,
    'ombh2': 0.02225,
    'H0': 67.3,
    'ns': 0.9645,
    'As': 2.2e-9,
    'mnu': 0.0, # NOTE NO NEUTRINOS IN DEFAULT
    'w0': -1.0,
    'tau':0.06,
    'nnu':3.046,
    'wa': 0.,
    'num_massive_neutrinos':3,

}
    

def Wkr_taylor(kR):
    xx = kR*kR
    return 1 - .1*xx + .00357142857143*xx*xx

def Wkr(k,R,taylor_switch=default_params['Wkr_taylor_switch']):
    kR = k*R
    ans = 3.*(np.sin(kR)-kR*np.cos(kR))/(kR**3.)
    ans[kR<taylor_switch] = Wkr_taylor(kR[kR<taylor_switch]) 
    return ans

def duffy_concentration(m,z,A=None,alpha=None,beta=None,h=None):
    A = default_params['duffy_A_mean'] if A is None else A
    alpha = default_params['duffy_alpha_mean'] if alpha is None else alpha
    beta = default_params['duffy_beta_mean'] if beta is None else beta
    h = default_params['H0'] / 100. if h is None else h
    return A*((h*m/2.e12)**alpha)*(1+z)**beta
    
class HaloCosmology(object):
    def __init__(self,zs,ks,ms=None,params={},mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False):
        self.mdef = mdef
        self.p = params
        self.mode = mass_function
        for param in default_params.keys():
            if param not in self.p.keys(): self.p[param] = default_params[param]
        self.zs = np.asarray(zs)
        self.ks = ks
        self.ms = ms
        
        # Cosmology
        self._init_cosmology(self.p,halofit)

        # Mass function
        if ms is not None: self.init_mass_function(ms)
        
        # Profiles
        self.uk_profiles = {}
        if not(skip_nfw): self.add_nfw_profile("nfw",numeric=nfw_numeric)
                    

    def update_param(self,param,value,halofit=None):
        cosmo_params = ['omch2','ombh2','H0','ns','As','mnu','w0','tau','nnu','wa','num_massive_neutrinos']
        self.p[param] = value
        if param in cosmo_params:
            self._init_cosmology(self.p,halofit)
            self._init_mass_function()
            # update profiles
            pass
        elif param in profile_params:
            # update profiles
            pass
        elif param in mass_function_params:
            self._init_mass_function()
        else:
            raise ValueError
        
    def _init_cosmology(self,params,halofit):
        try:
            theta = params['theta100']/100.
            H0 = None
            print("WARNING: Using theta100 parameterization. H0 ignored.")
        except:
            H0 = params['H0']
            theta = None
        
        self.pars = camb.set_params(ns=params['ns'],As=params['As'],H0=H0,
                                    cosmomc_theta=theta,ombh2=params['ombh2'],
                                    omch2=params['omch2'], mnu=params['mnu'],
                                    tau=params['tau'],nnu=params['nnu'],
                                    num_massive_neutrinos=
                                    params['num_massive_neutrinos'],
                                    w=params['w0'],wa=params['wa'],
                                    dark_energy_model='ppf',
                                    halofit_version=self.p['default_halofit'] if halofit is None else halofit,
                                    AccuracyBoost=2)
        self.results = camb.get_background(self.pars)
        self.params = params
        self.h = self.params['H0']/100.
        omh2 = self.params['omch2']+self.params['ombh2'] # FIXME: neutrinos
        self.om0 = omh2 / (self.params['H0']/100.)**2.
        self.chis = self.results.comoving_radial_distance(self.zs)
        self.Hzs = self.results.hubble_parameter(self.zs)
        self.Pzk = self._get_matter_power(self.zs,self.ks,nonlinear=False)
        if halofit is not None: self.nPzk = self._get_matter_power(self.zs,self.ks,nonlinear=True)

        
    def _get_matter_power(self,zs,ks,nonlinear=False):
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear, 
                                                hubble_units=False,
                                                k_hunit=False, kmax=ks.max(),
                                                zmax=zs.max()+1.)
        return PK.P(zs, ks, grid=True)

        
    def rho_critical_z(self,z):
        Hz = self.results.hubble_parameter(z) * 3.241e-20 # SI # FIXME: constants need checking
        G = 6.67259e-11 # SI
        rho = 3.*(Hz**2.)/8./np.pi/G # SI
        return rho * 1.477543e37 # in msolar / megaparsec3
    
    def rho_matter_z(self,z):
        return self.rho_critical_z(0.) * self.om0 \
            * (1+np.atleast_1d(z))**3. # in msolar / megaparsec3
    def omz(self,z): return self.rho_matter_z(z)/self.rho_critical_z(z)
    def deltav(self,z): # Duffy virial actually uses this from Bryan and Norman 1997
        # return 178. * self.omz(z)**(0.45) # Eke et al 1998
        x = self.omz(z) - 1.
        d = 18.*np.pi**2. + 82.*x - 39. * x**2.
        return d
    def rvir(self,m,z):
        if self.mdef == 'vir':
            return R_from_M(m,self.rho_critical_z(z),delta=self.deltav(z))
        elif self.mdef == 'mean':
            return R_from_M(m,self.rho_matter_z(z),delta=200.)
    
    def R_of_m(self,ms):
        return R_from_M(ms,self.rho_matter_z(0),delta=1.) # note rhom0
    
    def get_sigma2(self):
        ms = self.ms
        kmin = self.p['sigma2_kmin']
        kmax = self.p['sigma2_kmax']
        numks = self.p['sigma2_numks']
        self.ks_sigma2 = np.geomspace(kmin,kmax,numks) # ks for sigma2 integral
        self.sPzk = self.P_lin(self.ks_sigma2,self.zs)
        ks = self.ks_sigma2[None,None,:]
        R = self.R_of_m(ms)[None,:,None]
        W2 = Wkr(ks,R,self.p['Wkr_taylor_switch'])**2.
        Ps = self.sPzk[:,None,:]
        integrand = Ps*W2*ks**2./2./np.pi**2.
        sigma2 = simps(integrand,ks,axis=-1)
        return sigma2
        
    def init_mass_function(self,ms):
        self.ms = ms
        self.sigma2 = self.get_sigma2()
        self.nzm = self.get_nzm()
        self.bh = self.get_bh()

    def get_fsigmaz(self):
        sigma2 = self.sigma2
        deltac = self.p['st_deltac']
        if self.mode=="sheth-torman":
            sigma = np.sqrt(sigma2)
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            return A*np.sqrt(2.*a/np.pi)*(1+((sigma2/a/deltac**2.)**p))*(deltac/sigma)*np.exp(-a*deltac**2./2./sigma2)
        elif self.mode=="tinker":
            nus = deltac/np.sqrt(sigma2)
            fnus = tinker.f_nu(nus,self.zs[:,None])
            return nus * fnus # note that f is actually nu*fnu !
        else:
            raise NotImplementedError
    
    def get_bh(self):
        sigma2 = self.sigma2
        deltac = self.p['st_deltac']
        if self.mode=="sheth-torman":
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            return 1. + (1./deltac)*((a*deltac**2./sigma2)-1.) + (2.*p/deltac)/(1.+(a*deltac**2./sigma2)**p)
        elif self.mode=="tinker":
            nus = deltac/np.sqrt(sigma2)
            return tinker.bias(nus)
        else:
            raise NotImplementedError

    def concentration(self,mode='duffy'):
        ms = self.ms
        if mode=='duffy':
            if self.mdef == 'mean':
                A = self.p['duffy_A_mean']
                alpha = self.p['duffy_alpha_mean']
                beta = self.p['duffy_beta_mean']
            elif self.mdef == 'vir':
                A = self.p['duffy_A_vir']
                alpha = self.p['duffy_alpha_vir']
                beta = self.p['duffy_beta_vir']
            return duffy_concentration(ms[None,:],self.zs[:,None],A,alpha,beta,self.h)
        else:
            raise NotImplementedError

    def get_nzm(self):
        sigma2 = self.sigma2
        ms = self.ms
        ln_sigma_inv = -0.5*np.log(sigma2)
        fsigmaz = self.get_fsigmaz()
        dln_sigma_dlnm = np.gradient(ln_sigma_inv,np.log(ms),axis=-1)
        ms = ms[None,:]
        return self.rho_matter_z(0) * fsigmaz * dln_sigma_dlnm / ms**2. 

    
    def D_growth(self, a):
        # From Moritz Munchmeyer?
        
        _amin = 0.001    # minimum scale factor
        _amax = 1.0      # maximum scale factor
        _na = 512        # number of points in interpolation arrays
        atab = np.linspace(_amin,
                           _amax,
                           _na)
        ks = np.logspace(np.log10(1e-5),np.log10(1.),num=100) 
        zs = a2z(atab)
        deltakz = self.results.get_redshift_evolution(ks, zs, ['delta_cdm']) #index: k,z,0
        D_camb = deltakz[0,:,0]/deltakz[0,0,0]
        _da_interp = interp1d(atab, D_camb, kind='linear')
        _da_interp_type = "camb"
        return _da_interp(a)/_da_interp(1.0)

    def add_battaglia_profile(self,name,family=None,param_override={},
                              nxs=None,
                              xmax=None):
        # Set default parameters
        if family is None: family = self.p['battaglia_gas_family'] # AGN or SH?
        pparams = {}
        pparams['battaglia_gas_gamma'] = self.p['battaglia_gas_gamma']
        pparams.update(battaglia_defaults[family])

        # Update with overrides
        for key in param_override:
            if key=='battaglia_gas_gamma':
                pparams[key] = param_override[key]
            elif key in battaglia_defaults[family]:
                pparams[key] = param_override[key]
            else:
                raise ValueError # param in param_override doesn't seem to be a Battaglia parameter

        # Convert masses to m200critz
        rhocritz = self.rho_critical_z(self.zs)
        if self.mdef=='vir':
            delta_rhos1 = rhocritz*self.deltav(self.zs)
        elif self.mdef=='mean':
            delta_rhos1 = self.rho_matter_z(self.zs)*200.
        rvirs = self.rvir(self.ms[None,:],self.zs[:,None])
        cs = self.concentration()
        delta_rhos2 = 200.*self.rho_critical_z(self.zs)
        m200critz = mdelta_from_mdelta(self.ms,cs,delta_rhos1,delta_rhos2)
        r200critz = R_from_M(m200critz,self.rho_critical_z(self.zs),delta=200.)

        # Generate profiles
        """
        The physical profile is rho(r) = f(2r/R200)
        We rescale this to f(x), so x = r/(R200/2) = r/rgs
        So rgs = R200/2 is the equivalent of rss in the NFW profile
        """
        omb = self.p['ombh2'] / self.h**2.
        omm = self.om0
        rhofunc = lambda x: rho_gas_generic_x(x,m200critz[...,None],self.zs[:,None,None],omb,omm,rhocritz[...,None,None],
                                    gamma=pparams['battaglia_gas_gamma'],
                                    rho0_A0=pparams['rho0_A0'],
                                    rho0_alpham=pparams['rho0_alpham'],
                                    rho0_alphaz=pparams['rho0_alphaz'],
                                    alpha_A0=pparams['alpha_A0'],
                                    alpha_alpham=pparams['alpha_alpham'],
                                    alpha_alphaz=pparams['alpha_alphaz'],
                                    beta_A0=pparams['beta_A0'],
                                    beta_alpham=pparams['beta_alpham'],
                                    beta_alphaz=pparams['beta_alphaz'])

        rgs = r200critz/2.
        cgs = rvirs/rgs
        ks,ukouts = generic_profile_fft(rhofunc,cgs,rgs[...,None],self.zs,self.ks,xmax,nxs)
        self.uk_profiles[name] = ukouts.copy()

        
                        
    def add_nfw_profile(self,name,numeric=False,
                        nxs=None,
                        xmax=None):

        """
        xmax should be thought of in "concentration units", i.e.,
        for a cluster with concentration 3., xmax of 100 is probably overkill
        since the integrals are zero for x>3. However, since we are doing
        a single FFT over all profiles, we need to choose a single xmax.
        xmax of 100 is very safe for m~1e9 msolar, but xmax of 200-300
        will be needed down to m~1e2 msolar.
        nxs is the number of samples from 0 to xmax of rho_nfw(x). Might need
        to be increased from default if xmax is increased and/or going down
        to lower halo masses.
        xmax decides accuracy on large scales
        nxs decides accuracy on small scales
        
        """
        if nxs is None: nxs = self.p['nfw_integral_numxs']
        if xmax is None: xmax = self.p['nfw_integral_xmax']
        cs = self.concentration()
        ms = self.ms
        rvirs = self.rvir(ms[None,:],self.zs[:,None])
        rss = (rvirs/cs)[...,None]
        if numeric:
            ks,ukouts = generic_profile_fft(lambda x: rho_nfw_x(x,rhoscale=1),cs,rss,self.zs,self.ks,xmax,nxs)
            self.uk_profiles[name] = ukouts.copy()
        else:
            cs = cs[...,None]
            mc = np.log(1+cs)-cs/(1.+cs)
            x = self.ks[None,None]*rss *(1+self.zs[:,None,None])# !!!!
            Si, Ci = scipy.special.sici(x)
            Sic, Cic = scipy.special.sici((1.+cs)*x)
            ukouts = (np.sin(x)*(Sic-Si) - np.sin(cs*x)/((1+cs)*x) + np.cos(x)*(Cic-Ci))/mc
            self.uk_profiles[name] = ukouts.copy()
        
        return self.ks,ukouts
        

    # def get_power_1halo_cross_galaxies(self,name="nfw",name_gal="nfw"):
    #     pass
    # def get_power_2halo_cross_galaxies(self,name="nfw",name_gal="nfw"):
    #     pass

    def get_power_1halo(self,name="nfw",name2=None):
        name2 = name if name2 is None else name2
        ms = self.ms[...,None]
        integrand = self.nzm[...,None] * ms**2. * self.uk_profiles[name]*self.uk_profiles[name2] /self.rho_matter_z(0)**2.
        return np.trapz(integrand,ms,axis=-2)*(1.-np.exp(-(self.ks/self.p['kstar_damping'])**2.))
    
    def get_power_2halo(self,name="nfw",name2=None,verbose=False):
        def _2haloint(profile):
            integrand = self.nzm[...,None] * ms * profile /self.rho_matter_z(0) * self.bh[...,None]
            integral = np.trapz(integrand,ms,axis=-2)
            return integral
            
        ms = self.ms[...,None]
        integral = _2haloint(self.uk_profiles[name])
        if (name2 is None) or (name2==name): integral2 = integral.copy()
        else: integral2 = _2haloint(self.uk_profiles[name2])
            
        # consistency relation : Correct for part that's missing from low-mass halos to get P(k->0) = Plinear
        consistency_integrand = self.nzm[...,None] * ms /self.rho_matter_z(0) * self.bh[...,None]
        consistency = np.trapz(consistency_integrand,ms,axis=-2)
        if verbose: print("Two-halo consistency: " , consistency)
        return self.Pzk * (integral+1-consistency)*(integral2+1-consistency)
        
    # def get_power_1halo_galaxy_auto(self):
    #     pass
    
    # def get_power_2halo_galaxy_auto(self):
    #     pass

    def gas_profile(self,r,m200meanz):
        pass

    def P_lin(self,ks,zs,knorm = 1e-4,kmax = 0.1):
        """
        This function will provide the linear matter power spectrum used in calculation
        of sigma2. It is written as
        P_lin(k,z) = norm(z) * T(k)**2
        where T(k) is the Eisenstein, Hu, 1998 transfer function.
        Care has to be taken about interpreting this beyond LCDM.
        For example, the transfer function can be inaccurate for nuCDM and wCDM cosmologies.
        If this function is only used to model sigma2 -> N(M,z) -> halo model power spectra at small
        scales, and cosmological dependence is obtained through an accurate CAMB based P(k),
        one should be fine.
        """
        tk = self.Tk(ks,'eisenhu_osc') 
        assert knorm<kmax
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=False, 
                                                     hubble_units=False, k_hunit=False, kmax=kmax,
                                                     zmax=zs.max()+1.)
        pnorm = PK.P(zs, knorm,grid=True)
        tnorm = self.Tk(knorm,'eisenhu_osc') * knorm**(self.params['ns'])
        plin = (pnorm/tnorm) * tk**2. * ks**(self.params['ns'])
        return plin
        
        
    def Tk(self,ks,type ='eisenhu_osc'):
        """
        Pulled from cosmicpy https://github.com/cosmicpy/cosmicpy/blob/master/LICENSE.rst
        """
        
        k = ks/self.h
        self.tcmb = 2.726
        T_2_7_sqr = (self.tcmb/2.7)**2
        h2 = self.h**2
        w_m = self.params['omch2'] + self.params['ombh2']
        w_b = self.params['ombh2']

        self._k_eq = 7.46e-2*w_m/T_2_7_sqr / self.h     # Eq. (3) [h/Mpc]
        self._z_eq = 2.50e4*w_m/(T_2_7_sqr)**2          # Eq. (2)

        # z drag from Eq. (4)
        b1 = 0.313*pow(w_m, -0.419)*(1.0+0.607*pow(w_m, 0.674))
        b2 = 0.238*pow(w_m, 0.223)
        self._z_d = 1291.0*pow(w_m, 0.251)/(1.0+0.659*pow(w_m, 0.828)) * \
            (1.0 + b1*pow(w_b, b2))

        # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
        self._R_d = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_d)
        # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
        self._R_eq = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_eq)
        # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
        self.sh_d = 2.0/(3.0*self._k_eq) * np.sqrt(6.0/self._R_eq) * \
            np.log((np.sqrt(1.0 + self._R_d) + np.sqrt(self._R_eq + self._R_d)) /
                (1.0 + np.sqrt(self._R_eq)))
        # Eq. (7) but in [hMpc^{-1}]
        self._k_silk = 1.6 * pow(w_b, 0.52) * pow(w_m, 0.73) * \
            (1.0 + pow(10.4*w_m, -0.95)) / self.h

        Omega_m = self.om0
        fb = self.params['ombh2'] / (self.params['omch2']+self.params['ombh2']) # self.Omega_b / self.Omega_m
        fc = self.params['omch2'] / (self.params['omch2']+self.params['ombh2']) # self.params['ombh2'] #(self.Omega_m - self.Omega_b) / self.Omega_m
        alpha_gamma = 1.-0.328*np.log(431.*w_m)*w_b/w_m + \
            0.38*np.log(22.3*w_m)*(fb)**2
        gamma_eff = Omega_m*self.h * \
            (alpha_gamma + (1.-alpha_gamma)/(1.+(0.43*k*self.sh_d)**4))

        res = np.zeros_like(k)

        if(type == 'eisenhu'):

            q = k * pow(self.tcmb/2.7, 2)/gamma_eff

            # EH98 (29) #
            L = np.log(2.*np.exp(1.0) + 1.8*q)
            C = 14.2 + 731.0/(1.0 + 62.5*q)
            res = L/(L + C*q*q)

        elif(type == 'eisenhu_osc'):
            # Cold dark matter transfer function

            # EH98 (11, 12)
            a1 = pow(46.9*w_m, 0.670) * (1.0 + pow(32.1*w_m, -0.532))
            a2 = pow(12.0*w_m, 0.424) * (1.0 + pow(45.0*w_m, -0.582))
            alpha_c = pow(a1, -fb) * pow(a2, -fb**3)
            b1 = 0.944 / (1.0 + pow(458.0*w_m, -0.708))
            b2 = pow(0.395*w_m, -0.0266)
            beta_c = 1.0 + b1*(pow(fc, b2) - 1.0)
            beta_c = 1.0 / beta_c

            # EH98 (19). [k] = h/Mpc
            def T_tilde(k1, alpha, beta):
                # EH98 (10); [q] = 1 BUT [k] = h/Mpc
                q = k1 / (13.41 * self._k_eq)
                L = np.log(np.exp(1.0) + 1.8 * beta * q)
                C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * pow(q, 1.08))
                T0 = L/(L + C*q*q)
                return T0

            # EH98 (17, 18)
            f = 1.0 / (1.0 + (k * self.sh_d / 5.4)**4)
            Tc = f * T_tilde(k, 1.0, beta_c) + \
                (1.0 - f) * T_tilde(k, alpha_c, beta_c)

            # Baryon transfer function
            # EH98 (19, 14, 21)
            y = (1.0 + self._z_eq) / (1.0 + self._z_d)
            x = np.sqrt(1.0 + y)
            G_EH98 = y * (-6.0 * x +
                          (2.0 + 3.0*y) * np.log((x + 1.0) / (x - 1.0)))
            alpha_b = 2.07 * self._k_eq * self.sh_d * \
                pow(1.0 + self._R_d, -0.75) * G_EH98

            beta_node = 8.41 * pow(w_m, 0.435)
            tilde_s = self.sh_d / pow(1.0 + (beta_node /
                                             (k * self.sh_d))**3, 1.0/3.0)

            beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * w_m)**2 + 1.0)

            # [tilde_s] = Mpc/h
            Tb = (T_tilde(k, 1.0, 1.0) / (1.0 + (k * self.sh_d / 5.2)**2) +
                  alpha_b / (1.0 + (beta_b/(k * self.sh_d))**3) *
                  np.exp(-pow(k / self._k_silk, 1.4))) * np.sinc(k*tilde_s/np.pi)

            # Total transfer function
            res = fb * Tb + fc * Tc
        return res
"""
Mass function
"""
def R_from_M(M,rho,delta): return (3.*M/4./np.pi/delta/rho)**(1./3.) 

"""
HOD functions from Matt Johnson and Moritz Munchmeyer (modified)
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
    # Function to compute halo mass as a function of the stellar mass. arxiv 1001.0015 Table 2
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

def avg_NsNsm1(log10mhalo,z,log10mstellar_thresh,corr="max"):
    if corr=='max':
        return 

"""
Profiles
"""

def Fcon(c): return (np.log(1.+c) - (c/(1.+c)))

def rhoscale_nfw(mdelta,rdelta,cdelta):
    rs = rdelta/cdelta
    V = 4.*np.pi * rs**3.
    return pref * mdelta / V / Fcon(cdelta)

def rho_nfw_x(x,rhoscale): return rhoscale/x/(1.+x)**2.

def rho_nfw(r,rhoscale,rs): return rho_nfw_x(r/rs,rhoscale)

def mdelta_from_mdelta(M1,C1,delta_rhos1,delta_rhos2,vectorized=True):
    """
    Fast/vectorized mass definition conversion

    Converts M1(m) to M2(z,m).
    Needs concentrations C1(z,m),
    cosmic densities delta_rhos1(z), e.g. delta_rhos1(z) = Delta_vir(z)*rhoc(z)
    cosmic densities delta_rhos2(z), e.g. delta_rhos2(z) = 200*rhom(z)

    The vectorized version is several orders of magnitude faster.
    """
    if vectorized:
        M1 = M1[None,:]+C1*0.
        M2outs =  mdelta_from_mdelta_unvectorized(M1.copy(),C1,delta_rhos1[:,None],delta_rhos2[:,None])
    else:
        M2outs = np.zeros(C1.shape)
        for i in range(C1.shape[0]):
            for j in range(C1.shape[1]):
                M2outs[i,j] = mdelta_from_mdelta_unvectorized(M1[j],C1[i,j],delta_rhos1[i],delta_rhos2[i])
    return M2outs

    
def mdelta_from_mdelta_unvectorized(M1,C1,delta_rhos1,delta_rhos2):
    """
    Implements mdelta_from_mdelta.
    The logMass is necessary for numerical stability.
    I thought I calculated the right derivative, but using it leads to wrong
    answers, so the secant method is used instead of Newton's method.
    In principle, both the first and second derivatives can be calculated
    analytically.

    The conversion is done by assuming NFW and solving the equation
    M1 F1 - M2 F2 = 0
    where F(conc) = 1 / (log(1+c) - c/(1+c))
    This equation is obtained when rhoscale is equated between the two mass
    definitions, where rhoscale = F(c) * m /(4pi rs**3) is the amplitude
    of the NFW profile. The scale radii are also the same. Equating
    the scale radii also provides
    C2 = ((M2/M1) * (delta_rhos1/delta_rhos2) * (rho1/rho2)) ** (1/3) C1
    which reduces the system to one unknown M2.
    """
    C2 = lambda logM2: C1*((np.exp(logM2-np.log(M1)))*(delta_rhos1/delta_rhos2))**(1./3.)
    F2 = lambda logM2: 1./Fcon(C2(logM2))
    F1 = 1./Fcon(C1)
    # the function whose roots to find
    func = lambda logM2: M1*F1 - np.exp(logM2)*F2(logM2)
    from scipy.optimize import newton
    # its analytical derivative
    #jaco = lambda logM2: -F2(logM2) + (C2(logM2)/(1.+C2(logM2)))**2. * C2(logM2)/3. * F2(logM2)**2.
    M2outs = newton(func,np.log(M1))#,fprime=jaco) # FIXME: jacobian doesn't work
    return np.exp(M2outs)

def battaglia_gas_fit(m200critz,z,A0x,alphamx,alphazx):
    # Any factors of h in M?
    return A0x * (m200critz/1.e14)**alphamx * (1.+z)**alphazx
    
def rho_gas(r,m200critz,z,omb,omm,rhocritz,
            gamma=default_params['battaglia_gas_gamma'],
            profile="AGN"):
    return rho_gas_generic(r,m200critz,z,omb,omm,rhocritz,
                           gamma=gamma,
                           rho0_A0=battaglia_defaults[profile]['rho0_A0'],
                           rho0_alpham=battaglia_defaults[profile]['rho0_alpham'],
                           rho0_alphaz=battaglia_defaults[profile]['rho0_alphaz'],
                           alpha_A0=battaglia_defaults[profile]['alpha_A0'],
                           alpha_alpham=battaglia_defaults[profile]['alpha_alpham'],
                           alpha_alphaz=battaglia_defaults[profile]['alpha_alphaz'],
                           beta_A0=battaglia_defaults[profile]['beta_A0'],
                           beta_alpham=battaglia_defaults[profile]['beta_alpham'],
                           beta_alphaz=battaglia_defaults[profile]['beta_alphaz'])

def rho_gas_generic(r,m200critz,z,omb,omm,rhocritz,
                    gamma=default_params['battaglia_gas_gamma'],
                    rho0_A0=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_A0'],
                    rho0_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alpham'],
                    rho0_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alphaz'],
                    alpha_A0=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_A0'],
                    alpha_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alpham'],
                    alpha_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alphaz'],
                    beta_A0=battaglia_defaults[default_params['battaglia_gas_family']]['beta_A0'],
                    beta_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alpham'],
                    beta_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alphaz'],
):
    """
    AGN and SH Battaglia 2016 profiles
    r: physical distance
    m200critz: M200_critical_z

    """
    R200 = R_from_M(m200critz,rhocritz,delta=200)
    x = 2*r/R200
    return rho_gas_generic_x(x,m200critz,z,omb,omm,rhocritz,gamma,
                             rho0_A0,rho0_alpham,rho0_alphaz,
                             alpha_A0,alpha_alpham,alpha_alphaz,
                             beta_A0,beta_alpham,beta_alphaz)

def rho_gas_generic_x(x,m200critz,z,omb,omm,rhocritz,
                    gamma=default_params['battaglia_gas_gamma'],
                    rho0_A0=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_A0'],
                    rho0_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alpham'],
                    rho0_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alphaz'],
                    alpha_A0=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_A0'],
                    alpha_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alpham'],
                    alpha_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alphaz'],
                    beta_A0=battaglia_defaults[default_params['battaglia_gas_family']]['beta_A0'],
                    beta_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alpham'],
                    beta_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alphaz'],
):
    rho0 = battaglia_gas_fit(m200critz,z,rho0_A0,rho0_alpham,rho0_alphaz)
    alpha = battaglia_gas_fit(m200critz,z,alpha_A0,alpha_alpham,alpha_alphaz)
    beta = battaglia_gas_fit(m200critz,z,beta_A0,beta_alpham,beta_alphaz)
    return (omb/omm) * rhocritz * rho0 * (x)**gamma * (1.+x**alpha)**(-(beta+gamma)/alpha)


"""
FFT routines
"""

def uk_fft(rhofunc,rvir,dr=0.001,rmax=100):
    rvir = np.asarray(rvir)
    rps = np.arange(dr,rmax,dr)
    rs = rps
    rhos = rhofunc(np.abs(rs))
    theta = np.ones(rhos.shape)
    theta[np.abs(rs)>rvir[...,None]] = 0 # CHECK
    integrand = rhos * theta
    m = np.trapz(integrand*rs**2.,rs,axis=-1)*4.*np.pi
    ks,ukt = fft_integral(rs,integrand)
    uk = 4.*np.pi*ukt/ks/m[...,None]
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
    assert x.ndim==1
    extent = x[-1]-x[0]
    N = x.size
    step = extent/N
    integrand = x*y
    uk = -np.fft.rfft(integrand,axis=axis).imag*step
    ks = np.fft.rfftfreq(N, step) *2*np.pi
    return ks,uk
    
def analytic_fft_integral(ks): return np.sqrt(np.pi/2.)*np.exp(-ks**2./2.)*ks


def a2z(a): return (1.0/a)-1.0

pcount = 0
def generic_profile_fft(rhofunc_x,cmaxs,rss,zs,ks,xmax,nxs):
    """
    Generic profile FFTing
    rhofunc_x: function that accepts vector spanning linspace(0,xmax,nxs)
    xmax:  some O(10-1000) dimensionless number specifying maximum of real space
    profile
    nxs: number of samples of the profile.
    cmaxs: typically an [nz,nm] array of the dimensionless cutoff for the profile integrals. 
    For NFW, for example, this is concentration(z,mass).
    For other profiles, you will want to do cmax = Rvir(z,m)/R_scale_radius where
    R_scale_radius is whatever you have divided the physical distance by in the profile to
    get the integration variable i.e. x = r / R_scale_radius.
    rss: R_scale_radius
    zs: [nz,] array to convert physical wavenumber to comoving wavenumber.
    ks: target comoving wavenumbers to interpolate the resulting FFT on to.
    
    """
    xs = np.linspace(0.,xmax,nxs+1)[1:]
    rhos = rhofunc_x(xs)
    if rhos.ndim==1:
        rhos = rhos[None,None]
    else:
        assert rhos.ndim==3
    rhos = rhos + cmaxs[...,None]*0.
    theta = np.ones(rhos.shape)
    theta[np.abs(xs)>cmaxs[...,None]] = 0 # CHECK
    # m
    integrand = theta * rhos * xs**2.
    mnorm = np.trapz(integrand,xs) # mass but off by norm same as rho is off by
    # u(kt)
    integrand = rhos*theta
    kts,ukts = fft_integral(xs,integrand)
    uk = ukts/kts[None,None,:]/mnorm[...,None]
    kouts = kts/rss/(1+zs[:,None,None]) # divide k by (1+z) here for comoving FIXME: check this!
    ukouts = np.zeros((uk.shape[0],uk.shape[1],ks.size))
    # sadly at this point we must loop to interpolate :(
    # from orphics import io
    # pl = io.Plotter(xyscale='loglog')
    for i in range(uk.shape[0]):
        for j in range(uk.shape[1]):
            pks = kouts[i,j]
            puks = uk[i,j]
            puks = puks[pks>0]
            pks = pks[pks>0]
            ukouts[i,j] = np.interp(ks,pks,puks,left=puks[0],right=0)
            #TODO: Add compulsory debug plot here
    #         pl.add(ks,ukouts[i,j])
    # pl.hline(y=1)
    # pl.done()
    return ks, ukouts
