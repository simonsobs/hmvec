import sys,os
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import camb
from camb import model
import numpy as np
import astropy.constants as const
from . import tinker,utils
from .cosmology import Cosmology
from . import cib

import scipy.constants as constants
from .params import default_params, battaglia_defaults
from .fft import generic_profile_fft
import scipy

from scipy.integrate import simps
from scipy.integrate import quad
import time
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

 In Fisher calculations, I want to calculate power spectra at a fiducial parameter set
 and at perturbed parameters (partial derivatives). The usual flowdown for a power
 spectrum calculation is:
 C1. initialize background cosmology and linear matter power
 C2. calculate mass function
 C3. calculate profiles and HODs
 with each step depending on the previous. This means if I change a parameter associated
 with C3, I don't need to recalculate C1 and C2. So a Fisher calculation with n-point derivatives
 should be
 done based on a parameter set p = {p1,p2,p3} for each of the above as follows:
 1. Calculate power spectra for fiducial p (1 C1,C2,C3 call each)
 2. Calculate power spectra for perturbed p3 (n C3 calls for each p3 parameter)
 3. Calculate power spectra for perturbed p2 (n C2,C3 calls for each p2 parameter)
 2. Calculate power spectra for perturbed p1 (n C1,C2,C3 calls for each p1 parameter)

 """


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

class HaloModel(Cosmology):
    def __init__(self,zs,ks,ms=None,params={},mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False):
        self.zs = np.asarray(zs)
        self.ks = ks
        Cosmology.__init__(self,params,halofit)

        self.mdef = mdef
        self.mode = mass_function
        self.ms = ms
        self.hods = {}
        self.cib_params = {}

        # Mass function
        if ms is not None: self.init_mass_function(ms)

        # Profiles
        self.uk_profiles = {}
        self.pk_profiles = {}
        if not(skip_nfw): self.add_nfw_profile("nfw",numeric=nfw_numeric)

    def _init_cosmology(self,params,halofit):
        Cosmology._init_cosmology(self,params,halofit)
        self.Pzk = self._get_matter_power(self.zs,self.ks,nonlinear=False)
        if halofit is not None: self.nPzk = self._get_matter_power(self.zs,self.ks,nonlinear=True)


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
            # return nus, fnus                 # debugging    
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
            # return nus, tinker.bias(nus)          # debugging
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


    def add_battaglia_profile(self,name,family=None,param_override={},
                              nxs=None,
                              xmax=None,ignore_existing=False):
        if not(ignore_existing): assert name not in self.uk_profiles.keys(), "Profile name already exists."
        assert name!='nfw', "Name nfw is reserved."

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
        r200critz = R_from_M(m200critz,self.rho_critical_z(self.zs)[:,None],delta=200.)

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

    def add_battaglia_pres_profile(self,name,family=None,param_override={},
                              nxs=None,
                              xmax=None,ignore_existing=False):
        if not(ignore_existing): assert name not in self.pk_profiles.keys(), "Profile name already exists."
        assert name!='nfw', "Name nfw is reserved."

        # Set default parameters
        if family is None: family = self.p['battaglia_pres_family'] # AGN or SH?
        pparams = {}
        pparams['battaglia_pres_gamma'] = self.p['battaglia_pres_gamma']
        pparams['battaglia_pres_alpha'] = self.p['battaglia_pres_alpha']
        pparams.update(battaglia_defaults[family])

        # Update with overrides
        for key in param_override:
            if key in ['battaglia_pres_gamma','battaglia_pres_alpha']:
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
        r200critz = R_from_M(m200critz,self.rho_critical_z(self.zs)[:,None],delta=200.)

        # Generate profiles
        """
        The physical profile is rho(r) = f(2r/R200)
        We rescale this to f(x), so x = r/(R200/2) = r/rgs
        So rgs = R200/2 is the equivalent of rss in the NFW profile
        """
        omb = self.p['ombh2'] / self.h**2.
        omm = self.om0
        presFunc = lambda x: P_e_generic_x(x,m200critz[...,None],r200critz[...,None],self.zs[:,None,None],omb,omm,rhocritz[...,None,None],
                                    alpha=pparams['battaglia_pres_alpha'],
                                    gamma=pparams['battaglia_pres_gamma'],
                                    P0_A0=pparams['P0_A0'],
                                    P0_alpham=pparams['P0_alpham'],
                                    P0_alphaz=pparams['P0_alphaz'],
                                    xc_A0=pparams['xc_A0'],
                                    xc_alpham=pparams['xc_alpham'],
                                    xc_alphaz=pparams['xc_alphaz'],
                                    beta_A0=pparams['beta_A0'],
                                    beta_alpham=pparams['beta_alpham'],
                                    beta_alphaz=pparams['beta_alphaz'])

        rgs = r200critz
        cgs = rvirs/rgs
        sigmaT=constants.physical_constants['Thomson cross section'][0] # units m^2
        mElect=constants.physical_constants['electron mass'][0] / default_params['mSun']# units kg
        ks,pkouts = generic_profile_fft(presFunc,cgs,rgs[...,None],self.zs,self.ks,xmax,nxs,do_mass_norm=False)
        self.pk_profiles[name] = pkouts.copy()*4*np.pi*(sigmaT/(mElect*constants.c**2))*(r200critz**3*((1+self.zs)**2/self.h_of_z(self.zs))[...,None])[...,None]

    def add_nfw_profile(self,name,numeric=False,
                        nxs=None,
                        xmax=None,ignore_existing=False):

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
        if not(ignore_existing): assert name not in self.uk_profiles.keys(), "Profile name already exists."
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

    def add_hod(self,name,mthresh=None,ngal=None,corr="max",
                satellite_profile_name='nfw',
                central_profile_name=None,ignore_existing=False,param_override={}):
        """
        Specify an HOD.
        This requires either a stellar mass threshold mthresh (nz,)
        or a number density ngal (nz,) from which mthresh is identified iteratively.
        You can either specify a corr="max" maximally correlated central-satellite
        model or a corr="min" minimally correlated model.
        Miscentering could be included through central_profile_name (default uk=1 for default name of None).
        """
        if not(ignore_existing):
            assert name not in self.uk_profiles.keys(), \
                "HOD name already used by profile."
        assert satellite_profile_name in self.uk_profiles.keys(), \
            "No matter profile by that name exists."
        if central_profile_name is not None:
            assert central_profile_name in self.uk_profiles.keys(), \
                "No matter profile by that name exists."
        if not(ignore_existing):
            assert name not in self.hods.keys(), "HOD with that name already exists."

        hod_params = ['hod_sig_log_mstellar','hod_bisection_search_min_log10mthresh',
                   'hod_bisection_search_max_log10mthresh','hod_bisection_search_rtol',
                   'hod_bisection_search_warn_iter','hod_alphasat','hod_Bsat',
                   'hod_betasat','hod_Bcut','hod_betacut']
        # Set default parameters
        pparams = {}
        for ip in hod_params:
            pparams[ip] = self.p[ip]

        # Update with overrides
        for key in param_override:
            if key in hod_params:
                pparams[key] = param_override[key]
            else:
                raise ValueError # param in param_override doesn't seem to be an HOD parameter



        self.hods[name] = {}
        if ngal is not None:
            try: assert ngal.size == self.zs.size
            except:
                raise ValueError("ngal has to be a vector of size self.zs")
            assert mthresh is None

            nfunc = lambda ilog10mthresh: ngal_from_mthresh(ilog10mthresh,
                                                            self.zs,
                                                            self.nzm,
                                                            self.ms,
                                                            sig_log_mstellar=pparams['hod_sig_log_mstellar'],
                                                            alphasat=pparams['hod_alphasat'],
                                                            Bsat=pparams['hod_Bsat'],betasat=pparams['hod_betasat'],
                                                            Bcut=pparams['hod_Bcut'],betacut=pparams['hod_betacut'])

            log10mthresh = utils.vectorized_bisection_search(ngal,nfunc,
                                                             [pparams['hod_bisection_search_min_log10mthresh'],
                                                              pparams['hod_bisection_search_max_log10mthresh']],
                                                             "decreasing",
                                                             rtol=pparams['hod_bisection_search_rtol'],
                                                             verbose=True,
                                                             hang_check_num_iter=pparams['hod_bisection_search_warn_iter'])
            mthresh = 10**log10mthresh

        try: assert mthresh.size == self.zs.size
        except:
            raise ValueError("mthresh has to be a vector of size self.zs")

        log10mhalo = np.log10(self.ms[None,:])
        log10mstellar_thresh = np.log10(mthresh[:,None])
        Ncs = avg_Nc(log10mhalo,self.zs[:,None],log10mstellar_thresh,sig_log_mstellar=pparams['hod_sig_log_mstellar'])
        Nss = avg_Ns(log10mhalo,self.zs[:,None],log10mstellar_thresh,Nc=Ncs,
                     sig_log_mstellar=pparams['hod_sig_log_mstellar'],
                     alphasat=pparams['hod_alphasat'],
                     Bsat=pparams['hod_Bsat'],betasat=pparams['hod_betasat'],
                     Bcut=pparams['hod_Bcut'],betacut=pparams['hod_betacut'])
        NsNsm1 = avg_NsNsm1(Ncs,Nss,corr)
        NcNs = avg_NcNs(Ncs,Nss,corr)

        self.hods[name]['Nc'] = Ncs
        self.hods[name]['Ns'] = Nss
        self.hods[name]['NsNsm1'] = NsNsm1
        self.hods[name]['NcNs'] = NcNs
        self.hods[name]['ngal'] = self.get_ngal(Ncs,Nss)
        self.hods[name]['bg'] = self.get_bg(Ncs,Nss,self.hods[name]['ngal'])
        self.hods[name]['satellite_profile'] = satellite_profile_name
        self.hods[name]['central_profile'] = central_profile_name
        self.hods[name]['log10mthresh'] = np.log10(mthresh[:,None])

    def set_cibParams(self, name=None, **params):
        """
        Values for parameters of CIB model. To use a pre-existing set of parameters, simply specify 'name'. To tweak a pre-existing set, specify the preset and add the different parameter values as keyword arguments. To use a completely newly set of parameters, don't give a name and give all of the new parameters.
        
        Required Arguments:
        name [string] : Name of parameter set. Presets: 'planck13' and 'vierro'

        Keyword Arguments:
        alpha [float] : SED - z evolution of dust temperature 
        beta [float] : SED - emissivity index at low frequency  
        gamma [float] : SED - frequency power law index at high frequency 
        Td_o [float] : SED - dust temp at z = 0 
        delta [float] : z evolution of normalization of L-M relation 
        logM_eff [float] : log(M_eff) in L-M relation 
        var [float] : variance of Gaussian part of L-M relation 
        L_o [float] : normalization constant for total luminosity
        """
        paramslist = ['alpha', 'beta', 'gamma', 'delta', 'Td_o', 'logM_eff', 'var', 'L_o']

        #Set up the Parameter Set
        if name.lower() == 'planck13':        # Planck 2013
            self.cib_params['alpha'] = 0.36
            self.cib_params['beta'] = 1.75
            self.cib_params['gamma'] = 1.7
            self.cib_params['delta'] = 3.6
            self.cib_params['Td_o'] = 24.4
            self.cib_params['logM_eff'] = 12.6
            self.cib_params['var'] = 0.5
            self.cib_params['L_o'] = 6.4e-8
        elif name.lower() == 'vierro':      # Vierro et al
            self.cib_params['alpha'] = 0.2
            self.cib_params['beta'] = 1.6
            self.cib_params['gamma'] = 1.7      # not in Viero, so using Planck13
            self.cib_params['delta'] = 2.4
            self.cib_params['Td_o'] = 20.7
            self.cib_params['logM_eff'] = 12.3
            self.cib_params['var'] = 0.3
            self.cib_params['L_o'] = 6.4e-8
        elif name==None and len(params)!=8:
            raise Exception("New sets of parameters require exactly 8 parameters")
        else:
            raise NotImplementedError("Need valid parameter set name")    

        #Add Specific Parameters
        for key in params:
            if key not in paramslist:
                raise ValueError(f'"{key}" is not a valid CIB parameter. Note that parameter names are case sensitive') 
            self.cib_params[key] = params[key]


    def get_ngal(self,Nc,Ns): return ngal_from_mthresh(nzm=self.nzm,ms=self.ms,Ncs=Nc,Nss=Ns)

    def get_bg(self,Nc,Ns,ngal):
        integrand = self.nzm * (Nc+Ns) * self.bh
        return np.trapz(integrand,self.ms,axis=-1)/ngal


    def _get_hod_common(self,name):
        hod = self.hods[name]
        cname = hod['central_profile']
        sname = hod['satellite_profile']
        uc = 1 if cname is None else self.uk_profiles[cname]
        us = self.uk_profiles[sname]
        return hod,uc,us

    def _get_hod_square(self,name):
        hod,uc,us = self._get_hod_common(name)
        return (2.*uc*us*hod['NcNs'][...,None]+hod['NsNsm1'][...,None]*us**2.)/hod['ngal'][...,None,None]**2.

    def _get_hod(self,name,lowklim=False):
        hod,uc,us = self._get_hod_common(name)
        if lowklim:
            uc = 1
            us = 1
        return (uc*hod['Nc'][...,None]+us*hod['Ns'][...,None])/hod['ngal'][...,None,None]

    def _get_matter(self,name,lowklim=False):
        ms = self.ms[...,None]
        uk = self.uk_profiles[name]
        if lowklim: uk = 1
        return ms*uk/self.rho_matter_z(0)

    def _get_pressure(self,name,lowklim=False):
        pk = self.pk_profiles[name].copy()
        if lowklim: pk[:,:,:] = pk[:,:,0][...,None]
        return pk

    """
    CIB Stuff
    """
    def get_flux(self, nu_obs, cenflag=True, satflag=True, satmf='Tinker'):
        """Gives CIB flux at observed frequency.

        Args:
            nu_obs (float): Observed frequency
            cenflag (bool, optional): Include central galaxies. Defaults to True.
            satflag (bool, optional): Include satellite galaxies. Defaults to True.
            satmf (str, optional): Subhalo mass function. Defaults to 'Tinker'.

        Returns:
            [array]: flux on whole z,m,k grid
        """
        chis = self.comoving_radial_distance(self.zs)
        fcen = 0.0
        fsat = 0.0
        nu_obs = np.array([nu_obs])

        #Luminosities
        if cenflag:
            fcen = self._get_fcen(nu_obs)
        if satflag:
            fsat = self._get_fsat(nu_obs, satmf=satmf)
        assert cenflag==True or satflag==True, "Pick a flux source: centrals and/or satellites"

        #Flux
        ftot = fcen + fsat
        decay = 1 / ((1+self.zs) * chis**2)

        return np.einsum('ijk,i->ijk', ftot, decay)


    def _get_fcen(self, nu):
        '''Function of M and z, but defined over whole z,M,k grid'''
        return cib.luminosity(self.zs, self.ms, len(self.ks), nu, self.cib_params) / (4.0*np.pi)

        # Lcen = cib.luminosity(self.zs, self.ms, len(self.ks), nu) / (4.0*np.pi)
        
        # #Flux Cut
        # maxflux
        # prefactor = (1+self.zs) / (4*np.pi * self.comoving_radial_distance(zs)**2)
        # flux = Lcen * prefactor[:, np.newaxis, np.newaxis]
        # fmask = np.where(flux <= maxflux, 1, 0)

        # return Lcen * fmask

    def _get_fsat(self, freq, cibinteg='trap', satmf='Tinker'):
        '''Function of M and z, but defined over whole z,M,k grid'''

        def integ(m, M):
            return sdndm(m, M, satmf) * cib.capitalSigma(m, self.cib_params['logM_eff'], self.cib_params['var'])

        #Integrate Subhalo Masses
#         import pdb; pdb.set_trace()
        Nsatm = len(self.ms) + 1
        Mmin = 1e10
        satms = np.geomspace(Mmin, self.ms, num=Nsatm, axis=-1)
        if cibinteg.lower() == 'trap':
            fsat_m = np.trapz(integ(satms, self.ms[...,None]), satms, axis=-1)
        elif cibinteg.lower() == 'simps':
            fsat_m = simps(integ(satms, self.ms[...,None]), satms, axis=-1)
        else: raise NotImplementedError('Invalid subhalo integration method (cibinteg)')
#         pdb.set_trace()
        
        #Get Redshift Dependencies
        a = self.cib_params['alpha']
        b = self.cib_params['beta']
        d = self.cib_params['delta']
        g = self.cib_params['gamma']
        T = self.cib_params['Td_o']
        Lo = self.cib_params['L_o']
        fsat_z = Lo * cib.capitalPhi(self.zs, d) * cib.capitalTheta(freq, 'obs', self.zs, a, b, g, T)

        #Calculate fsat on Entire Grid
        fsat_k = np.ones(len(self.ks))
        fsat_zz, fsat_mm, _ = np.meshgrid(fsat_z, fsat_m, fsat_k, indexing='ij')
        fsat = fsat_zz * fsat_mm

        return fsat / (4.0*np.pi)

    def _get_cib(self, freq, satflag=True, cibinteg='trap', satmf='Tinker'):
        '''Assumes central galaxies are at the center of the halo'''
        uhalo = self.uk_profiles['nfw']
        fcen = self._get_fcen(freq)
        if satflag:
            fsat = self._get_fsat(freq, cibinteg, satmf)
        else:
            fsat = 0.
        return fcen + uhalo*fsat
    
    def _get_cib_square(self, freq, satflag=True, cibinteg='trap', satmf='Tinker'):
        '''Assumes central galaxies are at the center of the halo'''
        if satflag:
            uhalo = self.uk_profiles['nfw']
            fcen1 = self._get_fcen(freq[0])
            fcen2 = self._get_fcen(freq[1])
            fsat1 = self._get_fsat(freq[0], cibinteg, satmf)
            fsat2 = self._get_fsat(freq[1], cibinteg, satmf)

            return (fcen1*fsat2*uhalo) + (fcen2*fsat1*uhalo) + (fsat1*fsat2*uhalo*uhalo)
        else:
            return 0.

    def testingCIB(self):
        def integ(m, M):
            return sdndm(m, M) * cib.capitalSigma(m, self.cib_params['logM_eff'], self.cib_params['var'])
        def quadinteg(Marray):
            Mcen = Marray[0]

            integral, err = quad(integ, self.ms[0], Mcen, args=(Mcen,))
            
            return [integral, err]

        timestring = f''

        start = time.time()
        #Gaussian Quadrature 1
        cenms = self.ms.reshape((len(self.ms), 1))
        fgausstable = np.apply_along_axis(quadinteg, 1, cenms)

        end = time.time()
        timestring += f'Gaussian Quadrature 1 time (s): {end-start} \n'
        
        start = time.time()
        #Gaussian Quadrature 2
        fquad = np.zeros(len(self.ms))
        quaderr = np.zeros(len(self.ms))
        for i, centralM in enumerate(self.ms):
            fquad[i], quaderr[i] = quad(integ, self.ms[0], centralM, args=(centralM,))
  
        end = time.time()
        timestring += f'Gaussian Quadrature 2 time (s): {end-start} \n'

        start = time.time()
        #Trapezoidal
        Nsubm = 500
        satms = np.geomspace(self.ms[0], self.ms, num=Nsubm, axis=-1)
        ftrap = np.trapz(integ(satms, self.ms[...,None]), satms, axis=-1)

        end = time.time()
        timestring += f'Trapezoidal time (s): {end-start} \n'
        
        start = time.time()
        #Simpson
        Nsubm = 500
        satms = np.geomspace(self.ms[0], self.ms, num=Nsubm, axis=-1)
        fsimps = simps(integ(satms, self.ms[...,None]), satms, axis=-1)

        end = time.time()
        timestring += f'Simpson time (s): {end-start} \n'

        #Error
        fgauss = fquad
        gausserr = quaderr
        traperr = np.abs(fgauss - ftrap) + np.abs(gausserr)
        simpserr = np.abs(fgauss - fsimps) + np.abs(gausserr)

        # #Plot
        # fig, ax = plt.subplots(2, 1, sharey=True, figsize=(10,20))
        # #Trap vs Simps
        # ax[0].loglog(self.ms, simpserr, label='Simpson')
        # ax[0].loglog(self.ms, traperr, label='Trapezoidal')
        # #Gauss vs Simps Error
        # ax[1].loglog(self.ms, gausserr, label='Gaussian Quadrature')
        # ax[1].loglog(self.ms, simpserr, label='Simpson')
        
        # #Gravy
        # ax[0].set_ylabel('Errors')
        # ax[1].set_ylabel('Errors')
        # ax[1].set_xlabel(r'Central Masses ($M_\odot$)')
        # ax[0].legend()
        # ax[1].legend()
        
        #Plot Errors
        plt.loglog(self.ms, traperr, label='Trapezoidal')
        plt.loglog(self.ms, simpserr, label='Simpson')
        plt.loglog(self.ms, gausserr, label='Gaussian Quadrature')
        plt.loglog(self.ms, ftrap, '--',label='Integral')
        
        #Gravy
        plt.ylabel('Errors')
        plt.xlabel(r'Central Masses ($M_\odot$)')
        plt.legend()
        plt.savefig('int_errs.pdf', dpi=900, bbox_inches='tight')
        
        print(timestring)

    def _freqtest(self, freq):
        """ Tests formatting of CIB frequencies """
        if len(freq) > 2:
            raise ValueError('Only 1 pair of frequencies to cross at a time')
        elif len(freq) == 1  and  freq.ndim == 1:
            return np.array([freq, freq])
        elif len(freq) == 1  and  freq.ndim == 2:
            return np.array([freq[0], freq[0]])
        elif freq.ndim != 2:
            raise ValueError('Need a 2D array for the frequency')
        else:
            return freq
    
    
    """
    Power Stuff
    """

    def get_power(self,name1,name2=None,nu_obs=None,verbose=False, subhalos=True, cibinteg='trap', satmf='Tinker'):
        '''
        CIB Keyword Arguments:
        nu_obs [2darray] : 1st axis - freq's to be cross correlated. 2nd axis - bandpass
        subhalos [bool]  : flag to add satellite galaxies
        cibinteg [str]   : integration method for subhalo masses for cib; either "trap" or "simps"
        satmf [str]      : subhalo mass function; either 'Tinker' or 'Jiang'
        '''
        if name2 is None: name2 = name1
        
        if name1.lower() == 'cib' or name2.lower() == 'cib':
            nu_obs = self._freqtest(nu_obs)
            return self.get_power_1halo(name1,name2, nu_obs, subhalos, cibinteg, satmf) + self.get_power_2halo(name1,name2,verbose, nu_obs, subhalos, cibinteg, satmf)
        else:
            return self.get_power_1halo(name1,name2) + self.get_power_2halo(name1,name2,verbose)

        
    def get_power_1halo(self,name="nfw",name2=None, nu_obs=None, subhalos=True, cibinteg='trap', satmf='Tinker'):
        '''
        Keyword Arguments:
        nu_obs [2darray] : 1st axis - freq's to be cross correlated. 2nd axis - bandpass
        subhalos [bool]  : flag to add satellite galaxies
        cibinteg [str]   : integration method for subhalo masses for cib; either "trap" or "simps"
        satmf [str]      : subhalo mass function; either 'Tinker' or 'Jiang'
        '''
        name2 = name if name2 is None else name2
        if name.lower() == 'cib' or name2.lower() == 'cib':
            nu_obs = self._freqtest(nu_obs)

        ms = self.ms[...,None]
        mnames = self.uk_profiles.keys()
        hnames = self.hods.keys()
        pnames =self.pk_profiles.keys()
        if (name in hnames) and (name2 in hnames):
            square_term = self._get_hod_square(name)
        elif (name in pnames) and (name2 in pnames):
            square_term = self._get_pressure(name)**2
        elif (name.lower()=='cib') and (name2.lower()=='cib'):
            if subhalos:
                square_term = self._get_cib_square(nu_obs, subhalos, cibinteg, satmf)
            else:
                square_term = 0.
        else:
            square_term=1.
            for nm in [name,name2]:
                if nm in hnames:
                    square_term *= self._get_hod(nm)
                elif nm in mnames:
                    square_term *= self._get_matter(nm)
                elif nm in pnames:
                    square_term *= self._get_pressure(nm)
                elif nm.lower()=='cib':
                    square_term *= self._get_cib(nu_obs[0], subhalos, cibinteg, satmf)
                else: raise ValueError

        integrand = self.nzm[...,None] * square_term
        return np.trapz(integrand,ms,axis=-2)*(1-np.exp(-(self.ks/self.p['kstar_damping'])**2.))

    
    def get_power_2halo(self,name="nfw",name2=None,verbose=False,nu_obs=None, subhalos=True, cibinteg='trap', satmf='Tinker'):
        '''
        Keyword Arguments:
        nu_obs [2darray] : 1st axis - freq's to be cross correlated. 2nd axis - bandpass
        subhalos [bool]  : flag to add satellite galaxies
        cibinteg [str]   : integration method for subhalo masses for cib; either "trap" or "simps"
        satmf [str]      : subhalo mass function; either 'Tinker' or 'Jiang'
        '''
        name2 = name if name2 is None else name2
        if name.lower() == 'cib' or name2.lower() == 'cib':
            nu_obs = self._freqtest(nu_obs)
            
        def _2haloint(iterm):
            integrand = self.nzm[...,None] * iterm * self.bh[...,None]
            integral = np.trapz(integrand,ms,axis=-2)
            return integral

        def _get_term(iname, inu):
            if iname in self.uk_profiles.keys():
                rterm1 = self._get_matter(iname)
                rterm01 = self._get_matter(iname,lowklim=True)
                b = 1
            elif iname in self.pk_profiles.keys():
                rterm1 = self._get_pressure(iname)
                rterm01 = self._get_pressure(iname,lowklim=True)
                print ('Check the consistency relation for tSZ')
                b = rterm01 =0
            elif iname in self.hods.keys():
                rterm1 = self._get_hod(iname)
                rterm01 = self._get_hod(iname,lowklim=True)
                b = self.get_bg(self.hods[iname]['Nc'],self.hods[iname]['Ns'],self.hods[iname]['ngal'])[:,None]
            elif iname.lower()=='cib':
                rterm1 = self._get_cib(nu_obs[inu], subhalos, cibinteg, satmf)
                rterm01 = 0
                b = 0
            else: raise ValueError

            return rterm1,rterm01,b

        ms = self.ms[...,None]
        
        #Integrand Terms
        freqctr = 0
        iterm1, iterm01, b1 = _get_term(name, freqctr)
        if name.lower() == 'cib':
            freqctr += 1
        iterm2, iterm02, b2 = _get_term(name2, freqctr)
        
        #Calculate Integral
        integral = _2haloint(iterm1)
        integral2 = _2haloint(iterm2)

        # consistency relation : Correct for part that's missing from low-mass halos to get P(k->0) = b1*b2*Plinear
        consistency1 = _2haloint(iterm01)
        consistency2 = _2haloint(iterm02)
        if verbose:
            print("Two-halo consistency1: " , consistency1,integral)
            print("Two-halo consistency2: " , consistency2,integral2)
        return self.Pzk * (integral+b1-consistency1)*(integral2+b2-consistency2)

    
    def get_sfrd(self, freq_range):
        kennicutt = 1.7e-10

        sfr = kennicutt * cib.luminosity(self.zs, self.ms, len(self.ks), freq_range, 'rest', **self.cib_params)

        return np.trapz(self.nzm * sfr[:,:,0], self.ms, axis=-1)
        


def sdndm(msat, mcen, funcname='Tinker'):
    '''Satellite halo mass function 
    Tinker: https://iopscience.iop.org/article/10.1088/0004-637X/719/1/88
    Jiang: '''

    if funcname.lower() == 'jiang':
        #Parameters
        gamma_1    = 0.13
        alpha_1    = -0.83
        gamma_2    = 1.33
        alpha_2    = -0.02
        beta_2     = 5.67
        zeta       = 1.19

        #Calculation
        dndm = 1/msat*(((gamma_1 * ((msat/mcen)**alpha_1)) +
            (gamma_2 * ((msat/mcen)**alpha_2))) *
            (np.exp(-(beta_2) * ((msat / mcen)**zeta))))

    elif funcname.lower() == 'tinker':
        # Extra factor of m as we need dndm not dndlnm
        gamma    = 0.3
        alpha    = -0.7
        beta     = -9.9
        zeta     = 2.5

        #Calculation
        dndm = 1/msat * ((gamma * ((msat/mcen)**alpha))*
            (np.exp((beta) * ((msat / mcen)**zeta))))

    else: raise NotImplementedError('Invalid subhalo mass function name')

    return dndm


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
    # FIXME: can the for loop be removed?
    # FIXME: is the zero indexing safe?

    log10mstar = np.linspace(-18,18,4000)[None,:]
    mh = Mhalo_stellar(z,log10mstar)
    mstar = np.zeros((z.shape[0],log10mhalo.shape[-1]))
    for i in range(z.size):
        mstar[i] = np.interp(log10mhalo[0],mh[i],log10mstar[0])
    return mstar

def Mhalo_stellar_core(log10mstellar,a,Mstar00,Mstara,M1,M1a,beta0,beta_a,gamma0,gamma_a,delta0,delta_a):
    log10M1 = M1 + M1a*(a-1)
    log10Mstar0 = Mstar00 + Mstara*(a-1)
    beta = beta0 + beta_a*(a-1)
    gamma = gamma0 + gamma_a*(a-1)
    delta = delta0 + delta_a*(a-1)
    log10mstar = log10mstellar
    log10mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
    return log10mh

def Mhalo_stellar(z,log10mstellar):
    # Function to compute halo mass as a function of the stellar mass. arxiv 1001.0015 Table 2
    # z = list of redshifts
    # log10mhalo = log of the halo mass

    output = np.zeros((z.size,log10mstellar.shape[-1]))

    a = 1./(1+z)
    log10mstellar = log10mstellar + z*0

    Mstar00=10.72
    Mstara=0.55
    M1=12.35
    M1a=0.28
    beta0=0.44
    beta_a=0.18
    gamma0=1.56
    gamma_a=2.51
    delta0=0.57
    delta_a=0.17

    sel1 = np.where(z.reshape(-1)<=0.8)
    output[sel1] = Mhalo_stellar_core(log10mstellar[sel1],a[sel1],Mstar00,Mstara,M1,M1a,beta0,beta_a,gamma0,gamma_a,delta0,delta_a)

    Mstar00=11.09
    Mstara=0.56
    M1=12.27
    M1a=-0.84
    beta0=0.65
    beta_a=0.31
    gamma0=1.12
    gamma_a=-0.53
    delta0=0.56
    delta_a=-0.12

    sel1 = np.where(z.reshape(-1)>0.8)
    output[sel1] = Mhalo_stellar_core(log10mstellar[sel1],a[sel1],Mstar00,Mstara,M1,M1a,beta0,beta_a,gamma0,gamma_a,delta0,delta_a)
    return output


def avg_Nc(log10mhalo,z,log10mstellar_thresh,sig_log_mstellar):
    """<Nc(m)>"""
    log10mstar = Mstellar_halo(z,log10mhalo)
    num = log10mstellar_thresh - log10mstar
    denom = np.sqrt(2.) * sig_log_mstellar
    return 0.5*(1. - erf(num/denom))


def hod_default_mfunc(mthresh,Bamp,Bind): return (10.**(12.))*Bamp*10**((mthresh-12)*Bind)

def avg_Ns(log10mhalo,z,log10mstellar_thresh,Nc=None,sig_log_mstellar=None,
           alphasat=None,Bsat=None,betasat=None,Bcut=None,betacut=None,
           Msat_override=None,Mcut_override=None):
    mthresh = Mhalo_stellar(z,log10mstellar_thresh)
    Msat = Msat_override if Msat_override is not None else hod_default_mfunc(mthresh,Bsat,betasat)
    Mcut = Mcut_override if Mcut_override is not None else hod_default_mfunc(mthresh,Bcut,betacut)
    Nc = avg_Nc(log10mhalo,z,log10mstellar_thresh,sig_log_mstellar=sig_log_mstellar) if Nc is None else Nc
    masses = 10**log10mhalo
    return Nc*((masses/Msat)**alphasat)*np.exp(-Mcut/(masses))


def avg_NsNsm1(Nc,Ns,corr="max"):
    if corr=='max':
        ret = Ns**2./Nc
        ret[np.isclose(Nc,0.)] = 0 #FIXME: is this kosher?
        return ret
    elif corr=='min':
        return Ns**2.

def avg_NcNs(Nc,Ns,corr="max"):
    if corr=='max':
        return Ns
    elif corr=='min':
        return Ns*Nc

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
    # Note the sign difference in the second gamma. Battaglia 2016 had a typo here.
    return (omb/omm) * rhocritz * rho0 * (x**gamma) * (1.+x**alpha)**(-(beta+gamma)/alpha)



def P_e(r,m200critz,z,omb,omm,rhocritz,
            alpha=default_params['battaglia_pres_alpha'],
            gamma=default_params['battaglia_pres_gamma'],
            profile="pres"):
    return P_e_generic(r,m200critz,z,omb,omm,rhocritz,
                           alpha=alpha,
                           gamma=gamma,
                           P0_A0=battaglia_defaults[profile]['P0_A0'],
                           P0_alpham=battaglia_defaults[profile]['P0_alpham'],
                           P0_alphaz=battaglia_defaults[profile]['P0_alphaz'],
                           xc_A0=battaglia_defaults[profile]['xc_A0'],
                           xc_alpham=battaglia_defaults[profile]['xc_alpham'],
                           xc_alphaz=battaglia_defaults[profile]['xc_alphaz'],
                           beta_A0=battaglia_defaults[profile]['beta_A0'],
                           beta_alpham=battaglia_defaults[profile]['beta_alpham'],
                           beta_alphaz=battaglia_defaults[profile]['beta_alphaz'])

def P_e_generic(r,m200critz,z,omb,omm,rhocritz,
                        alpha=default_params['battaglia_pres_alpha'],
                        gamma=default_params['battaglia_pres_gamma'],
                           P0_A0=battaglia_defaults[default_params['battaglia_pres_family']]['P0_A0'],
                           P0_alpham=battaglia_defaults[default_params['battaglia_pres_family']]['P0_alpham'],
                           P0_alphaz=battaglia_defaults[default_params['battaglia_pres_family']]['P0_alphaz'],
                           xc_A0=battaglia_defaults[default_params['battaglia_pres_family']]['xc_A0'],
                           xc_alpham=battaglia_defaults[default_params['battaglia_pres_family']]['xc_alpham'],
                           xc_alphaz=battaglia_defaults[default_params['battaglia_pres_family']]['xc_alphaz'],
                           beta_A0=battaglia_defaults[default_params['battaglia_pres_family']]['beta_A0'],
                           beta_alpham=battaglia_defaults[default_params['battaglia_pres_family']]['beta_alpham'],
                           beta_alphaz=battaglia_defaults[default_params['battaglia_pres_family']]['beta_alphaz']):
    """
    AGN and SH Battaglia 2016 profiles
    r: physical distance
    m200critz: M200_critical_z

    """
    R200 = R_from_M(m200critz,rhocritz,delta=200)
    x = r/R200
    return P_e_generic_x(x,m200critz,R200,z,omb,omm,rhocritz,alpha,gamma,
                             P0_A0,P0_alpham,P0_alphaz,
                             xc_A0,xc_alpham,xc_alphaz,
                             beta_A0,beta_alpham,beta_alphaz)

def P_e_generic_x(x,m200critz,R200critz,z,omb,omm,rhocritz,
                        alpha=default_params['battaglia_pres_alpha'],
                        gamma=default_params['battaglia_pres_gamma'],
                           P0_A0=battaglia_defaults['pres']['P0_A0'],
                           P0_alpham=battaglia_defaults['pres']['P0_alpham'],
                           P0_alphaz=battaglia_defaults['pres']['P0_alphaz'],
                           xc_A0=battaglia_defaults['pres']['xc_A0'],
                           xc_alpham=battaglia_defaults['pres']['xc_alpham'],
                           xc_alphaz=battaglia_defaults['pres']['xc_alphaz'],
                           beta_A0=battaglia_defaults['pres']['beta_A0'],
                           beta_alpham=battaglia_defaults['pres']['beta_alpham'],
                           beta_alphaz=battaglia_defaults['pres']['beta_alphaz']):
    P0 = battaglia_gas_fit(m200critz,z,P0_A0,P0_alpham,P0_alphaz)
    xc = battaglia_gas_fit(m200critz,z,xc_A0,xc_alpham,xc_alphaz)
    beta = battaglia_gas_fit(m200critz,z,beta_A0,beta_alpham,beta_alphaz)
    # to convert to p_e
    XH=.76
    eFrac=2.0*(XH+1.0)/(5.0*XH+3.0)
    # print (gamma,alpha,beta[0,0],xc[0,0],P0[0,0])
    # print (gamma,alpha,beta[0,50],xc[0,50],P0[0,50],(P0 *(x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta))[0,50,-350])
    G_newt = constants.G/(default_params['parsec']*1e6)**3*default_params['mSun']
    return eFrac*(omb/omm)*200*m200critz*G_newt* rhocritz/(2*R200critz) * P0 * (x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta)





def a2z(a): return (1.0/a)-1.0


def ngal_from_mthresh(log10mthresh=None,zs=None,nzm=None,ms=None,
                      sig_log_mstellar=None,Ncs=None,Nss=None,
                      alphasat=None,Bsat=None,betasat=None,
                      Bcut=None,betacut=None):
    if (Ncs is None) and (Nss is None):
        log10mstellar_thresh = log10mthresh[:,None]
        log10mhalo = np.log10(ms[None,:])
        Ncs = avg_Nc(log10mhalo,zs[:,None],log10mstellar_thresh,sig_log_mstellar)
        Nss = avg_Ns(log10mhalo,zs[:,None],log10mstellar_thresh,Ncs,sig_log_mstellar,alphasat,Bsat,betasat,Bcut,betacut)
    else:
        assert log10mthresh is None
        assert zs is None
        assert sig_log_mstellar is None
    integrand = nzm * (Ncs+Nss)
    return np.trapz(integrand,ms,axis=-1)
