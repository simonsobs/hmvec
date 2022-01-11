import sys,os
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import camb
from camb import model
import numpy as np
from . import tinker,utils,hod
from .cosmology import Cosmology

import scipy.constants as constants
from .params import default_params, battaglia_defaults
from .fft import generic_profile_fft
from .mstar_mhalo import Mstellar_halo, Mhalo_stellar
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
    def __init__(self,zs,ks,ms=None,params=None,mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False,accurate_sigma2=False):
        self.zs = np.asarray(zs)
        self.ks = ks
        self.accurate_sigma2 = accurate_sigma2
        Cosmology.__init__(self,params,halofit)

        self.mdef = mdef
        self.mode = mass_function
        self.hods = {}

        # Mass function
        if ms is not None:
            self.ms = np.asarray(ms)
            self.init_mass_function(self.ms)

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

    def get_sigma2(self, vectorize_z=True):
        _MAX_INTEGRAND_SIZE = 1e8

        ms = self.ms
        kmin = self.p['sigma2_kmin']
        kmax = self.p['sigma2_kmax']
        numks = self.p['sigma2_numks']
        self.ks_sigma2 = np.geomspace(kmin,kmax,numks) # ks for sigma2 integral
        if self.accurate_sigma2:
            self.sPzk = self.P_lin_slow(self.ks_sigma2,self.zs,kmax=kmax)
        else:
            self.sPzk = self.P_lin(self.ks_sigma2,self.zs)
        ks = self.ks_sigma2[None,None,:]
        R = self.R_of_m(ms)[None,:,None]
        W2 = Wkr(ks,R,self.p['Wkr_taylor_switch'])**2.
        Ps = self.sPzk[:,None,:]

        # If N_z * N_m * N_k is large, compute sigma^2 separately for each z
        # to avoid storing the integrands for each z in memory simultaneously.
        # Otherwise, we perform the integrals on a single 3d integrand array.
        if (Ps.shape[0] * np.prod(W2.shape[1:]) > _MAX_INTEGRAND_SIZE) or not vectorize_z:
            sigma2 = np.zeros((Ps.shape[0], W2.shape[1]), dtype=Ps.dtype)
            for zi in range(Ps.shape[0]):
                integrand = Ps[zi] * W2[0] * ks[0]**2. / 2. / np.pi**2.
                sigma2[zi] = simps(integrand, ks[0], axis=-1)
        else:
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


    def add_battaglia_profile(self,name,family=None,param_override=None,
                              nxs=None,
                              xmax=None,ignore_existing=False,vectorize_z=True,
                              verbose=False):
        if not(ignore_existing): assert name not in self.uk_profiles.keys(), "Profile name already exists."
        assert name!='nfw', "Name nfw is reserved."
        if nxs is None: nxs = self.p['electron_density_profile_integral_numxs']
        if xmax is None: xmax = self.p['electron_density_profile_integral_xmax']

        # Set default parameters
        if family is None: family = self.p['battaglia_gas_family'] # AGN or SH?
        pparams = {}
        pparams['battaglia_gas_gamma'] = self.p['battaglia_gas_gamma']
        pparams.update(battaglia_defaults[family])

        # Update with overrides
        if param_override is not None:
            if verbose: print(param_override)
            for key in param_override.keys():
                if key=='battaglia_gas_gamma':
                    pparams[key] = param_override[key]
                elif key in battaglia_defaults[family]:
                    pparams[key] = param_override[key]
                else:
                    #raise ValueError # param in param_override doesn't seem to be a Battaglia parameter
                    pass

        # Convert masses to m200critz
        rhocritz = self.rho_critical_z(self.zs)
        if self.mdef=='vir':
            delta_rhos1 = rhocritz*self.deltav(self.zs)
        elif self.mdef=='mean':
            delta_rhos1 = self.rho_matter_z(self.zs)*200.
        rvirs = self.rvir(self.ms[None,:],self.zs[:,None]) # Packed as [z,m]
        cs = self.concentration()
        delta_rhos2 = 200.*self.rho_critical_z(self.zs)
        m200critz = mdelta_from_mdelta(self.ms,cs,delta_rhos1,delta_rhos2) # Packed as [z,m]
        r200critz = R_from_M(
            m200critz,self.rho_critical_z(self.zs)[:,None],delta=200.
        ) # Packed as [z,m]

        # Generate profiles
        """
        The physical profile is rho(r) = f(2r/R200)
        We rescale this to f(x), so x = r/(R200/2) = r/rgs
        So rgs = R200/2 is the equivalent of rss in the NFW profile
        """
        omb = self.p['ombh2'] / self.h**2.
        omm = self.om0

        if vectorize_z:
            rhofunc = lambda x: rho_gas_generic_x(
                x,
                m200critz[..., None],
                self.zs[:, None, None],
                omb,
                omm,
                rhocritz[..., None, None],
                gamma=pparams['battaglia_gas_gamma'],
                rho0_A0=pparams['rho0_A0'],
                rho0_alpham=pparams['rho0_alpham'],
                rho0_alphaz=pparams['rho0_alphaz'],
                alpha_A0=pparams['alpha_A0'],
                alpha_alpham=pparams['alpha_alpham'],
                alpha_alphaz=pparams['alpha_alphaz'],
                beta_A0=pparams['beta_A0'],
                beta_alpham=pparams['beta_alpham'],
                beta_alphaz=pparams['beta_alphaz']
            )

            rgs = r200critz/2.
            cgs = rvirs/rgs
            ks,ukouts = generic_profile_fft(rhofunc,cgs,rgs[...,None],self.zs,self.ks,xmax,nxs)

            self.uk_profiles[name] = ukouts.copy()

        else:
            # If we are computing for many redshifts, the memory cost of vectorizing over
            # z may be substantial, so we also provide the option (via vectorize_z)
            # to serialize over z
            for zi, z in enumerate(self.zs):
                rhofunc = lambda x: rho_gas_generic_x(
                    x,
                    m200critz[zi : zi+1, :, None],
                    self.zs[zi : zi+1, None, None],
                    omb,
                    omm,
                    rhocritz[zi : zi+1, None, None],
                    gamma=pparams['battaglia_gas_gamma'],
                    rho0_A0=pparams['rho0_A0'],
                    rho0_alpham=pparams['rho0_alpham'],
                    rho0_alphaz=pparams['rho0_alphaz'],
                    alpha_A0=pparams['alpha_A0'],
                    alpha_alpham=pparams['alpha_alpham'],
                    alpha_alphaz=pparams['alpha_alphaz'],
                    beta_A0=pparams['beta_A0'],
                    beta_alpham=pparams['beta_alpham'],
                    beta_alphaz=pparams['beta_alphaz']
                )
                rgs = r200critz[zi : zi + 1, :] / 2. # Packed as [z,m]
                cgs = rvirs[zi : zi + 1, :] / rgs # Packed as [z,m]
                ks, ukouts = generic_profile_fft(
                    rhofunc, cgs, rgs[..., None], self.zs[zi : zi+1], self.ks, xmax, nxs
                )

                if zi == 0:
                    self.uk_profiles[name] = np.zeros(
                        (len(self.zs), len(self.ms), ukouts.shape[-1]), dtype=np.float64
                    )

                self.uk_profiles[name][zi, :] = ukouts[:]



    def add_battaglia_pres_profile(self,name,family=None,param_override=None,
                              nxs=None,
                              xmax=None,ignore_existing=False):
        if not(ignore_existing): assert name not in self.pk_profiles.keys(), "Profile name already exists."
        assert name!='nfw', "Name nfw is reserved."
        if nxs is None: nxs = self.p['electron_pressure_profile_integral_numxs']
        if xmax is None: xmax = self.p['electron_pressure_profile_integral_xmax']

        # Set default parameters
        if family is None: family = self.p['battaglia_pres_family'] # AGN or SH?
        pparams = {}
        pparams['battaglia_pres_gamma'] = self.p['battaglia_pres_gamma']
        pparams['battaglia_pres_alpha'] = self.p['battaglia_pres_alpha']
        pparams.update(battaglia_defaults[family])

        # Update with overrides
        if param_override is not None:
            for key in param_override.keys():
                if key in ['battaglia_pres_gamma','battaglia_pres_alpha']:
                    pparams[key] = param_override[key]
                elif key in battaglia_defaults[family]:
                    pparams[key] = param_override[key]
                else:
                    #raise ValueError # param in param_override doesn't seem to be a Battaglia parameter
                    pass

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

    def add_hod(
        self,
        name,
        family=hod.Leauthaud12HOD,
        corr="max",
        satellite_profile_name='nfw',
        central_profile_name=None,
        ignore_existing=False,
        param_override=None,
        **kwargs
    ):
        """Precompute and store quantities related to a given HOD.

        Possible HODs are defined as subclasses of HODBase in hod.py.

        Parameters
        ----------
        name : string
            Name for HOD. Quantities are fetched from name item
            in hods dict.
        family : hod.HODBase
            Name of HOD class (e.g. hod.Leauthaud12HOD).
        corr : string, optional
            Either "min" or "max", describing correlations in central-satellite model.
            Default: "max"
        satellite_profile_name : string, optional
            Density profile for satellites. Default: "nfw"
        central_profile_name : string, optional
            Density profile for centrals, used to specify miscentering.
            Default: None (correponds to uk=1)
        ignore_existing : bool, optional
            Whether to overwrite existing HOD with given name. Default: False
        **param_override : dict, optional
            Dict of parameter values to override defaults with. Default: None
        """

        # Check for existing profiles or HODs with same name
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

        # Make dict entry to store info for new HOD, and store HOD object.
        self.hods[name] = {}
        self.hods[name]["hod"] = family(
            self.zs, self.ms, params=self.p, param_override=param_override, **kwargs
        )

        # Store precomputed HOD quantities.
        # TODO: This is a bit redundant, since these quantities are also stored
        # in self.hods[name]['hod'], but it's left here for consistency with previous
        # routines
        self.hods[name]['Nc'] = self.hods[name]['hod'].Nc
        self.hods[name]['Ns'] = self.hods[name]['hod'].Ns
        self.hods[name]['NsNsm1'] = self.hods[name]['hod'].NsNsm1
        self.hods[name]['NcNs'] = self.hods[name]['hod'].NcNs
        self.hods[name]['ngal'] = self.get_ngal(
            self.hods[name]['Nc'], self.hods[name]['Ns']
        )
        self.hods[name]['bg'] = self.get_bg(
            self.hods[name]['Nc'],
            self.hods[name]['Ns'],
            self.hods[name]['ngal']
        )
        self.hods[name]['satellite_profile'] = satellite_profile_name
        self.hods[name]['central_profile'] = central_profile_name

    def get_ngal(self, Nc, Ns):
        integrand = self.nzm * (Nc + Ns)
        return np.trapz(integrand, self.ms, axis=-1)

    def get_bg(self, Nc, Ns, ngal):
        integrand = self.nzm * (Nc + Ns) * self.bh
        return np.trapz(integrand, self.ms, axis=-1) / ngal

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


    def get_power(
        self, name, name2=None, verbose=False, b1=None, b2=None, m_integrand=False
    ):
        if name2 is None: name2 = name
        return (
            self.get_power_1halo(name,name2,m_integrand=m_integrand)
            + self.get_power_2halo(name,name2,verbose,b1,b2,m_integrand=m_integrand)
        )

    def get_power_1halo(self, name="nfw", name2=None, m_integrand=False):
        name2 = name if name2 is None else name2
        ms = self.ms[...,None]
        mnames = self.uk_profiles.keys()
        hnames = self.hods.keys()
        pnames =self.pk_profiles.keys()
        if (name in hnames) and (name2 in hnames):
            square_term = self._get_hod_square(name)
        elif (name in pnames) and (name2 in pnames):
            square_term = self._get_pressure(name)**2
        else:
            square_term=1.
            for nm in [name,name2]:
                if nm in hnames:
                    square_term *= self._get_hod(nm)
                elif nm in mnames:
                    square_term *= self._get_matter(nm)
                elif nm in pnames:
                    square_term *= self._get_pressure(nm)
                else: raise ValueError(
                    f"Profile {nm} not computed! Available profiles are {hnames}; {mnames}; {pnames}"
                )

        integrand = self.nzm[...,None] * square_term

        if not m_integrand:
            # Integrate in m, and return result packed as [z,k]
            return (
                np.trapz(integrand,ms,axis=-2)
                * (1-np.exp(-(self.ks/self.p['kstar_damping'])**2.))
            )
        else:
            # Return full integrand, packed as [z,m,k]
            return (
                integrand * (
                    1-np.exp(-(self.ks/self.p['kstar_damping'])**2.)
                )[np.newaxis, np.newaxis, :]
            )

    def get_power_2halo(
        self,name="nfw",name2=None,verbose=False,b1_in=None,b2_in=None,m_integrand=False
    ):
        name2 = name if name2 is None else name2

        def _2halointegrand(iterm):
            return self.nzm[...,None] * iterm * self.bh[...,None]

        def _2haloint(iterm):
            integrand = _2halointegrand(iterm)
            integral = np.trapz(integrand,self.ms[..., None],axis=-2)
            return integral

        def _get_term(iname):
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
            else: raise ValueError
            return rterm1,rterm01,b


        iterm1,iterm01,b1 = _get_term(name)
        iterm2,iterm02,b2 = _get_term(name2)
        if b1_in is not None:
            b1 = b1_in.reshape((b1_in.shape[0],1))
        if b2_in is not None:
            b2 = b2_in.reshape((b1_in.shape[0],1))

        integral = _2haloint(iterm1)
        integral2 = _2haloint(iterm2)

        # consistency relation : Correct for part that's missing from low-mass
        # halos to get P(k->0) = b1*b2*Plinear
        consistency1 = _2haloint(iterm01)
        consistency2 = _2haloint(iterm02)

        if m_integrand:
            # Return dP_2h / dM, packed as [z,m,k].
            # There is a question of how to incorporate the
            # normalization from the consistency relation into this. The
            # prescription below, which takes d(consistency)/dM=0, seems to work,
            # in that integrating it in M gives a result that's pretty close
            # to the full result.
            prefactor = (
                _2halointegrand(iterm1) * (integral2+b2-consistency2)[..., np.newaxis, :]
                + (integral+b1-consistency1)[..., np.newaxis, :] * _2halointegrand(iterm2)
            )
            return prefactor * self.Pzk[..., np.newaxis, :]
        else:
            # Return P_2h packed as [z,k]
            if verbose:
                print("Two-halo consistency1: " , consistency1,integral)
                print("Two-halo consistency2: " , consistency2,integral2)
            return self.Pzk * (integral+b1-consistency1)*(integral2+b2-consistency2)

    def sigma_1h_profiles(self,thetas,Ms,concs,sig_theta=None,delta=200,rho='mean',rho_at_z=True):
        import clusterlensing as cl
        zs = self.zs
        Ms = np.asarray(Ms)
        concs = np.asarray(concs)
        chis = self.angular_diameter_distance(zs)
        rbins = chis * thetas
        offsets = chis * sig_theta if sig_theta is not None else None
        if rho=='critical': rhofunc = self.rho_critical_z
        elif rho=='mean': rhofunc = self.rho_matter_z
        rhoz = zs if rho_at_z else zs * 0
        Rdeltas = R_from_M(Ms,rhofunc(rhoz),delta=delta)
        rs = Rdeltas / concs
        rhocrits = self.rho_critical_z(zs)
        delta_c =  Ms / 4 / np.pi / rs**3 / rhocrits / Fcon(concs)
        smd = cl.nfw.SurfaceMassDensity(rs, delta_c, rhocrits,rbins=rbins,offsets=offsets)
        sigma = smd.sigma_nfw()
        return sigma

    def kappa_1h_profiles(self,thetas,Ms,concs,zsource,sig_theta=None,delta=200,rho='mean',rho_at_z=True):
        sigma = self.sigma_1h_profiles(thetas,Ms,concs,sig_theta=sig_theta,delta=delta,rho=rho,rho_at_z=rho_at_z)
        sigmac = self.sigma_crit(self.zs,zsource)
        return sigma / sigmac

    def kappa_2h_profiles(self,thetas,Ms,zsource,delta=200,rho='mean',rho_at_z=True,lmin=100,lmax=10000,verbose=True):
        from scipy.special import j0
        zlens = self.zs
        sigmac = self.sigma_crit(zlens,zsource)
        rhomz = self.rho_matter_z(zlens)
        chis = self.comoving_radial_distance(zlens)
        DAz = self.results.angular_diameter_distance(zlens)
        ells = self.ks*chis
        sel = np.logical_and(ells>lmin,ells<lmax)
        ells = ells[sel]
        #Array indexing is as follows:
        #[z,M,k/r]
        Ps = self.Pzk[:,sel]
        bhs = []
        for i in range(zlens.shape[0]): # vectorize this
            bhs.append( interp1d(self.ms,self.bh[i])(Ms))
        bhs = np.asarray(bhs)
        if verbose:
            print("bias ",bhs)
            print("sigmacr ", sigmac)
        ints = []
        for theta in thetas: # vectorize
            integrand = rhomz * bhs * Ps / (1+zlens)**3. / sigmac / DAz**2 * j0(ells*theta) * ells / 2./ np.pi
            ints.append( np.trapz(integrand,ells) )
        return np.asarray(ints)

"""
Mass function
"""
def R_from_M(M,rho,delta): return (3.*M/4./np.pi/delta/rho)**(1./3.)


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
