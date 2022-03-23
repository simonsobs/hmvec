import sys,os
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d, RectBivariateSpline
import camb
from camb import model
import numpy as np
from . import tinker,utils,hod
from .cosmology import Cosmology, R_from_M

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

def maccio_HI_concentration(m, z, c_HI0=None, gamma=None):
    """Maccio et al. 2007 concentration-mass relation, adapted to HI.

    We use the form fitted to HI profiles in Padmanabhan et al. 2017 (1611.06235),
    Eq. 3.

    Parameters
    ----------
    m : array_like
        Halo masses to evaluate at. Must be broadcastable against z.
    z : array_like
        Redshifts to evaluate at. Must be broadcastable against m.
    c_HI0, gamma: float, optional
        Fit parameters in c-m relation. If not specified, use fitted values from paper.
        Default: None.

    Returns
    -------
    c_HI : array_like
        HI concentrations, with shape determined by broadcasting m and z against each
        other.
    """
    # Use values from Table 3 of 1611.06235
    c_HI0 = default_params["maccio_HI_cHI0"] if c_HI0 is None else c_HI0
    gamma = default_params["maccio_HI_gamma"] if gamma is None else gamma

    return c_HI0 * (m / 1e11) ** -0.109 * 4 / (1 + z) ** gamma


class HaloModel(Cosmology):

    def __init__(self,zs,ks,ms=None,mus=None,params=None,mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False,accurate_sigma2=False):
        self.zs = np.asarray(zs)
        self.ks = ks
        self.mus = mus
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

    def rvir(self,m,z,mdef=None):

        if mdef is None:
            mdef = self.mdef

        if mdef == 'vir':
            return R_from_M(m,self.rho_critical_z(z),delta=self.deltav(z))
        elif mdef == 'mean':
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

    def concentration(self, mode='duffy'):
        """Compute concentration-mass relation.

        Parameters
        ----------
        mode : one of {'duffy', 'maccio_HI'}, optional
            Which c-m relation to use. Default: 'duffy'.

        Returns
        -------
        Concentration values, packed as [z,m].
        """
        ms = self.ms
        if mode == 'duffy':
            if self.mdef == 'mean':
                A = self.p['duffy_A_mean']
                alpha = self.p['duffy_alpha_mean']
                beta = self.p['duffy_beta_mean']
            elif self.mdef == 'vir':
                A = self.p['duffy_A_vir']
                alpha = self.p['duffy_alpha_vir']
                beta = self.p['duffy_beta_vir']
            return duffy_concentration(
                ms[None, :], self.zs[:, None], A, alpha, beta, self.h
            )
        elif mode == 'maccio_HI':
            return maccio_HI_concentration(ms[None, :], self.zs[:, None])
        else:
            raise NotImplementedError(
                "Concentration-mass relation %s not implemented!" % mode
            )

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

    def add_HI_profile(
        self, name, numeric=False, nxs=None, xmax=None, ignore_existing=False
    ):
        """Precompute and store Fourier transform of HI density profile.

        Parameters
        ----------
        name : one of {"padmanabhan17"}, optional
            Name for HI profile. Default: "padmanabhan17".
        numeric : bool, optional
            Whether to use analytical form of Fourier-space profile (False), or evaluate
            FFT of position-space profile (True). Default: False.
        nxs : integer, optional
            Number of radial samples to use for FFT, linearly spaced from 0 to xmax.
            If not specified, value taken from "HI_integral_numxs" entry of params dict.
        xmax : float, optional
            Maximum dimensionless radius to use in FFT, scaled accordingly for a given
            profiles (e.g. for padmanabhan17, x = r / r_s). If not specified, value
            taken from "HI_integral_xmax" entry of params dict.
        ignore_existing : bool, optional
            Whether to overwrite existing profile with given name. Default: False

        Returns
        -------
        ks : array_like
            Array of k values (in Mpc^-1) that u(k) is evaluated at.
        ukouts : array_like
            Output Fourier-space profiles, packed as [z,m,k].
        """
        if name not in ["padmanabhan17"]:
            raise NotImplementedError("HI profile %s not implemented!" % name)

        if not ignore_existing and name in self.uk_profiles.keys():
            raise ValueError("Profile %s already exists!" % name)

        if nxs is None: nxs = self.p["HI_integral_numxs"]
        if xmax is None: xmax = self.p["HI_integral_xmax"]

        if name == "padmanabhan17":

            if not numeric:
                # Halo concentrations, packed as [z,m]
                con = self.concentration(mode="maccio_HI")
                # Halo masses, packed as [m]
                ms = self.ms
                # Virial radii, packed as [z,m]
                rvir = self.rvir(ms[None, :], self.zs[:, None])
                # Scale radii, packed as [z,m]
                rs = rvir / con

                # Eq. 7 of 1611.06235, normalized to unity at k=0
                x = self.ks[None, None, :] * rs[:, :, None] * (1 + self.zs[:,None,None])
                self.uk_profiles[name] = (1 + x ** 2) ** -2

            else:
                raise NotImplementedError(
                    "Numeric transform of Padmanabhan17 HI profile not implemented!"
                )

        return self.ks, self.uk_profiles[name]

    def add_hod(
        self,
        name,
        family=hod.Leauthaud12_HOD,
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
        family : hod.HODBase, optional
            Name of HOD class. Default: hod.Leauthaud12_HOD.
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
            self.zs,
            self.ms,
            params=self.p,
            nzm=self.nzm,
            param_override=param_override,
            corr=corr,
            **kwargs
        )

        # Store precomputed HOD quantities.
        # TODO: This is a bit redundant, since these quantities are also stored
        # in self.hods[name]['hod'], but it's left here for consistency with previous
        # routines
        self.hods[name]['Nc'] = self.hods[name]['hod'].Nc
        self.hods[name]['Ns'] = self.hods[name]['hod'].Ns
        self.hods[name]['NsNsm1'] = self.hods[name]['hod'].NsNsm1
        self.hods[name]['NsNsm1Nsm2'] = self.hods[name]['hod'].NsNsm1Nsm2
        self.hods[name]['NcNs'] = self.hods[name]['hod'].NcNs
        self.hods[name]['NcNsNsm1'] = self.hods[name]['hod'].NcNsNsm1
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

    def sigma2_FoG(self, m, z):
        """Compute squared mass-dependent Finger of God damping scale.

        We follow Schaan & White 2021 (2103.01964, discussion between Eqs. 2.7 and 2.8)
        in taking the damping scale to be sigma_d = sigma_{v,1d} / (aH), where the
        line-of-sight velocity dispersion is that of a singular isothermal sphere with
        radius r_vir and mass m.

        Parameters
        ----------
        m : array_like, 1d
            Halo masses to compute for, in Msun.
        z : array_like, 1d
            Redshifts to compute for.

        Returns
        -------
        sigma_d2 : array_like
            Sigma_d^2, packed as [z,m].
        """

        _GNEWTON = 4.30091e-9 # Mpc Msun^-1 km^2 s^-1; value from Wikipedia

        # Compute sigma_v^2, packed as [z,m]. We need this r_vir to be physical,
        # rather than comoving, but this should be what self.rvir() computes.
        # Units of sigma_v^2 are km^2 s^-2
        sigma_v2 = _GNEWTON * m[np.newaxis, :] / (
            2 * self.rvir(m[np.newaxis, :], z[:, np.newaxis], mdef="vir")
        )
        # Compute sigma_d^2 = sigma_v^2 / (aH)^2, packed as [z,m].
        # hubble_parameter(z) has units of km s^-1 Mpc^-1, so sigma_d2 has units of
        # Mpc^2
        sigma_d2 = sigma_v2 * (((1 + z) / self.hubble_parameter(z))**2)[:, np.newaxis]

        return sigma_d2

    def DFoG(self, kmu, func="Schaan"):
        """Mass-dependent Finger of God damping function.

        We follow Schaan & White 2021 (2103.01964, Eq. 2.7) in taking the damping
        function to be an exponential with damping scale sigma_d, computed in the
        sigma2_FoG routine. Lorentzian damping is also implemented

        Parameters
        ----------
        kmu : array_like
            k*mu values to compute for, packed as [k,mu].
        func : {"Schaan", "Lorentzian"}, optional
            Form of damping function. Default: "Schaan".

        Returns
        -------
        factor : array_like
            D_FoG(k, m, z, mu), packed as [z,m,k,mu].
        """

        if func == "Schaan":
            factor = np.exp(
                -0.5
                * kmu[np.newaxis, np.newaxis, :, :] ** 2
                * self.sigma2_FoG(self.ms, self.zs)[:, :, np.newaxis, np.newaxis]
            )
        elif func == "Lorentzian":
            factor = 1 / (
                1 + 0.5 * kmu[np.newaxis, np.newaxis, :, :] ** 2
                * self.sigma2_FoG(self.ms, self.zs)[:, :, np.newaxis, np.newaxis]
            )
        else:
            raise NotImplementedError(
                "Specified DFoG function (%s) not implemented" % func
            )
        return factor

    def get_bg(self, Nc, Ns, ngal, rsd=False, u_name="nfw", vectorize_mu=True):
        """Galaxy bias, optionally including effects of Fingers of God.

        We follow Schaan & White 2021 (2103.01964, Eq. A.27) in incorporating a
        mass-dependent Finger of God damping inside the integral over halo mass.
        Alternatively, this can be ignored, such that b_g is only a function of z
        instead of z, k, and mu.

        Parameters
        ----------
        Nc, Ns : array_like
            N_c and N_s from a given HOD, packed as [z,m].
        ngal : array_like
            Mean galaxy number density, packed as [z].
        rsd : bool, optional
            Whether to include Finger of God effects. Affects dimensionality of output.
            Default: False.
        u_name : string, optional
            Internal name for satellite galaxy profile. Default: "nfw".
        vectorize_mu : bool, optional
            Whether to vectorize in mu, which is faster than looping over mu but takes
            more memory. Regardless of this option, we loop over mu if vectorization
            would take an excessive amount of memory. Default: True.

        Returns
        -------
        bg : array_like
            Galaxy bias. If ignoring RSD, packed as [z]; if including RSD, packed as
            [z,k,mu].
        """

        _MAX_INTEGRAND_SIZE = 1e8

        if not rsd:
            # Get integrand, packed as [z, m]
            integrand = self.nzm * (Nc + Ns) * self.bh
            # Integrate in m to get b_g(z)
            bg = np.trapz(integrand, self.ms, axis=-1) / ngal

        else:

            if self.mus is None:
                raise RuntimeError("mu array not defined!")

            # Get u(k,m,z) corresponding to satellite profile, packed as [z,m,k]
            uk = self.uk_profiles[u_name]
            # If full integrand, packed as [z,m,k,mu] would be too large
            # to hold in memory at once, loop over mu:
            if (
                (np.prod(uk.shape) * self.mu.shape[0] > _MAX_INTEGRAND_SIZE)
                or not vectorize_mu
            ):
                bg = np.zeros((uk.shape[0], uk.shape[2],) + self.mus.shape)
                for mui, mu in enumerate(self.mu):
                    # Form n(m) * b(m), packed as [z, m]
                    integrand = self.nzm * self.bh
                    # Multiply by (N_c(m) + N_s(m) D_FoG(k*mu)).
                    # Result is packed as [z,m,k,mu], with a length-1 mu axis.
                    integrand = (
                        integrand[:, :, None, None]
                        * (
                            Nc[:, :, None, None]
                            + Ns[:, :,  None, None]
                            * uk[:, :, :, None]
                            * self.DFoG(
                                self.ks[:, None]
                                * self.mu[mui : mui + 1][None, :]
                            )
                        )
                    )
                    # Integrate in m, to get b_g(z, k, mu)
                    bg[:, :, mui : mui + 1] = np.trapz(
                        integrand, self.ms, axis=1
                    ) / ngal[:, None,  None]

            # Otherwise, form whole integrand and integrate all at once
            else:
                # Form n(m) * b(m), packed as [z, m]
                integrand = self.nzm * self.bh
                # Multiply by (N_c(m) + N_s(m) D_FoG(k*mu)).
                # Result is packed as [z,m,k,mu].
                integrand = integrand[:, :, None, None] * (
                    Nc[:, :, None, None]
                    + Ns[:, :, None, None]
                    * uk[:, :, :, None]
                    * self.DFoG(self.ks[:, None] * self.mus[None, :])
                )
                # Integrate in m to get b_g(z,k,mu)
                bg = np.trapz(integrand, self.ms, axis=1) / ngal[:, None,  None]

        return bg

    def get_fgrowth(self, rsd=False, vectorize_mu=True):
        """Logarithmic growth rate f(z), optionally Finger-of-God effects.

        We follow Schaan & White 2021 (2103.01964, Eq. A.28) in incorporating a
        mass-dependent Finger of God damping inside an integral over halo mass.
        Alternatively, this can be ignored, such that f is only a function of z
        instead of z, k, and mu.

        Parameters
        ----------
        rsd : bool, optional
            Whether to include Finger of God effects. Affects dimensionality of output.
            Default: False.
        vectorize_mu : bool, optional
            Whether to vectorize in mu, which is faster than looping over mu but takes
            more memory. Regardless of this option, we loop over mu if vectorization
            would take an excessive amount of memory. Default: True.

        Returns
        -------
        f : array_like
            Logarithmic growth rate. If ignoring RSD, packed as [z]; if including RSD,
            packed as [z,k,mu].
        """
        _MAX_INTEGRAND_SIZE = 1e8

        # Compute scale-independent growth rate f(z) = a D'(a) / D(a).
        fz = self.f_growth(1/(1+self.zs))

        if not rsd:
            return fz

        else:

            if self.mus is None:
                raise RuntimeError("mu array not defined!")

            # Get (m / rhobar_m) u_matter(k,m,z), packed as [z,m,k]
            uk = self._get_matter("nfw")

            # Compute normalization factor, \int dm n(m) (m / rhobar_m),
            # packed as [z].
            integrand = self.nzm[:, :] * uk[:, :, 0]
            norm = np.trapz(integrand, self.ms, axis=1)

            # If full integrand, packed as [z,m,k,mu] would be too large
            # to hold in memory at once, loop over mu:
            if (
                (np.prod(uk.shape) * self.mu.shape[0] > _MAX_INTEGRAND_SIZE)
                or not vectorize_mu
            ):
                f = np.zeros((uk.shape[0], uk.shape[2],) + self.mus.shape)
                for mui, mu in enumerate(self.mu):
                    # Form n(m) * (m / rhobar_m) u_matter(k,m,z), packed as [z,m,k]
                    integrand = self.nzm[:, :, None] * uk
                    # Multiply by D_FoG(k*mu).
                    # Result is packed as [z,m,k,mu], with a length-1 mu axis.
                    integrand = integrand[:, :, :, None] * self.DFoG(
                        self.ks[:, None]
                        * self.mu[mui : mui + 1][None, :]
                    )
                    # Integrate in m to get f(z, k, mu) / f_{scale-independent}(z)
                    f[:, :, mui : mui + 1] = np.trapz(integrand, self.ms, axis=1)

                # Finally, multiply by f_{scale-independent}(z), and divide by
                # normalization factor
                f *= (fz / norm)[:, None, None]

            # Otherwise, form whole integrand and integrate all at once
            else:
                # Form n(m) * (m / rhobar_m) u_matter(k,m,z), packed as [z,m,k]
                integrand = self.nzm[:, :, None] * uk
                # Multiply by D_FoG(k*mu).
                # Result is packed as [z,m,k,mu].
                integrand = (
                    integrand[:, :, None, None]
                    * self.DFoG(self.ks[:, None] * self.mu[None, :])
                )
                # Integrate in m to get f(z, k, mu) / f_{scale-independent}(z)
                f = np.trapz(integrand, self.ms, axis=1)
                # Finally, multiply by f_{scale-independent}(z)
                f *= (fz / norm)[:, None, None]

            return f


    def _get_hod_common(self, name):
        """Fetch dict of HOD quantities and u_c, u_s profiles.

        Parameters
        ----------
        name : string
            Internal name for HOD.

        Returns
        -------
        hod : dict
            Dict of precomputed HOD quantities.
        uc, us : array_like
            Central and satellite profiles, packed as [z,m,k].
        """
        hod = self.hods[name]
        cname = hod['central_profile']
        sname = hod['satellite_profile']
        uc = 1 if cname is None else self.uk_profiles[cname]
        us = self.uk_profiles[sname]
        return hod, uc, us

    def _get_hod_square(self, name, rsd=False, mui=None):
        """Fetch HOD-specific factors in P_1h integrand.

        Without RSD, this computes
            (2 u_c u_s N_c N_s + u_s^2 N_s^2) / n_gal^2 .
        Including RSD, this becomes (see Schaan & White 2021 (2103.01964), Eq. A.29)
            (2 u_c u_s N_c N_c D_FoG + u_s^2 N_s^2 D_FoG^2) / n_gal^2 .

        Parameters
        ----------
        name : string
            Internal name for HOD.
        rsd : bool, optional
            Whether to include RSD: Affects dimensionality of output. Default: False.
        mui : int, optional
            If specified, we compute for mu = self.mus[mui] instead of all mu's at once.
            Default: None.

        Returns
        -------
        s : array_like
            Product of factors in P_1h integrand. If ignoring RSD, packed as [z,m,k];
            if including RSD, packed as [z,m,k,mu].
        """
        # uc, us packed as [z,m,k]
        hod, uc, us = self._get_hod_common(name)

        if rsd:
            # Make array of k*mu, packed as [k,mu]
            if mui is None:
                kmu = self.ks[:, None] * self.mus[None, :]
            else:
                kmu = self.ks[:, None] * self.mus[None, mui : mui+1]
            # Compute (2 u_c u_s N_c N_c D_FoG + u_s^2 N_s^2 D_FoG^2) / n_gal^2,
            # packed as [z,m,k,mu]
            if np.asarray(uc).size > 1: uc = uc[..., None]
            return (
                2
                * uc
                * us[..., None]
                * hod['NcNs'][..., None, None]
                * self.DFoG(kmu)
                + hod['NsNsm1'][..., None, None]
                * us[..., None] ** 2
                * self.DFoG(kmu) ** 2
            ) / hod['ngal'][..., None, None, None] ** 2
        else:
            # Compute (2 u_c u_s N_c N_s + u_s^2 N_s^2) / n_gal^2, packed as [z,m,k]
            return (
                2 * uc * us * hod['NcNs'][..., None]
                + hod['NsNsm1'][..., None] * us ** 2
            ) / hod['ngal'][..., None, None] ** 2

    def _get_hod(self, name, lowklim=False, rsd=False, mui=None):
        """Fetch HOD-specific factors for single-tracer contribution to P_1h.

        Without RSD, this computes
            (u_c N_c + u_s N_s) / n_gal .
        Including RSD, this becomes (see Schaan & White 2021 (2103.01964), Eq. A.29)
            (u_c N_c + u_s N_s D_FoG) / n_gal .

        Parameters
        ----------
        name : string
            Internal name for HOD.
        lowklim : bool, optional
            If True, take low-k limit of profiles and FoG damping. Default: False.
        rsd : bool, optional
            Whether to include RSD: Affects dimensionality of output. Default: False.
        mui : int, optional
            If specified, we compute for mu = self.mus[mui] instead of all mu's at once.
            Default: None.

        Returns
        -------
        s : array_like
            Product of factors in P_1h integrand. If ignoring RSD, packed as [z,m,k];
            if including RSD, packed as [z,m,k,mu].
        """
        # uc, us packed as [z,m,k]
        hod, uc, us = self._get_hod_common(name)
        if lowklim:
            uc = 1
            us = 1

        if rsd:
            # Make array of k*mu, packed as [k,mu]
            if mui is None:
                kmu = self.ks[:, None] * self.mus[None, :]
            else:
                kmu = self.ks[:, None] * self.mus[None, mui : mui+1]
            # Compute (u_c N_c + u_s N_s D_FoG), packed as [z,m,k,mu]
            if np.asarray(uc).size > 1: uc = uc[..., None]
            if np.asarray(us).size > 1: us = us[..., None]
            if lowklim:
                dfog = 1
            else:
                dfog = self.DFoG(kmu)
            return (
                uc * hod['Nc'][..., None, None]
                + us * hod['Ns'][..., None, None] * dfog
            ) / hod['ngal'][..., None, None, None]
        else:
            # Compute (u_c N_c + u_s N_s) / n_gal, packed as [z,m,k]
            return (
                (uc * hod['Nc'][..., None] + us * hod['Ns'][...,None])
                / hod['ngal'][..., None, None]
            )

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
        self,
        name,
        name2=None,
        verbose=False,
        b1=None,
        b2=None,
        m_integrand=False,
        rsd=False,
    ):
        """Compute halo model power spectrum for specified profiles.

        This is the sum of the 1-halo and 2-halo terms.

        Parameters
        ----------
        name, name2 : string, optional
            Internal tracer names. Default: "nfw" for name1, None for name2.
        verbose : bool, optional
            Whether to print debugging information. Default: False.
        b1, b2 : array_like, optional
            Low-k linear bias for each tracer. If specified, override internal
            computations of these linear biases. Not applied to factors involving RSD.
            Default: None.
        m_integrand : bool, optional
            Whether to return dP/dm. Affects dimensionality of output.
            Default: False.
        rsd : bool, optional
            Whether to include RSD (Kaiser and FoG effects). RSD is only included in
            factors computed based on an HOD. Affects dimensionality of output.
            Default: False.

        Returns
        -------
        P : array_like
            Sum of 1-halo and 2-halo terms. Packing depends on inclusion of RSD and
            whether m integrand is requested:
                - No RSD, m integral: [z,k].
                - No RSD, m integrand: [z,m,k].
                - RSD, m integral: [z,k,mu].
                - RSD, m integrand: not implemented, but would be [z,m,k,mu].
        """
        if name2 is None: name2 = name
        return (
            self.get_power_1halo(name, name2, m_integrand=m_integrand, rsd=rsd)
            + self.get_power_2halo(
                name, name2, verbose, b1, b2, m_integrand=m_integrand, rsd=rsd
            )
        )

    def get_power_1halo(self, name="nfw", name2=None, m_integrand=False, rsd=False):
        """Compute P_1h(k,z) for specified profiles.

        Parameters
        ----------
        name, name2 : string, optional
            Internal tracer names. Default: "nfw" for name1, None for name2.
        m_integrand : bool, optional
            Whether to return dP_2h/dm. Affects dimensionality of output. Not
            implemented if RSD is turned on. Default: False.
        rsd : bool, optional
            Whether to include RSD (Kaiser and FoG effects). RSD is only included in
            factors computed based on an HOD. Affects dimensionality of output.
            Default: False.

        Returns
        -------
        P_1h : array_like
            1-halo term. Packing depends on inclusion of RSD and whether m integrand
            is requested:
                - No RSD, m integral: [z,k].
                - No RSD, m integrand: [z,m,k].
                - RSD, m integral: [z,k,mu].
                - RSD, m integrand: not implemented, but would be [z,m,k,mu].
        """
        if rsd and m_integrand:
            raise NotImplementedError("M integrand output not implemented with RSD")

        name2 = name if name2 is None else name2
        hnames = self.hods.keys()

        if rsd and (name in hnames or name2 in hnames):
            return self._get_power_1halo_rsd(name=name, name2=name2)
        else:
            return self._get_power_1halo_norsd(
                name=name, name2=name2, m_integrand=m_integrand
            )

    def _get_power_1halo_norsd(self, name="nfw", name2=None, m_integrand=False):
        """Compute P_1h(k,z) for specified profiles, without RSD.

        Parameters
        ----------
        name, name2 : string, optional
            Internal tracer names. Default: "nfw" for name1, None for name2.
        m_integrand : bool, optional
            Whether to return dP_2h/dm. Affects dimensionality of output.
            Default: False.

        Returns
        -------
        P_1h : array_like
            1-halo term. Packing depends on whether m integrand is requested:
                - M integral: [z,k].
                - M integrand: [z,m,k].
        """
        name2 = name if name2 is None else name2
        ms = self.ms[...,None]
        mnames = self.uk_profiles.keys()
        hnames = self.hods.keys()
        pnames = self.pk_profiles.keys()

        if (name in hnames) and (name2 in hnames):
            square_term = self._get_hod_square(name)
        elif (name in pnames) and (name2 in pnames):
            square_term = self._get_pressure(name)**2
        else:
            square_term = 1
            for nm in [name, name2]:
                if nm in hnames:
                    square_term *= self._get_hod(nm)
                elif nm in mnames:
                    square_term *= self._get_matter(nm)
                elif nm in pnames:
                    square_term *= self._get_pressure(nm)
                else:
                    raise ValueError("Profile %s not computed!" % nm)

        integrand = self.nzm[..., None] * square_term

        if m_integrand:
            # Return full integrand, packed as [z,m,k]
            return (
                integrand * (
                    1 - np.exp(-(self.ks / self.p['kstar_damping']) ** 2)
                )[None, None, :]
            )
        else:
            # Integrate in m, and return result packed as [z,k]
            return (
                np.trapz(integrand,ms,axis=-2)
                * (1-np.exp(-(self.ks/self.p['kstar_damping'])**2.))
            )

    def _get_power_1halo_rsd(self, name="nfw", name2=None):
        """Compute P_1h(k,z) for specified profiles, including RSD.

        Parameters
        ----------
        name, name2 : string, optional
            Internal tracer names. Default: "nfw" for name1, None for name2.

        Returns
        -------
        P_1h : array_like
            1-halo term, packed as [z,k,mu].
        """
        _MAX_INTEGRAND_SIZE = 1e8

        name2 = name if name2 is None else name2
        mnames = self.uk_profiles.keys()
        hnames = self.hods.keys()
        pnames = self.pk_profiles.keys()

        term1 = None
        term2 = None

        # Fetch matter and/or pressure profiles, if either are desired
        if name in mnames:
            term1 = self._get_matter(name)[..., None]
        elif name in pnames:
            term1 = self._get_pressure(name)[..., None]

        if name2 in mnames:
            term2 = self._get_matter(name2)[..., None]
        elif name2 in pnames:
            term2 = self._get_pressure(name2)[..., None]

        # If full integrand would be too large to store in memory, loop over mu,
        # performing m integral separately for each mu
        integrand_size = len(self.zs) * len(self.ms) * len(self.ks) * len(self.mus)
        if integrand_size > _MAX_INTEGRAND_SIZE:

            p1h = np.zeros((len(self.zs), len(self.ks), len(self.mus)))

            for mui, mu in enumerate(self.mus):
                # Get m integrand
                if (name in hnames) and (name2 in hnames):
                    square_term = self._get_hod_square(name, rsd=True, mui=mui)
                elif (name in hnames) and (name2 not in hnames):
                    square_term = self._get_hod(name, rsd=True, mui=mui) * term2
                elif (name not in hnames) and (name2 in hnames):
                    square_term = term1 * self._get_hod(name2, rsd=True, mui=mui)
                integrand = self.nzm[..., None, None] * square_term
                # Do m integral
                p1h[:, :, mui : mui + 1] = np.trapz(integrand, self.ms, axis=1) * (
                    1 - np.exp(
                        - (self.ks[None, :, None] / self.p['kstar_damping']) ** 2
                    )
                )

        else:
            # Get m integrand
            if (name in hnames) and (name2 in hnames):
                square_term = self._get_hod_square(name, rsd=rsd)
            elif (name in hnames) and (name2 not in hnames):
                square_term = self._get_hod(name, rsd=rsd) * term2
            elif (name not in hnames) and (name2 in hnames):
                square_term = term1 * self._get_hod(name2, rsd=rsd)
            integrand = self.nzm[..., None, None] * square_term
            # Do m integral
            p1h = np.trapz(integrand, ms, axis=1) * (1 - np.exp(
                - (self.ks[None, :, None] / self.p['kstar_damping']) ** 2
            ))

        return p1h


    def get_power_2halo(
        self,
        name="nfw",
        name2=None,
        verbose=False,
        b1_in=None,
        b2_in=None,
        m_integrand=False,
        rsd=False
    ):
        """Compute P_2h(k,z) for specified profiles.

        This implements a low-k consistency condition that ensures that P_2h is equal
        to b_A b_B P_lin in the low-k limit, where A and B are the 2 tracers we are
        computing for.

        Parameters
        ----------
        name, name2 : string, optional
            Internal tracer names. Default: "nfw" for name1, None for name2.
        verbose : bool, optional
            Whether to print debugging information. Default: False.
        b1_in, b2_in : array_like, optional
            Low-k linear bias for each tracer. If specified, override internal
            computations of these linear biases. If using RSD and computing for
            galaxies, adjust overall normalization of b_g(z,k,mu) to equal the input
            bias in the low-k limit. Default: None.
        m_integrand : bool, optional
            Whether to return dP_2h/dm. Affects dimensionality of output.
            Default: False.
        rsd : bool, optional
            Whether to include RSD (Kaiser and FoG effects). RSD is only included in
            factors computed based on an HOD. Affects dimensionality of output.
            Default: False.

        Returns
        -------
        P_2h : array_like
            2-halo term. Packing depends on inclusion of RSD and whether m integrand
            is requested:
                - No RSD, m integral: [z,k].
                - No RSD, m integrand: [z,m,k].
                - RSD, m integral: [z,k,mu].
                - RSD, m integrand: not implemented, but would be [z,m,k,mu].
        """
        name2 = name if name2 is None else name2

        def _2halointegrand(iterm):
            # Compute n(m) * b_h(m) * [Fourier-space profile]
            return self.nzm[..., None] * iterm * self.bh[..., None]

        def _2haloint(iterm):
            # Compute \int dm n(m) b_h(m) [Fourier-space profile]
            integrand = _2halointegrand(iterm)
            integral = np.trapz(integrand, self.ms[..., None], axis=-2)
            return integral

        def _get_term(iname):
            # Get Fourier-space profile, low-k limit of this profile, and linear bias
            if iname in self.uk_profiles.keys():
                # Matter profile (m / rhobar_m) u(k,m,z), with b=1
                rterm1 = self._get_matter(iname)
                rterm01 = self._get_matter(iname, lowklim=True)
                b = 1
            elif iname in self.pk_profiles.keys():
                # Pressure profile, with b=0 and low-k limit also set to 0
                rterm1 = self._get_pressure(iname)
                rterm01 = self._get_pressure(iname, lowklim=True)
                print ('Check the consistency relation for tSZ')
                b = rterm01 = 0
            elif iname in self.hods.keys():
                # Profile from HOD, with b also computed from HOD
                rterm1 = self._get_hod(iname)
                rterm01 = self._get_hod(iname, lowklim=True)
                b = self.get_bg(
                    self.hods[iname]['Nc'],
                    self.hods[iname]['Ns'],
                    self.hods[iname]['ngal']
                )[:,None]
            else: raise ValueError("Profile %s not defined!" % iname)
            return rterm1, rterm01, b

        if rsd and (name not in self.hods.keys() and name2 not in self.hods.keys()):
            raise RuntimeError("RSD can only be applied to galaxy factors in P_2h!")

        if rsd and m_integrand:
            raise NotImplementedError("M integrand output not implemented with RSD")

        # If needed for either tracer, compute F(k,mu,z)
        if rsd and (name in self.hods.keys() or name2 in self.hods.keys()):
            if verbose: print("Computing f")
            fg = self.get_fgrowth(rsd=True)

        # Compute effective bias factor for tracer 1
        if rsd and (name in self.hods.keys()):
            if verbose: print("Tracer 1: computing b")
            b1 = self.get_bg(
                self.hods[name]['Nc'],
                self.hods[name]['Ns'],
                self.hods[name]['ngal'],
                rsd=True,
                u_name=self.hods[name]['satellite_profile']
            )
            if b1_in is not None:
                b1 *= (b1_in[:, None] / b1[:, 0])[:, None, :]
            factor1 = (b1 + fg * self.mu[np.newaxis, np.newaxis, :] ** 2)
        else:
            # Compute get Fourier-space profile for name, name2
            iterm1, iterm01, b1 = _get_term(name)
            # Set linear bias factors to inputs, if specified
            if b1_in is not None:
                b1 = b1_in.reshape((b1_in.shape[0],1))

            # Compute \int dm n(m) b_h(m) [Fourier-space profile] for name, name2
            integral = _2haloint(iterm1)
            # For halo bias consistency relation, compute
            # \int dm n(m) b_h(m) [Fourier-space profile]
            # using low-k limit of profile
            consistency1 = _2haloint(iterm01)
            # Compute bias factor for P_2h
            factor1 = integral + b1 - consistency1
            if rsd:
                factor1 = factor1[..., np.newaxis]

        # Compute effective bias factor for tracer 2
        if rsd and (name2 in self.hods.keys()):
            if verbose: print("Tracer 2: computing b")
            b2 = self.get_bg(
                self.hods[name2]['Nc'],
                self.hods[name2]['Ns'],
                self.hods[name2]['ngal'],
                rsd=True,
                u_name=self.hods[name2]['satellite_profile']
            )
            if b2_in is not None:
                b2 *= (b2_in[:, None] / b2[:, 0])[:, None, :]
            factor2 = (b2 + fg * self.mu[np.newaxis, np.newaxis, :] ** 2)
        else:
            iterm2, iterm02, b2 = _get_term(name2)
            if b2_in is not None:
                b2 = b2_in.reshape((b2_in.shape[0],1))

            integral2 = _2haloint(iterm2)
            consistency2 = _2haloint(iterm02)
            factor2 = integral2 + b2 - consistency2
            if rsd:
                factor2 = factor2[..., None]


        if m_integrand:
            # Return dP_2h / dM, packed as [z,m,k].
            # There is a question of how to incorporate the
            # normalization from the consistency relation into this. The
            # prescription below, which takes d(consistency)/dM=0, seems to work,
            # in that integrating it in M gives a result that's pretty close
            # to the full result.
            prefactor = (
                _2halointegrand(iterm1)
                * (integral2+b2-consistency2)[..., None, :]
                + (integral+b1-consistency1)[..., None, :]
                * _2halointegrand(iterm2)
            )
            return prefactor * self.Pzk[..., None, :]

        else:
            if rsd:
                # Result packed as [z,k,mu].
                return self.Pzk[:, :, None] * factor1 * factor2
            else:
                # Subtract the low-k limit from each integral, and then add the linear
                # bias. This is redundant for P_gg, but for P_mm it ensures that P_2h
                # is equal to P_lin in the low-k limit.
                # The resulting P_2h is packed as [z,k]
                # if verbose:
                #     print("Two-halo consistency1: " , consistency1,integral)
                #     print("Two-halo consistency2: " , consistency2,integral2)
                return self.Pzk * factor1 * factor2

    def get_bispectrum(
        self,
        name="g",
        name2=None,
        name3=None,
        b1_in=None,
        verbose=False,
        zi=None
    ):
        """Compute halo model bispectrum for specified profiles

        Currently only implemented for B_ggg, without RSD.

        Parameters
        ----------
        name, name2, name3 : string, optional
            Internal tracer names. Default: "g" for name1, None for name2 and name3.
        b1_in : array_like, optional
            Low-k linear bias. If specified, override internal computations of linear
            bias. Default: None.
        verbose : bool, optional
            Whether to print debugging information. Default: False.
        zi : int, optional
            If specified, only compute bispectrum for redshift corresponding to
            self.zs[zi]. Affects dimensionality of output. Default: None.

        Returns
        -------
        B : array_like
            Bispectrum, packed as [z,k1,k2,k3] if zi=None or [k1,k2,k3] if zi is
            specified.
        """
        return (
            self.get_bispectrum_3halo(
                name, name2=name2, name3=name3, b1_in=b1_in, zi=zi
            )
            + self.get_bispectrum_2halo(
                name, name2=name2, name3=name3, b1_in=b1_in, verbose=verbose, zi=zi
            )
            + self.get_bispectrum_1halo(
                name, name2=name2, name3=name3, verbose=verbose, zi=zi
            )
        )

    def get_bispectrum_3halo(
        self,
        name="g",
        name2=None,
        name3=None,
        b1_in=None,
        zi=None
    ):
        """Compute B_3h(k1,k2,k3,z) for specified profiles.

        Currently only implemented for B_ggg, without RSD.

        Parameters
        ----------
        name, name2, name3 : string, optional
            Internal tracer names. Default: "g" for name1, None for name2 and name3.
        b1_in : array_like, optional
            Low-k linear bias. If specified, override internal computations of linear
            bias. Default: None.
        zi : int, optional
            If specified, only compute bispectrum for redshift corresponding to
            self.zs[zi]. Affects dimensionality of output. Default: None.

        Returns
        -------
        B_3h : array_like
            3-halo term, packed as [z,k1,k2,k3] if zi=None or [k1,k2,k3] if zi is
            specified.
        """
        name2 = name if name2 is None else name2
        name3 = name if name3 is None else name3

        def _3halointegrand(iterm):
            # Compute n(m) * b_h(m) * [Fourier-space profile]
            if zi is None:
                return self.nzm[..., None] * iterm * self.bh[..., None]
            else:
                return self.nzm[zi, :, None] * iterm * self.bh[zi, :, None]

        def _3haloint(iterm):
            # Compute \int dm n(m) b_h(m) [Fourier-space profile]
            integrand = _3halointegrand(iterm)
            integral = np.trapz(integrand, self.ms[..., None], axis=-2)
            return integral

        def _get_term(iname):
            # Get Fourier-space profile, low-k limit of this profile, and linear bias.
            # Profile from HOD, with b also computed from HOD.
            # Output of _get_hod() includes 1/ngal factor
            rterm1 = self._get_hod(iname)
            rterm01 = self._get_hod(iname, lowklim=True)
            b = self.get_bg(
                self.hods[iname]['Nc'],
                self.hods[iname]['Ns'],
                self.hods[iname]['ngal']
            )[:,None]

            return rterm1, rterm01, b

        if (
            (name not in self.hods.keys())
            or (name2 not in self.hods.keys())
            or (name3 not in self.hods.keys())
        ):
            raise NotImplementedError(
                "Bispectrum only implemented for galaxy autos! "
                "name = %s, name2 = %s, name3 = %s" % (name, name2, name3)
            )

        if (name != name2) or (name != name3):
            raise NotImplementedError("Bispectrum only implemented for tracer autos!")

        # Compute effective bias factor
        ## Get Fourier-space profile
        iterm1, iterm01, b1 = _get_term(name)
        if zi is not None:
            iterm1 = iterm1[zi, ...]
            iterm01 = iterm01[zi, ...]
            b1 = b1[zi]
        ## Set linear bias factors to inputs, if specified
        if b1_in is not None:
            b1 = b1_in.reshape((b1_in.shape[0],1))
            if zi is not None:
                b1 = b1[zi, ...]
        ## Compute \int dm n(m) b_h(m) [Fourier-space profile
        integral = _3haloint(iterm1)
        ## For halo bias consistency relation, compute
        ## \int dm n(m) b_h(m) [Fourier-space profile]
        ## using low-k limit of profile
        consistency1 = _3haloint(iterm01)
        ## Compute bias factor
        factor1 = integral + b1 - consistency1

        if zi is None:
            return (
                self.get_bispectrum_matter_tree()
                * factor1[:, :, None, None]
                * factor1[:, None, :, None]
                * factor1[:, None, None, :]
            )
        else:
            return (
                self.get_bispectrum_matter_tree(zi=zi)
                * factor1[:, None, None]
                * factor1[None, :, None]
                * factor1[None, None, :]
            )

    def get_bispectrum_matter_tree(self, zi=None):
        """Compute tree-level matter bispectrum.

        This is evaluated at every (k1,k2,k3) triplet for k_i in self.ks.

        Parameters
        ----------
        zi : int, optional
            If specified, only compute bispectrum for redshift corresponding to
            self.zs[zi]. Affects dimensionality of output. Default: None.

        Returns
        -------
        B : array_like
            Tree-level bispectrum, packed as [z,k1,k2,k3] if zi=None or [k1,k2,k3] if
            zi is specified.
        """

        def _F2(k1, k2, k3):
            costheta12 = 0.5 * (k3 ** 2 - k1 ** 2 - k2 ** 2) / (k1 * k2)
            return (
                5.0 / 7
                + 0.5 * costheta12 * (k1 / k2 + k2 / k1)
                + 2.0 / 7 * costheta12 ** 2
            )

        # Get coordinate meshes for k1, k2, k3 and flatten each mesh
        k1_mesh, k2_mesh, k3_mesh = np.meshgrid(
            self.ks, self.ks, self.ks, indexing="ij"
        )
        # k1, k2, k3 = k1.flatten(), k2.flatten(), k3.flatten()

        # Get required permutations of F_2 kernel
        F2_123 = _F2(k1_mesh, k2_mesh, k3_mesh)
        F2_231 = _F2(k2_mesh, k3_mesh, k1_mesh)
        F2_312 = _F2(k3_mesh, k1_mesh, k2_mesh)

        # Compute B_tree
        if zi is None:
            B = 2 * (
                F2_123[None, :, :, :]
                * self.Pzk[:, :, None, None] # P(k1)
                * self.Pzk[:, None, :, None] # P(k2)
                + F2_231[None, :, :, :]
                * self.Pzk[:, None, :, None] # P(k2)
                * self.Pzk[:, None, None, :] # P(k3)
                + F2_312[None, :, :, :]
                * self.Pzk[:, None, None, :] # P(k3)
                * self.Pzk[:, :, None, None] # P(k1)
            )
        else:
            B = 2 * (
                F2_123
                * self.Pzk[zi, :, None, None] # P(k1)
                * self.Pzk[zi, None, :, None] # P(k2)
                + F2_231
                * self.Pzk[zi, None, :, None] # P(k2)
                * self.Pzk[zi, None, None, :] # P(k3)
                + F2_312
                * self.Pzk[zi, None, None, :] # P(k3)
                * self.Pzk[zi, :, None, None] # P(k1)
            )

        return B


    def get_bispectrum_2halo(
        self, name="g", name2=None, name3=None, b1_in=None, verbose=False, zi=None
    ):
        """Compute B_2h(k1,k2,k3,z) for specified profiles.

        Parameters
        ----------
        name, name2, name3 : string, optional
            Internal tracer names. Default: "g" for name1, None for name2 and name3.
        b1_in : array_like, optional
            Low-k linear bias. If specified, override internal computations of linear
            bias. Default: None.
        verbose : bool, optional
            Whether to print debugging information. Default: False.
        zi : int, optional
            If specified, only compute bispectrum for redshift corresponding to
            self.zs[zi]. Affects dimensionality of output. Default: None.

        Returns
        -------
        B_2h : array_like
            2-halo term, packed as [z,k1,k2,k3] if zi=None or [k1,k2,k3] if
            zi is specified.
        """
        name2 = name if name2 is None else name2
        name3 = name if name3 is None else name3

        if (
            (name not in self.hods.keys())
            or (name2 not in self.hods.keys())
            or (name3 not in self.hods.keys())
        ):
            raise NotImplementedError("Bispectrum only implemented for galaxy autos!")

        if (name != name2) or (name != name3):
            raise NotImplementedError("Bispectrum only implemented for tracer autos!")

        def _3halointegrand(iterm):
            # Compute n(m) * b_h(m) * [Fourier-space profile]
            if zi is None:
                return self.nzm[..., None] * iterm * self.bh[..., None]
            else:
                return self.nzm[zi, :, None] * iterm * self.bh[zi, :, None]

        def _3haloint(iterm):
            # Compute \int dm n(m) b_h(m) [Fourier-space profile]
            integrand = _3halointegrand(iterm)
            integral = np.trapz(integrand, self.ms[..., None], axis=-2)
            return integral

        def _get_term(iname):
            # Get Fourier-space profile, low-k limit of this profile, and linear bias.
                # Profile from HOD, with b also computed from HOD
            rterm1 = self._get_hod(iname)
            rterm01 = self._get_hod(iname, lowklim=True)
            b = self.get_bg(
                self.hods[iname]['Nc'],
                self.hods[iname]['Ns'],
                self.hods[iname]['ngal']
            )[:,None]

            return rterm1, rterm01, b

        # Compute bias factor as for 3h term, as function of k
        if verbose:
            print("Computing effective bias factor")
        ## Get Fourier-space profile
        iterm1, iterm01, b1 = _get_term(name)
        if zi is not None:
            iterm1 = iterm1[zi, ...]
            iterm01 = iterm01[zi, ...]
            b1 = b1[zi]
        ## Set linear bias factors to inputs, if specified
        if b1_in is not None:
            b1 = b1_in.reshape((b1_in.shape[0],1))
            if zi is not None:
                b1 = b1[zi, ...]
        ## Compute \int dm n(m) b_h(m) [Fourier-space profile
        integral = _3haloint(iterm1)
        ## For halo bias consistency relation, compute
        ## \int dm n(m) b_h(m) [Fourier-space profile]
        ## using low-k limit of profile
        consistency1 = _3haloint(iterm01)
        ## Compute bias factor
        factor1 = integral + b1 - consistency1

        # Compute NcNs term, as function of k
        if verbose:
            print("Computing NcNs term")
        ## Get uc, us packed as [z,m,k]
        hod, uc, us = self._get_hod_common(name)
        if uc != 1:
            raise NotImplementedError("Bispectrum not implemented for nontrivial u_c!")
        ## Compute m integrand, packed as [z,m,k] or [m,k] for specific z
        if zi is None:
            integrand = (
                self.nzm[:, :, None]
                * self.bh[:, :, None]
                * hod['NcNs'][:, :, None]
                * us
                / hod['ngal'][:, None, None]
            )
        else:
            integrand = (
                self.nzm[zi, :, None]
                * self.bh[zi, :, None]
                * hod['NcNs'][zi, :, None]
                * us[zi]
                / hod['ngal'][zi, None, None]
            )
        ## Integrate in m, and apply low-k damping
        ncns_term = np.trapz(integrand, self.ms, axis=-2)
        del integrand
        lowk_damping = 1 - np.exp(-(self.ks/self.p['kstar_damping']) ** 2.)
        if zi is None:
            ncns_term *= lowk_damping[None, :]
        else:
            ncns_term *= lowk_damping

        # Compute NsNsm1 term, as function of k1,k2
        if verbose:
            print("Computing NsNsm1 term")
        ## Compute m integrand, packed as [z,m,k1,k2]
        if zi is None:
            integrand = (
                self.nzm[:, :, None, None]
                * self.bh[:, :, None, None]
                * hod['NsNsm1'][:, :, None, None]
                * us[:, :, :, None]
                * us[:, :, None, :]
                / hod['ngal'][:, None, None, None]
            )
        else:
            integrand = (
                self.nzm[zi, :, None, None]
                * self.bh[zi, :, None, None]
                * hod['NsNsm1'][zi, :, None, None]
                * us[zi, :, :, None]
                * us[zi, :, None, :]
                / hod['ngal'][zi, None, None, None]
            )
        ## Integrate in m, and apply low-k damping
        nsnsm1_term = np.trapz(integrand, self.ms, axis=-3)
        del integrand
        if zi is None:
            nsnsm1_term *= lowk_damping[None, :, None] * lowk_damping[None, None, :]
        else:
            nsnsm1_term *= lowk_damping[:, None] * lowk_damping[None, :]

        # Accumulate 3 permutations, packed as [z,k1,k2,k3]
        if verbose:
            print("Computing B")

        if zi is None:
            B = (
                (
                    ncns_term[:, :, None, None]
                    + ncns_term[:, None, :, None]
                    + nsnsm1_term[:, :, :, None]
                ) * factor1[:, None, None, :] * self.Pzk[:, None, None, :]
                + (
                    ncns_term[:, None, :, None]
                    + ncns_term[:, None, None, :]
                    + nsnsm1_term[:, None, :, :]
                ) * factor1[:, :, None, None] * self.Pzk[:, :, None, None]
                + (
                    ncns_term[:, None, None, :]
                    + ncns_term[:, :, None, None]
                    + nsnsm1_term[:, :, None, :]
                ) * factor1[:, None, :, None] * self.Pzk[:, None, :, None]
            )
        else:
            B = (
                (
                    ncns_term[:, None, None]
                    + ncns_term[None, :, None]
                    + nsnsm1_term[:, :, None]
                ) * factor1[None, None, :] * self.Pzk[zi, None, None, :]
                + (
                    ncns_term[None, :, None]
                    + ncns_term[None, None, :]
                    + nsnsm1_term[None, :, :]
                ) * factor1[:, None, None] * self.Pzk[zi, :, None, None]
                + (
                    ncns_term[None, None, :]
                    + ncns_term[:, None, None]
                    + nsnsm1_term[:, None, :]
                ) * factor1[None, :, None] * self.Pzk[zi, None, :, None]
            )

        return B


    def get_bispectrum_1halo(
        self, name="g", name2=None, name3=None, verbose=False, zi=None
    ):
        """Compute B_1h(k1,k2,k3,z) for specified profiles.

        Parameters
        ----------
        name, name2, name3 : string, optional
            Internal tracer names. Default: "g" for name1, None for name2 and name3.
        verbose : bool, optional
            Whether to print debugging information. Default: False.
        zi : int, optional
            If specified, only compute bispectrum for redshift corresponding to
            self.zs[zi]. Affects dimensionality of output. Default: None.

        Returns
        -------
        B_1h : array_like
            1-halo term, packed as [z,k1,k2,k3] if zi=None or [k1,k2,k3] if
            zi is specified.
        """
        name2 = name if name2 is None else name2
        name3 = name if name3 is None else name3

        if (
            (name not in self.hods.keys())
            or (name2 not in self.hods.keys())
            or (name3 not in self.hods.keys())
        ):
            raise NotImplementedError("Bispectrum only implemented for galaxy autos!")

        if (name != name2) or (name != name3):
            raise NotImplementedError("Bispectrum only implemented for tracer autos!")

        # Get uc, us packed as [z,m,k]
        hod, uc, us = self._get_hod_common(name)
        if uc != 1:
            raise NotImplementedError("Bispectrum not implemented for nontrivial u_c!")

        b1h = []

        # Loop over z, for memory reasons
        for zii, z in enumerate(self.zs):
            if zi is not None and zii != zi:
                continue

            if verbose:
                print("Computing for redshift index %d (z = %g)" % (zii, z))
                if zii == 0:
                    print(
                        "\tIntegrand shape:",
                        (
                            hod['NcNsNsm1'].shape[1],
                            len(self.ks),
                            len(self.ks),
                            len(self.ks)
                        )
                    )
            # Compute integrand, packed as [m,k1,k2,k3]
            if verbose:
                print("\tComputing NcNsNsm1 term")
            integrand = (
                hod['NcNsNsm1'][zii, :, None, None, None]
                * (
                    us[zii, :, :, None, None] * us[zii, :, None, :, None]
                    + us[zii, :, None, :, None] * us[zii, :, None, None, :]
                    + us[zii, :, None, None, :] * us[zii, :, :, None, None]
                )
            )
            if verbose:
                print("\tComputing NsNsm1Nsm2 term")
            integrand += (
                hod['NsNsm1Nsm2'][zii, :, None, None, None]
                * us[zii, :, :, None, None]
                * us[zii, :, None, :, None]
                * us[zii, :, None, None, :]
            )
            integrand
            integrand *= self.nzm[zii, :, None, None, None]

            # Integrate in m, and append to b1h
            lowk_damping = 1 - np.exp(-(self.ks/self.p['kstar_damping']) ** 2.)
            b1h.append(
                np.trapz(integrand,self.ms,axis=0)
                * lowk_damping[:, None, None]
                * lowk_damping[None, :, None]
                * lowk_damping[None, None, :]
            )

        # Divide final result by n_gal^3
        if zi is None:
            b1h = np.asarray(b1h) / hod['ngal'][:, None, None, None] ** 3
        else:
            b1h = np.asarray(b1h[0]) / hod['ngal'][zi, None, None, None] ** 3

        return b1h



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
Profiles
"""

def Fcon(c): return (np.log(1.+c) - (c/(1.+c)))

def rhoscale_nfw(mdelta,rdelta,cdelta):
    rs = rdelta/cdelta
    V = 4.*np.pi * rs**3.
    return pref * mdelta / V / Fcon(cdelta)

def rho_nfw_x(x,rhoscale): return rhoscale/x/(1.+x)**2.

def rho_nfw(r,rhoscale,rs): return rho_nfw_x(r/rs,rhoscale)

def rhoHI_vn18(r, alphastar, r0, rho0=None):
    """Compute position-space HI density profile from Villaescusa-Navarro et al. 2018
    (1804.09180).

    Parameters
    ----------
    r : array_like
        1d array of radii to evaluate at, in Mpc.
    alphastar, r0 : array_like
        Arrays of alpha_* and r_0 parameters to evaluate at. Typically, these
        will be evaluated at different redshifts and/or halo masses.
    rho0 : array_like, optional
        1d array of rho_0 parameter to evaluate at, with same shape as alphastar and r0.
        If not specified, we use rho_0=1.

    Returns
    -------
    rho : array_like
        rho_HI values, packed as [r,...]).
    """
    # TODO: there must be a better way to do this...
    r_newshape = (len(r), ) + tuple(np.ones(len(alphastar.shape), dtype=int))
    r = r.reshape(r_newshape)

    rho = r ** -alphastar[None, ...] * np.exp(-r0[None, ...] / r)

    if rho0 is not None:
        rho *= rho0[None, ...]

    return rho


def rhoHI_vn18_x(x, alphastar, r0, rho0=None):
    """Compute position-space HI density profile from Villaescusa-Navarro et al. 2018
    (1804.09180), as function of x=r/r0.

    Parameters
    ----------
    x : array_like
        1d array of dimensionless to evaluate at.
    alphastar, r0 : array_like
        1d arrays of alpha_* and r_0 parameters to evaluate at. Typically, these
        will be evaluated at different halo masses.
    rho0 : array_like, optional
        1d array of rho_0 parameter to evaluate at, with same shape as alphastar and r0.
        If not specified, we use rho_0=1.

    Returns
    -------
    rho : array_like
        rho_HI values, packed as [x,...].
    """
    # TODO: there must be a better way to do this...
    x_newshape = (len(x), ) + tuple(np.ones(len(alphastar.shape), dtype=int))
    x = x.reshape(x_newshape)

    rho = (x * r0[None, ...]) ** -alphastar[None, ...] * np.exp(-1 / x)

    if rho0 is not None:
        rho *= rho0[None, ...]

    return rho


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
