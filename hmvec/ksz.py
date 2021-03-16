"""
Halo model for ksz

We use linear matter power for k<0.1 Mpc-1 used in
calculations of large-scale Pgv, Pvv and Pgg.

We use the halo model for k>0.1 Mpc-1 used in calculations
of small-scale Pge, Pee and Pgg.

"""

import warnings

from .params import default_params
from .hmvec import HaloModel
from . import utils
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, dblquad

defaults = {'min_mass':1e6, 'max_mass':1e16, 'num_mass':1000}
constants = {
    'thompson_SI': 6.6524e-29,
    'meter_to_megaparsec': 3.241e-23,
    'G_SI': 6.674e-11,
    'mProton_SI': 1.673e-27,
    'H100_SI': 3.241e-18
}

def Ngg(ngalMpc3): 
    return (1./ngalMpc3)

def pgv(kls,):
    pass

def pge_err_core(pgv_int,kstar,chistar,volume_gpc3,kss,ks_bin_edges,pggtot,Cls):

    """
    pgv_int: \int dkl kl^2 Pgv^2/Pggtot
    kstar: kSZ radial weight function at chistar
    chistar: comoving distance to galaxy survey
    volume_gpc3: volume in gpc3
    kss: short wavelength k on which pggtot and cltot are defined
    
    
    """
    volume = volume_gpc3 * 1e9
    ints = []
    cltot = get_interpolated_cls(Cls,chistar,kss)
    integrand = (kss/(pggtot * cltot))
    for kleft,kright in zip(ks_bin_edges[:-1],ks_bin_edges[1:]):
        sel = np.s_[np.logical_and(kss>kleft,kss<=kright)]
        y = _sanitize(integrand[sel])
        x = kss[sel]
        ints.append(np.trapz(y,x))
    return (volume * kstar**2 / 12 / np.pi**3 / chistar**2. * pgv_int * np.asarray(ints))**(-0.5)


def get_kmin(volume_gpc3):
    vol_mpc3 = volume_gpc3 * 1e9
    return np.pi/vol_mpc3**(1./3.)


def chi(Yp,NHe):
    val = (1-Yp*(1-NHe/4.))/(1-Yp/2.)
    return val

def ne0_shaw(ombh2,Yp,NHe=0,me = 1.14,gasfrac = 0.9):
    '''
    Average electron density today
    Eq 3 of 1109.0553
    Units: 1/meter**3
    '''
    omgh2 = gasfrac* ombh2
    mu_e = 1.14 # mu_e*mass_proton = mean mass per electron
    ne0_SI = chi(Yp,NHe)*omgh2 * 3.*(constants['H100_SI']**2.)/constants['mProton_SI']/8./np.pi/constants['G_SI']/mu_e
    return ne0_SI

def ksz_radial_function(z,ombh2, Yp, gasfrac = 0.9,xe=1, tau=0, params=None):
    """
    K(z) = - T_CMB sigma_T n_e0 x_e(z) exp(-tau(z)) (1+z)^2
    Eq 4 of 1810.13423
    """
    if params is None: params = default_params
    T_CMB_muk = params['T_CMB'] # muK
    thompson_SI = constants['thompson_SI']
    meterToMegaparsec = constants['meter_to_megaparsec']
    ne0 = ne0_shaw(ombh2,Yp)
    return T_CMB_muk*thompson_SI*ne0*(1.+z)**2./meterToMegaparsec  * xe  *np.exp(-tau)

def _sanitize(inp):
    inp[~np.isfinite(inp)] = 0
    return inp

class kSZ(HaloModel):
    def __init__(self,zs,volumes_gpc3,ngals_mpc3,
                 kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                 num_kS_bins=101,num_mu_bins=102,ms=None,params=None,mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False,
                 electron_profile_name='e',electron_profile_family='AGN',
                 skip_electron_profile=False,electron_profile_param_override=None,
                 electron_profile_nxs=None,electron_profile_xmax=None,
                 skip_hod=False,hod_name="g",hod_corr="max",hod_param_override=None,
                 mthreshs_override=None,
                 verbose=False,
                 b1=None,b2=None,sigz=None):

        if ms is None: ms = np.geomspace(defaults['min_mass'],defaults['max_mass'],defaults['num_mass'])
        volumes_gpc3 = np.atleast_1d(volumes_gpc3)
        assert len(zs)==len(volumes_gpc3)==len(ngals_mpc3)
        ngals_mpc3 = np.asarray(ngals_mpc3)
        ks = np.geomspace(kS_min,kS_max,num_kS_bins)
        self.ks = ks
        self.mu = np.linspace(-1.,1.,num_mu_bins)
        if verbose: print('Defining HaloModel')
        HaloModel.__init__(self,zs,ks,ms=ms,params=params,mass_function=mass_function,
                 halofit=halofit,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=skip_nfw)
        if verbose: print('Defining HaloModel: finished')
        self.kS = self.ks
        if not(skip_electron_profile):
            if verbose: print('Defining electron profile')
            self.add_battaglia_profile(name=electron_profile_name,
                                            family=electron_profile_family,
                                            param_override=electron_profile_param_override,
                                            nxs=electron_profile_nxs,
                                            xmax=electron_profile_xmax,ignore_existing=False)
            if verbose: print('Defining electron profile: finished')
        
        if not(skip_hod):
            if verbose: print('Defining HOD')
            self.add_hod(hod_name,mthresh=mthreshs_override,ngal=ngals_mpc3,corr=hod_corr,
                         satellite_profile_name='nfw',
                         central_profile_name=None,ignore_existing=False,param_override=hod_param_override)
            if verbose: print('Defining HOD: finished')
        
        self.Pmms = []
        self.fs = []
        self.adotf = []
        self.d2vs = []
        
        self.sigma_z_func = lambda z : sigz * (1.+z)
        zhs,hs = np.loadtxt("fiducial_cosmology_Hs.txt",unpack=True)
        self.Hphotoz = interp1d(zhs,hs)

        # Define log-spaced array of k values
        self.kLs = np.geomspace(get_kmin(np.max(volumes_gpc3)),kL_max,num_kL_bins)
        # # kr = mu * kL ; this is an array of krs of shape (num_mus,num_kLs)
        self.krs = self.mu.reshape((self.mu.size,1)) * self.kLs.reshape((1,self.kLs.size))
        
        self.sigz = sigz
        if not skip_hod:
            self.sPggs = self.get_power(hod_name,name2=hod_name,verbose=verbose,b1=b1,b2=b1)
            self.sPges = self.get_power(hod_name,name2=electron_profile_name,verbose=verbose,b1=b1) 
            if sigz is not None:
                oPggs = self.sPggs.copy()
                oPges = self.sPges.copy()
                self.sPggs = []
                self.sPges = []
                for zindex in range(oPggs.shape[0]):
                    self.sPggs.append( oPggs[zindex] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1))**2.) )
                    self.sPges.append( oPges[zindex] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1))) )
                self.sPggs = np.asarray(self.sPggs)
                self.sPges = np.asarray(self.sPges)
            
        # Warn user that k_min is the same for all zs (thanks to Simon Foreman for speed-up here)
        if np.max(volumes_gpc3) != np.min(volumes_gpc3):
            warnings.warn('Using equal k_min at each z, despite different volumes at each z')
            
        # get P_linear and f(z)
        # on grid in z and k
        p = self._get_matter_power(self.zs,self.kLs,nonlinear=False)
        growth = self.results.get_redshift_evolution(
            self.kLs, 
            self.zs, 
            ['growth']
        )[:,:,0]

        self.kstars = []
        self.chistars = []
        self.Vs = volumes_gpc3
        self.vrec = []
        self.sPggtot = []
        self.sPge = []
        self.bgs = []
        aPgg = self.get_power('g','g',verbose=verbose)
        aPge = self.get_power('g','e',verbose=verbose)
        for zindex,volume_gpc3 in enumerate(volumes_gpc3):
            self.Pmms.append(np.resize(p[zindex].copy(),(self.mu.size,self.kLs.size)))
            self.fs.append(growth[:,zindex].copy())
            z = self.zs[zindex]
            a = 1./(1.+z)
            H = self.results.h_of_z(z)
            self.kstars.append(self.ksz_radial_function(zindex))
            self.d2vs.append(  self.fs[zindex]*a*H / self.kLs )
            self.adotf.append(self.fs[zindex]*a*H)

            self.chistars.append( self.results.comoving_radial_distance(z) )

            # Compute P_gg + N_gg and P_gv for fiducial and "true" parameters, as functions of k_L
            bg = self.hods['g']['bg'][zindex]
            self.bgs.append(bg)
            ngal = ngals_mpc3[zindex]
            ngg = Ngg(ngal)
            flPgg = self.lPgg(zindex,bg1=bg,bg2=bg)[0,:] + ngg
            flPgv = self.lPgv(zindex,bg=bg)[0,:]
            # Construct integrand (without prefactor) as function of tabulated k_L values,
            # and integrate
            kls = self.kLs
            integrand = _sanitize((kls**2.)*(flPgv*flPgv)/flPgg)
            vrec = np.trapz(integrand,kls)
            self.vrec.append(vrec.copy())
            
            if verbose: print("Calculating small scale Pgg...")
            Pgg = aPgg[zindex].copy()
            if sigz is not None:
                Pgg = Pgg[None,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1))**2.)
            Pggtot  = Pgg + ngg
            self.sPggtot.append(Pggtot.copy())
            Pge = aPge[zindex].copy()
            if sigz is not None:
                Pge = Pge[None,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1)))
            self.sPge.append(Pge.copy())
        
        self.ngals_mpc3 = ngals_mpc3

    def Pge_err(self,zindex,ks_bin_edges,Cls):
        kstar = self.kstars[zindex]
        chistar = self.chistars[zindex]
        volume = self.Vs[zindex]
        pgv_int  = self.vrec[zindex]
        kss = self.ks
        pggtot = self.sPggtot[zindex][0]
        return pge_err_core(pgv_int,kstar,chistar,volume,kss,ks_bin_edges,pggtot,Cls)        
    
    def lPvv(self,zindex,bv1=1,bv2=1):
        """The long-wavelength power spectrum of vxv.
        This is calculated as:
        (faH/kL)**2*Pmm(kL)
        to return a 1D array [mu,kL] with identical
        copies over mus.

        Here Pmm is the non-linear power for all halos.
        bv1 and bv2 are the velocity biases in each bin.
        """
        Pvv = (self.d2vs[zindex])**2. * self.Pmms[zindex] *bv1*bv2
        return Pvv

    def lPgg(self,zindex,bg1,bg2):
        """The long-wavelength power spectrum of gxg.
        Here Pmm is the non-linear power for all halos.
        bg1 and bg2 are the linear galaxy biases in each bin.
        """
        Pgg =  self.Pmms[zindex] *bg1*bg2
        if not(self.sigz is None):
            Pgg = Pgg[...,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1))**2.)
        return Pgg

    def lPgv(self,zindex,bg,bv=1):
        """The long-wavelength power spectrum of gxg.
        Here Pmm is the non-linear power for all halos.
        bg1 and bg2 are the linear galaxy biases in each bin.
        """
        Pgv =  self.Pmms[zindex] *bg*bv * (self.d2vs[zindex])
        if not(self.sigz is None):
            Pgv = Pgv[...,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1)))
        return Pgv
    

    def ksz_radial_function(self,zindex, gasfrac = 0.9,xe=1, tau=0, params=None):
        return ksz_radial_function(self.zs[zindex],self.pars.ombh2, self.pars.YHe, gasfrac = gasfrac,xe=xe, tau=tau, params=params)

    def Wphoto(self,zindex):
        krs = self.krs
        z = self.zs[zindex]
        H = self.Hphotoz(z)
        return np.exp(-self.sigma_z_func(z)**2.*krs**2./2./H**2.) # (mus,kLs)
    

    def Nvv(self,zindex,Cls):
        chi_star = self.chistars[zindex]
        Fstar = self.ksz_radial_function(zindex)
        return Nvv_core_integral(chi_star,Fstar,self.mu,self.kLs,self.kS,Cls,
                                 self.sPge[zindex],self.sPggtot[zindex],
                                 Pgg_photo_tot=None,errs=False,
                                 robust_term=False,photo=True)

    
def Nvv_core_integral(chi_star,Fstar,mu,kL,kSs,Cls,Pge,Pgg_tot,Pgg_photo_tot=None,errs=False,
                      robust_term=False,photo=True):
    """
    Returns velocity recon noise Nvv as a function of mu,kL
    Uses Pgg, Pge function of mu,kL,kS and integrates out kS

    if errs is True: sets Pge=1, so can be reused for Pge error calc

    Cls is an array for C_tot starting at l=0.
    e.g. C_tot = C_CMB + C_fg + (C_noise/beam**2 )
    """

    if robust_term:
        if photo: print("WARNING: photo_zs were True for an Nvv(robust_term=True) call. Overriding to False.")
        photo = False

    if errs:
        ret_Pge = Pge.copy()
        Pge = 1.

    amu = np.resize(mu,(kL.size,mu.size)).T
    prefact = amu**(-2.) * 2. * np.pi * chi_star**2. / Fstar**2.


    Clkstot = get_interpolated_cls(Cls,chi_star,kSs)
    integrand = _sanitize(kSs * ( Pge**2. / (Pgg_tot * Clkstot)))

    if robust_term:
        assert Pgg_photo_tot is not None
        integrand = _sanitize(integrand * (Pgg_photo_tot/Pgg_tot))

    integral = np.trapz(integrand,kSs)
    Nvv = prefact / integral
    assert np.all(np.isfinite(Nvv))
    if errs:
        return Nvv,ret_Pge
    else:
        return Nvv



def get_ksz_template_signal_snapshot(ells,volume_gpc3,z,ngal_mpc3,bg,fparams=None,params=None,
                            kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                            num_kS_bins=101,num_mu_bins=102,ms=None,mass_function="sheth-torman",
                            mdef='vir',nfw_numeric=False,
                            electron_profile_family='AGN',
                            electron_profile_nxs=None,electron_profile_xmax=None):
    """
    Get C_ell_That_T, the expected cross-correlation between a kSZ template
    and the CMB temperature.
    """

    # Define kSZ object corresponding to fiducial parameters
    fksz = kSZ([z],[volume_gpc3],[ngal_mpc3],
               kL_max=kL_max,num_kL_bins=num_kL_bins,kS_min=kS_min,kS_max=kS_max,
               num_kS_bins=num_kS_bins,num_mu_bins=num_mu_bins,ms=ms,params=fparams,mass_function=mass_function,
               halofit=None,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=False,
               electron_profile_name='e',electron_profile_family=electron_profile_family,
               skip_electron_profile=False,electron_profile_param_override=fparams,
               electron_profile_nxs=electron_profile_nxs,electron_profile_xmax=electron_profile_xmax,
               skip_hod=False,hod_name="g",hod_corr="max",hod_param_override=None)

    # Define kSZ object corresponding to "true" parameters, if specified
    if params is not None:
        pksz = kSZ([z],[volume_gpc3],[ngal_mpc3],
               kL_max=kL_max,num_kL_bins=num_kL_bins,kS_min=kS_min,kS_max=kS_max,
               num_kS_bins=num_kS_bins,num_mu_bins=num_mu_bins,ms=ms,params=params,mass_function=mass_function,
               halofit=None,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=False,
               electron_profile_name='e',electron_profile_family=electron_profile_family,
               skip_electron_profile=False,electron_profile_param_override=params,
               electron_profile_nxs=electron_profile_nxs,electron_profile_xmax=electron_profile_xmax)
    else:
        pksz = fksz
        
    # Get galaxy shot power as 1/nbar
    ngg = Ngg(ngal_mpc3)
    
    # Get P_gg + N_gg and P_ge as a function of k_S, for fiducial parameters
    fsPgg = fksz.sPggs[0] + ngg
    fsPge = fksz.sPges[0]

    # !!!
    # fsPgg = fksz._get_matter_power(fksz.zs[0],fksz.kS,nonlinear=True)[0] * bg**2. + ngg
    # fsPge = fksz._get_matter_power(fksz.zs[0],fksz.kS,nonlinear=True)[0] * bg
    # !!!
    
    # Get P_ge as a function of k_S, for "true" parameters
    psPge = pksz.sPges[0] if params is not None else fsPge
    
    # Get comoving distance to redshift z
    chistar = pksz.results.comoving_radial_distance(z)

    # Get interpolating function for P_ge^fid * P_ge^true / P_gg^{tot,fid}
    iPk = utils.interp(fksz.kS,_sanitize(fsPge * psPge / fsPgg))
    # Get product above at k = ell/chi_* for specified ells
    Pks = np.asarray([iPk(k) for k in ells/chistar])
    
    # Get kSZ radial weight function K(z) for fiducial and "true" parameters,
    # at input z
    fFstar = fksz.ksz_radial_function(zindex=0)
    pFstar = pksz.ksz_radial_function(zindex=0) if params is not None else fFstar
    
    # Get volume in Mpc^3
    V = volume_gpc3 * 1e9
    
    # Compute prefactor: K^fid K^true V^{1/3} / (6 \pi^2 \chi_*^2)
    pref = fFstar * pFstar * (V**(1/3.)) / 6 / np.pi**2 / chistar**2

    # Compute P_gg + N_gg and P_gv for fiducial and "true" parameters, as functions of k_L
    flPgg = fksz.lPgg(zindex=0,bg1=bg,bg2=bg)[0,:] + ngg
    flPgv = fksz.lPgv(zindex=0,bg=bg)[0,:]
    plPgv = pksz.lPgv(zindex=0,bg=bg)[0,:] if params is not None else flPgv
    
    # Construct integrand (without prefactor) as function of tabulated k_L values,
    # and integrate
    kls = fksz.kLs
    integrand = _sanitize((kls**2.)*(flPgv*plPgv)/flPgg)
    vrec = np.trapz(integrand,kls)
    
    # Return full integral as function of input ell values, and other info
    return pref * Pks * vrec, fksz, pksz


def get_interpolated_cls(Cls,chistar,kss):
    ls = np.arange(Cls.size)
    Cls[ls<2] = 0
    def _Cls(ell):
        if ell<=ls[-1]:
            return Cls[int(ell)]
        else:
            return np.inf
    # TODO: vectorize
    return np.array([_Cls(chistar*k) for k in kss])



def get_ksz_snr(volume_gpc3,z,ngal_mpc3,Cls,bg=None,params=None,
                kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                num_kS_bins=101,num_mu_bins=102,ms=None,mass_function="sheth-torman",
                mdef='vir',nfw_numeric=False,
                electron_profile_family='AGN',
                electron_profile_nxs=None,electron_profile_xmax=None,sigz=None):

    """
    SNR = \int 2pi k_L^2 dk_L dmu (1/(2pi)^3) Pgv(mu,kL)^2 / Pggtot(mu,kL)^2 / Nvv(mu,kL)
    """
    fksz = kSZ([z],[volume_gpc3],[ngal_mpc3],
               kL_max=kL_max,num_kL_bins=num_kL_bins,kS_min=kS_min,kS_max=kS_max,
               num_kS_bins=num_kS_bins,num_mu_bins=num_mu_bins,ms=ms,params=params,mass_function=mass_function,
               halofit=None,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=False,
               electron_profile_name='e',electron_profile_family=electron_profile_family,
               skip_electron_profile=False,electron_profile_param_override=params,
               electron_profile_nxs=electron_profile_nxs,electron_profile_xmax=electron_profile_xmax,
               skip_hod=False,hod_name="g",hod_corr="max",hod_param_override=None,sigz=sigz)
    V = volume_gpc3 * 1e9
    ngg = Ngg(ngal_mpc3)
    Nvv = fksz.Nvv(0,Cls)
    if bg is None:
        bg = fksz.bgs[0]
    lPgg = fksz.lPgg(zindex=0,bg1=bg,bg2=bg)
    lPgv = fksz.lPgv(zindex=0,bg=bg)
    if sigz is not None:
        lPgg = lPgg[...,0]
        lPgv = lPgv[...,0]
    ltPgg = lPgg + ngg
    kls = fksz.kLs
    integrand = _sanitize((kls**2.)*(lPgv**2)/ltPgg/Nvv)
    result = np.trapz(integrand,kls)
    snr2 = np.trapz(result,fksz.mu) / (2.*np.pi)**2.
    return np.sqrt(V*snr2),fksz


def get_ksz_auto_signal_mafry(ells,volume_gpc3,zs,ngal_mpc3,bg,params=None,
                        k_max = 100., num_k_bins = 200,
#                             kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                            num_kS_bins=101,num_mu_bins=102,ms=None,mass_function="sheth-torman",
                            mdef='vir',nfw_numeric=False,
                            electron_profile_family='AGN',
                            electron_profile_nxs=None,electron_profile_xmax=None,
                       verbose=False, pksz_in=None, save_debug_files=False):
    """
    Get C_ell_^kSZ, the CMB kSZ auto power, as described by Eq. (B28) and the following
    (unnumbered) equation in Smith et al:
    
        C_\ell = \frac{1}{2} (\frac{\sigma_T \bar{n}_{e,0}}{c})^2
                    \int \frac{d\chi}{\chi^4 a(\chi)^2} 
                    \exp(-2\tau) P_{q_\perp}(k=\ell/\chi, \chi)
                    
        P_{q_\perp}(k,z) = \dot{a}^2 f^2 \int \frac{d^3 k'}{(2\pi)^3} 
                            P_{ee}^{NL}(|\vec{k}-\vec{k}'|,z)
                            P_{\delta\delta}^{lin}(k',z)
                            \frac{k(k-2k'\mu')(1-\mu'^2)}{k'^2(k^2+k'^2-2kk'\mu}

    C_ell^kSZ is returned in uK^2.
    """

    # Make sure input redshifts are sorted
    zs = np.sort(np.asarray(zs))
    
    # Make arrays for volume and galaxy number density, for feeding to kSZ object
    volumes_gpc3 = volume_gpc3 * np.ones_like(zs)
    ngals_mpc3 = ngal_mpc3 * np.ones_like(zs)
    
    # Define kSZ object, if not specified as input
    if pksz_in is not None:
        pksz = pksz_in
    else:
        if verbose: print('Initializing kSZ objects')
        pksz = kSZ(
            zs, 
            volumes_gpc3, 
            ngals_mpc3,
            kL_max=k_max,                 # Same k_max for kL and kS
            num_kL_bins=num_k_bins, 
            kS_min=get_kmin(volume_gpc3), # Same k_min for kL and kS
            kS_max=k_max, 
            num_kS_bins=num_k_bins,
            num_mu_bins=num_mu_bins,
            ms=ms, 
            params=params, 
            mass_function=mass_function,
            halofit=None, 
            mdef=mdef, 
            nfw_numeric=nfw_numeric, 
            skip_nfw=False,
            electron_profile_name='e', 
            electron_profile_family=electron_profile_family,
            skip_electron_profile=False, 
            electron_profile_param_override=params,
            electron_profile_nxs=electron_profile_nxs,
            electron_profile_xmax=electron_profile_xmax,
            skip_hod=True,                # Skip HOD computation to save time
            verbose=verbose
        )
        
    # Get ks and mus that P_{q_perp} integrand is evaluated at
    ks = pksz.kS
    mus = pksz.mu
        
    # Get P_ee as a function of z and k (packed as [z,k])
    sPee = pksz.get_power('e',name2='e',verbose=False)
    
    # Get P_linear as a function of z and k (packed as [z,k])
    Pmm = np.asarray(pksz.Pmms)
    Pmm = Pmm[:,0,:]

    # Make meshes of mu and k, packed as [k,mu]
    mu_mesh,k_mesh = np.meshgrid(mus, ks)
    
    # Define function that returns fraction in P_{q_perp} integrand,
    # and also |\vec{k} - \vec{k}'|
    def Pqperp_igr_poly(k,kp,mu,z):
        
        frac = k * (k - 2*kp*mu) * (1 - mu**2)
        frac /= (kp**2 * (kp**2 + k**2 - 2*k*kp*mu))
        
        kmkp = np.sqrt(kp**2 + k**2 - 2*k*kp*mu)
        
        igr = kp**2 * frac
        
        return igr, kmkp
    
    # Compute P_{q_perp} values on grid in k,z
    if verbose: print('Computing P_{q_perp} on grid in k,z')
    Pqperp = np.zeros((ks.shape[0], zs.shape[0]))
    for iz,z in enumerate(zs):
    
        # Define interpolating functions for P_ee and P_mm at this z
        isPee = interp1d(ks, sPee[iz], bounds_error=False, fill_value=0.)
        iPmm = interp1d(ks, Pmm[iz], bounds_error=False, fill_value=0.)
    
        for ik,k in enumerate(ks):
        
            # Compute \dot{a} f = a H f at this redshift
            adotf = pksz.adotf[iz][0]
            
            if True:
                # Get P_{q_perp} integrand on [k,mu] mesh
                Pqperp_igr_mesh, kmkp_mesh = Pqperp_igr_poly(k,k_mesh,mu_mesh,z)
                Pee_mesh = isPee(kmkp_mesh.flatten()).reshape(kmkp_mesh.shape)
                Pmm_mesh = iPmm(k_mesh.flatten()).reshape(kmkp_mesh.shape)
                Pqperp_igr_mesh *= Pmm_mesh * Pee_mesh

                # If desired, save some meshes to disk for debugging
                if save_debug_files and ik == 0 and iz == 0:
                    np.savetxt('debug_files/kmkp_mesh.dat', kmkp_mesh)
                    np.savetxt('debug_files/pee_mesh.dat', Pee_mesh)
                    np.savetxt('debug_files/pqperp_igr_mesh.dat', Pqperp_igr_mesh)

                # Integrate integrand mesh along k axis with trapezoid rule
                integral = np.trapz(np.nan_to_num(Pqperp_igr_mesh), ks, axis=0) 
                
                # If desired, save partial integral to disk for debugging
                if save_debug_files and ik == 0 and iz == 0:
                    np.savetxt('debug_files/pqperp_igr_mu.dat', np.transpose([mus,integral]))

                # Integrate along mu axis with trapezoid rule
                integral = np.trapz(integral, mus)

            else:
                # Could also do double integral with dblquad, but in practice
                # this takes forever
                integral = dblquad(lambda kp,mu: Pqperp_igr(k,kp,mu,z),
                                        -1, 1, ks[0], ks[-1])[0]
            
            # Include prefactors for integral
            Pqperp[ik,iz] = adotf**2 * (2*np.pi)**-2 * integral
        
    # Make 2d interpolating function for P_{q_\perp}, with arguments z,k.
    # Resulting interpolating function automatically sorts arguments if
    # arrays are fed in, but we'll only call iPqperp with one (z,k) pair
    # at a time, so we'll be fine.
    iPqperp = interp2d(zs, ks, Pqperp)
    
    # Compute C_ell integral at each ell
    if verbose: print('Computing C_ell')
    cl = np.zeros(ells.shape[0])
    for iell,ell in enumerate(ells):
        
        # Set chi_min based on k=30Mpc^-1, and chi_max from max redshift
        chi_min = ell/30.
        chi_max = pksz.results.comoving_radial_distance(zs[-1])
        chi_int = np.geomspace(chi_min, chi_max, 100)
        k_int = ell/chi_int
        z_int = pksz.results.redshift_at_comoving_radial_distance(chi_int)
        
        # Get integrand evaluated at z,k corresponding to Limber integral
        integrand = np.zeros(k_int.shape[0])
        for ki,k in enumerate(k_int):
            integrand[ki] = iPqperp(z_int[ki], k)
        integrand /= chi_int**2 * 1/(1+z_int)**4
        
        # Include prefactors
        ne0 = ne0_shaw(pksz.pars.ombh2, pksz.pars.YHe)
        integrand *= 0.5 
        # Units: (m^2 * m^-3 * Mpc^-1 m^1)^2
        integrand *= (constants['thompson_SI'] \
                      * ne0 \
                      * 1/constants['meter_to_megaparsec'] )**2
        integrand *= (pksz.pars.TCMB * 1e6)**2
        
        if True:
            # Do C_ell integral via trapezoid rule
            cl[iell] = np.trapz(integrand, chi_int)
        else:
            # Doing integral of an interpolating function gives
            # equivalent results at the precision we care about
            igr_interp = interp1d(chi_int, integrand)
            cl[iell] = quad(igr_interp, chi_int[0], chi_int[-1])[0]

    # If desired, save some files for debugging
    if save_debug_files:
        np.savetxt('debug_files/zs.dat', zs)
        np.savetxt('debug_files/k_invMpc.dat', ks)
        np.savetxt('debug_files/pee.dat', sPee)
        np.savetxt('debug_files/pmm.dat', Pmm)
        np.savetxt('debug_files/pqperp.dat', Pqperp)
        
    # Return kSZ object (in case we want to use it later) and C_ell array
    return pksz, cl


def get_ksz_auto_squeezed(ells,volume_gpc3,zs,ngals_mpc3,bgs,params=None,
                        k_max = 100., num_k_bins = 200,
#                             kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                            num_kS_bins=101,num_mu_bins=102,ms=None,mass_function="sheth-torman",
                            mdef='vir',nfw_numeric=False,
                            electron_profile_family='AGN',
                            electron_profile_nxs=None,electron_profile_xmax=None,
                       verbose=False, pksz_in=None, save_debug_files=False,
                                template=False,
                         ngals_mpc3_for_v=None):
    """
    Get C_ell_^kSZ, the CMB kSZ auto power, as described by the squeezed limit
    in Ma & Fry, with some altered notation:
    
        C_\ell = \int \frac{d\chi}{\chi^2 H_0^2} \tilde{K}(z[\chi])^2
                 P_{q_r}(k=\ell/\chi, \chi)
    
        \tilde{K}(z) = T_{CMB} \bar{n}_{e,0} \sigma_T (1+z)^2 \exp(-\tau(z))
        
        P_{q_r}(k,z) = \frac{1}{6\pi^2} \int dk' (k')^2 P_{vv}(k',z) P_{ee}(k,z)
    
    C_ell^kSZ is returned in uK^2.
    """

    # Define empty dict for storing spectra
    spec_dict = {}
    
    # Widen search range for setting lower mass threshold from nbar
    if params is None:
        params = default_params
    params['hod_bisection_search_min_log10mthresh'] = 1
    
    # Make sure input redshifts are sorted
    zs = np.sort(np.asarray(zs))
    
    # Make arrays for volume, for feeding to kSZ object
    volumes_gpc3 = volume_gpc3 * np.ones_like(zs)
    
    if ngals_mpc3_for_v is None:
        ngals_mpc3_for_v = ngals_mpc3
    
    # If not computing for a kSZ template, skip HOD computation to save time
    if template:
        skip_hod = False
    else:
        skip_hod = True
    
    # Define kSZ object, if not specified as input
    if pksz_in is not None:
        pksz = pksz_in
    else:
        if verbose: print('Initializing kSZ objects')
        pksz = kSZ(
            zs, 
            volumes_gpc3, 
            ngals_mpc3,
            kL_max=k_max,                 # Same k_max for kL and kS
            num_kL_bins=num_k_bins, 
            kS_min=get_kmin(volume_gpc3), # Same k_min for kL and kS
            kS_max=k_max, 
            num_kS_bins=num_k_bins,
            num_mu_bins=num_mu_bins,
            ms=ms, 
            params=params, 
            mass_function=mass_function,
            halofit=None, 
            mdef=mdef, 
            nfw_numeric=nfw_numeric, 
            skip_nfw=False,
            electron_profile_name='e', 
            electron_profile_family=electron_profile_family,
            skip_electron_profile=False, 
            electron_profile_param_override=params,
            electron_profile_nxs=electron_profile_nxs,
            electron_profile_xmax=electron_profile_xmax,
            skip_hod=skip_hod,
            verbose=verbose,
            b1=bgs,
            b2=bgs
        )
        
    # Get ks and that P_{q_perp} integrand is evaluated at
    ks = pksz.kS
    spec_dict['ks'] = ks
        
    # If not computing for a kSZ template, get P_ee and P_vv
    # on grids in z and k
    if not template:
        
        # Get P_ee as a function of z and k (packed as [z,k])
        sPee = pksz.get_power('e',name2='e',verbose=False)

        # Get P_vv as a function of z and k (packed as [z,k]),
        # by getting it for each z individually
        lPvv0 = pksz.lPvv(zindex=0)[0,:]
        lPvv = np.zeros((len(zs), lPvv0.shape[0]))
        lPvv[0,:] = lPvv0
        for zi in range(1, len(zs)):
            lPvv[zi,:] = pksz.lPvv(zindex=zi)[0,:]
            
        spec_dict['sPee'] = sPee
        spec_dict['lPvv'] = lPvv
            
    # If computing for a kSZ template, get P_gg^total, P_ge, and
    # P_vv on grids in z and k
    else:
        
        # Get small-scale P_gg and P_ee as functions of z and k 
        # (packed as [z,k])
        sPgg_for_e = pksz.sPggs
        sPgg_for_v = sPgg_for_e.copy()
        for zi in range(zs.shape[0]):
            sPgg_for_e[zi] += 1/ngals_mpc3[zi]
            sPgg_for_v[zi] += 1/ngals_mpc3_for_v[zi]
        sPge = pksz.sPges
        
        # Get large-scale P_vv as a function of z and k (packed as [z,k]),
        # by getting it for each z individually
        lPgv0 = pksz.lPgv(zindex=0,bg=bgs[0])[0,:]
        lPgv = np.zeros((len(zs), lPgv0.shape[0]))
        lPgv[0,:] = lPgv0
        for zi in range(1, len(zs)):
            lPgv[zi,:] = pksz.lPgv(zindex=zi,bg=bgs[zi])[0,:]
            
        # Same for large-scale Pgg
        lPgg0 = pksz.lPgg(0,bgs[0],bgs[0])[0,:]
        lPgg = np.zeros((len(zs), lPgg0.shape[0]))
        lPgg[0,:] = lPgg0
        for zi in range(zs.shape[0]):
            lPgg[zi,:] = pksz.lPgg(zi,bgs[zi],bgs[zi])[0,:]
            lPgg[zi] += 1/ngals_mpc3_for_v[zi]
            
        spec_dict['sPgg'] = sPgg_for_e
        spec_dict['sPge'] = sPge
        spec_dict['lPgv'] = lPgv
        spec_dict['lPgg'] = lPgg
        
    # Compute P_{q_r} values on grid in k,z
    if verbose: print('Computing P_{q_r} on grid in k,z')
    Pqr = np.zeros((ks.shape[0], zs.shape[0]))
    for zi,z in enumerate(zs):
        
        # Get P_gv^2 / P_gg^total or P_vv, and integrate in k
        kls = pksz.kLs
        if template:
#             integrand = _sanitize((kls**2.)*lPgv[zi]**2/lPgg[zi])
            integrand = _sanitize((kls**2.)*lPgv[zi]**2/sPgg_for_v[zi])
        else:
            integrand = _sanitize((kls**2.)*lPvv[zi])
        vint = np.trapz(integrand,kls)
        
        # Get P_ge^2 / P_gg^total or P_ee
        if template:
            Pqr[:,zi] = sPge[zi]**2 / sPgg_for_e[zi]
        else:
            Pqr[:,zi] = sPee[zi]
        
        # Multiply by numerical prefactor and integral from above
        Pqr[:,zi] *= (6*np.pi**2)**-1 * vint
    
    # Make 2d interpolating function for P_{q_r}, with arguments z,k.
    # The resulting interpolating function automatically sorts arguments if
    # arrays are fed in, but we'll only call iPqperp with one (z,k) pair
    # at a time, so we'll be fine.
    iPqr = interp2d(zs, ks, Pqr)
    
    # Compute C_ell integral at each ell
    if verbose: print('Computing C_ell')
    cl = np.zeros(ells.shape[0])
    for iell,ell in enumerate(ells):
        
        # Set chi_min based on k=30Mpc^-1, and chi_max from max redshift
        chi_min = ell/30.
        chi_max = pksz.results.comoving_radial_distance(zs[-1])
        chi_int = np.geomspace(chi_min, chi_max, 100)
        k_int = ell/chi_int
        z_int = pksz.results.redshift_at_comoving_radial_distance(chi_int)
        
        # Get integrand evaluated at z,k corresponding to Limber integral
        integrand = np.zeros(k_int.shape[0])
        for ki,k in enumerate(k_int):
            integrand[ki] = iPqr(z_int[ki], k)
        integrand /= chi_int**2
        integrand *= (1+z_int)**4
        
        # Include prefactors
        ne0 = ne0_shaw(pksz.pars.ombh2, pksz.pars.YHe)
        # Units: (m^2 * m^-3 * Mpc^-1 m^1)^2
        integrand *= (constants['thompson_SI'] \
                      * ne0 \
                      * 1/constants['meter_to_megaparsec'] )**2
        integrand *= (pksz.pars.TCMB * 1e6)**2
        
        if True:
            # Do C_ell integral via trapezoid rule
            cl[iell] = np.trapz(integrand, chi_int)
        else:
            # Doing integral of an interpolating function gives
            # equivalent results at the precision we care about
            igr_interp = interp1d(chi_int, integrand)
            cl[iell] = quad(igr_interp, chi_int[0], chi_int[-1])[0]

    # If desired, save some files for debugging
    if save_debug_files and not template:
        np.savetxt('debug_files/zs.dat', zs)
        np.savetxt('debug_files/k_invMpc.dat', ks)
        np.savetxt('debug_files/pee.dat', sPee)
        np.savetxt('debug_files/pvv.dat', lPvv)
        np.savetxt('debug_files/pqr.dat', Pqr)
        
    # Return kSZ object (in case we want to use it later), C_ell array,
    # and dict of spectra used
    return pksz, cl, spec_dict


def Nvv(z,vol_gpc3,ngals_mpc3,Cl_total,sigz=None,
        kL_max=0.1,num_kL_bins=100,
        kS_min=0.1,
        kS_max=10.0,
        num_kS_bins=101,
        num_mu_bins=102):
    """
    Get the reconstruction noise N_vv on the radial velocity field
    as reconstructed using kSZ tomography using a CMB survey
    and a galaxy survey.

    This function provides a convenience wrapper for very basic usage.
    More advanced usage (e.g. photo-zs) involves using the 'kSZ' class and/or
    the 'Nvv_core_integral' function.

    Parameters
    ----------
    
    z : float
        The central redshift of the galaxy survey's "box"
    vol_gpc3 : float
        The overlap volume of the galaxy survey box with the CMB survey in Gpc^3
    ngals_mpc3 : float
        The comoving number density of the galaxy survey in Mpc^{-3}
    Cl_total : (nells,) float
        The total power spectrum of the CMB survey including lensed 
        CMB, kSZ, beam-deconvolved noise and foregrounds
    sigz : float, optional
        The Gaussian scatter for photometric redshifts. The assumed scatter
        will be sigz x (1+z).

    Returns
    -------

    mus : (nmus,) float
        Angle to the line-of-sight k.n from -1 to 1, corresponding to 
        the first dimension of the returned N_vv
    kLs : (nkls,) float
        An array of long-wavelength wavenumbers in Mpc^{-1}, corresponding
        to the second dimension of the returned N_vv
    N_vv : (nmus,nkls) float
        A 2d array containing the reconstruction noise power as a function
        of angle to the line-of-sight along the first dimension
        and long-wavelength wavenumbers along the second dimension
    

    """
    zs = [z]
    volumes_gpc3 = [vol_gpc3]
    ngals_mpc3 = [ngals_mpc3]
    hksz = kSZ(zs,volumes_gpc3,ngals_mpc3,
                   kL_max=kL_max,num_kL_bins=num_kL_bins,
                   kS_min=kS_min,
                   kS_max=kS_max,
                   num_kS_bins=num_kS_bins,
                   num_mu_bins=num_mu_bins,sigz=sigz)
    return hksz.mu,hksz.kLs,hksz.Nvv(0,Cl_total)



def get_ksz_snr_survey(zs,dndz,zedges,Cls,fsky,Ngals,bs=None,sigz=None):
    """

    Get the total kSZ SNR from survey specifications.
    Provide the redshift distribution through (zs,dndz)
    Divide into "boxes" by specifying the redshift bin edges.
    This allows the survey overlap volumes in each bin
    to be computed from the overlap sky fraction fsky.
    Provide the total CMB+foreground+noise power in Cls.
    Provide sigma_z/(1+z) if this is a photometric survey.
    Provide the total number of galaxies in the overlap
    region in Ngals.
    Provide the galaxy biases in each bin in bs.

    """

    from astropy.cosmology import WMAP9 as cosmo
    
    nbins = len(zedges) - 1
    if not(bs is None):
        if len(bs)!=nbins: raise Exception
    vols_gpc3 = []
    ngals_mpc3 = []
    snrs = []
    zcents = []
    tdndz = np.trapz(dndz,zs)
    bgs = []
    for i in range(nbins):
        # Calculate bin volumes in Gpc3
        zmin = zedges[i]
        zmax = zedges[i+1]
        zcent = (zmax+zmin)/2.
        chimin = cosmo.comoving_distance(zmin).value
        chimax = cosmo.comoving_distance(zmax).value
        vols_gpc3.append( fsky * (4./3.) * np.pi * (chimax**3. - chimin**3.) / 1e9)
        
        # Calculate comoving number densities
        sel = np.logical_and(zs>zmin,zs<=zmax)
        fracz = np.trapz(dndz[sel],zs[sel]) / tdndz
        Ng = Ngals * fracz
        ngals_mpc3.append( Ng / (vols_gpc3[i]*1e9) )

        # Calculate SNRs
        snr,fksz = get_ksz_snr(vols_gpc3[i],zcent,ngals_mpc3[i],Cls,bs[i] if not(bs is None) else None,sigz=sigz)
        bgs.append(fksz.bgs[0])
        snrs.append(snr)
        zcents.append(zcent)

    snrs = np.asarray(snrs)
    totsnr = np.sqrt(np.sum(snrs**2.))

    return vols_gpc3,ngals_mpc3,zcents,bgs,snrs,totsnr
        
