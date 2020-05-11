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

def pge_err(pgv_int,kstar,zstar,volume,kss,ks_bin_edges,pggtot,cltot):

    """
    pgv_int: \int dkl kl^2 Pgv^2/Pggtot
    kstar: kSZ radial weight function at zstar
    zstar: mean redshift of galaxy survey
    volume: volume in gpc3
    kss: short wavelength k on which pggtot and cltot are defined
    
    
    """
    pass


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
                b1=None,b2=None):

        if ms is None: ms = np.geomspace(defaults['min_mass'],defaults['max_mass'],defaults['num_mass'])
        volumes_gpc3 = np.atleast_1d(volumes_gpc3)
        assert len(zs)==len(volumes_gpc3)==len(ngals_mpc3)
        ngals_mpc3 = np.asarray(ngals_mpc3)
        ks = np.geomspace(kS_min,kS_max,num_kS_bins)
        self.mu = np.linspace(-1.,1.,num_mu_bins)
        self.kLs = []
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
        if not skip_hod:
            self.sPggs = self.get_power(hod_name,name2=hod_name,verbose=False,b1=b1,b2=b1) 
            self.sPges = self.get_power(hod_name,name2=electron_profile_name,verbose=False,b1=b1) 
            
        # SF: Mat's code used a different k_min for each z if the volumes are different, but
        # for speed, I've changed it to use a single k_min for all z - this lets us evaluate
        # all P_matter values we need in one shot
        if np.max(volumes_gpc3) != np.min(volumes_gpc3):
            warnings.warn('Using equal k_min at each z, despite different volumes at each z')
 
        # Define log-spaced array of k values, and get P_linear and f(z)
        # on grid in z and k
        kL = np.geomspace(get_kmin(np.max(volumes_gpc3)),kL_max,num_kL_bins)
        p = self._get_matter_power(self.zs,kL,nonlinear=False)
        growth = self.results.get_redshift_evolution(
            kL, 
            self.zs, 
            ['growth']
        )[:,:,0]
    
        for zindex,volume_gpc3 in enumerate(volumes_gpc3):
            self.kLs.append(kL.copy())
            self.Pmms.append(np.resize(p[zindex],(self.mu.size,self.kLs[zindex].size)))
            self.fs.append(growth[:,zindex])
            z = self.zs[zindex]
            a = 1./(1.+z)
            H = self.results.h_of_z(z)
            self.d2vs.append(  self.fs[zindex]*a*H / kL )
            self.adotf.append(self.fs[zindex]*a*H)



######## MAT'S VERSION - slow for many z's 
#         for zindex,volume_gpc3 in enumerate(volumes_gpc3):
#             if verbose: print('Getting properties for zindex %d' % zindex)
#             kL = np.geomspace(get_kmin(volume_gpc3),kL_max,num_kL_bins)
#             self.kLs.append(kL.copy())
#             p = self._get_matter_power(self.zs[zindex],self.kLs[zindex],nonlinear=False)
#             self.Pmms.append(np.resize(p,(self.mu.size,kL.size)))
#             self.fs.append( self.results.get_redshift_evolution(self.kLs[zindex], self.zs[zindex], ['growth']).ravel() )
#             z = self.zs[zindex]
#             a = 1./(1.+z)
#             H = self.results.h_of_z(z)
#             self.d2vs.append(  self.fs[zindex]*a*H / kL )
#             self.adotf.append(self.fs[zindex]*a*H)
########    

        # self.D = self.cc.D_growth(self.cc.z2a(self.z),"camb_anorm")
        # self.T = lambda x: self.cc.transfer(x)
        # self.alpha_func = lambda x: (2. * x**2. * self.T(x)) / (3.* self.cc.Omega_m * self.cc.results.h_of_z(0)**2.) * self.D
        
        # self.chi_star = self.cc.results.comoving_radial_distance(redshift)
        
        # # kr = mu * kL ; this is an array of krs of shape (num_mus,num_kLs)
        # self.krs = self.mus.reshape((self.mus.size,1)) * self.kLs.reshape((1,self.kLs.size))
            
    
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
        return Pgg

    def lPgv(self,zindex,bg,bv=1):
        """The long-wavelength power spectrum of gxg.
        Here Pmm is the non-linear power for all halos.
        bg1 and bg2 are the linear galaxy biases in each bin.
        """
        Pgv =  self.Pmms[zindex] *bg*bv * (self.d2vs[zindex])
        return Pgv
    

    def ksz_radial_function(self,zindex, gasfrac = 0.9,xe=1, tau=0, params=None):
        return ksz_radial_function(self.zs[zindex],self.pars.ombh2, self.pars.YHe, gasfrac = gasfrac,xe=xe, tau=tau, params=params)

    def Nvv(self,Cls):
        pass

    
def Nvv_core_integral(chi_star,Fstar,mu,kL,ngalMpc3,kSs,Cls,Pge,Pgg,Pgg_photo=None,errs=False,
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
    Pgg_tot = Pgg + Ngg(ngalMpc3)

    ls = np.arange(Cls.size)
    Cls[ls<2] = 0
    def _Cls(ell):
        if ell<=ls[-1]:
            return Cls[ell]
        else:
            return np.inf

    ClksTot = np.array([_Cls(chi_star*k) for k in kSs])
    integrand = _sanitize(kSs * ( Pge**2. / (Pgg_tot * ClksTot)))

    if robust_term:
        assert Pgg_photo is not None
        Pgg_photo_tot = Pgg_photo + Ngg(ngalMpc3)
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
    kls = fksz.kLs[0]
    integrand = _sanitize((kls**2.)*(flPgv*plPgv)/flPgg)
    vrec = np.trapz(integrand,kls)
    
    # Return full integral as function of input ell values, and other info
    return pref * Pks * vrec, fksz, pksz





def get_ksz_snr(volume_gpc3,z,ngal_mpc3,bg,Cls,params=None,
                kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                num_kS_bins=101,num_mu_bins=102,ms=None,mass_function="sheth-torman",
                mdef='vir',nfw_numeric=False,
                electron_profile_family='AGN',
                electron_profile_nxs=None,electron_profile_xmax=None):


    fksz = kSZ([z],[volume_gpc3],[ngal_mpc3],
               kL_max=kL_max,num_kL_bins=num_kL_bins,kS_min=kS_min,kS_max=kS_max,
               num_kS_bins=num_kS_bins,num_mu_bins=num_mu_bins,ms=ms,params=params,mass_function=mass_function,
               halofit=None,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=False,
               electron_profile_name='e',electron_profile_family=electron_profile_family,
               skip_electron_profile=False,electron_profile_param_override=params,
               electron_profile_nxs=electron_profile_nxs,electron_profile_xmax=electron_profile_xmax,
               skip_hod=False,hod_name="g",hod_corr="max",hod_param_override=None)
    
    Fstar = fksz.ksz_radial_function(zindex=0)
    V = volume_gpc3 * 1e9
    ngg = Ngg(ngal_mpc3)
    
    lPgg = fksz.lPgg(zindex=0,bg1=bg,bg2=bg)[0,:] + ngg
    lPgv = fksz.lPgv(zindex=0,bg=bg)[0,:]
    kls = fksz.kLs[0]
    integrand = _sanitize((kls**2.)*(lPgv**2)/lPgg)
    vrec = np.trapz(integrand,kls)


    sPgg = fksz.sPggs[0] + ngg
    sPge = fksz.sPges[0]
    chistar = fksz.results.comoving_radial_distance(z)

    
    
    kss = fksz.kS
    ls = np.arange(Cls.size)
    Cls[ls<2] = 0
    def _Cls(ell):
        if ell<=ls[-1]:
            return Cls[int(ell)]
        else:
            return np.inf
    Clkstot = np.array([_Cls(chistar*k) for k in kss])
    integrand = _sanitize(kss * sPge**2. / sPgg / Clkstot)
    ksint = np.trapz(integrand,kss)

    return np.sqrt(V*Fstar**2. * vrec * ksint / 12 / np.pi**3 / chistar**2.),fksz


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


def get_ksz_auto_squeezed(ells,volume_gpc3,zs,ngal_mpc3,bgs,params=None,
                        k_max = 100., num_k_bins = 200,
#                             kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                            num_kS_bins=101,num_mu_bins=102,ms=None,mass_function="sheth-torman",
                            mdef='vir',nfw_numeric=False,
                            electron_profile_family='AGN',
                            electron_profile_nxs=None,electron_profile_xmax=None,
                       verbose=False, pksz_in=None, save_debug_files=False,
                                template=False):
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
    
    # Make arrays for volume and galaxy number density, for feeding to kSZ object
    volumes_gpc3 = volume_gpc3 * np.ones_like(zs)
    ngals_mpc3 = ngal_mpc3 * np.ones_like(zs)
    
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
        sPgg = pksz.sPggs
        for zi in range(zs.shape[0]):
            sPgg[zi] += 1/ngals_mpc3[zi]
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
            lPgg[zi] += 1/ngals_mpc3[zi]
            
        spec_dict['sPgg'] = sPgg
        spec_dict['sPge'] = sPge
        spec_dict['lPgv'] = lPgv
        spec_dict['lPgg'] = lPgg
        
    # Compute P_{q_r} values on grid in k,z
    if verbose: print('Computing P_{q_r} on grid in k,z')
    Pqr = np.zeros((ks.shape[0], zs.shape[0]))
    for zi,z in enumerate(zs):
        
        # Get P_gv^2 / P_gg^total or P_vv, and integrate in k
        kls = pksz.kLs[0]
        if template:
#             integrand = _sanitize((kls**2.)*lPgv[zi]**2/lPgg[zi])
             integrand = _sanitize((kls**2.)*lPgv[zi]**2/sPgg[zi])
        else:
            integrand = _sanitize((kls**2.)*lPvv[zi])
        vint = np.trapz(integrand,kls)
        
        # Get P_ge^2 / P_gg^total or P_ee
        if template:
            Pqr[:,zi] = sPge[zi]**2 / sPgg[zi]
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
#     return pksz

