"""
Halo model for ksz

We use linear matter power for k<0.1 Mpc-1 used in
calculations of large-scale Pgv, Pvv and Pgg.

We use the halo model for k>0.1 Mpc-1 used in calculations
of small-scale Pge, Pee and Pgg.

"""

from .params import default_params
from .hmvec import HaloModel
from . import utils
import numpy as np

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
                 mthreshs_override=None):

        if ms is None: ms = np.geomspace(defaults['min_mass'],defaults['max_mass'],defaults['num_mass'])
        volumes_gpc3 = np.atleast_1d(volumes_gpc3)
        assert len(zs)==len(volumes_gpc3)==len(ngals_mpc3)
        ngals_mpc3 = np.asarray(ngals_mpc3)
        ks = np.geomspace(kS_min,kS_max,num_kS_bins)
        self.mu = np.linspace(-1.,1.,num_mu_bins)
        self.kLs = []
        HaloModel.__init__(self,zs,ks,ms=ms,params=params,mass_function=mass_function,
                 halofit=halofit,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=skip_nfw)
        self.kS = self.ks
        if not(skip_electron_profile):
            self.add_battaglia_profile(name=electron_profile_name,
                                            family=electron_profile_family,
                                            param_override=electron_profile_param_override,
                                            nxs=electron_profile_nxs,
                                            xmax=electron_profile_xmax,ignore_existing=False)

        if not(skip_hod):
            self.add_hod(hod_name,mthresh=mthreshs_override,ngal=ngals_mpc3,corr=hod_corr,
                         satellite_profile_name='nfw',
                         central_profile_name=None,ignore_existing=False,param_override=hod_param_override)
            
        self.Pmms = []
        self.fs = []
        self.d2vs = []
        self.sPggs = self.get_power(hod_name,name2=hod_name,verbose=False) 
        self.sPges = self.get_power(hod_name,name2=electron_profile_name,verbose=False) 
        for zindex,volume_gpc3 in enumerate(volumes_gpc3):
            kL = np.geomspace(get_kmin(volume_gpc3),kL_max,num_kL_bins)
            self.kLs.append(kL.copy())
            p = self._get_matter_power(self.zs[zindex],self.kLs[zindex],nonlinear=False)
            self.Pmms.append(np.resize(p,(self.mu.size,kL.size)))
            self.fs.append( self.results.get_redshift_evolution(self.kLs[zindex], self.zs[zindex], ['growth']).ravel() )
            z = self.zs[zindex]
            a = 1./(1.+z)
            H = self.results.h_of_z(z)
            self.d2vs.append(  self.fs[zindex]*a*H / kL )
        
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



def get_ksz_template_signal(ells,volume_gpc3,z,ngal_mpc3,bg,fparams=None,params=None,
                            kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                            num_kS_bins=101,num_mu_bins=102,ms=None,mass_function="sheth-torman",
                            mdef='vir',nfw_numeric=False,
                            electron_profile_family='AGN',
                            electron_profile_nxs=None,electron_profile_xmax=None):
    """
    Get C_ell_That_T, the expected cross-correlation between a kSZ template
    and the CMB temperature.
    """

    fksz = kSZ([z],[volume_gpc3],[ngal_mpc3],
               kL_max=kL_max,num_kL_bins=num_kL_bins,kS_min=kS_min,kS_max=kS_max,
               num_kS_bins=num_kS_bins,num_mu_bins=num_mu_bins,ms=ms,params=fparams,mass_function=mass_function,
               halofit=None,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=False,
               electron_profile_name='e',electron_profile_family=electron_profile_family,
               skip_electron_profile=False,electron_profile_param_override=fparams,
               electron_profile_nxs=electron_profile_nxs,electron_profile_xmax=electron_profile_xmax,
               skip_hod=False,hod_name="g",hod_corr="max",hod_param_override=None)

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

    ngg = Ngg(ngal_mpc3)
    
    fsPgg = fksz.sPggs[0] + ngg
    fsPge = fksz.sPges[0]

    # !!!
    # fsPgg = fksz._get_matter_power(fksz.zs[0],fksz.kS,nonlinear=True)[0] * bg**2. + ngg
    # fsPge = fksz._get_matter_power(fksz.zs[0],fksz.kS,nonlinear=True)[0] * bg
    # !!!
    
    psPge = pksz.sPges[0] if params is not None else fsPge
    chistar = pksz.results.comoving_radial_distance(z)

    iPk = utils.interp(fksz.kS,_sanitize(fsPge * psPge / fsPgg))
    Pks = np.asarray([iPk(k) for k in ells/chistar])
    
    fFstar = fksz.ksz_radial_function(zindex=0)
    pFstar = pksz.ksz_radial_function(zindex=0) if params is not None else fFstar
    V = volume_gpc3 * 1e9
    pref = fFstar * pFstar * (V**(1/3.)) / 6 / np.pi**2 / chistar**2

    flPgg = fksz.lPgg(zindex=0,bg1=bg,bg2=bg)[0,:] + ngg
    flPgv = fksz.lPgv(zindex=0,bg=bg)[0,:]
    plPgv = pksz.lPgv(zindex=0,bg=bg)[0,:] if params is not None else flPgv
    kls = fksz.kLs[0]
    integrand = _sanitize((kls**2.)*(flPgv*plPgv)/flPgg)
    vrec = np.trapz(integrand,kls)

    return pref * Pks * vrec,fksz,pksz






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
