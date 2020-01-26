"""
Halo model for ksz

We use linear matter power for k<0.1 Mpc-1 used in
calculations of large-scale Pgv, Pvv and Pgg.

We use the halo model for k>0.1 Mpc-1 used in calculations
of small-scale Pge, Pee and Pgg.

"""

from .params import default_params
from .hmvec import HaloModel
import numpy as np

defaults = {'min_mass':1e8, 'max_mass':1e6, 'num_mass':1000}
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
    return TcmbMuK*thompson_SI*ne0*(1.+z)**2./meterToMegaparsec  * xe  *np.exp(-tau)


class kSZ(HaloModel):
    def __init__(self,zs,volumes_gpc3,
                 kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                 num_kS_bins=101,num_mu_bins=102,ms=None,params={},mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False,
                 electron_profile_name='e',electron_profile_family='AGN',
                 skip_electron_profile=False,electron_profile_param_override={},
                 electron_profile_nxs=None,electron_profile_xmax=None):

        if ms is None: ms = np.geomspace(defaults['min_mass'],defaults['max_mass'],defaults['num_mass'])
        self.zs = np.atleast_1d(zs)
        volumes_gpc3 = np.atleast_1d(volumes_gpc3)
        assert len(self.zs)==len(volumes_gpc3)
        self.kS = np.geomspace(kS_min,kS_max,num_kS_bins)
        self.mu = np.linspace(-1.,1.,num_mu_bins)
        self.kLs = []
        HaloModel.__init__(self,self.zs,self.kS,ms=ms,params=params,mass_function=mass_function,
                 halofit=halofit,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=skip_nfw)

        if not(skip_electron_profile):
            self.add_battaglia_profile(name=electron_profile_name,
                                            family=electron_profile_family,
                                            param_override=electron_profile_param_override,
                                            nxs=electron_profile_nxs,
                                            xmax=electron_profile_xmax,ignore_existing=False)
        
        self.Pmms = []
        self.fs = []
        self.d2vs = []
        for zindex,volume_gpc3 in enumerate(volumes_gpc3):
            kL = np.geomspace(get_kmin(volume_gpc3),kL_max,num_kL_bins)
            self.kLs.append(kL.copy())
            p = self._get_matter_power(self.zs[zindex],self.kLs[zindex],nonlinear=False)
            self.Pmms.append(np.resize(p,(self.mu.size,kL.size)))
            self.fs.append( self.results.get_redshift_evolution(self.kLs[zindex], self.zs[zindex], ['growth']).ravel() )
            z = self.zs[aindex]
            a = 1./(1.+z)
            self.d2vs.append(  self.fs[zindex](kL)*a*self.H / kL )
        
        self.D = self.cc.D_growth(self.cc.z2a(self.z),"camb_anorm")
        self.T = lambda x: self.cc.transfer(x)
        self.alpha_func = lambda x: (2. * x**2. * self.T(x)) / (3.* self.cc.Omega_m * self.cc.results.h_of_z(0)**2.) * self.D
        
        self.chi_star = self.cc.results.comoving_radial_distance(redshift)
        
        # kr = mu * kL ; this is an array of krs of shape (num_mus,num_kLs)
        self.krs = self.mus.reshape((self.mus.size,1)) * self.kLs.reshape((1,self.kLs.size))
            

    
    def Pvv(self,zindex,bv1=1,bv2=1):
        """The long-wavelength power spectrum of vxv.
        This is calculated as:
        (faH/kL)**2*Pmm(kL)
        to return a 1D array [mu,kL] with identical
        copies over mus.

        Here Pmm is the non-linear power for all halos.
        bv1 and bv2 are the velocity biases in each bin.
        """
        Pvv = (self.d2v_func(kL))**2. * self.Pmms[zindex] *bv1*bv2
        return Pvv



    
    
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
    integrand = kSs * ( Pge**2. / (Pgg_tot * ClksTot))
    integrand[~np.isfinite(integrand)] = 0

    if robust_term:
        assert Pgg_photo is not None
        Pgg_photo_tot = Pgg_photo + Ngg(ngalMpc3)
        integrand = integrand * (Pgg_photo_tot/Pgg_tot)
        integrand[~np.isfinite(integrand)] = 0

    integral = np.trapz(integrand,kSs)
    Nvv = prefact / integral
    assert np.all(np.isfinite(Nvv))
    if errs:
        return Nvv,ret_Pge
    else:
        return Nvv



def get_ksz_template_signal(ells,volume_gpc3,z,ngal_mpc3,fparams,params=None,
                            kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                            num_kS_bins=101,num_mu_bins=102,ms=None,params={},mass_function="sheth-torman",
                            mdef='vir',nfw_numeric=False,
                            electron_profile_family='AGN',
                            electron_profile_nxs=None,electron_profile_xmax=None):
    
    fksz = kSZ([z],[volume_gpc3],
               kL_max=kL_max,num_kL_bins=num_kL_bins,kS_min=kS_min,kS_max=kS_max,
               num_kS_bins=num_kS_bins,num_mu_bins=num_mu_bins,ms=ms,params=fparams,mass_function=mass_function,
               halofit=None,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=False,
               electron_profile_name='e',electron_profile_family=electron_profile_family,
               skip_electron_profile=False,electron_profile_param_override=fparams,
               electron_profile_nxs=electron_profile_nxs,electron_profile_xmax=electron_profile_xmax)

    if params is not None:
        pksz = kSZ([z],[volume_gpc3],
               kL_max=kL_max,num_kL_bins=num_kL_bins,kS_min=kS_min,kS_max=kS_max,
               num_kS_bins=num_kS_bins,num_mu_bins=num_mu_bins,ms=ms,params=params,mass_function=mass_function,
               halofit=None,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=False,
               electron_profile_name='e',electron_profile_family=electron_profile_family,
               skip_electron_profile=False,electron_profile_param_override=params,
               electron_profile_nxs=electron_profile_nxs,electron_profile_xmax=electron_profile_xmax)


    ngg = Ngg(ngal_mpc3)
    fPgg = fksz.Pgg(zindex=0) + ngg
    fPge = fksz.Pge(zindex=0)
    pPge = pksz.Pge(zindex=0) if params is not None else fPge






