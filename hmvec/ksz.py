"""
Halo model for ksz

We use linear matter power for k<0.1 Mpc-1 used in
calculations of large-scale Pgv, Pvv and Pgg.

We use the halo model for k>0.1 Mpc-1 used in calculations
of small-scale Pge, Pee and Pgg.

"""

from hmvec import HaloModel
import numpy as np

defaults = {'min_mass':1e8, 'max_mass':1e6, 'num_mass':1000}

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

class kSZ(HaloModel):
    def __init__(self,zs,volumes_gpc3,
                 kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
                 num_kS_bins=101,num_mu_bins=102,ms=None,params={},mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False,
                 electron_profile_name='e',electron_profile_family='AGN',
                 skip_electron_profile=False,electron_profile_param_override={},
                 electron_profile_nxs=None,electron_profile_xmax=None):

        if ms is None: ms = np.geomspace(defaults['min_mass'],defaults['max_mass'],defaults['num_mass'])
        zs = np.atleast_1d(zs)
        volumes_gpc3 = np.atleast_1d(volumes_gpc3)
        assert len(zs)==len(volumes_gpc3)
        self.kS = np.geomspace(kS_min,kS_max,num_kS_bins)
        self.mu = np.linspace(-1.,1.,num_mu_bins)
        self.kLs = []
        for volume_gpc3 in volumes_gpc3:
            self.kLs.append(np.geomspace(get_kmin(volume_gpc3),kL_max,num_kL_bins))
        HaloModel.__init__(self,zs,self.kS,ms=ms,params=params,mass_function=mass_function,
                 halofit=halofit,mdef=mdef,nfw_numeric=nfw_numeric,skip_nfw=skip_nfw)

        if not(skip_electron_profile):
            self.add_battaglia_profile(name=electron_profile_name,
                                            family=electron_profile_family,
                                            param_override=electron_profile_param_override,
                                            nxs=electron_profile_nxs,
                                            xmax=electron_profile_xmax,ignore_existing=False)
        
    
#Pgv = 
