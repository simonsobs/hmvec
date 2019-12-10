"""
Halo model for ksz

"""

from .params import default_params
from .hmvec import HaloModel

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



class kSZ(HaloModel):

    def __init__(self,zs,ks,ms=None,params={},mass_function="sheth-torman",
                 halofit=None,mdef='vir',nfw_numeric=False,skip_nfw=False):
        """
        Initialize NFW and electron profile
        Get growth rate
        If HOD specified, calculate linear galaxy bias
        If k and HOD specified, calculate all power spectra
        """
        pass


    def chi(self,Yp,NHe):
        val = (1-Yp*(1-NHe/4.))/(1-Yp/2.)
        print(val)
        return val

    def ne0_shaw(self,ombh2,Yp,NHe=0,me = 1.14,gasfrac = 0.9):
        '''
        Average electron density today
        Eq 3 of 1109.0553
        Units: 1/meter**3
        '''
        omgh2 = gasfrac* ombh2
        mu_e = 1.14 # mu_e*mass_proton = mean mass per electron
        ne0_SI = chi(Yp,NHe)*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/mu_e
        return ne0_SI


    def ksz_radial_function(self,z,ombh2, gasfrac = 0.9,xe=1, tau=0, params=None):
        """
        K(z) = - T_CMB sigma_T n_e0 x_e(z) exp(-tau(z)) (1+z)^2
        Eq 4 of 1810.13423
        """

        if params is None: params = default_params
        T_CMB_muk = params['T_CMB'] # muK
        thompson_SI = params['thompson_SI']
        meterToMegaparsec = params['meterToMegaparsec']

        ne0 = ne0_shaw(ombh2)
        return TcmbMuK*thompson_SI*ne0*(1.+z)**2./meterToMegaparsec  * xe  *np.exp(-tau)






