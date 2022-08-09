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
from . import utils, hod
from .utils import sanitize
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
    """Compute 3d galaxy shot noise power spectrum from mean number density.

    Parameters
    ----------
    ngalMpc3 : float
        Mean galaxy number density, in Mpc^3.

    Returns
    -------
    P : float
        Shot noise power spectrum, equal to 1/ngal, in Mpc^-3.
    """
    return (1./ngalMpc3)


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
        y = sanitize(integrand[sel])
        x = kss[sel]
        ints.append(np.trapz(y,x))
    return (volume * kstar**2 / 12 / np.pi**3 / chistar**2. * pgv_int * np.asarray(ints))**(-0.5)


def get_kmin(volume_gpc3):
    """Compute k_min for a given cubic volume.

    Uses k_min = pi / V^{1/3}.

    Parameters
    ----------
    volume_gpc3 : float
        Comoving volume, in Gpc^3.

    Returns
    -------
    k_min : float
        Corresponding k_min, in Mpc^-1.
    """
    vol_mpc3 = volume_gpc3 * 1e9
    return np.pi/vol_mpc3**(1./3.)


def chi(Yp, NHe):
    """Fraction of total number of electrons that are ionized after reionization.

    Taken from Eq. 4 of 1109.0553,
        chi = (1 - Y_p * (1 - N_He / 4)) / (1 - Y_p / 2) ,
    where Y_p = primoridal He abundance, N_He = number of ionized He electrons.

    Parameters
    ----------
    Yp : float
        Primordial helium abundance.
    NHe : float
        Number of ionized He electrons.

    Returns
    -------
    val : float
        Chi as described above.
    """
    val = (1-Yp*(1-NHe/4.))/(1-Yp/2.)
    return val


def ne0_shaw(ombh2, Yp, NHe=0, mu_e = 1.14, gasfrac = 0.9):
    """Mean free-electron number density at z = 0.

    Taken from Eq. 3 of 1109.0553,
        n_e = (chi * rho_g) / (mu_e * m_p),
    where
        chi = fraction of total number of electrons that are ionized,
        rho_g = mean gas density at z=0,
        mu_e * m_p = mean gas mass per electron.

    Parameters
    ----------
    ombh2 : float
        Physical baryon density, Omega_b h^2.
    Yp : float
        Primordial helium abundance.
    NHe : float, optional
        Number of ionized He electrons. Default: 0.
    mu_e : float, optional
        Ratio of mean gas mass per electron to proton mass.
        Default: 1.14.
    gasfrac : float, optional
        Fraction of baryons in gas form. Default: 0.9.

    Returns
    -------
    ne0_SI : float
        Mean free-electron number density, in m^-3.
    """
    omgh2 = gasfrac* ombh2
    ne0_SI = chi(Yp,NHe)*omgh2 * 3.*(constants['H100_SI']**2.)/constants['mProton_SI']/8./np.pi/constants['G_SI']/mu_e
    return ne0_SI


def ksz_radial_function(z, ombh2, Yp, gasfrac = 0.9, xe=1, tau=0, params=None):
    """Radial kSZ weight function from Smith et al.

    Taken from Eq. 4 of 1810.13423,
        K(z) = - T_CMB sigma_T n_e0 x_e(z) exp(-tau(z)) (1+z)^2 .

    Parameters
    ----------
    z : float
        Redshift.
    ombh2 : float
        Physical baryon density, Omega_b h^2.
    Yp : float
        Primordial helium abundance.
    gasfrac : float, optional
        Fraction of baryons in gas form. Default: 0.9.
    xe : float, optional
        Mean ionization fraction. Default: 1.
    tau : float, optional
        Mean optical depth to electrons sourcing kSZ effect.
        Default: 0.
    **params : dict
        Dictionary containing CMB temperature T_CMB as item.
        Default: None.

    Returns
    -------
    K : float
        kSZ weight function, in uK Mpc^-1.
    """
    if params is None: params = default_params
    T_CMB_muk = params['T_CMB'] # muK
    thompson_SI = constants['thompson_SI'] # m^2
    meterToMegaparsec = constants['meter_to_megaparsec'] # Mpc m^-1
    ne0 = ne0_shaw(ombh2,Yp) # m^-3
    return T_CMB_muk * thompson_SI * ne0 * (1.+z)**2. / meterToMegaparsec  * xe  * np.exp(-tau)


class kSZ(HaloModel):
    """Class containing halo model kSZ ingredients.

    Reminder: all k's are in Mpc^-1, all masses are in Msun.

    Parameters
    ----------
    zs : array_like
        Array of redshifts to compute at.
    volumes_gpc3, ngals_mpc3 : array_like
        Array of comoving volume (in Gpc^3) corresponding to redshifts in zs.
    ngals_mpc3 : array_like, optional
        Array of galaxy number density (in Mpc^3) corresponding to redshifts in zs.
        If not specified, n_gal computed from HOD is used. Default: None
    use_hod_default_ngal : bool, optional
        Use default HOD parameters instead of setting lower mass threshold from input
        values of ngals_mpc3. Default: False.
    mthreshs_override : array_like, optional
        Array of mass thresholds to use instead of ngals_mpc3 in HOD. Default: None.
    rsd : bool, optional
        Whether to use (Kaiser) RSD in halo model computations. Default: False.
    fog : bool, optional
        Whether to include Fingers of God in halo model computations. Default: False.
    kL_max : float, optional
        Maximum k to consider as k_long (for computing velocity power spectrum).
        Default: 0.1.
    num_kL_bins : int, optional
        Number of k_long bins for computations. Bins will be log-spaced between k_min
        (determined from input volumes) and kL_max. Default: 100.
    kS_min, kS_max : float, optional
        Minimum and maximum k to consider as k_short (for computing electron power
        spectrum). Default: 0.1.
    num_kS_bins : int, optional
        Number of k_short bins for computations. Bins will be log-spaced between kS_min
        and kS_max. Default: 101.
    num_mu_bins : int, optional
        Number of mu bins for computations. Bins will be linear between -1 and 1.
        Default: 102.
    ms : array_like, optional
        Array of halo masses to compute over, in Msun.
        Default: Log-spaced array determined by parameters in `defaults` dict.
    **params : dict, optional
        Optional dict of parameters for halo model and radial kSZ weight computations.
        Default: None.
    mass_function : {'sheth-torman', 'tinker'}, optional
        Mass function to use. Default: 'sheth-torman'
    halofit : int, optional
        If anything other than None, use Halofit for matter power spectrum.
        Default: None
    mdef : {'vir', 'mean'}
        Halo mass definition. 'mean' = M200m.
    nfw_numeric : bool, optional
        Compute Fourier transform of NFW profile numerically instead of analytically.
        Default : False.
    skip_nfw : bool, optional
        Skip computation of NFW profile.
        Default : False.
    electron_profile_name : str, optional
        Internal identifier for electron profile. Default: 'e'.
    electron_profile_family : {'AGN', 'SH', 'pres'}, optional
        Electron profile to use. Default: 'AGN'.
    skip_electron_profile : bool, optional
        Skip computing electron profile. Default: False.
    **electron_profile_param_override : dict, optional
        Dictionary of override parameters for electron profile. Default: None.
    electron_profile_nxs : int, optional
        Number of samples of electron profile for FFT. Default: None.
    electron_profile_xmax : float, optional
        X_max for electron profile in FFT. Default: None.
    skip_hod : bool, optional
        Skip computation of HOD to save time. Default: False.
    hod_name : str, optional
        Internal identifier for galaxy HOD. Default: 'g'.
    hod_family : hod.HODBase
        Name of HOD class to use for galaxy HOD. Default: hod.Leauthaud12_HOD.
    hod_corr : {'max', 'min'}, optional
        Correlations between centrals and satellites in HOD. Default: 'max'.
    **hod_param_override : dict, optional
        Dictionary of override parameters for electron profile. Default: None.
    satellite_profile_name : str, optional
        Internal identifier for satellite galaxy profile. Default: 'nfw'.
    verbose : bool, optional
        Print progress of computations. Default: False.
    b1 : float, optional
        Linear galaxy bias. Default: None.
    b2 : float, optional
        Currently unused. Default: None.
    sigz : float, optional
        The Gaussian uncertainty for photometric redshifts. The assumed scatter
        will be sigz * (1+z). Default: None.
    physical_truncate : bool, optional
        Truncate the electron (gas) profile such that the enclosed mass is equal to
        f_b m_200c. Default: True.
    """
    def __init__(
        self,
        zs,
        volumes_gpc3,
        ngals_mpc3=None,
        use_hod_default_ngal=False,
        mthreshs_override=None,
        rsd=False,
        fog=False,
        kL_max=0.1,
        num_kL_bins=100,
        kS_min=0.1,
        kS_max=10.0,
        num_kS_bins=101,
        num_mu_bins=102,
        ms=None,
        params=None,
        mass_function="sheth-torman",
        halofit=None,
        mdef='vir',
        nfw_numeric=False,
        skip_nfw=False,
        electron_profile_name='e',
        electron_profile_family='AGN',
        skip_electron_profile=False,
        electron_profile_param_override=None,
        electron_profile_nxs=None,
        electron_profile_xmax=None,
        skip_hod=False,
        hod_name="g",
        hod_family=hod.Leauthaud12_HOD,
        hod_corr="max",
        hod_param_override=None,
        satellite_profile_name="nfw",
        verbose=False,
        b1=None,
        b2=None,
        sigz=None,
        physical_truncate=True,
    ):

        # Store rsd and fog preferences
        if fog and not rsd:
            raise ValueError("Cannot use FoG without RSD!")
        self.rsd = rsd
        self.fog = fog

        # Define masses to compute for
        if ms is None: ms = np.geomspace(
            defaults['min_mass'], defaults['max_mass'], defaults['num_mass']
        )

        # Define comoving volume and galaxy number density at each redshift
        volumes_gpc3 = np.atleast_1d(volumes_gpc3)
        zs = np.atleast_1d(zs)
        if ngals_mpc3 is not None:
            self.ngals_mpc3 = np.atleast_1d(ngals_mpc3)
        else:
            self.ngals_mpc3 = ngals_mpc3
        if len(zs) != len(volumes_gpc3):
            raise ValueError("zs and volumes_gpc3 must have same length")

        if self.ngals_mpc3 is not None:
            self.ngals_mpc3 = np.asarray(self.ngals_mpc3)
            if len(zs) != len(self.ngals_mpc3):
                raise ValueError("zs and ngals_mpc3 must have same length")

        # Warn user if k_min is the same for all zs
        if np.max(volumes_gpc3) != np.min(volumes_gpc3):
            warnings.warn(
                "Using equal k_min at each z, despite different volumes at each z"
            )

        # Define k_short and mu values to compute for
        ks = np.geomspace(kS_min,kS_max,num_kS_bins)
        self.ks = ks
        self.kS = self.ks
        self.mu = np.linspace(-1.,1.,num_mu_bins)

        # Define k_long values to compute for
        self.kLs = np.geomspace(get_kmin(np.max(volumes_gpc3)),kL_max,num_kL_bins)
        # Define k_{long,radial} = k_long * mu values to compute for, packed as [mu,k].
        self.krs = self.mu[:, None] * self.kLs[None, :]

        # Initialize HaloModel object
        if verbose: print('Defining HaloModel')
        HaloModel.__init__(
            self,
            zs,
            ks,
            ms=ms,
            mus=self.mu,
            params=params,
            mass_function=mass_function,
            halofit=halofit,
            mdef=mdef,
            nfw_numeric=nfw_numeric,
            skip_nfw=skip_nfw,
        )
        if verbose: print('Defining HaloModel: finished')

        # Initialize electron profile
        if not skip_electron_profile:
            if verbose: print('Defining electron profile')
            self.add_battaglia_profile(
                name=electron_profile_name,
                family=electron_profile_family,
                param_override=electron_profile_param_override,
                nxs=electron_profile_nxs,
                xmax=electron_profile_xmax,
                ignore_existing=False,
                vectorize_z=False,
                verbose=verbose,
                physical_truncate=physical_truncate,
            )
            if verbose: print('Defining electron profile: finished')

        # Define galaxy HOD
        if not skip_hod:
            if verbose: print('Defining HOD')
            if use_hod_default_ngal:
                ngal_for_hod = None
            else:
                ngal_for_hod = self.ngals_mpc3

            if satellite_profile_name != 'nfw':
                try:
                    self.add_HI_profile(name=satellite_profile_name)
                except:
                    try:
                        self.add_HI_profile(name=satellite_profile_name, numeric=True)
                    except:
                        raise ValueError(
                            "Couldn't initialize satellite profile %s"
                            % satellite_profile_name
                        )

            self.add_hod(
                hod_name,
                family=hod_family,
                mthresh=mthreshs_override,
                ngal=ngal_for_hod,
                corr=hod_corr,
                satellite_profile_name=satellite_profile_name,
                central_profile_name=None,
                ignore_existing=False,
                param_override=hod_param_override,
            )
            if self.ngals_mpc3 is None:
                self.ngals_mpc3 = self.hods[hod_name]['ngal']

            if verbose: print('Defining HOD: finished')

        # Set quantities needed for photo-z uncertainty calculations
        self.sigz = sigz
        if self.sigz is not None:
            self.sigma_z_func = lambda z : self.sigz * (1.+z)
            try:
                zhs,hs = np.loadtxt("fiducial_cosmology_Hs.txt",unpack=True)
            except:
                warnings.warn(
                    "File fiducial_cosmology_Hs.txt not found. Computing H(z) for "
                    "W_photoz internally"
                )
                zhs = self.zs
                hs = self.h_of_z(zhs)
            self.Hphotoz = interp1d(zhs,hs)

        # Get P_linear and f(z) on grid in z and k
        p = self._get_matter_power(self.zs,self.kLs,nonlinear=False)
        growth = self.results.get_redshift_evolution(
            self.kLs, self.zs, ['growth']
        )[:,:,0]

        # Compute some cosmological quantities needed for large-scale spectra
        self.Pmms = []
        self.fs = []
        self.adotf = []
        self.d2vs = []
        self.kstars = []
        self.chistars = []
        for zindex,volume_gpc3 in enumerate(volumes_gpc3):
            self.Pmms.append(np.resize(p[zindex].copy(),(self.mu.size,self.kLs.size)))
            self.fs.append(growth[:,zindex].copy())
            z = self.zs[zindex]
            a = 1./(1.+z)
            H = self.results.h_of_z(z)
            self.kstars.append(self.ksz_radial_function(zindex))
            self.d2vs.append(self.fs[zindex]*a*H / self.kLs)
            self.adotf.append(self.fs[zindex]*a*H)
            self.chistars.append(self.results.comoving_radial_distance(z))

        # Compute 3d power spectra
        if not skip_hod:
            # Compute P_gg and P_ge, packed as [z,k] without RSD or [z,k,mu] with RSD.
            if verbose: print("Calculating small scale Pgg and Pge...")
            self.sPgg_1h = self.get_power_1halo(hod_name, name2=hod_name, fog=fog)
            if rsd and not fog:
                self.sPgg_1h = self.sPgg_1h[:, :, None]
            self.sPge_1h = self.get_power_1halo(
                hod_name, name2=electron_profile_name, fog=fog
            )
            # If using Kaiser RSD but not FoG, extend axes from [z,k] to [z,k,mu]
            if rsd and not fog:
                self.sPge_1h = self.sPge_1h[:, :, None]
            self.sPgg_2h = self.get_power_2halo(
                hod_name,
                name2=hod_name,
                verbose=verbose,
                b1_in=b1,
                b2_in=b1,
                rsd=rsd,
                fog=fog
            )
            self.sPge_2h = self.get_power_2halo(
                hod_name,
                name2=electron_profile_name,
                verbose=verbose,
                b1_in=b1,
                rsd=rsd,
                fog=fog
            )
            self.sPgg = self.sPgg_1h + self.sPgg_2h
            self.sPge = self.sPge_1h + self.sPge_2h

            # Incorporate photo-z uncertainty
            if self.sigz is not None:
                for zi in range(self.sPgg.shape[0]):
                    if self.rsd:
                        # With RSD, sPgg and sPge are packed as [z,k,mu], while
                        # Wphoto(z) is packed as [mu,k], so we need to transpose it
                        self.sPgg[zi] *= self.Wphoto(zi).T ** 2
                        self.sPge[zi] *= self.Wphoto(zi).T
                    else:
                        # Without RSD, sPgg and sPge are packed as [z,k], and after
                        # incorporating W_photo(z), the results are packed as [z,k,mu]
                        self.sPgg[zi] = self.sPgg[zi][:, None] * self.Wphoto(zi).T ** 2
                        self.sPge[zi] = self.sPge[zi][:, None] * self.Wphoto(zi).T

            # Compute further power spectra
            # TODO: clean this up and add more comments
            self.Vs = volumes_gpc3
            self.vrec = []
            self.sPggtot = []
            # self.sPge = []
            self.bgs = []
            # aPgg = self.get_power('g','g',verbose=verbose, )
            # aPge = self.get_power('g','e',verbose=verbose)
            for zindex, volume_gpc3 in enumerate(volumes_gpc3):
                # Compute P_gg + N_gg and P_gv for fiducial and "true" parameters, as
                # functions of [z,k_L,mu]
                bg = self.hods['g']['bg'][zindex]
                self.bgs.append(bg)
                ngg = Ngg(self.ngals_mpc3[zindex])
                flPgg = self.lPgg(zindex,bg1=bg,bg2=bg,rsd=rsd)[0,:] + ngg
                flPgv = self.lPgv(zindex,bg=bg,rsd=rsd)[0,:]
                # Construct integrand (without prefactor) as function of tabulated k_L
                # values, and integrate
                kls = self.kLs
                integrand = sanitize((kls**2.)*(flPgv*flPgv)/flPgg)
                vrec = np.trapz(integrand,kls)
                self.vrec.append(vrec.copy())

                # Pgg = aPgg[zindex].copy()
                # if self.sigz is not None:
                #     Pgg = Pgg[None,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1))**2.)
                # Pggtot  = Pgg + ngg
                # self.sPggtot.append(Pggtot.copy())
                self.sPggtot.append(self.sPgg[zindex] + ngg)
                # Pge = aPge[zindex].copy()
                # if self.sigz is not None:
                #     Pge = Pge[None,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1)))
                # self.sPge.append(Pge.copy())


    def Pge_err(self,zindex,ks_bin_edges,Cls):
        kstar = self.kstars[zindex]
        chistar = self.chistars[zindex]
        volume = self.Vs[zindex]
        pgv_int  = self.vrec[zindex]
        kss = self.ks
        pggtot = self.sPggtot[zindex][0]
        return pge_err_core(pgv_int,kstar,chistar,volume,kss,ks_bin_edges,pggtot,Cls)

    def lPvv(self, zindex, bv1=1, bv2=1):
        """The long-wavelength velocity auto spectrum, P_vv.

        This is calculated as:
            (faH/kL)**2*Pmm(kL)
        to return a 1D array [mu,kL] with identical copies over mus.

        Here Pmm is the non-linear power for all halos.
        bv1 and bv2 are the velocity biases in each bin.

        Parameters
        ----------
        zindex : int
            Z index to compute at.
        bv1, bv2 : float, optional
            Linear bias of each velocity field.

        Returns
        -------
        Pvv : array_like
            P_vv, packed as [mu,k].
        """
        Pvv = self.d2vs[zindex]**2. * self.Pmms[zindex] * bv1 * bv2
        return Pvv

    def lPgg(self, zindex, bg1, bg2, rsd=False):
        """The long-wavelength galaxy auto spectrum, P_gg.

        Without RSD, this is calculated as:
            b_1 * b2 * Pmm(kL)
        to return a 1D array [mu,kL] with identical copies over mus.

        With RSD, it is calculated as:
            (b_1 + f mu^2) * (b_2 + f mu^2) * Pmm(kL) ,
        assuming that FoG are irrelevant on the scale we'll use this for.

        Here Pmm is the non-linear power for all halos.
        bg1 and bg2 are the linear galaxy biases in each bin.

        Parameters
        ----------
        zindex : int
            Z index to compute at.
        bg1, bg2 : float
            Linear bias of each galaxy field.

        Returns
        -------
        Pgg : array_like
            P_gg, packed as [mu,k].
        """
        if rsd:
            f = self.fs[zindex]
            # Recall that self.Pmms[zindex] is packed as [mu,k]
            Pgg = (
                self.Pmms[zindex]
                * (bg1 + f * self.mu[:, None] ** 2)
                * (bg2 + f * self.mu[:, None] ** 2)
            )
            if self.sigz is not None:
                Pgg *= self.Wphoto(zindex) ** 2
        else:
            Pgg = self.Pmms[zindex] * bg1 * bg2
            if self.sigz is not None:
                Pgg *= self.Wphoto(zindex) ** 2

        # if self.sigz is not None:
        #     Pgg = Pgg[...,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1))**2.)

        return Pgg

    def lPgv(self, zindex, bg, bv=1, rsd=False):
        """The long-wavelength galaxy-velocity cross spectrum, P_gv.

        Parameters
        ----------
        zindex : int
            Z index to compute at.
        bg, bv : float
            Linear biases of galaxy and velocity field.

        Returns
        -------
        Pgv : array_like
            P_gv, packed as [mu,k].
        """
        if rsd:
            f = self.fs[zindex]
            # Recall that self.Pmms[zindex] is packed as [mu,k]
            Pgv = (
                self.Pmms[zindex]
                * (bg + f * self.mu[:, None] ** 2)
                * bv
                * self.d2vs[zindex]
            )
            if self.sigz is not None:
                Pgv *= self.Wphoto(zindex)

        else:
            Pgv = self.Pmms[zindex] * bg * bv * (self.d2vs[zindex])
            if self.sigz is not None:
                Pgv *= self.Wphoto(zindex)

        # if not(self.sigz is None):
        #     Pgv = Pgv[...,None] * (self.Wphoto(zindex).reshape((self.mu.size,self.kLs.size,1)))

        return Pgv

    def ksz_radial_function(self, zindex, gasfrac = 0.9, xe=1, tau=0, params=None):
        """Radial kSZ weight function from Smith et al.

        See ksz.ksz_radial_function for information and parameter descriptions.

        Returns
        -------
        K : float
            kSZ weight function value at specified redshift.
        """
        return ksz_radial_function(
            self.zs[zindex],
            self.pars.ombh2,
            self.pars.YHe,
            gasfrac=gasfrac,
            xe=xe,
            tau=tau,
            params=params
        )

    def Wphoto(self,zindex):
        """Radial photo-z uncertainty window.

        This is \exp(-sigma_z^2 k^2 \mu^2 / (2 H(z)^2)).

        Parameters
        ----------
        zindex : int
            Index of stored redshift to compute for

        Returns
        -------
        W : array_like
            W_photo, packed as [mu,k].
        """
        if self.sigz is None:
            # Returns ones if photo-z error not specified. Recall that self.krs is an
            # array of k*mu, packed as [mu,k].
            return np.ones_like(self.krs)
        else:
            z = self.zs[zindex]
            H = self.Hphotoz(z)
            return np.exp(-self.sigma_z_func(z) ** 2 * self.krs ** 2 / 2 / H ** 2)

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
    integrand = sanitize(kSs * ( Pge**2. / (Pgg_tot * Clkstot)))

    if robust_term:
        assert Pgg_photo_tot is not None
        integrand = sanitize(integrand * (Pgg_photo_tot/Pgg_tot))

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
                            electron_profile_nxs=None,electron_profile_xmax=None,
                            hod_family=hod.Leauthaud12_HOD):
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
               skip_hod=False,hod_name="g",hod_family=hod_family,hod_corr="max",hod_param_override=None)

    # Define kSZ object corresponding to "true" parameters, if specified
    ## TODO: need to specify HOD params below?
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
    fsPgg = fksz.sPgg[0] + ngg
    fsPge = fksz.sPge[0]

    # !!!
    # fsPgg = fksz._get_matter_power(fksz.zs[0],fksz.kS,nonlinear=True)[0] * bg**2. + ngg
    # fsPge = fksz._get_matter_power(fksz.zs[0],fksz.kS,nonlinear=True)[0] * bg
    # !!!

    # Get P_ge as a function of k_S, for "true" parameters
    psPge = pksz.sPges[0] if params is not None else fsPge

    # Get comoving distance to redshift z
    chistar = pksz.results.comoving_radial_distance(z)

    # Get interpolating function for P_ge^fid * P_ge^true / P_gg^{tot,fid}
    iPk = utils.interp(fksz.kS,sanitize(fsPge * psPge / fsPgg))
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
    flPgg = fksz.lPgg(0,bg1=bg,bg2=bg)[0,:] + ngg
    flPgv = fksz.lPgv(0,bg=bg)[0,:]
    plPgv = pksz.lPgv(0,bg=bg)[0,:] if params is not None else flPgv

    # Construct integrand (without prefactor) as function of tabulated k_L values,
    # and integrate
    kls = fksz.kLs
    integrand = sanitize((kls**2.)*(flPgv*plPgv)/flPgg)
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
                electron_profile_nxs=None,electron_profile_xmax=None,sigz=None,
                hod_family=hod.Leauthaud12_HOD):

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
               skip_hod=False,hod_name="g",hod_family=hod_family,
               hod_corr="max",hod_param_override=None,sigz=sigz)
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
    integrand = sanitize((kls**2.)*(lPgv**2)/ltPgg/Nvv)
    result = np.trapz(integrand,kls)
    snr2 = np.trapz(result,fksz.mu) / (2.*np.pi)**2.
    return np.sqrt(V*snr2),fksz


def get_ksz_auto_mafry(
    ells,
    volume_gpc3,
    zs,
    ngals_mpc3=None,
    bgs=None,
    params=None,
    k_max = 100.,
    num_k_bins = 200,
    num_mu_bins=102,
    ms=None,
    mass_function="sheth-torman",
    mdef='vir',
    nfw_numeric=False,
    electron_profile_family='AGN',
    electron_profile_nxs=None,
    electron_profile_xmax=None,
    n_int = 100,
    verbose=False,
    pksz_in=None,
    slow_chi_integral=False,
    save_cl_integrand=False,
    physical_truncate=True,
):
    """Compute kSZ angular auto power spectrum, C_ell, as in Ma & Fry.

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

    Parameters
    ----------
    ells : array_like
        Array of ell values to compute C_ell at.
    volumes_gpc3, ngals_mpc3 : array_like
        Array of comoving volume (in Gpc^3) corresponding to redshifts in zs.
    ngals_mpc3 : array_like, optional
        Array of galaxy number density (in Mpc^3) corresponding to redshifts in zs.
        If not specified, n_gal computed from HOD is used. Default: None
    zs : array_like
        Array of redshifts to compute at.
    bgs : array_like, optional
        Array of linear galaxy bias at each redshift. Default: None.
    **params : dict, optional
        Optional dict of parameters for halo model and radial kSZ weight computations.
        Default: None.
    k_max : float, optional
        Maximum k to consider as k_long and k_short (same value used for both).
        Default: 100.
    num_k_bins : int, optional
        Number of k_long and k_short bins for computations. Default: 200.
    num_mu_bins : int, optional
        Number of mu bins for computations. Bins will be linear between -1 and 1.
        Default: 102.
    ms : array_like, optional
        Array of halo masses to compute over, in Msun.
        Default: Log-spaced array determined by parameters in `defaults` dict.
    mass_function : {'sheth-torman', 'tinker'}, optional
        Mass function to use. Default: 'sheth-torman'
    mdef : {'vir', 'mean'}
        Halo mass definition. 'mean' = M200m.
    nfw_numeric : bool, optional
        Compute Fourier transform of NFW profile numerically instead of analytically.
        Default : False.
    electron_profile_family : {'AGN', 'SH', 'pres'}, optional
        Electron profile to use. Default: 'AGN'.
    electron_profile_nxs : int, optional
        Number of samples of electron profile for FFT. Default: None.
    electron_profile_xmax : float, optional
        X_max for electron profile in FFT. Default: None.
    n_int : int, optional
        Number of samples to use in Limber integral. Default: 100.
    verbose : bool, optional
        Print progress of computations. Default: False.
    pksz_in : kSZ object, optional
        Predefined kSZ object to use for computations, instead of initializing
        a new one. Default: None.
    slow_chi_integral : bool, optional
        Use slower quad method for Limber integral for C_ell, instead of faster
        trapz. Default: False.
    save_cl_integrand : bool, optional
        Save integrand of C_ell to spec_dict, along with coordinate values
        that integrand was evaluated at, w.r.t. either chi or z.
    physical_truncate : bool, optional
        Truncate the electron (gas) profile such that the enclosed mass is equal to
        f_b m_200c. Default: True.

    Returns
    -------
    pksz : kSZ object
        kSZ object initialized during the routine. (Can be used as input to
        later calls to save time.)
    cl : np.ndarray
        Computed C_ell^kSZ, evaluated at input ells.
    **spec_dict : dict
        Dict containing various 3d power spectra used for kSZ computation.
    """
    _LIMBER_KMAX = 30 # Mpc

    # Define empty dict for storing spectra
    spec_dict = {}

    # Widen search range for setting lower mass threshold from nbar
    if params is None:
        params = default_params
    params['hod_bisection_search_min_log10mthresh'] = 1

    # Make sure input redshifts are sorted
    zs = np.sort(np.asarray(zs))

    # Make array for volumes, for feeding to kSZ object
    volumes_gpc3 = volume_gpc3 * np.ones_like(zs)

    # Define kSZ object, if not specified as input
    if pksz_in is not None:
        pksz = pksz_in
    else:
        if verbose: print('Initializing kSZ object')
        pksz = kSZ(
            zs,
            volumes_gpc3,
            ngals_mpc3=ngals_mpc3,
            rsd=False,
            fog=False,
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
            skip_hod=True, # Skip HOD, since we're computing true kSZ signal
            verbose=verbose,
            b1=bgs,
            b2=bgs,
            physical_truncate=physical_truncate,
        )

    # Update ngal. Also, if different galaxy number density for velocity template is not
    # specified, set equal to density for electron template
    ngals_mpc3 = pksz.ngals_mpc3

    # Get k_short values that P_{q_perp} integrand is evaluated at
    ks = pksz.kS
    spec_dict['ks'] = ks

    # Also get mu values
    mus = pksz.mu

    # Get P_ee (packed as [z,k])
    if verbose: print("Computing P_ee and P_vv")
    sPee = pksz.get_power('e',name2='e',verbose=False)

    # Get P_vv (packed as [z,k]), by getting it for each z individually
    lPvv0 = pksz.lPvv(zindex=0)[0,:]
    lPvv = np.zeros((len(zs), lPvv0.shape[0]), dtype=lPvv0.dtype)
    lPvv[0,:] = lPvv0
    for zi in range(1, len(zs)):
        lPvv[zi,:] = pksz.lPvv(zindex=zi)[0,:]

    # Get P_linear as a function of z and k (packed as [z,k])
    Pmm = np.asarray(pksz.Pmms)
    Pmm = Pmm[:,0,:]

    # Save 3d power spectra
    spec_dict['sPee'] = sPee
    spec_dict['lPvv'] = lPvv
    spec_dict['Pmm'] = Pmm

    # Make meshes of mu and k, packed as [k,mu]
    mu_mesh, k_mesh = np.meshgrid(mus, ks)

    # Define function that returns fraction in P_{q_r} integrand,
    # and also |\vec{k} - \vec{k}'|
    def Pqr_igr_poly(k, kp, mu, z):

        frac = k * (k - 2 * kp * mu) * (1 - mu ** 2)
        frac /= (kp ** 2 * (kp ** 2 + k ** 2 - 2 * k * kp * mu))

        kmkp = (kp ** 2 + k ** 2 - 2 * k * kp * mu) ** 0.5

        igr = kp ** 2 * frac

        return igr, kmkp

    # P_{q_r} will be packed as [k,z]
    if verbose:
        print('Computing P_{q_r} on grid in k,z')
    Pqr = np.zeros((ks.shape[0], zs.shape[0]))

    for zi, z in enumerate(zs):

        # Define interpolating functions for P_ee and P_mm at this z
        isPee = interp1d(ks, sPee[zi], bounds_error=False, fill_value=0.)
        iPmm = interp1d(ks, Pmm[zi], bounds_error=False, fill_value=0.)

        # Compute \dot{a} f = a H f at this redshift
        adotf = pksz.adotf[zi][0]

        for ik,k in enumerate(ks):

            if True:
                # Get P_{q_r} integrand on [k,mu] mesh
                Pqr_igr_mesh, kmkp_mesh = Pqr_igr_poly(k, k_mesh, mu_mesh, z)
                Pee_mesh = isPee(kmkp_mesh.flatten()).reshape(kmkp_mesh.shape)
                Pmm_mesh = iPmm(k_mesh.flatten()).reshape(kmkp_mesh.shape)
                Pqr_igr_mesh *= Pmm_mesh * Pee_mesh

                # Integrate integrand mesh along k axis with trapezoid rule
                integral = np.trapz(np.nan_to_num(Pqr_igr_mesh), ks, axis=0)

                # Integrate along mu axis with trapezoid rule
                integral = np.trapz(integral, mus)

            else:
                # Could also do double integral with dblquad, but in practice
                # this takes forever
                integral = dblquad(
                    lambda kp,mu: Pqr_igr(k, kp, mu, z), -1, 1, ks[0], ks[-1]
                )[0]

            # Include prefactors for integral
            Pqr[ik, zi] = adotf**2 * (2*np.pi)**-2 * integral

    spec_dict['Pqr'] = Pqr

    # Make 2d interpolating function for P_{q_r}, with arguments z,k.
    # The resulting interpolating function automatically sorts arguments if
    # arrays are fed in, but we'll only call iPqperp with one (z,k) pair
    # at a time, so we'll be fine.
    iPqr = interp2d(zs, ks, Pqr, fill_value=0.)

    # Compute C_ell integral at each ell
    if verbose: print('Computing C_ell')
    cl = np.zeros(ells.shape[0])
    for iell, ell in enumerate(ells):

        # Set chi_min from min redshift, or from k=30Mpc^-1 if the lowest redshift
        # translates to k>30Mpc^-1
        chi_min = max(pksz.results.comoving_radial_distance(zs[0]), ell / _LIMBER_KMAX)
        # Set chi_max from max redshift
        chi_max = pksz.results.comoving_radial_distance(zs[-1])
        chi_int = np.geomspace(chi_min, chi_max, n_int)
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
        integrand *= (
            constants['thompson_SI']
            * ne0
            * 1/constants['meter_to_megaparsec']
        )**2
        integrand *= (pksz.pars.TCMB * 1e6)**2
        integrand *= 0.5

        # If desired, save integrand for each ell to spec_dict.
        # We save the integrand w.r.t either chi and z, along with the
        # chi and z value it is evaluated at
        if save_cl_integrand:
            # Make empty arrays to hold output
            if iell == 0:
                spec_dict['ClkSZ_integrand_chi'] = np.zeros(
                    (len(ells), len(chi_int)), dtype=chi_int.dtype
                )
                spec_dict['ClkSZ_integrand_z'] = np.zeros_like(
                    spec_dict['ClkSZ_integrand_chi']
                )

                spec_dict['ClkSZ_dchi_integrand'] = np.zeros(
                    (len(ells), len(integrand)), dtype=integrand.dtype
                )
                spec_dict['ClkSZ_dz_integrand'] = np.zeros_like(
                    spec_dict['ClkSZ_dchi_integrand']
                )

            # Save integrand w.r.t. chi
            spec_dict['ClkSZ_integrand_chi'][iell] = chi_int
            spec_dict['ClkSZ_dchi_integrand'][iell] = integrand

            # Save integrand w.r.t z, which is dchi/dz * integrand_dchi
            spec_dict['ClkSZ_integrand_z'][iell] = z_int
            _DERIV_DZ = 0.001
            dchi_dz_int = (
                (
                    pksz.results.comoving_radial_distance(z_int + _DERIV_DZ)
                    - pksz.results.comoving_radial_distance(z_int - _DERIV_DZ)
                ) / (2 * _DERIV_DZ)
            )
            spec_dict['ClkSZ_dz_integrand'][iell] = integrand * dchi_dz_int

        if not slow_chi_integral:
            # Do C_ell integral via trapezoid rule
            cl[iell] = np.trapz(integrand, chi_int)
        else:
            # Doing integral of an interpolating function gives
            # equivalent results at the precision we care about
            igr_interp = interp1d(chi_int, integrand)
            cl[iell] = quad(igr_interp, chi_int[0], chi_int[-1])[0]

    # Return kSZ object (in case we want to use it later), C_ell array,
    # and dict of spectra used
    return pksz, cl, spec_dict


def get_ksz_auto_squeezed(
    ells,
    volume_gpc3,
    zs,
    ngals_mpc3=None,
    rsd=False,
    fog=False,
    isotropize_muS=False,
    bgs=None,
    params=None,
    k_max = 100.,
    num_k_bins = 200,
#    kL_max=0.1,num_kL_bins=100,kS_min=0.1,kS_max=10.0,
    num_mu_bins=102,
    ms=None,
    mass_function="sheth-torman",
    mdef='vir',
    nfw_numeric=False,
    electron_profile_family='AGN',
    electron_profile_nxs=None,
    electron_profile_xmax=None,
    hod_family=hod.Leauthaud12_HOD,
    hod_corr="max",
    satellite_profile_name="nfw",
    n_int = 100,
    verbose=False,
    template=False,
    pksz_in=None,
    save_debug_files=False,
    ngals_mpc3_for_v=None,
    slow_chi_integral=False,
    save_cl_integrand=False,
    pgg_noise_function=None,
    use_pee_in_template=False,
    sigz=None,
    mthreshs_override=None,
    use_hod_default_ngal=False,
    physical_truncate=True,
):
    """Compute kSZ angular auto power spectrum, C_ell, as in Ma & Fry.

    We use the squeezed limit from Ma & Fry, with some altered notation:

        C_\ell = \int \frac{d\chi}{\chi^2 H_0^2} \tilde{K}(z[\chi])^2
                 P_{q_r}(k=\ell/\chi, \chi)

        \tilde{K}(z) = T_{CMB} \bar{n}_{e,0} \sigma_T (1+z)^2 \exp(-\tau(z))

        P_{q_r}(k,z) = \frac{1}{6\pi^2} \int dk' (k')^2 P_{vv}(k',z) P_{ee}(k,z)

    If RSD is included, the k' integral turns into a double integral over k' and mu,
    while P_{ee} is computed as a function of mu and then we take the monopole over mu.

    C_ell^kSZ is returned in uK^2.

    Parameters
    ----------
    ells : array_like
        Array of ell values to compute C_ell at.
    volumes_gpc3, ngals_mpc3 : array_like
        Array of comoving volume (in Gpc^3) corresponding to redshifts in zs.
    ngals_mpc3 : array_like, optional
        Array of galaxy number density (in Mpc^3) corresponding to redshifts in zs.
        If not specified, n_gal computed from HOD is used. Default: None
    zs : array_like
        Array of redshifts to compute at.
    rsd : bool, optional
        Whether to include (Kaiser) RSD in 3d power spectra involving galaxies.
        Default: False.
    fog : bool, optional
        Whether to include Fingers of God in 3d power spectra involving galaxies.
        Default: False.
    isotropize_muS : bool, optional
        Whether to average Pge^2(k, mu) / Pggtot(k, mu) over mu. If False, we take
        mu = 0 when we compute C_ell^kSZ. Default: False.
    bgs : array_like, optional
        Array of linear galaxy bias at each redshift. Default: None.
    **params : dict, optional
        Optional dict of parameters for halo model and radial kSZ weight computations.
        Default: None.
    k_max : float, optional
        Maximum k to consider as k_long and k_short (same value used for both).
        Default: 100.
    num_k_bins : int, optional
        Number of k_long and k_short bins for computations. Default: 200.
    num_mu_bins : int, optional
        Number of mu bins for computations. Bins will be linear between -1 and 1.
        Default: 102.
    ms : array_like, optional
        Array of halo masses to compute over, in Msun.
        Default: Log-spaced array determined by parameters in `defaults` dict.
    mass_function : {'sheth-torman', 'tinker'}, optional
        Mass function to use. Default: 'sheth-torman'
    mdef : {'vir', 'mean'}
        Halo mass definition. 'mean' = M200m.
    nfw_numeric : bool, optional
        Compute Fourier transform of NFW profile numerically instead of analytically.
        Default : False.
    electron_profile_family : {'AGN', 'SH', 'pres'}, optional
        Electron profile to use. Default: 'AGN'.
    electron_profile_nxs : int, optional
        Number of samples of electron profile for FFT. Default: None.
    electron_profile_xmax : float, optional
        X_max for electron profile in FFT. Default: None.
    hod_family : hod.HODBase
        Name of HOD class to use for galaxy HOD. Default: hod.Leauthaud12_HOD.
    hod_corr : {'max', 'min'}, optional
        Correlations between centrals and satellites in HOD. Default: 'max'.
    satellite_profile_name : str, optional
        Internal identifier for satellite galaxy profile. Default: 'nfw'.
    n_int : int, optional
        Number of samples to use in Limber integral. Default: 100.
    verbose : bool, optional
        Print progress of computations. Default: False.
    template : bool, optional
        Whether to assume that the v and e fields have been constructed from templates
        based on galaxy surveys (True), or v and e are the true fields (False).
        Default: False.
    pksz_in : kSZ object, optional
        Predefined kSZ object to use for computations, instead of initializing
        a new one. Default: None.
    save_debug_files : bool, optional
        Save some computed spectra to disk for offline debugging. Default: False.
    ngals_mpc3_for_v : array_like, optional
        If specified, use these galaxy number densities for the velocity
        template. Default: None.
    slow_chi_integral : bool, optional
        Use slower quad method for Limber integral for C_ell, instead of faster
        trapz. Default: False.
    save_cl_integrand : bool, optional
        Save integrand of C_ell to spec_dict, along with coordinate values
        that integrand was evaluated at, w.r.t. either chi or z.
    pgg_noise_function : function(z, k), optional
        Callable function of z and k that gives the noise power spectrum
        for P_gg. If specified, replaces 1/ngals. Default: None.
    use_pee_in_template : bool, optional
        For template calculation, use P_ee instead of P_ge^2 / P_gg^tot.
        This tests the effect of the g-v correlation on the template
        amplitude. Default: False.
    sigz : float, optional
        The Gaussian scatter for photometric redshifts. The assumed scatter
        will be sigz * (1+z).
    mthreshs_override : array_like, optional
        Array of mass thresholds to use instead of ngal in HOD. Default: None.
    use_hod_default_ngal : bool, optional
        If set, don't use input ngals_mpc3 to fix mthresh for HOD. Default: False.
    physical_truncate : bool, optional
        Truncate the electron (gas) profile such that the enclosed mass is equal to
        f_b m_200c. Default: True.

    Returns
    -------
    pksz : kSZ object
        kSZ object initialized during the routine. (Can be used as input to
        later calls to save time.)
    cl : np.ndarray
        Computed C_ell^kSZ, evaluated at input ells.
    **spec_dict : dict
        Dict containing various 3d power spectra used for kSZ computation.
    """
    _LIMBER_KMAX = 30 # Mpc

    # Define empty dict for storing spectra
    spec_dict = {}

    # Widen search range for setting lower mass threshold from nbar
    if params is None:
        params = default_params
    params['hod_bisection_search_min_log10mthresh'] = 1

    # Make sure input redshifts are sorted
    zs = np.sort(np.asarray(zs))

    # Make array for volumes, for feeding to kSZ object
    volumes_gpc3 = volume_gpc3 * np.ones_like(zs)

    # If not computing for a kSZ template, skip HOD computation to save time
    if template:
        skip_hod = False
    else:
        skip_hod = True

    # Define kSZ object, if not specified as input
    if pksz_in is not None:
        pksz = pksz_in
    else:
        if verbose: print('Initializing kSZ object')
        pksz = kSZ(
            zs,
            volumes_gpc3,
            ngals_mpc3=ngals_mpc3,
            rsd=rsd,
            fog=fog,
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
            hod_family=hod_family,
            hod_corr=hod_corr,
            satellite_profile_name=satellite_profile_name,
            skip_electron_profile=False,
            electron_profile_param_override=params,
            electron_profile_nxs=electron_profile_nxs,
            electron_profile_xmax=electron_profile_xmax,
            skip_hod=skip_hod,
            verbose=verbose,
            b1=bgs,
            b2=bgs,
            sigz=sigz,
            mthreshs_override=mthreshs_override,
            use_hod_default_ngal=use_hod_default_ngal,
            physical_truncate=physical_truncate,
        )

    # Update ngal. Also, if different galaxy number density for velocity template is not
    # specified, set equal to density for electron template
    ngals_mpc3 = pksz.ngals_mpc3
    if ngals_mpc3_for_v is None:
        ngals_mpc3_for_v = ngals_mpc3

    # Get k_short values that P_{q_perp} integrand is evaluated at
    ks = pksz.kS
    spec_dict['ks'] = ks

    # If not computing for a kSZ template, get P_ee and P_vv
    # on grids in z and k
    if not template:
        # Get P_ee (packed as [z,k])
        if verbose: print("Computing P_ee and P_vv")
        sPee = pksz.get_power('e',name2='e',verbose=False)

        # Get P_vv (packed as [z,k]), by getting it for each z individually
        lPvv0 = pksz.lPvv(zindex=0)[0,:]
        lPvv = np.zeros((len(zs), lPvv0.shape[0]), dtype=lPvv0.dtype)
        lPvv[0,:] = lPvv0
        for zi in range(1, len(zs)):
            lPvv[zi,:] = pksz.lPvv(zindex=zi)[0,:]

        spec_dict['sPee'] = sPee
        spec_dict['lPvv'] = lPvv

    # If computing for a kSZ template, get P_gg^total, P_ge, and
    # P_gv on grids in z and k (and mu, if using RSD)
    else:

        # If no galaxy bias was specified, take from galaxy HOD
        if bgs is None:
            bgs = pksz.hods['g']['bg']

        # Get small-scale P_gg and P_ge (packed as [z,k] without RSD, or [z,k,mu] with
        # RSD)
        if verbose: print("Fetching small-scale P_gg and P_ge")
        sPgg_for_e = pksz.sPgg
        sPgg_for_v = sPgg_for_e.copy()
        spec_dict['sPgg'] = sPgg_for_e.copy()
        for zi, z in enumerate(zs):
            if pgg_noise_function is None:
                sPgg_for_e[zi] += 1/ngals_mpc3[zi]
                sPgg_for_v[zi] += 1/ngals_mpc3_for_v[zi]
            else:
                if rsd:
                    sPgg_for_e[zi] += pgg_noise_function(z, ks)[..., None]
                    sPgg_for_v[zi] += pgg_noise_function(z, ks)[..., None]
                else:
                    sPgg_for_e[zi] += pgg_noise_function(z, ks)
                    sPgg_for_v[zi] += pgg_noise_function(z, ks)
        sPge = pksz.sPge

        # Get large-scale P_gv and P_gg (packed as [z,k] without RSD, or [z,k,mu] with
        # RSD), by getting them for each z individually
        if verbose:
            print("Computing P_gv and large-scale P_gg")

        if rsd or (pksz.sigz is not None):
            lPgv = np.zeros((len(zs), len(pksz.ks), len(pksz.mus)))
            lPgg = np.zeros_like(lPgv)
            lPggtot = np.zeros_like(lPgv)

            for zi, z in enumerate(zs):
                # lPgv() and lPgg() returns results as [mu,k], so we need to transpose
                lPgv[zi] = pksz.lPgv(zi, bg=bgs[zi], rsd=rsd).T
                lPgg[zi] = pksz.lPgg(zi, bgs[zi], bgs[zi], rsd=rsd).T
                if pgg_noise_function is None:
                    lPggtot[zi] = lPgg[zi] + 1/ngals_mpc3_for_v[zi]
                else:
                    lPggtot[zi] = lPgg[zi] + pgg_noise_function(z, pksz.kLs)[..., None]
        else:
            lPgv = np.zeros((len(zs), len(pksz.ks)))
            lPgg = np.zeros_like(lPgv)
            lPggtot = np.zeros_like(lPgv)

            for zi, z in enumerate(zs):
                # lPgv() and lPgg() returns results as [mu,k], but each mu has the same
                # value, so we don't need the full mu axis
                lPgv[zi] = pksz.lPgv(zi, bg=bgs[zi], rsd=False)[0]
                lPgg[zi] = pksz.lPgg(zi, bgs[zi], bgs[zi], rsd=False)[0]
                if pgg_noise_function is None:
                    lPggtot[zi] = lPgg[zi] + 1/ngals_mpc3_for_v[zi]
                else:
                    lPggtot[zi] = lPgg[zi] + pgg_noise_function(z, pksz.kLs)

        # lPgv0 = pksz.lPgv(zindex=0,bg=bgs[0])[0,:]
        # lPgv = np.zeros((len(zs), lPgv0.shape[0]), dtype=lPgv0.dtype)
        # lPgv[0,:] = lPgv0
        # for zi in range(1, len(zs)):
        #     lPgv[zi,:] = pksz.lPgv(zindex=zi,bg=bgs[zi])[0,:]

        # # Same for large-scale Pgg
        # lPgg0 = pksz.lPgg(0,bgs[0],bgs[0])[0,:]
        # lPgg = np.zeros((len(zs), lPgg0.shape[0]), dtype=lPgg0.dtype)
        # lPgg[0,:] = lPgg0
        # for zi, z in enumerate(zs):
        #     lPgg[zi,:] = pksz.lPgg(zi,bgs[zi],bgs[zi])[0,:]
        #     if pgg_noise_function is None:
        #         lPgg[zi] += 1/ngals_mpc3_for_v[zi]
        #     else:
        #         lPgg[zi] += pgg_noise_function(z, pksz.kLs)

        spec_dict['sPggtot'] = sPgg_for_e.copy()
        spec_dict['sPge'] = sPge
        spec_dict['lPgv'] = lPgv
        spec_dict['lPgg'] = lPgg
        spec_dict['lPggtot'] = lPggtot

    # P_{q_r} will be packed as [k,z]
    if verbose: print('Computing P_{q_r} on grid in k,z')
    Pqr = np.zeros((ks.shape[0], zs.shape[0]))

    if template and use_pee_in_template:
        # In this case, P_ee won't have been computed before,
        # so we do it here
        sPee = pksz.get_power('e',name2='e',verbose=False)

    if verbose:
        print("Computing P_{q_r}")

    for zi, z in enumerate(zs):
        # Get P_gv^2 / P_gg^total or P_vv, and integrate in k (and mu if necessary)
        kls = pksz.kLs
        if template:
            if rsd or (pksz.sigz is not None):
                # Integrand at given z is packed as [k,mu]
                integrand = sanitize(
                    kls[:, None] ** 2
                    * pksz.mu[None, :] ** 2
                    * lPgv[zi] ** 2
                    / sPgg_for_v[zi]
                )
                # At each k, integrate over mu, multiplying by 3/2 such that the
                # later step of dividing by 6pi^2 actually divides by 4pi^2
                integrand = 1.5 * np.trapz(integrand, pksz.mu, axis=-1)
            else:
                # Integrand at given z is only function of k
                integrand = sanitize(kls ** 2 * lPgv[zi] ** 2 / sPgg_for_v[zi])
        else:
            # Integrand at given z is only function of k
            integrand = sanitize(kls ** 2 * lPvv[zi])

        # Integrate over k
        vint = np.trapz(integrand, kls)

        # Get P_ge^2 / P_gg^total or P_ee
        if template and not use_pee_in_template:
            if rsd or (pksz.sigz is not None):
                # Pge and Pgg at given z are packed as [k,mu]
                integrand = sPge[zi]**2 / sPgg_for_e[zi]
                if isotropize_muS:
                    # At each k, integrate over mu, multiplying by 1/2 so that we compute
                    # the monopole over mu
                    Pqr[:,zi] = 0.5 * np.trapz(integrand, pksz.mu, axis=-1)
                else:
                    # Take mu = 0
                    mu0_idx = np.argmin(np.abs(pksz.mu))
                    Pqr[:,zi] = integrand[:, mu0_idx]
            else:
                # Pge and Pgg at given z are function of k only
                Pqr[:,zi] = sPge[zi]**2 / sPgg_for_e[zi]
        else:
            Pqr[:,zi] = sPee[zi]

        # Multiply by numerical prefactor and integral from above
        Pqr[:,zi] *= (6*np.pi**2)**-1 * vint

    spec_dict['Pqr'] = Pqr

    # Make 2d interpolating function for P_{q_r}, with arguments z,k.
    # The resulting interpolating function automatically sorts arguments if
    # arrays are fed in, but we'll only call iPqperp with one (z,k) pair
    # at a time, so we'll be fine.
    iPqr = interp2d(zs, ks, Pqr, fill_value=0.)

    # Compute C_ell integral at each ell
    if verbose: print('Computing C_ell')
    cl = np.zeros(ells.shape[0])
    for iell, ell in enumerate(ells):

        # Set chi_min from min redshift, or from k=30Mpc^-1 if the lowest redshift
        # translates to k>30Mpc^-1
        chi_min = max(pksz.results.comoving_radial_distance(zs[0]), ell / _LIMBER_KMAX)
        # Set chi_max from max redshift
        chi_max = pksz.results.comoving_radial_distance(zs[-1])
        chi_int = np.geomspace(chi_min, chi_max, n_int)
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
        integrand *= (
            constants['thompson_SI']
            * ne0
            * 1/constants['meter_to_megaparsec']
        )**2
        integrand *= (pksz.pars.TCMB * 1e6)**2

        # If desired, save integrand for each ell to spec_dict.
        # We save the integrand w.r.t either chi and z, along with the
        # chi and z value it is evaluated at
        if save_cl_integrand:
            # Make empty arrays to hold output
            if iell == 0:
                spec_dict['ClkSZ_integrand_chi'] = np.zeros(
                    (len(ells), len(chi_int)), dtype=chi_int.dtype
                )
                spec_dict['ClkSZ_integrand_z'] = np.zeros_like(
                    spec_dict['ClkSZ_integrand_chi']
                )

                spec_dict['ClkSZ_dchi_integrand'] = np.zeros(
                    (len(ells), len(integrand)), dtype=integrand.dtype
                )
                spec_dict['ClkSZ_dz_integrand'] = np.zeros_like(
                    spec_dict['ClkSZ_dchi_integrand']
                )

            # Save integrand w.r.t. chi
            spec_dict['ClkSZ_integrand_chi'][iell] = chi_int
            spec_dict['ClkSZ_dchi_integrand'][iell] = integrand

            # Save integrand w.r.t z, which is dchi/dz * integrand_dchi
            spec_dict['ClkSZ_integrand_z'][iell] = z_int
            _DERIV_DZ = 0.001
            dchi_dz_int = (
                (
                    pksz.results.comoving_radial_distance(z_int + _DERIV_DZ)
                    - pksz.results.comoving_radial_distance(z_int - _DERIV_DZ)
                ) / (2 * _DERIV_DZ)
            )
            spec_dict['ClkSZ_dz_integrand'][iell] = integrand * dchi_dz_int

        if not slow_chi_integral:
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


def get_ksz_auto_squeezed_m_integrand(
    ells,
    volume_gpc3,
    zs,
    ngals_mpc3,
    params=None,
    k_max = 100.,
    num_k_bins = 200,
    num_mu_bins=102,
    ms=None,
    mass_function="sheth-torman",
    mdef='vir',
    nfw_numeric=False,
    electron_profile_family='AGN',
    electron_profile_nxs=None,
    electron_profile_xmax=None,
    hod_family=hod.Leauthaud12_HOD,
    n_int=100,
    verbose=False,
    pksz_in=None,
    save_cl_integrand=False,
    physical_truncate=True,
):
    """Compute kSZ angular auto power spectrum M derivative, dC_\ell / dM_halo.

    We use the squeezed limit from Ma & Fry, with some altered notation:

        C_\ell = \int \frac{d\chi}{\chi^2 H_0^2} \tilde{K}(z[\chi])^2
                 P_{q_r}(k=\ell/\chi, \chi)

        \tilde{K}(z) = T_{CMB} \bar{n}_{e,0} \sigma_T (1+z)^2 \exp(-\tau(z))

        P_{q_r}(k,z) = \frac{1}{6\pi^2} \int dk' (k')^2 P_{vv}(k',z) P_{ee}(k,z)

    dC_\ell / dM_halo is returned in uK^2 Msun^-1.

    Parameters
    ----------
    ells : array_like
        Array of ell values to compute C_ell at.
    volumes_gpc3, ngals_mpc3 : array_like
        Arrays of comoving volume (in Gpc^3) and galaxy number density (in Mpc^3)
        corresponding to redshifts in zs.
    zs : array_like
        Array of redshifts to compute at.
    bgs : array_like
        Array of linear galaxy bias at each redshift.
    **params : dict, optional
        Optional dict of parameters for halo model and radial kSZ weight computations.
        Default: None.
    k_max : float, optional
        Maximum k to consider as k_long and k_short (same value used for both).
        Default: 100.
    num_k_bins : int, optional
        Number of k_long and k_short bins for computations. Default: 200.
    num_mu_bins : int, optional
        Number of mu bins for computations. Bins will be linear between -1 and 1.
        Default: 102.
    ms : array_like, optional
        Array of halo masses to compute over, in Msun.
        Default: Log-spaced array determined by parameters in `defaults` dict.
    mass_function : {'sheth-torman', 'tinker'}, optional
        Mass function to use. Default: 'sheth-torman'
    mdef : {'vir', 'mean'}
        Halo mass definition. 'mean' = M200m.
    nfw_numeric : bool, optional
        Compute Fourier transform of NFW profile numerically instead of analytically.
        Default : False.
    electron_profile_family : {'AGN', 'SH', 'pres'}, optional
        Electron profile to use. Default: 'AGN'.
    electron_profile_nxs : int, optional
        Number of samples of electron profile for FFT. Default: None.
    electron_profile_xmax : float, optional
        X_max for electron profile in FFT. Default: None.
    hod_family : hod.HODBase, optional
        Name of HOD class. Default: hod.Leauthaud12_HOD.
    n_int : int, optional
        Number of samples to use in Limber integral. Default: 100.
    verbose : bool, optional
        Print progress of computations. Default: False.
    pksz_in : kSZ object, optional
        Predefined kSZ object to use for computations, instead of initializing
        a new one. Default: None.
    save_cl_integrand : bool, optional
        Save integrand of C_ell to spec_dict, along with coordinate values
        that integrand was evaluated at, w.r.t. either chi or z.
    physical_truncate : bool, optional
        Truncate the electron (gas) profile such that the enclosed mass is equal to
        f_b m_200c. Default: True.

    Returns
    -------
    pksz : kSZ object
        kSZ object initialized during the routine. (Can be used as input to
        later calls to save time.)
    cl : np.ndarray
        Computed dC_ell^kSZ / dM_halo, evaluated at input ells.
        Packed as [ell, m].
    **spec_dict : dict
        Dict containing various 3d power spectra used for kSZ computation.
    """

    # Define empty dict for storing spectra
    spec_dict = {}

    # Widen search range for setting lower mass threshold from nbar
    if params is None:
        params = default_params
    params['hod_bisection_search_min_log10mthresh'] = 1

    # Make sure input redshifts are sorted
    zs = np.sort(np.asarray(zs))

    # Make array for volumes, for feeding to kSZ object
    volumes_gpc3 = volume_gpc3 * np.ones_like(zs)

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
            hod_family=hod_family,
            verbose=verbose,
            physical_truncate=physical_truncate,
        )

    # Get k_short values that P_{q_perp} integrand is evaluated at
    ks = pksz.kS
    spec_dict['ks'] = ks

    # Get P_ee (packed as [z,M,k])
    sPee = pksz.get_power('e', name2='e', verbose=False, m_integrand=True)

    # Get P_vv (packed as [z,k]), by getting it for each z individually
    lPvv0 = pksz.lPvv(zindex=0)[0,:]
    lPvv = np.zeros((len(zs), lPvv0.shape[0]), dtype=lPvv0.dtype)
    lPvv[0,:] = lPvv0
    for zi in range(1, len(zs)):
        lPvv[zi,:] = pksz.lPvv(zindex=zi)[0,:]

    spec_dict['sPee'] = sPee
    spec_dict['lPvv'] = lPvv

    # Compute P_{q_r} values on grid in k,M,z
    if verbose: print('Computing P_{q_r} on grid in k,M,z')
    Pqr = np.zeros((ks.shape[0], pksz.ms.shape[0], zs.shape[0]))
    for zi,z in enumerate(zs):

        # Get P_vv, and integrate in k
        kls = pksz.kLs
        integrand = sanitize((kls**2.)*lPvv[zi])
        vint = np.trapz(integrand,kls)

        # Get P_ee, packed as [k,M,z]
        Pqr[:,:,zi] = sPee[zi].T

        # Multiply by numerical prefactor and integral from above
        Pqr[:,:,zi] *= (6*np.pi**2)**-1 * vint

    log10ms = np.log10(pksz.ms)
    log10ks = np.log10(ks)

    if verbose: print('Computing C_ell')
    cl = np.zeros((len(ells), len(log10ms)))
    if save_cl_integrand:
        spec_dict['ClkSZ_integrand_chi'] = np.zeros(
            (len(ells), n_int), dtype=np.float64
        )
        spec_dict['ClkSZ_integrand_z'] = np.zeros_like(
            spec_dict['ClkSZ_integrand_chi']
        )

        spec_dict['ClkSZ_dchi_integrand'] = np.zeros(
            (len(ells), len(log10ms), n_int), dtype=np.float64
        )
        spec_dict['ClkSZ_dz_integrand'] = np.zeros_like(
            spec_dict['ClkSZ_dchi_integrand']
        )

    # Loop over M. This is slow, but enables us to use scipy's interp2d
    # to interpolate over z,k at each M
    for ilogm, logm in enumerate(log10ms):
        nm = len(log10ms)
        if verbose and (ilogm % (nm // 10) == 0):
            print(f"\tComputing for M {ilogm} / {len(log10ms)}")
        iPqr = interp2d(zs, log10ks, Pqr[:, ilogm, :], fill_value=0.)

        # Compute C_ell integral at each ell
        for iell,ell in enumerate(ells):

            # Set chi_min based on k=30Mpc^-1, and chi_max from max redshift
            chi_min = ell/30.
            chi_max = pksz.results.comoving_radial_distance(zs[-1])
            chi_int = np.geomspace(chi_min, chi_max, n_int)
            k_int = ell/chi_int
            z_int = pksz.results.redshift_at_comoving_radial_distance(chi_int)

            # Get integrand evaluated at z,k corresponding to Limber integral
            integrand = np.zeros(k_int.shape[0])
            for ki,k in enumerate(k_int):
                # integrand[ki] = iPqr(z_int[ki], k)
                integrand[ki] = iPqr(z_int[ki], np.log10(k))
            integrand /= chi_int**2
            integrand *= (1+z_int)**4

            # Include prefactors
            ne0 = ne0_shaw(pksz.pars.ombh2, pksz.pars.YHe)
            # Units: (m^2 * m^-3 * Mpc^-1 m^1)^2
            integrand *= (
                constants['thompson_SI']
                * ne0
                * 1/constants['meter_to_megaparsec']
            )**2
            integrand *= (pksz.pars.TCMB * 1e6)**2

            # If desired, save integrand for each ell to spec_dict.
            # We save the integrand w.r.t either chi and z, along with the
            # chi and z value it is evaluated at
            if save_cl_integrand:

                # Save integrand w.r.t. chi
                spec_dict['ClkSZ_integrand_chi'][iell] = chi_int
                spec_dict['ClkSZ_dchi_integrand'][iell, ilogm] = integrand

                # Save integrand w.r.t z, which is dchi/dz * integrand_dchi
                spec_dict['ClkSZ_integrand_z'][iell] = z_int
                _DERIV_DZ = 0.001
                dchi_dz_int = (
                    (
                        pksz.results.comoving_radial_distance(z_int + _DERIV_DZ)
                        - pksz.results.comoving_radial_distance(z_int - _DERIV_DZ)
                    ) / (2 * _DERIV_DZ)
                )
                spec_dict['ClkSZ_dz_integrand'][iell, ilogm] = integrand * dchi_dz_int

            # Do C_ell chi integral via trapezoid rule
            cl[iell, ilogm] = np.trapz(integrand, chi_int)

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


def get_ksz_halomodel_spectra(pksz, b1=None, rsd=False, fog=False):
    """Get separate 1h and 2h terms for P_ee, P_ge, and P_gg.

    Parameters
    ----------
    pksz : kSZ object
        Predefined kSZ object to use for computations.
    b1 : array_like, optional
        Linear galaxy bias at desired redshifts. Default: None.
    rsd : bool, optional
        Whether to include (Kaiser) RSD. Default: False.
    fog : bool, optional
        Whether to include Fingers of God. Default: False.

    Returns
    -------
    **spec_dict : dict
        Dict containing 3d power spectra.
    """

    spec_dict = {}

    ks = pksz.kS
    spec_dict['ks'] = pksz.kS

    spec_dict["sPee_1h"] = pksz.get_power_1halo('e', name2='e')
    spec_dict["sPee_2h"] = pksz.get_power_2halo('e', name2='e')

    spec_dict["sPgg_shot"] = 1/pksz.ngals_mpc3
    spec_dict["sPgg_1h"] = pksz.get_power_1halo('g', name2='g', fog=fog)
    spec_dict["sPgg_2h"] = pksz.get_power_2halo(
        'g', name2='g', rsd=rsd, fog=fog, b1_in=b1, b2_in=b1
    )

    spec_dict["sPge_1h"] = pksz.get_power_1halo('g', name2='e', fog=fog)
    spec_dict["sPge_2h"] = pksz.get_power_2halo(
        'g', name2='e', rsd=rsd, fog=fog, b1_in=b1
    )

    return spec_dict
