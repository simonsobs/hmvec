from .params import default_params
from . import utils
from .cosmology import Cosmology
from .mstar_mhalo import Mstellar_halo, Mhalo_stellar

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, dblquad
from scipy.special import erf, erfc

"""
HOD class and related routines.

Implementation of Leauthaud12_HOD is based on code from to Matt Johnson and Moritz
Munchmeyer.
"""


class HODBase(Cosmology):
    """Base class for halo occupation distribution.

    The following methods must be defined by subclasses:
        - hod_params
        - avg_Nc
        - avg_Ns
        - avg_NsNsm1
        - avg_NcNs
    Also, __init__ of subclass must call init_mean_occupations() as final step.

    Parameters
    ----------
    zs : array_like, 1d
        Array of redshifts to compute at.
    ms : array_like, 1d
        Array of halo masses to compute over, in Msun.
    **params : dict, optional
        Dictionary of HOD parameters, likely inherited from calling
        HaloModel object. Default: None
    **param_override : dict, optional
        Dictionary of override parameters. Default: None.
    halofit : int, optional
        If anything other than None, use Halofit for matter power spectrum.
        Default: None
    corr : string, optional
        Either "min" or "max", describing correlations in central-satellite model.
        Default: "max"
    nzm : array_like,
        Precomputed halo mass function, packed as [z,m]. Default: None.
    """

    def __init__(
        self,
        zs,
        ms,
        params=None,
        param_override=None,
        halofit=None,
        corr="max",
        nzm=None,
        **kwargs
    ):

        # Store redshifts, halo masses, corr model, halo mass function
        self.zs = zs
        self.ms = ms
        self.log10mhalo = np.log10(self.ms)
        self.corr = corr
        self.nzm = nzm

        self.Nc = None
        self.Ns = None

        # Initialize cosmology
        Cosmology.__init__(self, params, halofit)

        # Set internal params based on input params, or default params if input
        # params not specified
        self.p = {}
        for ip in self.hod_params:
            if params is not None and ip in params:
                self.p[ip] = params[ip]
            else:
                try:
                    self.p[ip] = default_params[ip]
                except:
                    raise ValueError("HOD parameter %s is has no default value!" % ip)

        # If necessary, override specific parameter values
        if param_override is not None:
            for ip in param_override:
                if ip in self.hod_params:
                    self.p[ip] = param_override[ip]
                else:
                    raise ValueError(
                        "Override parameter %s is not defined for this HOD!" % ip
                    )

    def init_mean_occupations(self):
        """Precompute mean occupations of centrals and satellites."""
        self.Nc = self.avg_Nc(self.log10mhalo, self.zs)
        self.Ns = self.avg_Ns(self.log10mhalo, self.zs)
        self.NsNsm1 = self.avg_NsNsm1(self.Nc, self.Ns, self.corr)
        self.NcNs = self.avg_NcNs(self.Nc, self.Ns, self.corr)

    @property
    def hod_params(self):
        """List of HOD-specific parameter names."""
        pass

    def avg_Nc(self, log10mhalo, z, **kwargs):
        """<N_c | m_h>."""
        pass

    def avg_Ns(self, log10mhalo, z, **kwargs):
        """<N_s | m_h>"""
        pass

    def avg_NsNsm1(self, Nc, Ns, corr="max"):
        """<N_s N_s | m_h>.

        Parameters
        ----------
        Nc, Ns : array_like
            Arrays of Nc and Ns, packed as [z,m].
        corr : string, optional
            Either "min" or "max", describing correlations in central-satellite model.
            Default: "max"

        Returns
        -------
        NsNs : array_like
            N_s N_s term, packed as [z,m].
        """
        if corr not in ["min", "max"]:
            raise ValueError("Invalid corr argument (%s) passed to avg_NcNs!" % corr)

        if corr == "max":
            ret = Ns ** 2.0 / Nc
            ret[np.isclose(Nc, 0.0)] = 0  # FIXME: is this kosher?
            return ret
        elif corr == "min":
            return Ns ** 2.0

    def avg_NcNs(self, Nc, Ns, corr="max"):
        """<N_c N_s | m_h>.

        Parameters
        ----------
        Nc, Ns : array_like
            Arrays of Nc and Ns, packed as [z,m].
        corr : string, optional
            Either "min" or "max", describing correlations in central-satellite model.
            Default: "max"

        Returns
        -------
        NcNs : array_like
            N_c N_s term, packed as [z,m].
        """
        if corr not in ["min", "max"]:
            raise ValueError("Invalid corr argument (%s) passed to avg_NcNs!" % corr)

        if corr == "max":
            return Ns
        elif corr == "min":
            return Ns * Nc

    def compute_ngal(self):
        """Compute n_g(z).

        Performs \int n(m_h) [N_c(m_h) + N_s(m_h)].
        """
        try:
            integrand = self.nzm * (self.Nc + self.Ns)
        except:
            raise RuntimeError("One of nzm, Nc, and Ns is missing!")
        return np.trapz(integrand, self.ms, axis=-1)


class Leauthaud12_HOD(HODBase):
    """HOD from Leauthaud et al. 2012 (1104.0928).

    This HOD was used for the kSZ computations in Smith et al. 2018
    (1810.13423), and was also the only HOD used in a previous version of
    hmvec.

    Parameters
    ----------
    mthresh : array_like, optional
        Array of lower stellar mass threshold values in Msun as function of z, packed
        as [z,m]. Overrides ngal. If neither mthresh nor ngal are specified, a
        z-independent value of 10^10.5 Msun is used. Default: None.
    ngal : array_like, optional
        Used to set mthresh by inverting the n_gal(m_*) relation. Should also be a
        function of z. Default: None.
    """

    def __init__(
        self,
        zs,
        ms,
        params=None,
        param_override=None,
        halofit=None,
        corr="max",
        nzm=None,
        mthresh=None,
        ngal=None,
    ):

        # Run base initialization
        super().__init__(
            zs,
            ms,
            params=params,
            param_override=param_override,
            halofit=halofit,
            corr=corr,
            nzm=nzm,
        )

        # mthresh overrides ngal, if both are specified
        if mthresh is not None:
            self.log10mstellar_thresh = np.log10(mthresh)
            if mthresh.ndim == 1:
                self.log10mstellar_thresh = np.expand_dims(
                    np.log10(self.log10mstellar_thresh), axis=-1
                )

        # If ngal is specified, need to compute corresponding mthresh
        elif ngal is not None:

            # Check for input halo mass function
            if nzm is None:
                raise ValueError("nzm must be specified if ngal is specified")
            self.nzm = nzm

            # Check input parameters
            if ngal.size != zs.size:
                raise ValueError("ngal and zs must have same size")
            if mthresh is not None:
                raise ValueError("Can only specify one of ngal or mthresh")

            # Set overrides for Msat and Mcut
            if "hod_Leau12_Msat_override" not in self.p.keys():
                self.p["hod_Leau12_Msat_override"] = None
            if "hod_Leau12_Mcut_override" not in self.p.keys():
                self.p["hod_Leau12_Mcut_override"] = None

            # Define n_gal(m_thresh) function
            nfunc = lambda ilog10mthresh: self.ngal_from_mthresh(
                ilog10mthresh, self.zs, self.ms
            )

            # Compute m_thresh from input n_gal
            log10mthresh = utils.vectorized_bisection_search(
                ngal,
                nfunc,
                [
                    self.p["hod_bisection_search_min_log10mthresh"],
                    self.p["hod_bisection_search_max_log10mthresh"],
                ],
                "decreasing",
                rtol=self.p["hod_bisection_search_rtol"],
                verbose=True,
                hang_check_num_iter=self.p["hod_bisection_search_warn_iter"],
            )
            mthresh = 10 ** (log10mthresh * self.p["hod_Leau12_A_log10mthresh"])
            self.log10mstellar_thresh = np.log10(mthresh)[:, None]

        # If ngal and mthresh are None, use default M_* threshold
        else:
            self.log10mstellar_thresh = (
                self.p["hod_Leau12_log10mstellar_thresh"] * np.ones_like(zs)[:, None]
            )

        # Finally, compute halo occupations
        self.init_mean_occupations()

    @property
    def hod_params(self):
        return [
            "hod_Leau12_sig_log_mstellar",
            "hod_bisection_search_min_log10mthresh",
            "hod_bisection_search_max_log10mthresh",
            "hod_bisection_search_rtol",
            "hod_bisection_search_warn_iter",
            "hod_Leau12_alphasat",
            "hod_Leau12_Bsat",
            "hod_Leau12_betasat",
            "hod_Leau12_Bcut",
            "hod_Leau12_betacut",
            "hod_Leau12_A_log10mthresh",
            "hod_Leau12_log10mstellar_thresh",
        ]

    def avg_Nc(self, log10mhalo, z, log10mstellar_thresh=None):
        """<N_c | m_h>.

        Parameters
        ----------
        log10mhalo : array_like
            Array of log10(m_h) values.
        z : array_like
            Array of redshifts.
        log10mstellar_thresh : array_like, optional
            Array of log10 of lower stellar mass threshold, as function of z (packed as
            [z,log10m]). If not specified, stored value is pulled from class.
            Default: None.

        Returns
        -------
        N_c : array_like
            Computed N_c values, packed as [z,m].
        """
        if log10mstellar_thresh is None:
            log10mstellar_thresh = self.log10mstellar_thresh

        if log10mstellar_thresh.ndim != 2:
            raise RuntimeError(
                "log10mstellar_thresh must be packed as [z,log10m]!"
                " Instead, it has shape",
                log10mstellar_thresh.shape,
            )

        log10mstar = Mstellar_halo(z, np.tile(log10mhalo, (len(self.zs), 1)))
        num = log10mstellar_thresh - log10mstar
        denom = np.sqrt(2.0) * self.p["hod_Leau12_sig_log_mstellar"]
        return 0.5 * (1.0 - erf(num / denom))

    def avg_Ns(self, log10mhalo, z, log10mstellar_thresh=None, Nc=None):
        """<N_s | m_h>.

        Parameters
        ----------
        log10mhalo : array_like
            Array of log10(m_h) values.
        z : array_like
            Array of redshifts.
        log10mstellar_thresh : array_like, optional
            Array of log10 of lower stellar mass threshold at each z, packed as
            [z,log10m]. If not specified, stored value is pulled from class.
            Default: None.
        N_c : array_like, optional
            Array of N_c, packed as [z,m]. If not specified, computed on the fly.
            Default: None.

        Returns
        -------
        N_s : array_like
            Computed N_s values, packed as [z,m].
        """
        masses = 10 ** log10mhalo
        if log10mstellar_thresh is None:
            log10mstellar_thresh = self.log10mstellar_thresh

        if log10mstellar_thresh.ndim != 2:
            raise RuntimeError(
                "log10mstellar_thresh must be packed as [z,log10m]!"
                " Instead, it has shape",
                log10mstellar_thresh.shape,
            )
        log10mthresh = Mhalo_stellar(z, log10mstellar_thresh)

        if (
            "hod_Leau12_Msat_override" in self.p.keys()
            and self.p["hod_Leau12_Msat_override"] is not None
        ):

            Msat = self.p["hod_Leau12_Msat_override"]
        else:
            Msat = self.hod_default_mfunc(
                log10mthresh, self.p["hod_Leau12_Bsat"], self.p["hod_Leau12_betasat"]
            )

        if (
            "hod_Leau12_Mcut_override" in self.p.keys()
            and self.p["hod_Leau12_Mcut_override"] is not None
        ):
            Mcut = self.p["hod_Leau12_Mcut_override"]
        else:
            Mcut = self.hod_default_mfunc(
                log10mthresh, self.p["hod_Leau12_Bcut"], self.p["hod_Leau12_betacut"]
            )

        if Nc is None:
            if self.Nc is not None:
                Nc = self.Nc
            else:
                Nc = self.avg_Nc(log10mhalo, z)

        return (
            Nc
            * ((masses / Msat) ** self.p["hod_Leau12_alphasat"])
            * np.exp(-Mcut / masses)
        )

    def hod_default_mfunc(self, log10mthresh, Bamp, Bind):
        """Internal Leauthaud12 function for Msat and Mcut parameters."""
        return (10.0 ** (12.0)) * Bamp * 10 ** ((log10mthresh - 12) * Bind)

    def ngal_from_mthresh(
        self,
        log10mthresh=None,
        zs=None,
        ms=None,
        Nc=None,
        Ns=None,
    ):
        """Compute n_gal from stellar mass.

        Can either specify log10mthresh, zs, and ms, in which case N_c and N_s are
        computed on the fly, or can specify Nc and Ns directly.

        Parameters
        ----------
        log10mthresh : array_like, optional
            Array of log10 of stellar mass values, in Msun, packed as [log10m].
            Default: None.
        zs : array_like, optional
            Array of redshifts. Default: None.
        ms : array_like, optional
            Array of halo masses, in Msun. Default: None.
        Nc : array_like, optional
            Array of <N_c | m_h>, packed as [z,m]. Default: None.
        Ns : array_like, optional
            Array of <N_s | m_h>, packed as [z,m]. Default: None.

        Returns
        -------
        ngal : array_like
            Array of n_gal values, as function of z.
        """
        if self.nzm is None:
            raise ValueError("Need a stored halo mass function for ngal_from_mthresh")

        if (Nc is None) and (Ns is None):
            log10mstellar_thresh = log10mthresh[:, None]
            Nc = self.avg_Nc(self.log10mhalo, self.zs, log10mstellar_thresh)
            Ns = self.avg_Ns(
                self.log10mhalo,
                self.zs,
                log10mstellar_thresh,
                Nc,
            )
        else:
            if not (log10mthresh is None and zs is None):
                raise ValueError(
                    "ngal_from_mthresh called with strange combination of arguments"
                )

        integrand = self.nzm * (Nc + Ns)
        return np.trapz(integrand, ms, axis=-1)


class Alam20_HMQ_ELG_HOD(HODBase):
    """HMQ ELG HOD from Alam et al. 2020 (1910.05095)."""

    def __init__(
        self,
        zs,
        ms,
        params=None,
        param_override=None,
        halofit=None,
        corr="max",
        nzm=None,
        mthresh=None,
        ngal=None,
    ):

        # Run base initialization
        super().__init__(
            zs,
            ms,
            params=params,
            param_override=param_override,
            halofit=halofit,
            corr=corr,
            nzm=nzm,
        )

        # Precompute model parameter A
        pmax = self.p["hod_Alam20_HMQ_ELG_pmax"]
        Q = self.p["hod_Alam20_HMQ_ELG_Q"]
        gamma = self.p["hod_Alam20_HMQ_ELG_gamma"]
        log10mthresh0 = self.p["hod_Alam20_HMQ_ELG_log10Mc"]
        self.A = (pmax - 1 / Q) / np.max(
            2
            * self._phi(self.log10mhalo[:, None], log10mthresh=log10mthresh0)
            * self._Phigamma(self.log10mhalo[:, None], log10mthresh=log10mthresh0)
        )

        if mthresh is not None:
            self.log10mthresh = np.log10(mthresh)
            if mthresh.ndim == 1:
                self.log10mthresh = np.expand_dims(np.log10(self.log10mthresh), axis=-1)

        # If ngal is specified, need to compute corresponding mthresh
        elif ngal is not None:

            # Check for input halo mass function
            if nzm is None:
                raise ValueError("nzm must be specified if ngal is specified")
            self.nzm = nzm

            # Check input parameters
            if ngal.size != zs.size:
                raise ValueError("ngal and zs must have same size")
            if mthresh is not None:
                raise ValueError("Can only specify one of ngal or mthresh")

            # Define n_gal(m_thresh) function
            nfunc = lambda ilog10mthresh: self.ngal_from_mthresh(
                ilog10mthresh, self.zs, self.ms
            )

            # Compute m_thresh from input n_gal
            log10mthresh = utils.vectorized_bisection_search(
                ngal,
                nfunc,
                [
                    self.p["hod_bisection_search_min_log10mthresh"],
                    self.p["hod_bisection_search_max_log10mthresh"],
                ],
                "decreasing",
                rtol=self.p["hod_bisection_search_rtol"],
                verbose=True,
                hang_check_num_iter=self.p["hod_bisection_search_warn_iter"],
            )
            self.log10mthresh = log10mthresh[:, None]

        # If ngal and mthresh are None, use default M_c parameter
        else:
            self.log10mthresh = (
                self.p["hod_Alam20_HMQ_ELG_log10Mc"] * np.ones_like(zs)[:, None]
            )

        # Finally, compute halo occupations
        self.init_mean_occupations()

    @property
    def hod_params(self):
        return [
            "hod_bisection_search_min_log10mthresh",
            "hod_bisection_search_max_log10mthresh",
            "hod_bisection_search_rtol",
            "hod_bisection_search_warn_iter",
            "hod_Alam20_HMQ_ELG_log10Mc",
            "hod_Alam20_HMQ_ELG_sigmaM",
            "hod_Alam20_HMQ_ELG_gamma",
            "hod_Alam20_HMQ_ELG_Q",
            "hod_Alam20_HMQ_ELG_log10M1",
            "hod_Alam20_HMQ_ELG_kappa",
            "hod_Alam20_HMQ_ELG_alpha",
            "hod_Alam20_HMQ_ELG_pmax",
            # "hod_Alam20_HMQ_ELG_A",
        ]

    def _phi(self, x, log10mthresh=None):
        if log10mthresh is None:
            log10mthresh = self.log10mthresh

        sigmaM = self.p["hod_Alam20_HMQ_ELG_sigmaM"]
        return (2 * np.pi * sigmaM ** 2) ** -0.5 * np.exp(
            -((x - log10mthresh) ** 2) / (2 * sigmaM ** 2)
        )

    def _Phigamma(self, x, log10mthresh=None):
        if log10mthresh is None:
            log10mthresh = self.log10mthresh

        sigmaM = self.p["hod_Alam20_HMQ_ELG_sigmaM"]
        return 0.5 * (
            1
            + erf(
                self.p["hod_Alam20_HMQ_ELG_gamma"]
                * (x - log10mthresh)
                / (2 ** 0.5 * sigmaM)
            )
        )

    def avg_Nc(self, log10mhalo, z, log10mthresh=None):
        if log10mthresh is None:
            log10mthresh = self.log10mthresh

        term1 = (
            2
            # * self.p["hod_Alam20_HMQ_ELG_A"]
            * self.A
            * self._phi(log10mhalo[None, :], log10mthresh=log10mthresh)
            * self._Phigamma(log10mhalo[None, :], log10mthresh=log10mthresh)
        )
        term2 = (
            1
            / (2 * self.p["hod_Alam20_HMQ_ELG_Q"])
            * (1 + erf((log10mhalo[None, :] - log10mthresh) / 0.01))
        )

        return term1 + term2

    def avg_Ns(self, log10mhalo, z, log10mthresh=None):
        masses = 10 ** log10mhalo
        if log10mthresh is None:
            log10mthresh = self.log10mthresh

        kappa = self.p["hod_Alam20_HMQ_ELG_kappa"]
        alpha = self.p["hod_Alam20_HMQ_ELG_alpha"]
        M1 = 10 ** self.p["hod_Alam20_HMQ_ELG_log10M1"]
        mthresh = 10 ** log10mthresh

        # TODO: rewrite this part more pythonically
        # Make empty Ns array, packed as [m,z]
        Ns = np.zeros((masses.shape[0], mthresh.shape[0]))
        # Loop over z
        for i in range(Ns.shape[1]):
            # Make mask to select masses greater than threshold
            mask = masses > mthresh[i]
            Ns[:, i][mask] = (
                (masses[masses > mthresh[i]] - kappa * mthresh[i]) / M1
            ) ** alpha
        Ns = Ns.T

        return Ns

    def ngal_from_mthresh(
        self,
        log10mthresh=None,
        zs=None,
        ms=None,
        Ncs=None,
        Nss=None,
    ):
        if self.nzm is None:
            raise ValueError("Need a stored halo mass function for ngal_from_mthresh")

        if (Ncs is None) and (Nss is None):
            log10mthresh = log10mthresh[:, None]
            Ncs = self.avg_Nc(self.log10mhalo, self.zs, log10mthresh)
            Nss = self.avg_Ns(self.log10mhalo, self.zs, log10mthresh)

        integrand = self.nzm * (Ncs + Nss)
        return np.trapz(integrand, ms, axis=-1)


class Alam20_ErfBase_HOD(HODBase):
    """Base class for LRG/QSO HOD from Alam et al. 2020 (1910.05095).

    The LRG and QSO HODs have the same functional forms but different parameters, so
    this class provides a common base for defining these HODs via derived classes.

    Each subclass must define a "tracer" method that returns the tracer type.
    """

    def __init__(
        self,
        zs,
        ms,
        params=None,
        param_override=None,
        halofit=None,
        corr="max",
        nzm=None,
        mthresh=None,
        ngal=None,
    ):

        # Run base initialization
        super().__init__(
            zs,
            ms,
            params=params,
            param_override=param_override,
            halofit=halofit,
            corr=corr,
            nzm=nzm,
        )

        if mthresh is not None:
            self.log10mthresh = np.log10(mthresh)
            if mthresh.ndim == 1:
                self.log10mthresh = np.expand_dims(np.log10(self.log10mthresh), axis=-1)

        # If ngal is specified, need to compute corresponding mthresh
        elif ngal is not None:

            # Check for input halo mass function
            if nzm is None:
                raise ValueError("nzm must be specified if ngal is specified")
            self.nzm = nzm

            # Check input parameters
            if ngal.size != zs.size:
                raise ValueError("ngal and zs must have same size")
            if mthresh is not None:
                raise ValueError("Can only specify one of ngal or mthresh")

            # Define n_gal(m_thresh) function
            nfunc = lambda ilog10mthresh: self.ngal_from_mthresh(
                ilog10mthresh, self.zs, self.ms
            )

            # Compute m_thresh from input n_gal
            log10mthresh = utils.vectorized_bisection_search(
                ngal,
                nfunc,
                [
                    self.p["hod_bisection_search_min_log10mthresh"],
                    self.p["hod_bisection_search_max_log10mthresh"],
                ],
                "decreasing",
                rtol=self.p["hod_bisection_search_rtol"],
                verbose=True,
                hang_check_num_iter=self.p["hod_bisection_search_warn_iter"],
            )
            self.log10mthresh = log10mthresh[:, None]

        # If ngal and mthresh are None, use default M_c parameter
        else:
            self.log10mthresh = (
                self.p["hod_Alam20_HMQ_%s_log10Mc" % self.tracer]
                * np.ones_like(zs)[:, None]
            )

        # Finally, compute halo occupations
        self.init_mean_occupations()

    @property
    def tracer(self):
        pass

    @property
    def hod_params(self):
        return [
            "hod_bisection_search_min_log10mthresh",
            "hod_bisection_search_max_log10mthresh",
            "hod_bisection_search_rtol",
            "hod_bisection_search_warn_iter",
            "hod_Alam20_HMQ_%s_log10Mc" % self.tracer,
            "hod_Alam20_HMQ_%s_sigmaM" % self.tracer,
            "hod_Alam20_HMQ_%s_log10M1" % self.tracer,
            "hod_Alam20_HMQ_%s_kappa" % self.tracer,
            "hod_Alam20_HMQ_%s_alpha" % self.tracer,
            "hod_Alam20_HMQ_%s_pmax" % self.tracer,
        ]

    def avg_Nc(self, log10mhalo, z, log10mthresh=None):
        if log10mthresh is None:
            log10mthresh = self.log10mthresh

        return (
            0.5
            * self.p["hod_Alam20_HMQ_%s_pmax" % self.tracer]
            * erfc(
                (log10mthresh - log10mhalo[None, :])
                / (
                    2 ** 0.5
                    * np.log10(np.e)
                    * self.p["hod_Alam20_HMQ_%s_sigmaM" % self.tracer]
                )
            )
        )

    def avg_Ns(self, log10mhalo, z, log10mthresh=None):
        masses = 10 ** log10mhalo
        if log10mthresh is None:
            log10mthresh = self.log10mthresh

        kappa = self.p["hod_Alam20_HMQ_%s_kappa" % self.tracer]
        alpha = self.p["hod_Alam20_HMQ_%s_alpha" % self.tracer]
        M1 = 10 ** self.p["hod_Alam20_HMQ_%s_log10M1" % self.tracer]
        mthresh = 10 ** log10mthresh

        # TODO: rewrite this part more pythonically
        # Make empty Ns array, packed as [m,z]
        Ns = np.zeros((masses.shape[0], mthresh.shape[0]))
        # Loop over z
        for i in range(Ns.shape[1]):
            # Make mask to select masses greater than threshold
            mask = masses > mthresh[i]
            Ns[:, i][mask] = (
                (masses[masses > mthresh[i]] - kappa * mthresh[i]) / M1
            ) ** alpha
        Ns = Ns.T

        return Ns

    def ngal_from_mthresh(
        self,
        log10mthresh=None,
        zs=None,
        ms=None,
        Ncs=None,
        Nss=None,
    ):
        if self.nzm is None:
            raise ValueError("Need a stored halo mass function for ngal_from_mthresh")

        if (Ncs is None) and (Nss is None):
            log10mthresh = log10mthresh[:, None]
            Ncs = self.avg_Nc(self.log10mhalo, self.zs, log10mthresh)
            Nss = self.avg_Ns(self.log10mhalo, self.zs, log10mthresh)

        integrand = self.nzm * (Ncs + Nss)
        return np.trapz(integrand, ms, axis=-1)


class Alam20_LRG_HOD(Alam20_ErfBase_HOD):
    """LRG HOD from Alam et al. 2020 (1910.05095)."""

    @property
    def tracer(self):
        return "LRG"


class Alam20_QSO_HOD(Alam20_ErfBase_HOD):
    """QSO HOD from Alam et al. 2020 (1910.05095)."""

    @property
    def tracer(self):
        return "QSO"


class Zheng05Base_HOD(HODBase):
    """Class for the HOD from Zheng et al. 2005 (astro-ph/0408564).

    Note that when using ngal to determine mthresh, the bisection search often fails
    to converge if rtol=1e-4, so overriding the value of hod_bisection_search_rtol
    to something higher (e.g. 1e-2) may be necessary.

    This class specifies everything except for the normalization of N_sat. A derived
    class needs to be defined for that.
    """

    def __init__(
        self,
        zs,
        ms,
        params=None,
        param_override=None,
        halofit=None,
        corr="max",
        nzm=None,
        mthresh=None,
        ngal=None,
    ):

        # Run base initialization
        super().__init__(
            zs,
            ms,
            params=params,
            param_override=param_override,
            halofit=halofit,
            corr=corr,
            nzm=nzm,
        )

        if mthresh is not None:
            self.log10mthreshC = np.log10(mthresh)
            self.log10mthreshS = self.log10mthreshC
            # self.log10mthreshS = self.p["hod_Zheng05_log10Mcut"] * np.ones_like(
            #     self.log10mthreshC
            # )
            if mthresh.ndim == 1:
                self.log10mthreshC = np.expand_dims(self.log10mthreshC, axis=-1)
                self.log10mthreshS = np.expand_dims(self.log10mthreshS, axis=-1)

        # If ngal is specified, need to compute corresponding mthresh
        elif ngal is not None:

            # Check for input halo mass function
            if nzm is None:
                raise ValueError("nzm must be specified if ngal is specified")
            self.nzm = nzm

            # Check input parameters
            if ngal.size != zs.size:
                raise ValueError("ngal and zs must have same size")
            if mthresh is not None:
                raise ValueError("Can only specify one of ngal or mthresh")

            # Define n_gal(m_thresh) function
            nfunc = lambda ilog10mthresh: self.ngal_from_mthresh(
                ilog10mthresh, self.zs, self.ms
            )

            # Compute m_thresh from input n_gal
            log10mthresh = utils.vectorized_bisection_search(
                ngal,
                nfunc,
                [
                    self.p["hod_bisection_search_min_log10mthresh"],
                    self.p["hod_bisection_search_max_log10mthresh"],
                ],
                "decreasing",
                rtol=self.p["hod_bisection_search_rtol"],
                verbose=True,
                hang_check_num_iter=self.p["hod_bisection_search_warn_iter"],
            )
            self.log10mthreshC = log10mthresh[:, None]
            self.log10mthreshS = self.log10mthreshC

        # If ngal and mthresh are None, use default M_c parameter
        else:
            self.log10mthreshC = (
                self.p["hod_Zheng05_log10Mth"] * np.ones_like(zs)[:, None]
            )
            self.log10mthreshS = self.p["hod_Zheng05_log10Mcut"] * np.ones_like(
                self.log10mthreshC
            )

        # Finally, compute halo occupations
        self.init_mean_occupations()

    @property
    def hod_params(self):
        pass

    def avg_Nc(self, log10mhalo, z, log10mthresh=None):
        if log10mthresh is None:
            log10mthresh = self.log10mthreshC

        # TODO: rewrite this part more pythonically
        # Make empty Nc array, packed as [m,z]
        Nc = np.zeros((log10mhalo.shape[0], log10mthresh.shape[0]))
        # Loop over z
        for i in range(Nc.shape[1]):
            # Make mask to select masses greater than threshold
            mask = log10mhalo > log10mthresh[i]
            Nc[:, i][mask] = 0.5 * (
                1
                + erf(
                    (log10mhalo[mask] - log10mthresh[i])
                    / self.p["hod_Zheng05_sigmalogM"]
                )
            )
        Nc = Nc.T

        return Nc

    def avg_Ns(self, log10mhalo, z, log10mthresh=None):
        pass

    def ngal_from_mthresh(
        self,
        log10mthresh=None,
        zs=None,
        ms=None,
        Ncs=None,
        Nss=None,
    ):
        if self.nzm is None:
            raise ValueError("Need a stored halo mass function for ngal_from_mthresh")

        if (Ncs is None) and (Nss is None):
            log10mthresh = log10mthresh[:, None]
            Ncs = self.avg_Nc(self.log10mhalo, self.zs, log10mthresh)
            Nss = self.avg_Ns(self.log10mhalo, self.zs, log10mthresh)

        integrand = self.nzm * (Ncs + Nss)
        return np.trapz(integrand, ms, axis=-1)


class Zheng05M1_HOD(Zheng05Base_HOD):
    """Class for the HOD from Zheng et al. 2005 (astro-ph/0408564).

    We use the Zheng (i.e. M_1) parameterization for amplitude of N_sat.
    """

    @property
    def hod_params(self):
        return [
            "hod_bisection_search_min_log10mthresh",
            "hod_bisection_search_max_log10mthresh",
            "hod_bisection_search_rtol",
            "hod_bisection_search_warn_iter",
            "hod_Zheng05_log10Mth",
            "hod_Zheng05_sigmalogM",
            "hod_Zheng05_log10Mcut",
            "hod_Zheng05_log10M1",
            "hod_Zheng05_alpha",
        ]

    def avg_Ns(self, log10mhalo, z, log10mthresh=None):
        masses = 10 ** log10mhalo
        if log10mthresh is None:
            log10mthresh = self.log10mthreshS

        mthresh = 10 ** log10mthresh
        alpha = self.p["hod_Zheng05_alpha"]
        M1 = 10 ** self.p["hod_Zheng05_log10M1"]

        # TODO: rewrite this part more pythonically
        # Make empty Ns array, packed as [m,z]
        Ns = np.zeros((masses.shape[0], mthresh.shape[0]))
        # Loop over z
        for i in range(Ns.shape[1]):
            # Make mask to select masses greater than threshold
            mask = masses > mthresh[i]
            Ns[:, i][mask] = ((masses[mask] - mthresh[i]) / M1) ** alpha
        Ns = Ns.T

        return Ns


class Zheng05beta_HOD(Zheng05Base_HOD):
    """Class for the HOD from Zheng et al. 2005 (astro-ph/0408564).

    We use the beta parameterization for amplitude of N_sat, i.e. M_1 = 10^beta M_cut.
    """

    @property
    def hod_params(self):
        return [
            "hod_bisection_search_min_log10mthresh",
            "hod_bisection_search_max_log10mthresh",
            "hod_bisection_search_rtol",
            "hod_bisection_search_warn_iter",
            "hod_Zheng05_log10Mth",
            "hod_Zheng05_sigmalogM",
            "hod_Zheng05_log10Mcut",
            "hod_Zheng05_beta",
            "hod_Zheng05_alpha",
        ]

    def avg_Ns(self, log10mhalo, z, log10mthresh=None):
        masses = 10 ** log10mhalo
        if log10mthresh is None:
            log10mthresh = self.log10mthreshS

        mthresh = 10 ** log10mthresh
        alpha = self.p["hod_Zheng05_alpha"]
        M1 = 10 ** self.p["hod_Zheng05_beta"] * mthresh

        # TODO: rewrite this part more pythonically
        # Make empty Ns array, packed as [m,z]
        Ns = np.zeros((masses.shape[0], mthresh.shape[0]))
        # Loop over z
        for i in range(Ns.shape[1]):
            # Make mask to select masses greater than threshold
            mask = masses > mthresh[i]
            Ns[:, i][mask] = ((masses[mask] - mthresh[i]) / M1[i]) ** alpha
        Ns = Ns.T

        return Ns
