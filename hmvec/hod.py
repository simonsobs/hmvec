from .params import default_params
from .hmvec import HaloModel
from . import utils
from .cosmology import Cosmology
from .mstar_mhalo import Mstellar_halo, Mhalo_stellar

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, dblquad
from scipy.special import erf

"""
HOD class and related routines.

Implementation of Leauthaud12HOD is based on code from to Matt Johnson and Moritz
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
    zs : array_like
        Array of redshifts to compute at.
    ms : array_like
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
    """

    def __init__(
        self,
        zs,
        ms,
        params=None,
        param_override=None,
        halofit=None,
        corr="max",
        **kwargs
    ):

        # Store redshifts, halo masses, corr model
        self.zs = zs
        self.ms = ms
        self.log10mhalo = np.log10(self.ms[None, :])
        self.corr = corr

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
        self.Nc = self.avg_Nc(self.log10mhalo, self.zs[:, None])
        self.Ns = self.avg_Ns(self.log10mhalo, self.zs[:, None])
        self.NsNsm1 = self.avg_NsNsm1(self.Nc, self.Ns, self.corr)
        self.NcNs = self.avg_NcNs(self.Nc, self.Ns, self.corr)

    @property
    def hod_params(self):
        """List of HOD-specific parameter names."""
        pass

    def avg_Nc(self, log10mhalo, z, **kwargs):
        """<N_c | m_h>"""
        pass

    def avg_Ns(self, log10mhalo, z, **kwargs):
        """<N_s | m_h>"""
        pass

    def avg_NsNsm1(self, Nc, Ns, corr="max"):
        """<N_s N_s | m_h>"""
        if corr not in ["min", "max"]:
            raise ValueError("Invalid corr argument (%s) passed to avg_NcNs!" % corr)

        if corr == "max":
            ret = Ns ** 2.0 / Nc
            ret[np.isclose(Nc, 0.0)] = 0  # FIXME: is this kosher?
            return ret
        elif corr == "min":
            return Ns ** 2.0

    def avg_NcNs(self, Nc, Ns, corr="max"):
        """<N_c N_s | m_h>"""
        if corr not in ["min", "max"]:
            raise ValueError("Invalid corr argument (%s) passed to avg_NcNs!" % corr)

        if corr == "max":
            return Ns
        elif corr == "min":
            return Ns * Nc


class Leauthaud12HOD(HODBase):
    """HOD from Leauthaud et al. 2012 (1104.0928).

    This HOD was used for the kSZ computations in Smith et al. 2018
    (1810.13423), and was also the only HOD used in a previous version of
    hmvec.
    """

    def __init__(
        self,
        zs,
        ms,
        params=None,
        param_override=None,
        halofit=None,
        corr="max",
        mthresh=None,
        ngal=None,
        nzm=None,
    ):

        # Run base initialization
        super().__init__(
            zs,
            ms,
            params=params,
            param_override=param_override,
            halofit=halofit,
            corr=corr,
        )

        # If ngal is specified, need to compute corresponding mthresh
        if ngal is not None:

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
            if "hod_Msat_override" not in self.p.keys():
                self.p["hod_Msat_override"] = None
            if "hod_Mcut_override" not in self.p.keys():
                self.p["hod_Mcut_override"] = None

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
            mthresh = 10 ** (log10mthresh * self.p["hod_A_log10mthresh"])

        # Do sanity check on mthresh
        if mthresh.size != zs.size:
            raise ValueError("mthresh and zs must have same size")

        self.log10mstellar_thresh = np.log10(mthresh[:, None])

        # Finally, compute halo occupations
        self.init_mean_occupations()

    @property
    def hod_params(self):
        return [
            "hod_sig_log_mstellar",
            "hod_bisection_search_min_log10mthresh",
            "hod_bisection_search_max_log10mthresh",
            "hod_bisection_search_rtol",
            "hod_bisection_search_warn_iter",
            "hod_alphasat",
            "hod_Bsat",
            "hod_betasat",
            "hod_Bcut",
            "hod_betacut",
            "hod_A_log10mthresh",
        ]

    def avg_Nc(self, log10mhalo, z, log10mstellar_thresh=None):
        if log10mstellar_thresh is None:
            log10mstellar_thresh = self.log10mstellar_thresh

        log10mstar = Mstellar_halo(z, log10mhalo)
        num = log10mstellar_thresh - log10mstar
        denom = np.sqrt(2.0) * self.p["hod_sig_log_mstellar"]
        return 0.5 * (1.0 - erf(num / denom))

    def avg_Ns(self, log10mhalo, z, log10mstellar_thresh=None, Nc=None):
        masses = 10 ** log10mhalo
        if log10mstellar_thresh is not None:
            mthresh = Mhalo_stellar(z, log10mstellar_thresh)
        else:
            mthresh = Mhalo_stellar(z, self.log10mstellar_thresh)

        if (
            "hod_Msat_override" in self.p.keys()
            and self.p["hod_Msat_override"] is not None
        ):

            Msat = self.p["hod_Msat_override"]
        else:
            Msat = self.hod_default_mfunc(
                mthresh, self.p["hod_Bsat"], self.p["hod_betasat"]
            )

        if (
            "hod_Mcut_override" in self.p.keys()
            and self.p["hod_Mcut_override"] is not None
        ):
            Mcut = self.p["hod_Mcut_override"]
        else:
            Mcut = self.hod_default_mfunc(
                mthresh, self.p["hod_Bcut"], self.p["hod_betacut"]
            )

        if Nc is None:
            if self.Nc is not None:
                Nc = self.Nc
            else:
                Nc = self.avg_Nc(log10mhalo, z)

        return Nc * ((masses / Msat) ** self.p["hod_alphasat"]) * np.exp(-Mcut / masses)

    def hod_default_mfunc(self, mthresh, Bamp, Bind):
        return (10.0 ** (12.0)) * Bamp * 10 ** ((mthresh - 12) * Bind)

    def ngal_from_mthresh(
        self,
        log10mthresh=None,
        zs=None,
        ms=None,
        sig_log_mstellar=None,
        Ncs=None,
        Nss=None,
    ):
        if self.nzm is None:
            raise ValueError("Need a stored halo mass function for ngal_from_mthresh")

        if (Ncs is None) and (Nss is None):
            log10mstellar_thresh = log10mthresh[:, None]
            Ncs = self.avg_Nc(self.log10mhalo, self.zs[:, None], log10mstellar_thresh)
            Nss = self.avg_Ns(
                self.log10mhalo,
                self.zs[:, None],
                log10mstellar_thresh,
                Ncs,
            )
        else:
            if not (log10mthresh is None and zs is None and sig_log_mstellar is None):
                raise ValueError(
                    "ngal_from_mthresh called with strange combination of arguments"
                )

        integrand = self.nzm * (Ncs + Nss)
        return np.trapz(integrand, ms, axis=-1)
