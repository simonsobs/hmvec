import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
import logging


def interp(x,y,bounds_error=False,fill_value=0.,**kwargs):
    return interp1d(x,y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)

def vectorized_bisection_search(x,inv_func,ybounds,monotonicity,rtol=1e-4,verbose=True,hang_check_num_iter=20):
    """
    You have a monotonic one-to-one relationship x <-> y
    You know the inverse function inv_func=x(y), 
    but you don't know y(x).
    Find y for a given x using a bisection search
    assuming y is bounded in ybounds=(yleft,yright)
    and with a relative tolerance on x of rtol.
    """
    assert monotonicity in ['increasing','decreasing']
    mtol = np.inf
    func = inv_func
    iyleft,iyright = ybounds
    yleft = x*0+iyleft
    yright = x*0+iyright
    i = 0
    warned = False
    while np.any(np.abs(mtol)>rtol):
        ynow = (yleft+yright)/2.
        xnow = func(ynow)
        mtol = (xnow-x)/x
        if monotonicity=='decreasing':
            yleft[mtol>0] = ynow[mtol>0]
            yright[mtol<=0] = ynow[mtol<=0]
        elif monotonicity=='increasing':
            yright[mtol>0] = ynow[mtol>0]
            yleft[mtol<=0] = ynow[mtol<=0]
        i += 1
        if (i>hang_check_num_iter) and not(warned):
            print("WARNING: Bisection search has done more than ", hang_check_num_iter,
                  " loops. Still searching...")
            warned = True
    if verbose: print("Bisection search converged in ", i, " iterations.")
    return ynow


def test_bisection_search():
    true_y_of_x = lambda x: x**2.
    x_of_y = lambda y: np.sqrt(y)
    xs = np.array([2.,4.,6.])
    eys = np.array([4.,16.,36.])
    d = vectorized_bisection_search(xs,x_of_y,(1,40),'increasing',rtol=1e-4,verbose=True)
    assert np.all(np.isclose(d,eys,rtol=1e-3))

def get_matter_power_interpolator_generic(ks, zs, pk, 
                                      return_z_k=False, log_interp=True, extrap_kmax=None, silent=False):
        r"""

        This is copied and adapted from CAMB/results.py

        It provides a way to get a PK.P interpolator, but with a generic power spectrum, allowing
        it to be used with CLASS as well.

        Docstring from CAMB:

        Assuming transfers have been calculated, return a 2D spline interpolation object to evaluate matter
        power spectrum as function of z and k/h (or k). Uses self.Params.Transfer.PK_redshifts as the spline node
        points in z. If fewer than four redshift points are used the interpolator uses a reduced order spline in z
        (so results at intermediate z may be inaccurate), otherwise it uses bicubic.
        Usage example:

        .. code-block:: python

           PK = results.get_matter_power_interpolator();
           print('Power spectrum at z=0.5, k/h=0.1 is %s (Mpc/h)^3 '%(PK.P(0.5, 0.1)))

        For a description of outputs for different var1, var2 see :ref:`transfer-variables`.

        :param nonlinear: include non-linear correction from halo model
        :param var1: variable i (index, or name of variable; default delta_tot)
        :param var2: variable j (index, or name of variable; default delta_tot)
        :param hubble_units: if true, output power spectrum in :math:`({\rm Mpc}/h)^{3}` units,
                             otherwise :math:`{\rm Mpc}^{3}`
        :param k_hunit: if true, matter power is a function of k/h, if false, just k (both :math:`{\rm Mpc}^{-1}` units)
        :param return_z_k: if true, return interpolator, z, k where z, k are the grid used
        :param log_interp: if true, interpolate log of power spectrum
                           (unless any values cross zero in which case ignored)
        :param extrap_kmax: if set, use power law extrapolation beyond kmax to extrap_kmax
                            (useful for tails of integrals)
        :param silent: Set True to silence warnings
        :return: An object PK based on :class:`~scipy:scipy.interpolate.RectBivariateSpline`,
                 that can be called with PK.P(z,k) or PK(z,log(k)) to get log matter power values.
                 If return_z_k=True, instead return interpolator, z, k where z, k are the grid used.
        """

        class PKInterpolator(RectBivariateSpline):
            islog: bool
            logsign: int

            def P(self, z, k, grid=None):
                if grid is None:
                    grid = not np.isscalar(z) and not np.isscalar(k)
                if self.islog:
                    return self.logsign * np.exp(self(z, np.log(k), grid=grid))
                else:
                    return self(z, np.log(k), grid=grid)

        class PKInterpolatorSingleZ(interp1d):
            islog: bool
            logsign: int

            def __init__(self, *args, **kwargs):
                self._single_z = np.array(args[0])
                super().__init__(*(args[1:]), kind=kwargs.get("ky"))

            def check_z(self, z):
                if not np.allclose(z, self._single_z):
                    raise ValueError(
                        "P(z,k) requested at z=%g, but only computed for z=%s. "
                        "Cannot extrapolate!" % (z, self._single_z))

            def __call__(self, *args):
                self.check_z(args[0])
                # NB returns dimensionality as the 2D one: 1 dimension if z single
                return (lambda x: x[0] if np.isscalar(args[0]) else x)(super().__call__(*(args[1:])))

            def P(self, z, k, **_kwargs):
                # grid kwarg is ignored
                if self.islog:
                    return self.logsign * np.exp(self(z, np.log(k)))
                else:
                    return self(z, np.log(k))

        #khs, zs, pk = self.get_linear_matter_power_spectrum(var1, var2, hubble_units, nonlinear=nonlinear)
        # print(ks.shape)
        # print(zs.shape)
        # print(pk.shape)
        # import sys
        # sys.exit()
        k_max = ks[-1]
        sign = 1
        if log_interp and np.any(pk <= 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_interp = False
        p_or_log_p = np.log(sign * pk) if log_interp else pk
        logk = np.log(ks)
        deg_z = min(len(zs) - 1, 3)
        kmax = ks[-1]
        PKInterpolator = PKInterpolator if deg_z else PKInterpolatorSingleZ
        if extrap_kmax and extrap_kmax > kmax:
            # extrapolate to ultimate power law
            # TODO: use more physical extrapolation function for linear case
            if not silent and (k_max < 3 and extrap_kmax > 2 and nonlinear or k_max < 0.4):
                logging.warning("Extrapolating to higher k with matter transfer functions "
                                "only to k=%.3g Mpc^{-1} may be inaccurate.\n " % (k_max ))
            if not log_interp:
                raise ValueError(
                    "Cannot use extrap_kmax with log_inter=False (e.g. PK crossing zero for %s, %s.)" % (var1, var2))

            logextrap = np.log(extrap_kmax)
            log_p_new = np.empty((pk.shape[0], pk.shape[1] + 2))
            log_p_new[:, :-2] = p_or_log_p
            delta = logextrap - logk[-1]

            dlog = (log_p_new[:, -3] - log_p_new[:, -4]) / (logk[-1] - logk[-2])
            log_p_new[:, -1] = log_p_new[:, -3] + dlog * delta
            log_p_new[:, -2] = log_p_new[:, -3] + dlog * delta * 0.9
            logk = np.hstack((logk, logextrap - delta * 0.1, logextrap))
            p_or_log_p = log_p_new

        deg_k = min(len(logk) - 1, 3)
        res = PKInterpolator(zs, logk, p_or_log_p, kx=deg_z, ky=deg_k)
        res.kmin = np.min(ks)
        res.kmax = kmax
        res.islog = log_interp
        res.logsign = sign
        res.zmin = np.min(zs)
        res.zmax = np.max(zs)
        if return_z_k:
            return res, zs, ks
        else:
            return res
