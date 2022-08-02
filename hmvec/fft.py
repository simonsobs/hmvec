import numpy as np

"""
FFT routines
"""

def uk_fft(rhofunc,rvir,dr=0.001,rmax=100):
    rvir = np.asarray(rvir)
    rps = np.arange(dr,rmax,dr)
    rs = rps
    rhos = rhofunc(np.abs(rs))
    theta = np.ones(rhos.shape)
    theta[np.abs(rs)>rvir[...,None]] = 0 # CHECK
    integrand = rhos * theta
    m = np.trapz(integrand*rs**2.,rs,axis=-1)*4.*np.pi
    ks,ukt = fft_integral(rs,integrand)
    uk = 4.*np.pi*ukt/ks/m[...,None]
    return ks,uk


def uk_brute_force(r,rho,rvir,ks):
    sel = np.where(r<rvir)
    rs = r[sel]
    rhos = rho[sel]
    m = np.trapz(rhos*rs**2.,rs)*4.*np.pi
    # rs in dim 0, ks in dim 1
    rs2d = rs[...,None]
    rhos2d = rhos[...,None]
    ks2d = ks[None,...]
    sinkr = np.sin(rs2d*ks2d)
    integrand = 4.*np.pi*rs2d*sinkr*rhos2d/ks2d
    return np.trapz(integrand,rs,axis=0)/m

def fft_integral(x,y,axis=-1):
    """
    Calculates
    \int dx x sin(kx) y(|x|) from 0 to infinity using an FFT,
    which appears often in fourier transforms of 1-d profiles.
    For y(x) = exp(-x**2/2), this has the analytic solution
    sqrt(pi/2) exp(-k**2/2) k
    which this function can be checked against.
    """
    assert x.ndim==1
    extent = x[-1]-x[0]
    N = x.size
    step = extent/N
    integrand = x*y
    uk = -np.fft.rfft(integrand,axis=axis).imag*step
    ks = np.fft.rfftfreq(N, step) *2*np.pi
    return ks,uk

def analytic_fft_integral(ks): return np.sqrt(np.pi/2.)*np.exp(-ks**2./2.)*ks


def generic_profile_fft(rhofunc_x, cmaxs, rss, zs, ks, xmax, nxs, do_mass_norm=True):
    """Evaluate FFT of generic density profile.

    Parameters
    ----------
    rhofunc_x : function
        Density profile as a function of dimensionless radius coordinate. Function must
        accept a vector of x = r/r_s values.
    cmaxs : array_like
        Dimensionless cutoff for the profile integrals, packed as [z,m].
        For NFW, for example, this is concentration(z,mass).
        For other profiles, you will want to do
        cmax = Rvir(z,m)/R_scale_radius where R_scale_radius is whatever you have
        divided the physical distance by in the profile to get the integration variable,
        i.e. x = r / R_scale_radius.
    rss : array_like
        Array of R_scale_radius, packed as [z,m].
    zs : array_like
        Array of redshifts (for converting physical wavenumber to comoving).
    ks : array_like
        Target comoving wavenumbers to interpolate the resulting FFT onto.
    xmax : float
        Maximum x coord of position-space profile.
    nxs : int
        Number of x samples to use in FFT.
    do_mass_norm : bool, optional
        Whether to normalize profile by enclosed mass. Default: True.

    Returns
    -------
    ks : array_like
        Output comoving wavenumbers (same as input).
    ukouts : array_like
        Output u(k), packed as [z,m,k].
    """
    # Define x array, and evaluate rho(x)
    xs = np.linspace(0.,xmax,nxs+1)[1:]
    rhos = rhofunc_x(xs)

    # Ensure that rho array is 3-dimensional with x as final axis
    if rhos.ndim == 1:
        rhos = rhos[None, None, :]
    else:
        assert rhos.ndim==3

    # Reshape rho array to be packed as [z,m,x]
    rhos = rhos + cmaxs[..., None] * 0.

    # Form rho(x) * x^2.
    # Enforce cmaxs constraint by setting x>cmax points to zero using "theta" mask
    theta = np.ones(rhos.shape)
    theta[np.abs(xs) > cmaxs[..., None]] = 0 # CHECK
    integrand = theta * rhos * xs ** 2.

    # Compute \int dx x^2 rho(x) to compute enclosed mass.
    # Normalization is wrong, but same wrong factor as rho.
    # mnorm is packed as [z,m]
    mnorm = np.trapz(integrand, xs)

    # Set normalization factor to unity if not using it
    if not(do_mass_norm):
        mnorm *= 0
        mnorm +=1

    # Form rho(x), only nonzero up to cmaxs
    integrand = rhos*theta

    # Compute \int dx x sin(kx) rho(x).
    # kts is packed as [k], ukts is packed as [z,m,k]
    kts,ukts = fft_integral(xs, integrand)

    # Divide result by k and mnorm
    uk = ukts / kts[None, None, :] / mnorm[..., None]

    # Convert kt to k by dividing by rss, and (1+z) for comoving (FIXME: check this!)
    kouts = kts / rss / (1 + zs[:, None, None])

    # Interpolate u(k) onto desired k grid
    ukouts = np.zeros((uk.shape[0],uk.shape[1],ks.size))
    for i in range(uk.shape[0]):
        for j in range(uk.shape[1]):
            pks = kouts[i, j]
            puks = uk[i, j]
            puks = puks[pks > 0]
            pks = pks[pks > 0]
            ukouts[i, j] = np.interp(ks, pks, puks, left=puks[0], right=0)

    return ks, ukouts
