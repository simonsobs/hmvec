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


def generic_profile_fft(rhofunc_x,cmaxs,rss,zs,ks,xmax,nxs,do_mass_norm=True):
    """
    Generic profile FFTing
    rhofunc_x: function that accepts vector spanning linspace(0,xmax,nxs)
    xmax:  some O(10-1000) dimensionless number specifying maximum of real space
    profile
    nxs: number of samples of the profile.
    cmaxs: typically an [nz,nm] array of the dimensionless cutoff for the profile integrals. 
    For NFW, for example, this is concentration(z,mass).
    For other profiles, you will want to do cmax = Rvir(z,m)/R_scale_radius where
    R_scale_radius is whatever you have divided the physical distance by in the profile to
    get the integration variable i.e. x = r / R_scale_radius.
    rss: R_scale_radius
    zs: [nz,] array to convert physical wavenumber to comoving wavenumber.
    ks: target comoving wavenumbers to interpolate the resulting FFT on to.
    
    """
    xs = np.linspace(0.,xmax,nxs+1)[1:]
    rhos = rhofunc_x(xs)
    if rhos.ndim==1:
        rhos = rhos[None,None]
    else:
        assert rhos.ndim==3
    rhos = rhos + cmaxs[...,None]*0.
    theta = np.ones(rhos.shape)
    theta[np.abs(xs)>cmaxs[...,None]] = 0 # CHECK
    # m
    integrand = theta * rhos * xs**2.
    mnorm = np.trapz(integrand,xs) # mass but off by norm same as rho is off by
    if not(do_mass_norm):
        mnorm *= 0
        mnorm +=1
    # u(kt)
    integrand = rhos*theta
    kts,ukts = fft_integral(xs,integrand)
    uk = ukts/kts[None,None,:]/mnorm[...,None]
    kouts = kts/rss/(1+zs[:,None,None]) # divide k by (1+z) here for comoving FIXME: check this!
    ukouts = np.zeros((uk.shape[0],uk.shape[1],ks.size))
    # sadly at this point we must loop to interpolate :(
    # from orphics import io
    # pl = io.Plotter(xyscale='loglog')
    for i in range(uk.shape[0]):
        for j in range(uk.shape[1]):
            pks = kouts[i,j]
            puks = uk[i,j]
            puks = puks[pks>0]
            pks = pks[pks>0]
            ukouts[i,j] = np.interp(ks,pks,puks,left=puks[0],right=0)
            #TODO: Add compulsory debug plot here
    #         pl.add(ks,ukouts[i,j])
    # pl.hline(y=1)
    # pl.done()
    return ks, ukouts
