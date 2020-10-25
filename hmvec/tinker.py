import numpy as np
from scipy.interpolate import interp1d
import os

"""
Implements Tinker et al 2010 and Tinker et al 2008

nu = deltac / sigma


nu and sigma have shape
(numzs,numms)

So a function that asks for redshifts zs in additions
expects consistent number of redshifts.
"""

constants = {
    'deltac': 1.686,
}

default_params = {
    'tinker_f_nu_alpha_z0_delta_200':0.368, # Tinker et al 2010 Table 4
    }

def bias(nu,delta=200.):
    # Eq 6 of Tinker 2010
    deltac = constants['deltac']
    y = np.log10(delta)
    A = lambda y: 1. + 0.24*y*np.exp(-(4./y)**4.)
    a = lambda y: 0.44*y-0.88
    B = 0.183
    b = 1.5
    C = lambda y: 0.019 + 0.107*y + 0.19 *np.exp(-(4./y)**4.)
    c = 2.4
    nua = nu**a(y)
    t1 = (nua)/(nua+deltac**a(y))
    t2 = nu**b
    t3 = nu**c
    return 1 - A(y)*t1 + B*t2 + C(y)*t3

    
def f_nu(nu,zs,delta=200.,norm_consistency=True,
         alpha=default_params['tinker_f_nu_alpha_z0_delta_200']):
    # This is the f of Tinker 2010
    # but not the f of Tinker 2008
    # Tinker 2008 f = g in Appendix = nu * f of Tinker 2010
    # \int b f dnu should be 1 (in fact, norm_consistency enforces this for z>0)
    # This should be equiavelnt to \int dm (m/rho) n b = 1 (bias consistency)
    # if n = (rho/m) nu f(nu) dlnsigmainv/dm
    assert np.isclose(delta,200.), "delta!=200 note implemented yet." # FIXME: interpolate for any value of delta
    # FIXME: set z>3 to z=3
    zs = zs*np.heaviside(3-zs,0)+3*np.heaviside(zs-3,0)
    beta0 = 0.589
    gamma0 = 0.864
    phi0 = -0.729
    eta0 = -0.243
    beta  = beta0  * (1+zs)**(0.20)
    phi   = phi0   * (1+zs)**(-0.08)
    eta   = eta0   * (1+zs)**(0.27)
    gamma = gamma0 * (1+zs)**(-0.01)
    unnormalized = (1. + (beta*nu)**(-2.*phi))*(nu**(2*eta))*np.exp(-gamma*nu**2./2.)
    if norm_consistency:
        aroot = os.path.dirname(__file__)+"/../data/alpha_consistency.txt"
        izs,ialphas = np.loadtxt(aroot,unpack=True) # FIXME: hardcoded
        alpha = interp1d(izs,ialphas,bounds_error=True)(zs)
    return alpha * unnormalized 

    
def simple_f_nu(nu,delta=200.):
    assert np.isclose(delta,200.), "delta!=200 note implemented yet." # FIXME: interpolate for any value of delta
    deltac = constants['deltac']
    sigma = deltac/nu
    A = 0.186
    a = 1.47
    b = 2.57
    c = 1.19
    return A* (1.+((sigma/b)**(-a))) * np.exp(-c/sigma**2.)


def NlnMsub(Msubs,Mhosts):
    """
    Eq 12 of the *published* version of J. L. Tinker and A. R. Wetzel, apj 719, 88 (2010),
    0909.1325
    Differs from arxiv in the 0.3 prefactor
    Accepts 1d array of Msubs and Mhosts
    and returns 2d array for (Msubs,Mhosts)
    """
    mrat = Msubs[:,None]/Mhosts[None,:]
    return 0.3 * (mrat**(-0.7)) * np.exp(-9.9 * (mrat**2.5))
