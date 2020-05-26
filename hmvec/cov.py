from .params import default_params
from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import binned_statistic as binnedstat

"""
Covariances
We implement the Gaussian covariance between bandpowers.
"""

def bin_annuli(ells,cls,bin_edges):
    numer = binnedstat(ells,ells*cls,bins=bin_edges,statistic=np.nanmean)[0]
    denom = binnedstat(ells,ells,bins=bin_edges,statistic=np.nanmean)[0]
    return numer/denom
    

default_binning = bin_annuli

def shot_noise(ngal):
    return 1./(ngal*1.18e7)

def lensing_shape_noise(ngal,shape_noise=0.3):
    return (shape_noise**2.)/2./shot_noise(ngal)

def get_avail_cls(acls,x,y):
    try: return acls[x+"_"+y]
    except:
        try:
            return self.cls[y+"_"+x]
        except:
            return 0

class GaussianCov(object):
    def __init__(self,bin_edges,binning_func=default_binning):
        self.cls = {}
        self.nls = {}
        ellmin,ellmax = bin_edges[0],bin_edges[-1]
        self.ells = np.arange(ellmin,ellmax+1,1)
        self.bin_edges = bin_edges
        self.dls = np.diff(self.bin_edges)
        self.ls = (self.bin_edges[1:]+self.bin_edges[:-1])/2.

    def add_cls(self,name1,name2,ells,cls,ellsn=None,ncls=None):
        assert "_" not in name1
        assert "_" not in name2
        assert name2+"_"+name1 not in self.cls.keys()
        self.cls[name1+"_"+name2] = bin_annuli(self.ells,interp1d(ells,cls)(self.ells),self.bin_edges)
        if (ellsn is not None) and (ncls is not None):
            self.nls[name1+"_"+name2] = bin_annuli(self.ells,interp1d(ellsn,ncls)(self.ells),self.bin_edges)

    def get_scls(self,x,y):
        return get_avail_cls(self.cls,x,y)
    
    def get_ncls(self,x,y):
        return get_avail_cls(self.nls,x,y)
    
    def get_tcls(self,x,y):
        return self.get_scls(x,y) + self.get_ncls(x,y)
        
    def get_cov(self,x,y,w,z,fsky):
        clsum = self.get_tcls(x,w)*self.get_tcls(y,z)+self.get_tcls(x,z)*self.get_tcls(y,w)
        covs = clsum / (2*self.ls+1.)/self.dls/fsky
        return covs

def KnoxCov(self,specTypeXY,specTypeWZ,ellBinEdges,fsky):
    '''
    returns cov(Cl_XY,Cl_WZ)
    '''
    def ClTot(spec,ell1,ell2):
        return self._bin_cls(spec,ell1,ell2,noise=True)

    X, Y = specTypeXY
    W, Z = specTypeWZ

    ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
    ellWidths = np.diff(ellBinEdges)

    covs = []
    sigs1 = []
    sigs2 = []

    for ell_left,ell_right in zip(ellBinEdges[:-1],ellBinEdges[1:]):
        ClSum = ClTot(X+W,ell_left,ell_right)*ClTot(Y+Z,ell_left,ell_right)+ClTot(X+Z,ell_left,ell_right)*ClTot(Y+W,ell_left,ell_right)
        ellMid = (ell_right+ell_left)/2.
        ellWidth = ell_right-ell_left
        var = ClSum/(2.*ellMid+1.)/ellWidth/fsky
        covs.append(var)
        sigs1.append(self._bin_cls(specTypeXY,ell_left,ell_right,noise=False)**2.*np.nan_to_num(1./var))
        sigs2.append(self._bin_cls(specTypeWZ,ell_left,ell_right,noise=False)**2.*np.nan_to_num(1./var))