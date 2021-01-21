import numpy as np
import hmvec as hm
import matplotlib
import matplotlib.pyplot as plt
from cobaya.run import run
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood

#Plot settings
matplotlib.rcParams['axes.labelsize'] = 'xx-large'
matplotlib.rcParams['xtick.labelsize'] = 'x-large'
matplotlib.rcParams['ytick.labelsize'] = 'x-large'
matplotlib.rcParams['legend.fontsize'] = 'x-large'
matplotlib.rcParams['axes.titlesize'] = 'xx-large'

class PowerSpectrum(Theory):
    params = {'alpha':None, 'beta':None, 'gamma':None, 'delta':None, 'Td_o':None, 'logM_eff':None, 'L_o':None, 'freq':None}

    def initialize(self):
        #Setup Grid
        Nz = 100                                 # num of redshifts
        Nm = 100                                 # num of masses
        Nk = 1000                                # num of wavenumbers
        self.redshifts = np.linspace(0.01, 6, Nz)             
        self.masses = np.geomspace(1.0e6, 1.0e15, Nm)          
        self.ks = np.geomspace(1.0e-3, 100.0, Nk)              # wavenumbers
        self.ells = np.linspace(10, 2000, 200)

        #Initialize Halo Model 
        self.hcos = hm.HaloModel(self.redshifts, self.ks, ms=self.masses)

    def get_can_provide(self):
        return ['Cl']

    def calculate(self, state, want_derived=True, **params):
        #Extract Just CIB Parameters
        cibparams = dict()
        for key, value in params.items():
            if key == 'freq':
                continue
            cibparams[key] = value

        #Set New Parameters
        self.hcos.set_cibParams('custom', **cibparams)

        #Power Spectrum
        Pjj_tot = self.hcos.get_power("cib", "cib", nu_obs= params['freq'], satmf= 'tinker') 
        state['Cl'] = self.hcos.C_ii(self.ells, self.redshifts, self.ks, Pjj_tot, dcdzflag= False)

class ChiSqLikelihood(Likelihood):
    params = {'covariance':None, 'data':None}

    def get_requirements(self):
        return ['Cl']

    def logp(self, **param_values):
        Cl = self.provider.get_result('Cl')
        dev = param_values['data'] - Cl

        chisquared = dev.T @ np.linalg.inv(param_values['covariance']) @ dev

        return -1.0/2.0 * chisquared
        

#Toy Data
filename_cov = ''
filename_data = ''
cov = np.load(filename_cov)
data = np.load(filename_data)

#Autocorrelation: 1 Freq
autofreq = np.array([545e9], dtype=np.double)    

#Cobaya Input File
info = {
    "likelihood": ChiSqLikelihood,
    
    "params": dict([
        #CIB Model Parameters
        ("alpha", {
            "prior": {"min": 0, "max": 1.3},
            "ref": {"min": 0.2, "max": 0.5},
            "latex": r"\alpha"}),
        ("beta", {
            "prior": {"min": 0, "max": 2.1},
            "ref": {"min": 1.2, "max": 1.7},
            "latex": r"\beta"}),
        ("gamma", {
            "prior": {"min": 0, "max": 2.7},
            "ref": {"min": 1.2, "max": 1.7},
            "latex": r"\gamma"}),
        ("delta", {
            "prior": {"min": 2.5, "max": 4.6},
            "ref": {"min": 3, "max": 4},
            "latex": r"\delta"}),
        ("Td_o", {
            "prior": {"min": 15, "max": 30},
            "ref": {"min": 18, "max": 22},
            "latex": r"T_{d,o}"}),
        ("logM_eff", {
            "prior": {"min": 11, "max": 14},
            "ref": {"min": 11.8, "max": 13},
            "latex": r"\text{log}(M_{\text{eff}})"}),
        ("L_o", {
            "prior": {"min": 1e-17, "max": 1e-13},
            "ref": {"min": 9e-16, "max": 9e-15},
            "latex": r"L_o"}),
        
        #Fixed Params for Theory/Likelihood
        ("covariance", cov),
        ("data", data),
        ("freq", autofreq)
    ]),

    "sampler": {
        "mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}
    }
}