import numpy as np
import hmvec as hm
import matplotlib.pyplot as plt
import matplotlib
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood

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
        dev = param_values['data'][:,1] - Cl

        chisquared = dev.T @ np.linalg.inv(param_values['covariance']) @ dev

        return -1.0/2.0 * chisquared
        