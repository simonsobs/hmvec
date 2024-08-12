import numpy as np
import os,sys
from scipy.interpolate import interp2d,interp1d
from .params import default_params
import camb
from camb import model
import scipy.interpolate as si
import scipy.constants as constants
from scipy.special import hyp2f1
try:
    from scipy.integrate import simps
except:
    from scipy.integrate import simpson as simps
from . import utils
import warnings

"""
This module attempts to abstract away the choice between
CAMB and CLASS.
It does this simply by providing a common interface.
It makes no guarantee that the same set
of parameters passed to the two engines will produce the same
results. It could be a test-bed for converging towards that. 

"""

cspeed = 299792.458 # km/s


def Wkr_taylor(kR):
    xx = kR*kR
    return 1 - .1*xx + .00357142857143*xx*xx

def Wkr(k,R,taylor_switch=default_params['Wkr_taylor_switch']):
    kR = k*R
    ans = 3.*(np.sin(kR)-kR*np.cos(kR))/(kR**3.)
    ans[kR<taylor_switch] = Wkr_taylor(kR[kR<taylor_switch]) 
    return ans

def get_eds_model(fb=0.15,H0=68.0,YHe=0.25):
    # Get an Einstein-de Sitter (Lambda=0) model
    # from baryon fraction and Hubble constant
    om = 1.0
    omb = fb*om
    omc = (1-fb)*om
    h0 = H0/100
    omch2 = omc*h0**2
    ombh2 = omb*h0**2
    return {'omch2':omch2,'ombh2':ombh2,'H0':H0,'mnu':0.,'YHe':YHe}

class Cosmology(object):

    def __init__(self,params={},halofit=None,engine='camb',accuracy='medium'):
        engine = engine.lower()
        if not(engine in ['camb','class']): raise ValueError
        self.accuracy = accuracy
        self.engine = engine
        if self.accuracy=='low' and (('S8' in params.keys()) or ('sigma8' in params.keys())): raise ValueError("Can't use S8 or sigma8 with low accuracy.")
        
        self.p = dict(params) if params is not None else {}
        for param in default_params.keys():
            if param not in self.p.keys(): self.p[param] = default_params[param]
        
        # Cosmology
        self._init_cosmology(self.p,halofit)

    def get_cmb_cls(self,lmax=3000,lens_potential_accuracy=4,nonlinear=True):
        if self.engine=='camb':
            if nonlinear:
                self._camb_pars.NonLinear = model.NonLinear_both
            else:
                self._camb_pars.NonLinear = model.NonLinear_none
            if not(nonlinear): lens_potential_accuracy = 0
            self._camb_pars.set_for_lmax(lmax=(lmax+500), lens_potential_accuracy=lens_potential_accuracy)
            self._camb_results.calc_power_spectra(self._camb_pars)
            powers = self.results.get_cmb_power_spectra(self._camb_pars, CMB_unit='muK',raw_cl=True)
        elif self.engine=='class':
            #continue
            #self.class_results
            pass

    def angular_diameter_distance(self,z1,z2=None):
        if not(z2 is None):
            if self.engine=='camb':
                return self._camb_results.angular_diameter_distance2(z1,z2)
            elif self.engine=='class':
                return self._class_results.angular_distance_from_to(z1, z2) # TODO: vectorize over z1
        else:
            if self.engine=='camb':
                return self._camb_results.angular_diameter_distance(z1)
            elif self.engine=='class':
                return np.vectorize(self._class_results.angular_distance)(z1)
                
    def sigma_crit(self,zlens,zsource):
        Gval = 4.517e-48 # Newton G in Mpc,seconds,Msun units
        cval = 9.716e-15 # speed of light in Mpc,second units
        Dd = self.angular_diameter_distance(zlens)
        Ds = self.angular_diameter_distance(zsource)
        Dds = np.asarray([self.angular_diameter_distance(zl,zsource) for zl in zlens])
        return cval**2 * Ds / 4 / np.pi / Gval / Dd / Dds
        

    def P_mm_linear(self,zs,ks):
        pass

    def P_mm_nonlinear(self,ks,zs,halofit_version='mead'):
        pass

    def comoving_radial_distance(self,z):
        if self.engine=='camb':
            return self._camb_results.comoving_radial_distance(z)
        elif self.engine=='class':
            return np.vectorize(self._class_results.comoving_distance)(z)

    def hubble_parameter(self,z):
        # H(z) in km/s/Mpc
        if self.engine=='camb':
            return self._camb_results.hubble_parameter(z)
        elif self.engine=='class':
            H = np.vectorize(self._class_results.Hubble)(z)
            return H*cspeed
    
    def h_of_z(self,z):
        # H(z) in 1/Mpc
        if self.engine=='camb':
            H = self._camb_results.h_of_z(z)
        elif self.engine=='class':
            H = np.vectorize(self._class_results.Hubble)(z)
        return H
    
    def bias_fnl(self,bg,fnl,z,ks,deltac=1.42):
        beta = 2. * deltac * (bg-1.)
        a = 1./(1+z)
        alpha = (2. * ks**2. * self.Tk(ks,type ='eisenhu_osc')) / (3.* self.omm0 * self.h_of_z(0)**2.) * self.D_growth(a,type="anorm",exact=False)
        return bg+fnl*(beta/alpha)

    def _init_cosmology(self,params,halofit):
        
        try:
            theta = params['theta100']/100.
            H0 = None
            print("WARNING: Using theta100 parameterization. H0 ignored.")
        except:
            H0 = params['H0']
            theta = None
            h = H0/100
        try:        
            omm = params['omm']        
            h = params['H0']/100.      
            params['omch2'] = omm*h**2-params['ombh2']     
            print("WARNING: omm specified. Ignoring omch2.")       
        except:        
            pass

        YHe = params['YHe'] if 'YHe' in params.keys() else None
        TCMB = params['TCMB'] if 'TCMB' in params.keys() else None
        if TCMB is None:
            TCMB = params['T_cmb'] if 'T_cmb' in params.keys() else None
        if self.engine=='camb':
            if ('sigma8' in params.keys()) or ('S8' in params.keys()):
                print("sigma8 or S8 not supported with CAMB. Use the CLASS engine.")
            self._camb_pars = camb.set_params(ns=params['ns'],As=params['As'],H0=H0,
                                        cosmomc_theta=theta,ombh2=params['ombh2'],
                                        omch2=params['omch2'], mnu=params['mnu'],
                                        omk = params['omk'],
                                        tau=params['tau'],nnu=params['nnu'],
                                        num_massive_neutrinos=
                                        params['num_massive_neutrinos'],
                                        w=params['w0'],wa=params['wa'],
                                        dark_energy_model='ppf',
                                        halofit_version=self.p['default_halofit'] if halofit is None else halofit,
                                        AccuracyBoost=2,pivot_scalar=params['pivot_scalar'],YHe=YHe)
            self._camb_pars.WantTransfer = True
            self._camb_results = camb.get_background(self._camb_pars)
        elif self.engine=='class':
            from classy import Class
            self._class_results = Class()
            passp = {}
            if ('sigma8' in params.keys()):
                passp['sigma8'] = params['sigma8']
                if ('S8' in params.keys()) or ('As' in params.keys()): 
                    warnings.warn("Using sigma8 from params and ignoring S8 and As.")                    
            elif ('S8' in params.keys()):
                passp['S8'] = params['S8']
                if ('sigma8' in params.keys()) or ('As' in params.keys()): 
                    warnings.warn("Using S8 from params and ignoring As.")
            else:
                passp['A_s'] = params['As']
            if theta is None:
                passp['h'] = h
            else:
                passp['theta_s_100'] = theta*100
                print("WARNING: Definitions of theta in CAMB and CLASS are different. Assuming CLASS definition.")
            for p in params.keys():
                if p[:6]=='class_':
                    passp[p[6:]] = params[p]
            
            passp['omega_cdm'] = params['omch2']
            passp['omega_b'] = params['ombh2']
            passp['Omega_k'] = params['omk']
            passp['n_s'] = params['ns']
            if not(YHe is None): passp['YHe'] = YHe
            if not(TCMB is None): passp['T_cmb'] = TCMB
            self._class_pars = dict(passp)
            self._class_results.set(passp)
            self._class_results.compute()
        self.params = params
        omh2 = self.params['omch2']+self.params['ombh2'] # FIXME: neutrinos
        self.h = h
        self.omm0 = omh2 / (self.params['H0']/100.)**2.
        self.omk0 = self.params['omk']
        self.oml0 = 1-self.omm0-self.omk0
        try: self.as8 = self.params['as8']        
        except: self.as8 = 1

        self.ombh2 = self.params['ombh2']
        if self.engine=='class':
            self.YHe = self._get_class_result('YHe')
        elif self.engine=='camb':
            self.YHe = self._camb_pars.YHe
        
    def _get_matter_power(self,zs,ks,nonlinear=False):
        PK = self.get_pk_interpolator(zs,kmax=ks.max(),var='total',nonlinear=nonlinear)
        return (self.as8**2.) * PK.P(zs, ks, grid=True)

        
    def rho_matter_z(self,z):
        return self.rho_critical_z(0.) * self.omm0 \
            * (1+np.atleast_1d(z))**3. # in msolar / megaparsec3

    def omz(self,z):
        return self.rho_matter_z(z)/self.rho_critical_z(z)
    
    def rho_critical_z(self,z):
        Hz = self.hubble_parameter(z) * 3.241e-20 # SI # FIXME: constants need checking
        G = 6.67259e-11 # SI
        rho = 3.*(Hz**2.)/8./np.pi/G # SI
        return rho * 1.477543e37 # in msolar / megaparsec3
    
    def get_sigma2_R(self,R,zs,
                     kmin=None,kmax=None,numks=None,
                     Ws=None,ret_pk=False):
        zs = np.atleast_1d(zs)
        R = np.asarray(R)
        if R.ndim==1: R = R[None,:,None]
        kmin = self.p['sigma2_kmin'] if kmin is None else kmin
        kmax = self.p['sigma2_kmax'] if kmax is None else kmax
        numks = self.p['sigma2_numks'] if numks is None else numks
        ks_sigma2 = np.geomspace(kmin,kmax,numks) # ks for sigma2 integral
        if self.accuracy=='high':
            self.sPzk = self.P_lin_slow(ks_sigma2,zs,kmax=kmax)
        elif self.accuracy=='medium':
            self.sPzk = self.P_lin(ks_sigma2,zs)
        elif self.accuracy=='low':
            self.sPzk = self.P_lin_approx(ks_sigma2,zs)
        ks = ks_sigma2[None,None,:]
        W2 = Wkr(ks,R,self.p['Wkr_taylor_switch'])**2. if Ws is None else Ws**2.
        Ps = self.sPzk[:,None,:]
        integrand = Ps*W2*ks**2./2./np.pi**2.
        sigma2 = simps(integrand,x=ks,axis=-1)
        if ret_pk:
            return sigma2, ks, Ps
        else:
            return sigma2
    
    def get_sigma8(self,zs,exact=False,kmin=1e-4,kmax=None,Ws=None,numks=1000,ret_pk=False):
        zs = np.atleast_1d(zs)
        if exact:
            if self.engine=='camb':
                return self._camb_results.get_sigma8() # fix this
            elif self.engine=='class':
                if kmax is None: kmax = self.p['sigma2_kmax']
                self._set_class_power(zs,kmax=kmax)
                return np.vectorize(lambda x : self._class_results.sigma(8./self.h,x))(zs)
        else:
            r = self.get_sigma2_R(8./self.params['H0']*100.,zs,
                                               kmin=kmin,kmax=kmax,Ws=Ws,numks=numks,ret_pk=ret_pk)
            if ret_pk:
                return np.sqrt(r[0]), r[1], r[2]
            else:
                return np.sqrt(r)
        
    def D_growth_exact_arbitrary_norm(self,a,k_camb=1e-5):
        if self.engine=='camb':
            deltakz = self._camb_results.get_redshift_evolution(k_camb, a2z(a), ['delta_cdm']) #index: z,0
            D = deltakz[:,0]
        elif self.engine=='class':
            D = np.vectorize(self._class_results.scale_independent_growth_factor)(a2z(a))
        return D
    

    def D_growth_approx(self,a):
        # Solution to Heath et al 1977
        # (inspired by
        # Klypin, Trujillo, Primack - Bolshoi paper 1 - Appendix A)
        # normed so that D(a)=a in matter domination
        # These assume LCDM; should work with non-flat models.
        # Haven't checked if it works for mnu!= or w!=-1.
        a = np.asarray(a)
        omm0 = self.omm0
        oml0 = self.oml0
        x = (oml0/omm0)**(1./3.) * a
        # Exact analytic integral
        Dovera = np.sqrt(1.+x**3.)*(hyp2f1(5/6.,3/2.,11/6.,-x**3.))
        # An approximation to this integral
        # oma = 1./(1+x**3)
        # oml = 1-oma
        # Dovera = (5./2*oma)/(oma**(4./7)-oml+(1+oma/2.)*(1+oml/70.))        
        return Dovera*a

    
    def D_growth(self, a,type="anorm",exact=False,k_camb=1e-5):
        if exact:
            Dfunc = lambda a: self.D_growth_exact_arbitrary_norm(a,k_camb=k_camb)
        else:
            Dfunc = self.D_growth_approx
        Dtoday = Dfunc(1)
        val = Dfunc(a)/Dtoday
        if type=="z0norm":
            mul = 1 #normed so that D(a=1)=1
        elif type=="anorm":
            mul = self.D_growth_approx(1)
            # mul = 0.7779 for oml=0.7, omm=0.3
            # This is different from the 0.76 often cited!
        else:
            raise ValueError
        return val*mul

    def get_bao_rs_dV(self,zs):
        zs = np.asarray(zs)
        if self.engine=='camb':
            retval = self._camb_results.get_BAO(zs,self._camb_pars)[:,0]
        elif self.engine=='class':
            Hzs = self.hubble_parameter(zs)/cspeed
            D_As = self.angular_diameter_distance(zs)
            D_Vs = ((1+zs)**2 * D_As**2 * zs/Hzs)**(1/3.)
            retval = self._class_results.rs_drag()/D_Vs
        return retval

    def get_growth_rate_f(self,zs):
        zs = np.atleast_1d(zs)
        if self.engine=='camb':
            raise NotImplementedError
        elif self.engine=='class':
            return np.vectorize(self._class_results.scale_independent_growth_factor_f)(zs)
    

    def P_lin(self,ks,zs,knorm = 1e-4,kmax = None):
        """
        This function will provide the linear matter power spectrum used in calculation
        of sigma2. It is written as
        P_lin(k,z) = norm(z) * T(k)**2
        where T(k) is the Eisenstein, Hu, 1998 transfer function.
        Care has to be taken about interpreting this beyond LCDM.
        For example, the transfer function can be inaccurate for nuCDM and wCDM cosmologies.
        If this function is only used to model sigma2 -> N(M,z) -> halo model power spectra at small
        scales, and cosmological dependence is obtained through an accurate CAMB based P(k),
        one should be fine.
        """
        zs = np.asarray(zs)
        ks = np.asarray(ks)
        tk = self.Tk(ks,'eisenhu_osc') 
        if kmax is None: kmax = ks.max()
        if knorm>=kmax: raise ValueError
        PK = self.get_pk_interpolator(zs,kmax=kmax,var='total',nonlinear=False)
        pnorm = PK.P(zs, knorm,grid=True)
        tnorm = self.Tk(knorm,'eisenhu_osc') * knorm**(self.params['ns'])
        plin = (pnorm/tnorm) * tk**2. * ks**(self.params['ns'])
        return (self.as8**2.) *plin
 
    def P_lin_slow(self,ks,zs,kmax = None):
        zs = np.asarray(zs)
        ks = np.asarray(ks)
        if kmax is None: kmax = ks.max()
        PK = self.get_pk_interpolator(zs,kmax=kmax,var='total',nonlinear=False)
        plin = PK.P(zs, ks,grid=True)
        return (self.as8**2.) * plin

    def get_Omega_nu(self):
        # check this
        if self.engine=='camb':
            return self._camb_results.get_Omega('nu')
        elif self.engine=='class':
            return self._class_results.Omega_nu
        
    def P_lin_approx(self,ks,zs,type='eisenhu_osc'):
        zs = np.atleast_1d(zs)
        ks = np.asarray(ks)
        tk = self.Tk(ks,type=type)[None,:]
        a = 1/(1+zs)
        Dzs = self.D_growth(a,type='anorm')[:,None]
        kp = self.params['pivot_scalar']
        ns = self.params['ns']
        omh2 = (self.params['omch2'] + self.params['ombh2'])*100**2. + self.get_Omega_nu()*self.params['H0']**2.
        kfacts = (ks/kp)**(ns-1.)  * ks
        pref = 8*np.pi**2*self.params['As']/25./omh2**2. * cspeed**4.
        return  pref * kfacts[None,:] * Dzs**2. * tk**2.

    def Tk(self,ks,type ='eisenhu_osc'):
        """
        Pulled from cosmicpy https://github.com/cosmicpy/cosmicpy/blob/master/LICENSE.rst
        """
        
        k = ks/self.h
        self.tcmb = 2.726
        T_2_7_sqr = (self.tcmb/2.7)**2
        h2 = self.h**2
        w_m = self.params['omch2'] + self.params['ombh2']
        w_b = self.params['ombh2']

        self._k_eq = 7.46e-2*w_m/T_2_7_sqr / self.h     # Eq. (3) [h/Mpc]
        self._z_eq = 2.50e4*w_m/(T_2_7_sqr)**2          # Eq. (2)

        # z drag from Eq. (4)
        b1 = 0.313*pow(w_m, -0.419)*(1.0+0.607*pow(w_m, 0.674))
        b2 = 0.238*pow(w_m, 0.223)
        self._z_d = 1291.0*pow(w_m, 0.251)/(1.0+0.659*pow(w_m, 0.828)) * \
            (1.0 + b1*pow(w_b, b2))

        # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
        self._R_d = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_d)
        # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
        self._R_eq = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_eq)
        # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
        self.sh_d = 2.0/(3.0*self._k_eq) * np.sqrt(6.0/self._R_eq) * \
            np.log((np.sqrt(1.0 + self._R_d) + np.sqrt(self._R_eq + self._R_d)) /
                (1.0 + np.sqrt(self._R_eq)))
        # Eq. (7) but in [hMpc^{-1}]
        self._k_silk = 1.6 * pow(w_b, 0.52) * pow(w_m, 0.73) * \
            (1.0 + pow(10.4*w_m, -0.95)) / self.h

        Omega_m = self.omm0
        fb = self.params['ombh2'] / (self.params['omch2']+self.params['ombh2']) # self.Omega_b / self.Omega_m
        fc = self.params['omch2'] / (self.params['omch2']+self.params['ombh2']) # self.params['ombh2'] #(self.Omega_m - self.Omega_b) / self.Omega_m
        alpha_gamma = 1.-0.328*np.log(431.*w_m)*w_b/w_m + \
            0.38*np.log(22.3*w_m)*(fb)**2
        gamma_eff = Omega_m*self.h * \
            (alpha_gamma + (1.-alpha_gamma)/(1.+(0.43*k*self.sh_d)**4))

        res = np.zeros_like(k)

        if(type == 'eisenhu'):

            q = k * pow(self.tcmb/2.7, 2)/gamma_eff

            # EH98 (29) #
            L = np.log(2.*np.exp(1.0) + 1.8*q)
            C = 14.2 + 731.0/(1.0 + 62.5*q)
            res = L/(L + C*q*q)

        elif(type == 'eisenhu_osc'):
            # Cold dark matter transfer function

            # EH98 (11, 12)
            a1 = pow(46.9*w_m, 0.670) * (1.0 + pow(32.1*w_m, -0.532))
            a2 = pow(12.0*w_m, 0.424) * (1.0 + pow(45.0*w_m, -0.582))
            alpha_c = pow(a1, -fb) * pow(a2, -fb**3)
            b1 = 0.944 / (1.0 + pow(458.0*w_m, -0.708))
            b2 = pow(0.395*w_m, -0.0266)
            beta_c = 1.0 + b1*(pow(fc, b2) - 1.0)
            beta_c = 1.0 / beta_c

            # EH98 (19). [k] = h/Mpc
            def T_tilde(k1, alpha, beta):
                # EH98 (10); [q] = 1 BUT [k] = h/Mpc
                q = k1 / (13.41 * self._k_eq)
                L = np.log(np.exp(1.0) + 1.8 * beta * q)
                C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * pow(q, 1.08))
                T0 = L/(L + C*q*q)
                return T0

            # EH98 (17, 18)
            f = 1.0 / (1.0 + (k * self.sh_d / 5.4)**4)
            Tc = f * T_tilde(k, 1.0, beta_c) + \
                (1.0 - f) * T_tilde(k, alpha_c, beta_c)

            # Baryon transfer function
            # EH98 (19, 14, 21)
            y = (1.0 + self._z_eq) / (1.0 + self._z_d)
            x = np.sqrt(1.0 + y)
            G_EH98 = y * (-6.0 * x +
                          (2.0 + 3.0*y) * np.log((x + 1.0) / (x - 1.0)))
            alpha_b = 2.07 * self._k_eq * self.sh_d * \
                pow(1.0 + self._R_d, -0.75) * G_EH98

            beta_node = 8.41 * pow(w_m, 0.435)
            tilde_s = self.sh_d / pow(1.0 + (beta_node /
                                             (k * self.sh_d))**3, 1.0/3.0)

            beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * w_m)**2 + 1.0)

            # [tilde_s] = Mpc/h
            Tb = (T_tilde(k, 1.0, 1.0) / (1.0 + (k * self.sh_d / 5.2)**2) +
                  alpha_b / (1.0 + (beta_b/(k * self.sh_d))**3) *
                  np.exp(-pow(k / self._k_silk, 1.4))) * np.sinc(k*tilde_s/np.pi)

            # Total transfer function
            res = fb * Tb + fc * Tc
        return res

    def lensing_window(self,ezs,zs,dndz=None):
        """
        Generates a lensing convergence window 
        W(z).

        zs: If (nz,) with nz>2 and dndz is not None, then these are the points
        at which dndz is defined. If nz=2 and no dndz is provided, it is (zmin,zmax)
        for a top-hat window. If a single number, and no dndz is provided,
        it is a delta function source at zs.
        """
        zs = np.array(zs).reshape(-1)
        H0 = self.h_of_z(0.)
        H = self.h_of_z(ezs)
        chis = self.comoving_radial_distance(ezs)
        chistar = self.comoving_radial_distance(zs)
        if zs.size==1:
            assert dndz is None
            integrand = (chistar - chis)/chistar
            integral = integrand
            integral[ezs>zs] = 0
        else:
            nznorm = np.trapz(dndz,zs)
            dndz = dndz/nznorm
            # integrand has shape (num_z,num_zs) to be integrated over zs
            integrand = (chistar[None,:] - chis[:,None])/chistar[None,:] * dndz[None,:]
            for i in range(integrand.shape[0]): integrand[i][zs<ezs[i]] = 0 # FIXME: vectorize this
            integral = np.trapz(integrand,zs,axis=-1)
            
        return 1.5*self.omm0*H0**2.*(1.+ezs)*chis/H * integral

    def C_kg(self,ells,zs,ks,Pgm,gzs,gdndz=None,lzs=None,ldndz=None,lwindow=None):
        gzs = np.array(gzs).reshape(-1)
        if lwindow is None: Wz1s = self.lensing_window(gzs,lzs,ldndz)
        else: Wz1s = lwindow
        chis = self.comoving_radial_distance(gzs)
        hzs = self.h_of_z(gzs) # 1/Mpc
        if gzs.size>1:
            nznorm = np.trapz(gdndz,gzs)
            Wz2s = gdndz/nznorm
        else:
            Wz2s = 1.
        return limber_integral(ells,zs,ks,Pgm,gzs,Wz1s,Wz2s,hzs,chis)

    def C_gg(self,ells,zs,ks,Pgg,gzs,gdndz=None,zmin=None,zmax=None):
        gzs = np.asarray(gzs)
        chis = self.comoving_radial_distance(gzs)
        hzs = self.h_of_z(gzs) # 1/Mpc
        if gzs.size>1:
            nznorm = np.trapz(gdndz,gzs)
            Wz1s = gdndz/nznorm
            Wz2s = gdndz/nznorm
        else:
            dchi = self.comoving_radial_distance(zmax) - self.comoving_radial_distance(zmin)
            Wz1s = 1.
            Wz2s = 1./dchi/hzs
        return limber_integral(ells,zs,ks,Pgg,gzs,Wz1s,Wz2s,hzs,chis)

    def C_kk(self,ells,zs,ks,Pmm,lzs1=None,ldndz1=None,lzs2=None,ldndz2=None,lwindow1=None,lwindow2=None):
        if lwindow1 is None: lwindow1 = self.lensing_window(zs,lzs1,ldndz1)
        if lwindow2 is None: lwindow2 = self.lensing_window(zs,lzs2,ldndz2)
        chis = self.comoving_radial_distance(zs)
        hzs = self.h_of_z(zs) # 1/Mpc
        return limber_integral(ells,zs,ks,Pmm,zs,lwindow1,lwindow2,hzs,chis)

    def C_gy(self,ells,zs,ks,Pgp,gzs,gdndz=None,zmin=None,zmax=None):
        gzs = np.asarray(gzs)
        chis = self.comoving_radial_distance(gzs)
        hzs = self.h_of_z(gzs) # 1/Mpc
        if gzs.size>1:
            nznorm = np.trapz(gdndz,gzs)
            Wz1s = dndz/nznorm
            Wz2s = gdndz/nznorm
        else:
            dchi = self.comoving_radial_distance(zmax) - self.comoving_radial_distance(zmin)
            Wz1s = 1.
            Wz2s = 1./dchi/hzs

        return limber_integral(ells,zs,ks,Ppy,gzs,1,Wz2s,hzs,chis)

    def C_ky(self,ells,zs,ks,Pym,lzs1=None,ldndz1=None,lzs2=None,ldndz2=None,lwindow1=None):
        if lwindow1 is None: lwindow1 = self.lensing_window(zs,lzs1,ldndz1)
        chis = self.comoving_radial_distance(zs)
        hzs = self.h_of_z(zs) # 1/Mpc
        return limber_integral(ells,zs,ks,Pym,zs,lwindow1,1,hzs,chis)

    def C_yy(self,ells,zs,ks,Ppp,dndz=None,zmin=None,zmax=None):
        chis = self.comoving_radial_distance(zs)
        hzs = self.h_of_z(zs) # 1/Mpc
        # Convert to y units
        # 

        return limber_integral(ells,zs,ks,Ppp,zs,1,1,hzs,chis)

    def total_matter_power_spectrum(self, Pnn, Pne, Pee):
        """
        Calculates the total matter auto-power spectrum.

        Parameters
        ==========

        - Pnn (array): The auto-power spectrum of cold dark matter (e.g. NFW).
        - Pne (array): Cross-power spectrum of cold dark matter (e.g. NFW) and gas (e.g. free electron overdensity).
        - Pee (array): The auto-power spectrum of gas (e.g. free electron overdensity).
        
        Returns
        =======

        - array: The total matter power spectrum.

        """
        omtoth2 = self.p['omch2'] + self.p['ombh2']
        fc = self.p['omch2'] / omtoth2
        fb = self.p['ombh2'] / omtoth2
        return fc ** 2. * Pnn + 2. * fc * fb * Pne + fb * fb * Pee
    
    def total_matter_power_spectrum(self,Pnn,Pne,Pee):
        """
        
        """

        omtoth2 = self.p['omch2'] + self.p['ombh2']
        fc = self.p['omch2']/omtoth2
        fb = self.p['ombh2']/omtoth2
        return fc**2.*Pnn + 2.*fc*fb*Pne + fb*fb*Pee

    def total_matter_galaxy_power_spectrum(self, Pgn, Pge):
        """
        Calculates the cross-spectrum between total matter and galaxies.

        Arguments
        =========

        - Pgn (array): Cross-power spectrum of cold dark matter (e.g. NFW) and galaxies.
        - Pge (array): Cross-power spectrum of galaxies and gas (e.g. free electron overdensity).

        Returns
        =======

        - array: Cross-power spectrum of total matter and galaxies.
        """
        omtoth2 = self.p['omch2'] + self.p['ombh2']
        fc = self.p['omch2'] / omtoth2
        fb = self.p['ombh2'] / omtoth2
        return fc * Pgn + fb * Pge
    
    def total_matter_galaxy_power_spectrum(self,Pgn,Pge):
        """
        
        """
        omtoth2 = self.p['omch2'] + self.p['ombh2']
        fc = self.p['omch2']/omtoth2
        fb = self.p['ombh2']/omtoth2
        return fc*Pgn + fb*Pge

    def cmb_lensing_kk_exact(self,lmax,lens_potential_accuracy=4):
        r"""
        Calculate the lensing convergence power spectrum C_l^{\kappa\kappa} using the exact calculation in CAMB.

        Arguments
        =========

        lmax: int
            maximum ell value to calculate to

        lens_potential_accuracy: int, optional
            accuracy of lensing potential calculation

        Returns
        =======

        ls: array
            ell values

        cl_kappa: array
            C_l^{\kappa\kappa} values

        """
        if self.engine=='class': raise NotImplementedError
        self._camb_pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)
        results = camb.get_results(self._camb_pars)
        cl = results.get_lens_potential_cls(lmax=lmax)[:,0]
        ells = np.arange(cl.size)
        return ells,cl*2.*np.pi/4.

    def _get_class_result(self,key):
        return self._class_results.get_current_derived_parameters([key])[key]

    def get_tau_star(self):
        r"""
        Get the conformal time at recombination.
        """
        if self.engine=='camb':
            return self._camb_results.tau_maxvis
        elif self.engine=='class':
            return self._get_class_result('tau_star')


    def z_of_tau(self,tau):
        r"""
        Get the redshift at conformal time tau.
        """
        if self.engine=='camb':
            return self._camb_results.redshift_at_comoving_radial_distance(tau)
        elif self.engine=='class':
            return np.vectorize(self._class_results.z_of_tau)(tau)
            
        
    def redshift_at_comoving_radial_distance(self,chi,zmax=1e4):
        r"""
        Get the redshift at comoving radial distance chi.

        TODO: Check if r and chi are the same in CLASS, esp.
        for non-flat cosmologies.
        """
        chi = np.asarray(chi)
        if self.engine=='camb':
            return self._camb_results.redshift_at_comoving_radial_distance(chi)
        elif self.engine=='class':
            # CLASS only has a chi(z) function so we have to invert it
            # with a bisection search
            zmin = 0.
            chi = np.atleast_1d(chi)
            ret = utils.vectorized_bisection_search(chi,self.comoving_radial_distance,[zmin,zmax],'increasing',verbose=False)
            if ret.size==1: return float(ret[0])
            else: return ret

    def conformal_time(self,z,zmintol=1e-5):
        r"""
        Get the conformal time at redshift z.

        Arguments
        =========

        z: float
            redshift. With the CLASS engine, if this is less than zmintol (default: 1e-5), it is assumed to be zero and the conformal age of the universe is returned.

        Returns
        =======

        float: conformal time
        """
        if self.engine=='camb':
            return self._camb_results.conformal_time(z)
        elif self.engine=='class':
            # CLASS only has a z(tau) function so we have to invert it
            # with a bisection search
            taumin = 0.
            taumax = self._get_class_result('conformal_age')
            z = np.atleast_1d(z)
            ret = z*0.
            mask = z<zmintol
            ret[mask] = taumax
            if ret[~mask].size>0:            
                print(z[~mask].shape)
                ret[~mask] = utils.vectorized_bisection_search(z[~mask],self.z_of_tau,[taumin,taumax],'decreasing',verbose=False)
            if ret.size==1: return float(ret[0])
            else: return ret

    def _set_class_power(self,zs,kmax):
        self._class_pars['output']='mPk, dTk'
        if zs.size>100: zs = np.geomspace(zs.min(),zs.max(),100) # FIXME: CLASS z limit
        self._class_pars['z_pk'] = ','.join([f'{z:.6f}' for z in zs]) # FIXME: z precision
        self._class_pars['P_k_max_h/Mpc'] = kmax / self.h
        self._class_results.set(self._class_pars)
        self._class_results.compute()
        
    def get_pk_interpolator(self,zs,kmax,var='weyl',nonlinear=False,return_z_k=False, k_per_logint=None, log_interp=True, extrap_kmax=None):
        var = var.lower()
        ozs = zs.copy()
        if self.engine=='camb':
            if var=='weyl': 
                cvar = model.Transfer_Weyl
            elif var=='total': 
                cvar = 'delta_tot'
            elif var=='cb':
                cvar = 'delta_nonu'
            else:
                raise ValueError
            PK = camb.get_matter_power_interpolator(self._camb_pars, nonlinear=nonlinear, 
                                                hubble_units=False, k_hunit=False, kmax=kmax,
                                                var1=cvar,var2=cvar, zmax=zs[-1])
        elif self.engine=='class':
            self._set_class_power(zs,kmax)
            if var=='weyl':
                pk,ks,zs = self._class_results.get_Weyl_pk_and_k_and_z(nonlinear=nonlinear, h_units=False)
            else:
                if var=='total':
                    onlyc = False
                elif var=='cb':
                    onlyc = True
                else:
                    raise ValueError
                pk,ks,zs = self._class_results.get_pk_and_k_and_z(nonlinear=nonlinear, only_clustering_species = onlyc, h_units=False)
                if zs.min()>ozs.min(): raise ValueError
                if zs.max()<ozs.max(): raise ValueError
                
            #pk  is k,z ordering and zs are in reverse order!!
            PK = utils.get_matter_power_interpolator_generic(ks, zs[::-1], 
                                                             pk.swapaxes(0,1)[::-1,:], 
                                                             return_z_k=return_z_k,
                                                             log_interp=log_interp,
                                                             extrap_kmax=extrap_kmax, silent=True)
    
        return PK
                

    def cmb_lensing_limber(self,lmax,nonlinear=False):
        r"""
        Calculate the lensing convergence power spectrum C_l^{\kappa\kappa} using Limber approximation, but using the Weyl potential power spectrum from CAMB. This code is adapted from the CAMB demo.

        Arguments
        =========

        lmax: int
            maximum ell value to calculate to

        nonlinear: bool, optional
            whether to use non-linear corrections

        Returns
        =======

        ls: array
            ell values

        cl_kappa: array 
            C_l^{\kappa\kappa} values
        """
        nz = 100 #number of steps to use for the radial/redshift integration
        kmax=10  #kmax to use
        chistar = self.conformal_time(0)- self.get_tau_star()
        chis = np.linspace(0,chistar,nz)
        zs=self.redshift_at_comoving_radial_distance(chis)
        #Calculate array of delta_chi, and drop first and last points where things go singular
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1]
        zs = zs[1:-1]

        #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
        #Here for lensing we want the power spectrum of the Weyl potential.
        PK = self.get_pk_interpolator(zs,kmax,var='weyl',nonlinear=nonlinear)

        #Get lensing window function (flat universe)
        win = ((chistar-chis)/(chis**2*chistar))**2
        #Do integral over chi
        ls = np.arange(2,lmax+1, dtype=np.float64)
        cl_kappa=np.zeros(ls.shape)
        w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
        for i, l in enumerate(ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=kmax]=0
            P = PK.P(zs, k, grid=False)
            cl_kappa[i] = np.dot(dchis, w*P*win/k**4)
        cl_kappa*= (ls*(ls+1))**2
        return ls,cl_kappa


def a2z(a): return (1.0/np.atleast_1d(a))-1.0

def limber_integral(ells,zs,ks,Pzks,gzs,Wz1s,Wz2s,hzs,chis):
    r"""
    Get C(ell) = \int dz (H(z)/c) W1(z) W2(z) Pzks(z,k=ell/chi) / chis**2.
    ells: (nells,) multipoles looped over
    zs: redshifts (npzs,) corresponding to Pzks
    ks: comoving wavenumbers (nks,) corresponding to Pzks
    Pzks: (npzs,nks) power specrum
    gzs: (nzs,) corersponding to Wz1s, W2zs, Hzs and chis
    Wz1s: weight function (nzs,)
    Wz2s: weight function (nzs,)
    hzs: Hubble parameter (nzs,) in *1/Mpc* (e.g. camb.results.h_of_z(z))
    chis: comoving distances (nzs,)

    We interpolate P(z,k)
    """
    hzs = np.array(hzs).reshape(-1)
    Wz1s = np.array(Wz1s).reshape(-1)
    Wz2s = np.array(Wz2s).reshape(-1)
    chis = np.array(chis).reshape(-1)

    prefactor = hzs * Wz1s * Wz2s   / chis**2.
    zevals = gzs
    if zs.size>1:            
         f = interp2d(ks,zs,Pzks,bounds_error=True)     
    else:      
         f = interp1d(ks,Pzks[0],bounds_error=True)
    Cells = np.zeros(ells.shape)
    for i,ell in enumerate(ells):
        kevals = (ell+0.5)/chis
        if zs.size>1:
            # hack suggested in https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
            # to get around scipy.interpolate limitations
            interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zevals)[0]
        else:
            interpolated = f(kevals)
        if zevals.size==1: Cells[i] = interpolated * prefactor
        else: Cells[i] = np.trapz(interpolated*prefactor,zevals)
    return Cells
    
