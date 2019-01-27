import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import camb
from camb import model,nonlinear
import numpy as np
import tinker
import scipy

"""

General vectorized FFT-based halo model implementation
Author(s): Mathew Madhavacheril
Credits: Follows approach in Matt Johnson and Moritz
Munchmeyer's implementation. Some of the HOD functions are copied from there.

Array indexing is as follows:
[z,M,k/r]

r is always in Mpc
k is always in Mpc-1
All rho densities are in Msolar/Mpc^3
All masses m are in Msolar
No h units anywhere

TODO: copy member functions like Duffy concentration to independent
barebones functions in separate script/library

FIXME: the profile integrals have ranges and sampling that depend on 
how low in mass one goes.

Known issues:
1. The 1-halo term will add some power at the largest scales. A softening term
has been added that effectively zeros the 1-halo term below k<0.01.
2. sigma2 becomes very innacurate for low masses, because lower masses require higher
k in the linear matter power. For this reason, low-mass halos
are never accounted for correctly, and thus consistency relations do not hold.
Currently the consistency relation is divided out (like Matt does) to get the 2-halo
power to agree with Plin at large scales.
3. Higher redshifts have less than expected 1-halo power. Matt's implementation
has power comparable to non-linear halofit.
5. Accuracy of sigma2 is very important for small scale 1-halo at high redshifts


1. not an fft issue (using analytical)
2. not a c(M,z) issue (tried Bhatt vs Duffy)
3. not a sigma2 accuracy issue? Using high k Eisenstein
4. ahhh this might be because the correction to the 2-halo term is additive not multiplicative!!!
5. mass functions all disagree

Limitations:
1. Only 200 * rho_mean(z) is supported except for gas profiles defined with rho_crit(z)
where the necessary conversion is done
2. Only Tinker 2010 is supported

"""

default_params = {
    'st_A': 0.3222,
    'st_a': 0.707,
    'st_p': 0.3,
    'st_deltac': 1.686,
    'duffy_A_vir':7.85, # for Mvir
    'duffy_alpha_vir':-0.081,
    'duffy_beta_vir':-0.71,
    'duffy_A_mean':10.14, # for M200rhomeanz
    'duffy_alpha_mean':-0.081,
    'duffy_beta_mean':-1.01,
    # 'kstar_damping':0.00001,
    'kstar_damping':0.01,
    
    
    'omch2': 0.1198,
    'ombh2': 0.02225,
    'H0': 67.3,
    'ns': 0.9645,
    'As': 2.2e-9,
    'mnu': 0.0, # NOTE NO NEUTRINOS IN DEFAULT
    'w0': -1.0,
    'tau':0.06,
    'nnu':3.046,
    'wa': 0.,
    'num_massive_neutrinos':3,
                        
    }


tunable_params = {
    'sigma2_kmin':1e-4,
    'sigma2_kmax':1000,
    'sigma2_numks':4000,
    'nfw_integral_default_numxs':30000,
    'nfw_integral_default_xmax':200,
    'Wkr_taylor_switch':0.01,
    #'Wkr_taylor_switch':0.1,
}
    

def Wkr_taylor(kR):
    xx = kR*kR
    return 1 - .1*xx + .00357142857143*xx*xx

def Wkr(k,R,taylor_switch=tunable_params['Wkr_taylor_switch']):
    kR = k*R
    ans = 3.*(np.sin(kR)-kR*np.cos(kR))/(kR**3.)
    ans[kR<taylor_switch] = Wkr_taylor(kR[kR<taylor_switch]) 
    return ans

def duffy_concentration(m,z,A,alpha,beta,h): return A*((h*m/2.e12)**alpha)*(1+z)**beta
    
class HaloCosmology(object):
    def __init__(self,zs,ks,params={},ms=None,mass_function="tinker",halofit="original"):
        self.p = params
        self.mode = mass_function
        self.nzm = None
        for param in default_params.keys():
            if param not in self.p.keys(): self.p[param] = default_params[param]
        self.zs = np.asarray(zs)
        self.ks = ks
        self._init_cosmology(self.p,halofit)
        self.Pzk = self._get_matter_power(self.zs,ks,nonlinear=False)
        self.nPzk = self._get_matter_power(self.zs,ks,nonlinear=True)
        self.uk_profiles = {}
        if ms is not None:
            kmin = tunable_params['sigma2_kmin']
            kmax = tunable_params['sigma2_kmax']
            numks = tunable_params['sigma2_numks']
            self.ks_sigma2 = np.geomspace(kmin,kmax,numks) # ks for sigma2 integral
            self.sPzk = self.P_lin(self.ks_sigma2,self.zs)
            self.initialize_mass_function(ms)
            
    
    def _init_cosmology(self,params,halofit):
        try:
            theta = params['theta100']/100.
            H0 = None
            print("WARNING: Using theta100 parameterization. H0 ignored.")
        except:
            H0 = params['H0']
            theta = None
        
        self.pars = camb.set_params(ns=params['ns'],As=params['As'],H0=H0, cosmomc_theta=theta,ombh2=params['ombh2'], omch2=params['omch2'], mnu=params['mnu'], tau=params['tau'],nnu=params['nnu'],num_massive_neutrinos=params['num_massive_neutrinos'],w=params['w0'],wa=params['wa'],dark_energy_model='ppf',halofit_version=halofit,AccuracyBoost=2,kmax=1) # !!
        self.results = camb.get_background(self.pars)
        self.params = params
        self.h = self.params['H0']/100.
        omh2 = self.params['omch2']+self.params['ombh2'] # FIXME: neutrinos
        self.om0 = omh2 / (self.params['H0']/100.)**2.
        self.chis = self.results.comoving_radial_distance(self.zs)
        self.Hzs = self.results.hubble_parameter(self.zs)

        
    def _get_matter_power(self,zs,ks,nonlinear=False):
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear, 
                                                     hubble_units=False, k_hunit=False, kmax=ks.max(),#+1., # !!!!)
                                                     zmax=zs.max()+1.) # FIXME: neutrinos !!!
        return PK.P(zs, ks, grid=True)

        
    def rho_critical_z(self,z):
        Hz = self.results.hubble_parameter(z) * 3.241e-20 # SI # FIXME: constants need checking
        G = 6.67259e-11 # SI
        rho = 3.*(Hz**2.)/8./np.pi/G # SI
        return rho * 1.477543e37 # in msolar / megaparsec3
    
    def rho_matter_z(self,z): return self.rho_critical_z(0.) * self.om0 * (1+np.atleast_1d(z))**3. # in msolar / megaparsec3
    def omz(self,z): return self.rho_matter_z(z)/self.rho_critical_z(z)
    #def deltav(self,z): return 178. * self.omz(z)**(0.45) # Eke et al 1998
    def deltav(self,z): # Duffy virial actually uses this from Bryan and Norman 1997
        x = self.omz(z) - 1.
        d = 18.*np.pi**2. + 82.*x - 39. * x**2.
        return d
    def rvir(self,m,z):
        # rvir has now been changed to R200rhomean(z)
        #return ((3.*m/4./np.pi)/200./self.rho_matter_z(z))**(1./3.)
        return ((3.*m/4./np.pi)/self.deltav(z)/self.rho_critical_z(z))**(1./3.)
    
    def R_of_m(self,ms): return (3.*ms/4./np.pi/self.rho_matter_z(0))**(1./3.) # note rhom0
    
    def get_sigma2(self,ms):
        ks = self.ks_sigma2[None,None,:]
        R = self.R_of_m(ms)[None,:,None]
        W2 = Wkr(ks,R)**2.
        Ps = self.sPzk[:,None,:]
        integrand = Ps*W2*ks**2./2./np.pi**2.
        # from orphics import io
        # pl = io.Plotter(xyscale='loglog')
        # for i in range(self.zs.size): pl.add(ks[0,-1],integrand[i,-1],label=str(self.zs[i]))
        # pl.done()
        from scipy.integrate import simps
        sigma2 = simps(integrand,ks,axis=-1)
        return sigma2
        

    def initialize_mass_function(self,ms):
        sigma2 = self.get_sigma2(ms)
        self.sigma2 = sigma2
        self.nzm = self.get_nzm(ms,sigma2)
        self.bh = self.get_bh(ms,sigma2)

    def get_fsigmaz(self,ms=None,sigma2=None):
        if sigma2 is None: sigma2 = self.get_sigma2(ms)
        deltac = self.p['st_deltac']
        if self.mode=="sheth-torman":
            sigma = np.sqrt(sigma2)
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            return A*np.sqrt(2.*a/np.pi)*(1+((sigma2/a/deltac**2.)**p))*(deltac/sigma)*np.exp(-a*deltac**2./2./sigma2)
        elif self.mode=="tinker":
            nus = deltac/np.sqrt(sigma2)
            fnus = tinker.f_nu(nus,self.zs[:,None])
            return nus * fnus # note that f is actually nu*fnu !
        else:
            raise NotImplementedError
    
    def get_bh(self,ms=None,sigma2=None):
        if sigma2 is None: sigma2 = self.get_sigma2(ms)
        deltac = self.p['st_deltac']
        if self.mode=="sheth-torman":
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            return 1. + (1./deltac)*((a*deltac**2./sigma2)-1.) + (2.*p/deltac)/(1.+(a*deltac**2./sigma2)**p)
        elif self.mode=="tinker":
            nus = deltac/np.sqrt(sigma2)
            return tinker.bias(nus)
        else:
            raise NotImplementedError

    def concentration(self,ms,mode='duffy'):
        if mode=='duffy':
            # A = self.p['duffy_A_mean']
            # alpha = self.p['duffy_alpha_mean']
            # beta = self.p['duffy_beta_mean']
            A = self.p['duffy_A_vir']
            alpha = self.p['duffy_alpha_vir']
            beta = self.p['duffy_beta_vir']
            return duffy_concentration(ms[None,:],self.zs[:,None],A,alpha,beta,self.h)
        else:
            raise NotImplementedError

    def get_nzm(self,ms,sigma2=None):
        if sigma2 is None: sigma2 = self.get_sigma2(ms)
        ln_sigma_inv = -0.5*np.log(sigma2)
        fsigmaz = self.get_fsigmaz(ms,sigma2)
        dln_sigma_dlnm = np.gradient(ln_sigma_inv,np.log(ms),axis=-1)
        ms = ms[None,:]
        self.ms = ms
        return self.rho_matter_z(0) * fsigmaz * dln_sigma_dlnm / ms**2. 

    def a2z(self,a):
        return (1.0/a)-1.0
    
    def D_growth(self, a):
        self._amin = 0.001    # minimum scale factor
        self._amax = 1.0      # maximum scale factor
        self._na = 512        # number of points in interpolation arrays
        self.atab = np.linspace(self._amin,
                             self._amax,
                             self._na)
        ks = np.logspace(np.log10(1e-5),np.log10(1.),num=100) 
        zs = self.a2z(self.atab)
        deltakz = self.results.get_redshift_evolution(ks, zs, ['delta_cdm']) #index: k,z,0
        D_camb = deltakz[0,:,0]/deltakz[0,0,0]
        self._da_interp = interp1d(self.atab, D_camb, kind='linear')
        self._da_interp_type = "camb"
        return self._da_interp(a)/self._da_interp(1.0)
            
    def add_nfw_profile(self,name,ms,
                        nxs=tunable_params['nfw_integral_default_numxs'],
                        xmax=tunable_params['nfw_integral_default_xmax']):
        """
        xmax should be thought of in "concentration units", i.e.,
        for a cluster with concentration 3., xmax of 100 is probably overkill
        since the integrals are zero for x>3. However, since we are doing
        a single FFT over all profiles, we need to choose a single xmax.
        xmax of 100 is very safe for m~1e9 msolar, but xmax of 200-300
        will be needed down to m~1e2 msolar.
        nxs is the number of samples from 0 to xmax of rho_nfw(x). Might need
        to be increased from default if xmax is increased and/or going down
        to lower halo masses.
        xmax decides accuracy on large scales
        nxs decides accuracy on small scales
        
        """
        cs = self.concentration(ms)

        # sigma = np.sqrt(self.sigma2)
        # deltac = 1.686
        # nus = deltac/sigma
        # a = 1./(1.+self.zs)
        # growths = self.D_growth(a)/self.D_growth(1)
        # cs = growths[:,None]**1.15 * 9. * nus**(-0.29)


        
        rvirs = self.rvir(ms[None,:],self.zs[:,None])
        rss = (rvirs/cs)[...,None]
        # xs = np.linspace(0.,xmax,nxs+1)[1:]
        # rhoscale = 1 # makes rho off by norm, see below
        # rhos = rho_nfw_x(xs,rhoscale)[None,None] + cs[...,None]*0.
        # theta = np.ones(rhos.shape)
        # theta[np.abs(xs)>cs[...,None]] = 0 # CHECK
        # # m
        # integrand = theta * rhos * xs**2.
        # mnorm = np.trapz(integrand,xs) # mass but off by norm same as rho is off by
        # # u(kt)
        # integrand = rhos*theta
        # kts,ukts = fft_integral(xs,integrand)
        # uk = ukts/kts[None,None,:]/mnorm[...,None]
        # ks = kts/rss/(1+self.zs[:,None,None]) # divide k by (1+z) here for comoving FIXME: check this!
        # ukouts = np.zeros((uk.shape[0],uk.shape[1],self.ks.size))
        # # sadly at this point we must loop to interpolate :(
        # for i in range(uk.shape[0]):
        #     for j in range(uk.shape[1]):
        #         pks = ks[i,j]
        #         puks = uk[i,j]
        #         puks = puks[pks>0]
        #         pks = pks[pks>0]
        #         ukouts[i,j] = np.interp(self.ks,pks,puks,left=puks[0],right=0)
        #         #TODO: Add compulsory debug plot here
        # self.uk_profiles[name] = ukouts.copy()

        cs = cs[...,None]
        mc = np.log(1+cs)-cs/(1.+cs)
        x = self.ks[None,None]*rss *(1+self.zs[:,None,None])# !!!!
        Si, Ci = scipy.special.sici(x)
        Sic, Cic = scipy.special.sici((1.+cs)*x)
        ukouts = (np.sin(x)*(Sic-Si) - np.sin(cs*x)/((1+cs)*x) + np.cos(x)*(Cic-Ci))/mc
        self.uk_profiles[name] = ukouts.copy()
        
        return self.ks,ukouts # !!!
        

    def get_power_1halo_cross_galaxies(self,name="matter"):
        pass
    def get_power_2halo_cross_galaxies(self,name="matter"):
        pass

    def get_power_1halo_auto(self,name="matter"):
        ms = self.ms[...,None]
        integrand = self.nzm[...,None] * ms**2. * self.uk_profiles[name]**2. /self.rho_matter_z(0)**2.
        return np.trapz(integrand,ms,axis=-2)*(1.-np.exp(-(self.ks/default_params['kstar_damping'])**2.))
    
    def get_power_2halo_auto(self,name="matter"):
        ms = self.ms[...,None]
        integrand = self.nzm[...,None] * ms * self.uk_profiles[name] /self.rho_matter_z(0) * self.bh[...,None]
        integral = np.trapz(integrand,ms,axis=-2)
        # consistency relation : Correct for part that's missing from low-mass halos to get P(k->0) = Plinear
        consistency_integrand = self.nzm[...,None] * ms /self.rho_matter_z(0) * self.bh[...,None]
        consistency = np.trapz(consistency_integrand,ms,axis=-2)
        print("Two-halo consistency: " , consistency)
        return self.Pzk * (integral+1-consistency)**2. 
        
    
    def get_power_1halo_galaxy_auto(self):
        pass
    def get_power_2halo_galaxy_auto(self):
        pass

    def P_lin(self,ks,zs,knorm = 1e-4,kmax = 0.1):
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

        norm(z) is calculated as follows:
        norm(z) = P_linear_CAMB(k_norm,z) / T_CAMB(k_norm)**2
        """
        tk = self.Tk(ks,'eisenhu_osc') 
        assert knorm<kmax
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=False, 
                                                     hubble_units=False, k_hunit=False, kmax=kmax,
                                                     zmax=zs.max()+1.)
        pnorm = PK.P(zs, knorm,grid=True) #/ tnorm**2. / knorm**(self.params['ns'])
        tnorm = self.Tk(knorm,'eisenhu_osc') * knorm**(self.params['ns'])
        plin = (pnorm/tnorm) * tk**2. * ks**(self.params['ns'])
        return plin
        
        
    def Tk(self,ks,type ='eisenhu_osc'):

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

        Omega_m = self.om0
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
"""
Mass function
"""



"""
HOD
"""
    
def Mstellar_halo(z,log10mhalo):
    # Function to compute the stellar mass Mstellar from a halo mass mv at redshift z.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    log10mstar = np.linspace(-18,18,1000)
    mh = Mhalo_stellar(z,log10mstar)
    mstar = np.interp(log10mhalo,mh,log10mstar)
    return mstar


def Mhalo_stellar(z,log10mstellar):
    # Function to compute halo mass as a function of the stellar mass.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    a = 1./(1+z) 
    Mstar00=10.72 ; Mstara=0.55 ; M1=12.35 ; M1a=0.28
    beta0=0.44 ; beta_a=0.18 ; gamma0=1.56 ; gamma_a=2.51
    delta0=0.57 ; delta_a=0.17
    log10M1 = M1 + M1a*(a-1)
    log10Mstar0 = Mstar00 + Mstara*(a-1)
    beta = beta0 + beta_a*(a-1)
    gamma = gamma0 + gamma_a*(a-1)
    delta = delta0 + delta_a*(a-1)
    log10mstar = log10mstellar
    log10mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
    return log10mh


def avg_Nc(log10mhalo,z,log10mstellar_thresh):
    """<Nc(m)>"""
    sig_log_mstellar = 0.2
    log10mstar = Mstellar_halo(z,log10mhalo)
    num = log10mstellar_thresh - log10mstar
    denom = np.sqrt(2.) * sig_log_mstellar
    return 0.5*(1. - erf(num/denom))

def avg_Ns(log10mhalo,z,log10mstellar_thresh,Nc=None):
    Bsat=9.04
    betasat=0.74
    alphasat=1.
    Bcut=1.65
    betacut=0.59
    mthresh = Mhalo_stellar(z,log10mstellar_thresh)
    Msat=(10.**(12.))*Bsat*10**((mthresh-12)*betasat)
    Mcut=(10.**(12.))*Bcut*10**((mthresh-12)*betacut)
    Nc = avg_Nc(log10mhalo,z,log10mstellar_thresh,sig_log_mstellar=0.2) if Nc is None else Nc
    masses = 10**log10mhalo
    return Nc*((masses/Msat)**alphasat)*np.exp(-Mcut/(masses))    

"""
Profiles
"""

def rho_nfw_x(x,rhoscale):
    return rhoscale/x/(1.+x)**2.

def rho_nfw(r,rhoscale,rs):
    rrs = r/rs
    return rho_nfw_x(rrs,rhoscale)

# def rho_gas(r,R200,M200,z,omega_b,omega_m,rho_critical_z,)

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


