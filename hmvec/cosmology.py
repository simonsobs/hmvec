import numpy as np
from scipy.interpolate import interp2d,interp1d
from .params import default_params
import camb
from camb import model
import scipy.interpolate as si
import scipy.constants as constants

"""
This module will (eventually) abstract away the choice of boltzmann codes.
However, it does it stupidly by simply providing a common
stunted interface. It makes no guarantee that the same set
of parameters passed to the two engines will produce the same
results. It could be a test-bed for converging towards that. 

"""

class Cosmology(object):

    def __init__(self,params=None,halofit=None,engine='camb'):
        assert engine in ['camb','class']
        if engine=='class': raise NotImplementedError
        
        self.p = dict(params) if params is not None else {}
        for param in default_params.keys():
            if param not in self.p.keys(): self.p[param] = default_params[param]
        
        # Cosmology
        self._init_cosmology(self.p,halofit)


    def sigma_crit(self,zlens,zsource):
        Gval = 4.517e-48 # Newton G in Mpc,seconds,Msun units
        cval = 9.716e-15 # speed of light in Mpc,second units
        Dd = self.angular_diameter_distance(zlens)
        Ds = self.angular_diameter_distance(zsource)
        Dds = np.asarray([self.results.angular_diameter_distance2(zl,zsource) for zl in zlens])
        return cval**2 * Ds / 4 / np.pi / Gval / Dd / Dds
        

    def P_mm_linear(self,zs,ks):
        pass

    def P_mm_nonlinear(self,ks,zs,halofit_version='mead'):
        pass

    def comoving_radial_distance(self,z):
        return self.results.comoving_radial_distance(z)

    def angular_diameter_distance(self,z):
        return self.results.angular_diameter_distance(z)

    def hubble_parameter(self,z):
        # H(z) in km/s/Mpc
        return self.results.hubble_parameter(z)
    
    def h_of_z(self,z):
        # H(z) in 1/Mpc
        return self.results.h_of_z(z)

    def _init_cosmology(self,params,halofit):
        try:
            theta = params['theta100']/100.
            H0 = None
            print("WARNING: Using theta100 parameterization. H0 ignored.")
        except:
            H0 = params['H0']
            theta = None
        try:        
            omm = params['omm']        
            h = params['H0']/100.      
            params['omch2'] = omm*h**2-params['ombh2']     
            print("WARNING: omm specified. Ignoring omch2.")       
        except:        
            pass        
        self.pars = camb.set_params(ns=params['ns'],As=params['As'],H0=H0,
                                    cosmomc_theta=theta,ombh2=params['ombh2'],
                                    omch2=params['omch2'], mnu=params['mnu'],
                                    tau=params['tau'],nnu=params['nnu'],
                                    num_massive_neutrinos=
                                    params['num_massive_neutrinos'],
                                    w=params['w0'],wa=params['wa'],
                                    dark_energy_model='ppf',
                                    halofit_version=self.p['default_halofit'] if halofit is None else halofit,
                                    AccuracyBoost=2)
        self.results = camb.get_background(self.pars)
        self.params = params
        self.h = self.params['H0']/100.
        omh2 = self.params['omch2']+self.params['ombh2'] # FIXME: neutrinos
        self.om0 = omh2 / (self.params['H0']/100.)**2.        
        try: self.as8 = self.params['as8']        
        except: self.as8 = 1
        
    def _get_matter_power(self,zs,ks,nonlinear=False):
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear, 
                                                hubble_units=False,
                                                k_hunit=False, kmax=ks.max(),
                                                zmax=zs.max()+1.)
        return (self.as8**2.) * PK.P(zs, ks, grid=True)

        
    def rho_matter_z(self,z):
        return self.rho_critical_z(0.) * self.om0 \
            * (1+np.atleast_1d(z))**3. # in msolar / megaparsec3

    def omz(self,z): 
        return self.rho_matter_z(z)/self.rho_critical_z(z)
    
    def rho_critical_z(self,z):
        Hz = self.hubble_parameter(z) * 3.241e-20 # SI # FIXME: constants need checking
        G = 6.67259e-11 # SI
        rho = 3.*(Hz**2.)/8./np.pi/G # SI
        return rho * 1.477543e37 # in msolar / megaparsec3
    
    def D_growth(self, a):
        # From Moritz Munchmeyer?
        
        _amin = 0.001    # minimum scale factor
        _amax = 1.0      # maximum scale factor
        _na = 512        # number of points in interpolation arrays
        atab = np.linspace(_amin,
                           _amax,
                           _na)
        ks = np.logspace(np.log10(1e-5),np.log10(1.),num=100) 
        zs = a2z(atab)
        deltakz = self.results.get_redshift_evolution(ks, zs, ['delta_cdm']) #index: k,z,0
        D_camb = deltakz[0,:,0]/deltakz[0,0,0]
        _da_interp = interp1d(atab, D_camb, kind='linear')
        _da_interp_type = "camb"
        return _da_interp(a)/_da_interp(1.0)

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
        """
        tk = self.Tk(ks,'eisenhu_osc') 
        assert knorm<kmax
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=False, 
                                                     hubble_units=False, k_hunit=False, kmax=kmax,
                                                     zmax=zs.max()+1.)
        pnorm = PK.P(zs, knorm,grid=True)
        tnorm = self.Tk(knorm,'eisenhu_osc') * knorm**(self.params['ns'])
        plin = (pnorm/tnorm) * tk**2. * ks**(self.params['ns'])
        return (self.as8**2.) *plin
 
    def P_lin_slow(self,ks,zs,kmax = 0.1):
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=False, 
                                                     hubble_units=False, k_hunit=False, kmax=kmax,
                                                     zmax=zs.max()+1.)
        plin = PK.P(zs, ks,grid=True)
        return (self.as8**2.) * plin

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
            
        return 1.5*self.om0*H0**2.*(1.+ezs)*chis/H * integral

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

    def total_matter_power_spectrum(self,Pnn,Pne,Pee):
        omtoth2 = self.p['omch2'] + self.p['ombh2']
        fc = self.p['omch2']/omtoth2
        fb = self.p['ombh2']/omtoth2
        return fc**2.*Pnn + 2.*fc*fb*Pne + fb*fb*Pee

    def total_matter_galaxy_power_spectrum(self,Pgn,Pge):
        omtoth2 = self.p['omch2'] + self.p['ombh2']
        fc = self.p['omch2']/omtoth2
        fb = self.p['ombh2']/omtoth2
        return fc*Pgn + fb*Pge



def a2z(a): return (1.0/a)-1.0

def limber_integral(ells,zs,ks,Pzks,gzs,Wz1s,Wz2s,hzs,chis):
    """
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
    
