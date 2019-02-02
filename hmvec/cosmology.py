
"""
This module abstracts away the choice of boltzmann codes.
However, it does it stupidly by simply providing a common
stunted interface. It makes no guarantee that the same set
of parameters passed to the two engines will produce the same
results. It could be a test-bed for converging towards that. 

"""

class Cosmology(object):

    def __init__(self,params,engine='camb'):
        assert engine in ['camb','class']

        pass


    def P_mm_linear(self,zs,ks):
        pass

    def P_mm_nonlinear(self,ks,zs,halofit_version='mead'):
        pass

    def comoving_radial_distance(self,z):
        pass

    def hubble(self,z):
        pass

    def _init_cosmology(self,params,halofit):
        try:
            theta = params['theta100']/100.
            H0 = None
            print("WARNING: Using theta100 parameterization. H0 ignored.")
        except:
            H0 = params['H0']
            theta = None
        
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
        self.chis = self.results.comoving_radial_distance(self.zs)
        self.Hzs = self.results.hubble_parameter(self.zs)
        self.Pzk = self._get_matter_power(self.zs,self.ks,nonlinear=False)
        if halofit is not None: self.nPzk = self._get_matter_power(self.zs,self.ks,nonlinear=True)

        
    def _get_matter_power(self,zs,ks,nonlinear=False):
        PK = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear, 
                                                hubble_units=False,
                                                k_hunit=False, kmax=ks.max(),
                                                zmax=zs.max()+1.)
        return PK.P(zs, ks, grid=True)

        
    def rho_matter_z(self,z):
        return self.rho_critical_z(0.) * self.om0 \
            * (1+np.atleast_1d(z))**3. # in msolar / megaparsec3
    def omz(self,z): return self.rho_matter_z(z)/self.rho_critical_z(z)
    
    def rho_critical_z(self,z):
        Hz = self.hubble(z) * 3.241e-20 # SI # FIXME: constants need checking
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
        return plin
        
        
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

    def lensing_window(self,zs=None,dndz=None):
        """
        Generates a lensing convergence window 
        W(z).

        zs: If (nz,) with nz>2 and dndz is not None, then these are the points
        at which dndz is defined. If nz=2 and no dndz is provided, it is (zmin,zmax)
        for a top-hat window. If a single number, and no dndz is provided,
        it is a delta function source at zs.
        """

        pass



    
    def C_kg(self,ells,ks,Pgm,gzs,gdndz=None,lzs=None,ldndz=None,lwindow=None):
        pass

    def C_gg(self,ells,ks,Pgg,gzs,dndz=None):
        pass

    def C_kk(self,ells,ks,Pmm,lzs=None,ldndz=None,lwindow=None):
        pass
