
battaglia_defaults = {}
battaglia_defaults['AGN'] = {
    'rho0_A0':4000.,
    'rho0_alpham':0.29,
    'rho0_alphaz':-0.66,
    'alpha_A0':0.88,
    'alpha_alpham':-0.03,
    'alpha_alphaz':0.19,
    'beta_A0':3.83,
    'beta_alpham':0.04,
    'beta_alphaz':-0.025
}
battaglia_defaults['SH'] = {
    'rho0_A0':19000.,
    'rho0_alpham':0.09,
    'rho0_alphaz':-0.95,
    'alpha_A0':0.70,
    'alpha_alpham':-0.017,
    'alpha_alphaz':0.27,
    'beta_A0':4.43,
    'beta_alpham':0.005,
    'beta_alphaz':0.037
}
    

battaglia_defaults['pres'] = {
    'P0_A0':18.1,
    'P0_alpham':0.154,
    'P0_alphaz':-0.758,
    'xc_A0':0.497,
    'xc_alpham':-0.00865,
    'xc_alphaz':0.731,
    'beta_A0':4.35,
    'beta_alpham':0.0393,
    'beta_alphaz':0.415
}
    

default_params = {
    
    # Mass function
    'st_A': 0.3222,
    'st_a': 0.707,
    'st_p': 0.3,
    'st_deltac': 1.686,
    'sigma2_kmin':1e-4,
    'sigma2_kmax':2000,
    'sigma2_numks':10000,
    'Wkr_taylor_switch':0.01,

    # Profiles
    'duffy_A_vir':7.85, # for Mvir
    'duffy_alpha_vir':-0.081,
    'duffy_beta_vir':-0.71,
    'duffy_A_mean':10.14, # for M200rhomeanz
    'duffy_alpha_mean':-0.081,
    'duffy_beta_mean':-1.01,
    'nfw_integral_numxs':40000, # not sufficient
    'nfw_integral_xmax':200,
    'electron_density_profile_integral_numxs':5000,
    'electron_density_profile_integral_xmax':20,
    'electron_pressure_profile_integral_numxs':5000,
    'electron_pressure_profile_integral_xmax':20,
    'battaglia_gas_gamma':-0.2,
    'battaglia_gas_family': 'AGN',

    'battaglia_pres_gamma' : -0.3,
    'battaglia_pres_alpha' : 1.,
    'battaglia_pres_family' : 'pres',
    # Power spectra
    'kstar_damping':0.01,
    'default_halofit':'mead',
    
    # Cosmology
    'omch2': 0.1198,
    'ombh2': 0.02225,
    'H0': 67.3,
    'ns': 0.9645,
    'As': 2.2e-9,
    'mnu': 0.0, # NOTE NO NEUTRINOS IN DEFAULT
    'omk': 0.0,
    'pivot_scalar': 0.05,
    'w0': -1.0,
    'tau':0.06,
    'nnu':3.046,
    'wa': 0.,
    'num_massive_neutrinos':3,
    'T_CMB':2.7255e6,
    'parsec': 3.08567758e16,
    'mSun': 1.989e30,
    'thompson_SI': 6.6524e-29,
    'meterToMegaparsec': 3.241e-23,
    'Yp': 0.24,

    # HOD
    'hod_A_log10mthresh': 1.0, # This parameter is used to vary log10mthresh
    'hod_sig_log_mstellar': 0.2,
    'hod_alphasat': 1.0,
    'hod_Bsat':9.04,
    'hod_betasat':0.74,
    'hod_Bcut':1.65,
    'hod_betacut':0.59,
    'hod_bisection_search_min_log10mthresh': 7.,
    'hod_bisection_search_max_log10mthresh': 14.,
    'hod_bisection_search_rtol': 1e-4,
    'hod_bisection_search_warn_iter': 20,

    # CLASS
    'class_output': ''
    

}
