import numpy as np

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
    'maccio_HI_cHI0': 28.65,
    'maccio_HI_gamma': 1.45,
    'nfw_integral_numxs':40000, # not sufficient
    'nfw_integral_xmax':200,
    'electron_density_profile_integral_numxs':5000,
    'electron_density_profile_integral_xmax':20,
    'electron_pressure_profile_integral_numxs':5000,
    'electron_pressure_profile_integral_xmax':20,
    'battaglia_gas_gamma':-0.2,
    'battaglia_gas_family': 'AGN',
    'HI_integral_numxs': 40000,
    'HI_integral_xmax': 200,

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

    # General HOD
    'hod_bisection_search_min_log10mthresh': 7.,
    'hod_bisection_search_max_log10mthresh': 14.,
    'hod_bisection_search_rtol': 1e-4,
    'hod_bisection_search_warn_iter': 20,

    # Leauthaud12 HOD
    # See column 2 of SIG_MOD1 section of Table 5 of 1104.0928
    'hod_Leau12_A_log10mthresh': 1.0, # This parameter is used to vary log10mthresh
    'hod_Leau12_sig_log_mstellar': 0.2,
    'hod_Leau12_alphasat': 1.0,
    'hod_Leau12_Bsat':9.04,
    'hod_Leau12_betasat':0.74,
    'hod_Leau12_Bcut':1.65,
    'hod_Leau12_betacut':0.59,
    'hod_Leau12_log10mstellar_thresh': 10.5,

    # HMQ ELG Alam2020 HOD
    # See Table 1 of 1910.05095
    # Paper uses h^-1 Msun, so we convert mass parameters to Msun
    'hod_Alam20_HMQ_ELG_log10Mc': 11.75 - np.log10(0.67),
    'hod_Alam20_HMQ_ELG_sigmaM': 0.58,
    'hod_Alam20_HMQ_ELG_gamma': 4.12,
    'hod_Alam20_HMQ_ELG_Q': 100,
    'hod_Alam20_HMQ_ELG_log10M1': 13.53 - np.log10(0.67),
    'hod_Alam20_HMQ_ELG_kappa': 1.0,
    'hod_Alam20_HMQ_ELG_alpha': 1.0,
    'hod_Alam20_HMQ_ELG_pmax': 0.33,

    # LRG Alam2020 HOD
    # See Table 1 of 1910.05095
    # Paper uses h^-1 Msun, so we convert mass parameters to Msun
    'hod_Alam20_LRG_log10Mc': 13.0 - np.log10(0.67),
    'hod_Alam20_LRG_sigmaM': 0.60,
    'hod_Alam20_LRG_log10M1': 14.24 - np.log10(0.67),
    'hod_Alam20_LRG_kappa': 0.98,
    'hod_Alam20_LRG_alpha': 0.4,
    'hod_Alam20_LRG_pmax': 0.33,

    # Erf-based ELG Alam2020 HOD
    # See Table 1 of 1910.05095
    # Paper uses h^-1 Msun, so we convert mass parameters to Msun
    'hod_Alam20_ELG_log10Mc': 11.88 - np.log10(0.67),
    'hod_Alam20_ELG_sigmaM': 0.56,
    'hod_Alam20_ELG_log10M1': 13.94 - np.log10(0.67),
    'hod_Alam20_ELG_kappa': 1.0,
    'hod_Alam20_ELG_alpha': 0.4,
    'hod_Alam20_ELG_pmax': 0.33,

    # QSO Alam2020 HOD
    # See Table 1 of 1910.05095
    # Paper uses h^-1 Msun, so we convert mass parameters to Msun
    'hod_Alam20_QSO_log10Mc': 12.21 - np.log10(0.67),
    'hod_Alam20_QSO_sigmaM': 0.60,
    'hod_Alam20_QSO_log10M1': 14.09 - np.log10(0.67),
    'hod_Alam20_QSO_kappa': 1.0,
    'hod_Alam20_QSO_alpha': 0.39,
    'hod_Alam20_QSO_pmax': 0.033,

    # LRG Yuan2022 HOD
    # See caption of Figure 2 of 2202.12911
    # Paper uses h^-1 Msun, so we convert mass parameters to Msun
    'hod_Alam20_Yuan22LRG_log10Mc': 12.7 - np.log10(0.67),
    'hod_Alam20_Yuan22LRG_sigmaM': 0.2 / np.log10(np.e),
    'hod_Alam20_Yuan22LRG_log10M1': 13.6 - np.log10(0.67),
    'hod_Alam20_Yuan22LRG_kappa': 0.08,
    'hod_Alam20_Yuan22LRG_alpha': 1.15,
    'hod_Alam20_Yuan22LRG_pmax': 0.8,

    # Zheng2005 HOD
    # See Sec. 2.3 of 2201.05076
    # Original reference: astro-ph/0408564
    'hod_Zheng05_log10Mth': 12.712,
    'hod_Zheng05_sigmalogM': 0.287,
    'hod_Zheng05_log10Mcut': 12.95,
    'hod_Zheng05_log10M1': 13.62,
    'hod_Zheng05_beta': 13.62 - 12.95,
    'hod_Zheng05_alpha': 0.98,

    # Walsh2019 HOD
    # See "w_p & P_0" column of Table 1 of 1905.07024
    # Mass units assumed to be Msun
    'hod_Walsh19_log10Mth': 13.18,
    'hod_Walsh19_sigmalogM': 0.55,
    'hod_Walsh19_log10Mcut': 4.87,
    'hod_Walsh19_log10M1': 14.28,
    'hod_Walsh19_beta': 14.28 - 13.18,
    'hod_Walsh19_alpha': 1.12,

    # Cochrane2017 HOD
    # See z=1.45 row of Table 5 of 1704.05472
    'hod_Cochrane17_log10Mc': 11.45,
    'hod_Cochrane17_sigmalogM': 0.6,
    'hod_Cochrane17_FcA': 0.7,
    'hod_Cochrane17_FcB': 0.7,
    'hod_Cochrane17_Fs': 0.005,
    'hod_Cochrane17_alpha': 1.0,

    # Hadzhiyska2020 HOD
    # These parameters correspond to an ad hoc fitting function matched to the upper
    # left panel of Figure 4 of 2011.05331
    'hod_Hadzhiyska20_log10Mc': 12.0, # 11.8 - np.log10(0.6774),
    'hod_Hadzhiyska20_A': 0.1,
    'hod_Hadzhiyska20_deltalogM': 0.15,
    'hod_Hadzhiyska20_sigmalogM': 0.5,
    'hod_Hadzhiyska20_log10M1': 12.5, # 12.3 - np.log10(0.6774),
    'hod_Hadzhiyska20_alpha1': 3.5,
    'hod_Hadzhiyska20_alpha2': 0.6,

    # Zhai2021 HOD
    # These parameters correspond to an ad hoc fitting function matched to the red line
    # for 1.0 < z < 1.1 in Figure 8 of 2103.11063
    'hod_Zhai21_log10Mc': 11.9, # 11.7 - np.log10(0.678),
    'hod_Zhai21_A1': 0.2,
    'hod_Zhai21_A2': 0.01,
    'hod_Zhai21_sigmalogM': 0.2,
    'hod_Zhai21_log10M1': 12.3, # 12.1 - np.log10(0.678),
    'hod_Zhai21_alpha1': 2.5,
    'hod_Zhai21_alpha2': 0.5,

    # Harikane17 HOD
    # See the z=3.8, m_UV^th=24.5 row of Table 4 of 1704.06535
    'hod_Harikane17_log10Mmin': 12.22,
    'hod_Harikane17_sigmalogM': 0.2,
    'hod_Harikane17_log10Msat': 14.23,
    'hod_Harikane17_beta': 14.23 - 12.22,
    'hod_Harikane17_alpha': 1.0,

    # Padmanabhan17 HI mass-halo mass relation
    'HIhod_Padmanabhan17_alpha': 0.09,
    'HIhod_Padmanabhan17_log10vc0': 1.56,
    'HIhod_Padmanabhan17_beta': -0.58,
    'HIhod_Padmanabhan17_gamma': 1.45,
}
