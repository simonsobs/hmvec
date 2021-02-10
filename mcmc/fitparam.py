import numpy as np
import hmvec as hm
import matplotlib
import matplotlib.pyplot as plt
from cobaya.run import run

#Cobaya Input File
info = {
    "theory": {"mcmc_class.PowerSpectrum": None},
    
    "likelihood": {"mcmc_class.ChiSqLikelihood": None},
    
    "params": {
        #CIB Model Parameters
#         "alpha": {
#             "prior": {"min": 0, "max": 1.3},
#             "ref": {"min": 0.2, "max": 0.5},
#             "latex": r"\alpha"},
#         "beta": {
#             "prior": {"min": 0, "max": 2.1},
#             "ref": {"min": 1.2, "max": 1.7},
#             "latex": r"\beta"},
#         "gamma": {
#             "prior": {"min": 0, "max": 2.7},
#             "ref": {"min": 1.2, "max": 1.7},
#             "latex": r"\gamma"},
#         "delta": {
#             "prior": {"min": 2.5, "max": 4.6},
#             "ref": {"min": 3, "max": 4},
#             "latex": r"\delta"},
#         "Td_o": {
#             "prior": {"min": 15, "max": 30},
#             "ref": {"min": 18, "max": 22},
#             "latex": r"T_{d,o}"},
#         "logM_eff": {
#             "prior": {"min": 11, "max": 14},
#             "ref": {"min": 11.8, "max": 13},
#             "latex": r"\text{log}(M_{\text{eff}})"},
        "L_o": {
            "prior": {"min": 1e-17, "max": 1e-13},
            "ref": {"min": 9e-16, "max": 9e-15},
            "latex": r"L_o"},
    },

    "sampler": {
        "mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}
    },

    "output": "toy/autocib"
}

#Run Cobaya
updated_info, sampler = run(info)