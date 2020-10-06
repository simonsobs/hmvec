import numpy as np
import hmvec as hm
import matplotlib.pyplot as plt
import matplotlib

#Plot settings
matplotlib.rcParams['axes.labelsize'] = 'xx-large'
matplotlib.rcParams['xtick.labelsize'] = 'x-large'
matplotlib.rcParams['ytick.labelsize'] = 'x-large'
matplotlib.rcParams['legend.fontsize'] = 'x-large'
matplotlib.rcParams['axes.titlesize'] = 'xx-large'

#Setup Grid
Nz = 200                                 # num of redshifts
Nm = 200                                 # num of masses
Nk = 1000                                # num of wavenumbers
redshifts = np.linspace(0.01, 10, Nz)             
masses = np.geomspace(1.0e6, 1.0e15, Nm)          
ks = np.geomspace(1.0e-3, 100.0, Nk)              # wavenumbers
ells = np.linspace(10, 2000, 200)

#Initialize Halo Model 
hcos = hm.HaloModel(redshifts, ks, ms=masses)

#Frequencies
freqarray = np.array([545e9], dtype=np.double)  

#Cobaya Input File
info = {
    "likelihood": {
        "gaussian_mixture": {
            "means": [0.2, 0],
            "covs": [[0.1, 0.05],
                     [0.05, 0.2]],
            "derived": True}},
    
    "params": dict([
        ("alpha", {
            "prior": {"min": 0, "max": 1.3},
            "ref": {"dist": "norm", "loc": 0.36, "scale": 0.05},
            "latex": r"\alpha"}),
        ("beta", {
            "prior": {"min": 0, "max": 2.5},
            "ref": {"dist": "norm", "loc": 1.75, "scale": 0.06},
            "latex": r"\beta"}),
        ("gamma", {
            "prior": {"min": 0, "max": 2.5},
            "ref": {"dist": "norm", "loc": 1.7, "scale": 0.2},
            "latex": r"\gamma"}),
        ("delta", {
            "prior": {"min": 2.5, "max": 4.5},
            "ref": {"dist": "norm", "loc": 3.6, "scale": 0.2},
            "latex": r"\delta"}),
        ("b", {
            "prior": {"min": 0, "max": 2.5},
            "ref": {"dist": "norm", "loc": 1.75, "scale": 0.06},
            "latex": r"\beta"}),
        ("b", {
            "prior": {"min": 0, "max": 2.5},
            "ref": {"dist": "norm", "loc": 1.75, "scale": 0.06},
            "latex": r"\beta"}),
        ("b", {
            "prior": {"min": 0, "max": 2.5},
            "ref": {"dist": "norm", "loc": 1.75, "scale": 0.06},
            "latex": r"\beta"})
        ]),
    "sampler": {
        "mcmc": None}}