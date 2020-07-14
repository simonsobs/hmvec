import numpy as np
import hmvec as hm
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
cosmo = Planck15
# import time

#Setup Grid
Nz = 210                                 # num of redshifts
Nm = 91                                 # num of masses
Nk = 1000                                # num of wavenumbers
redshifts = np.linspace(0.012, 10.22, Nz)             # redshifts
logmass = np.arange(6,15.005,0.1)
masses = 10**logmass           # masses
ells = np.linspace(150., 2000., 20)
ks = np.array([])
chis = cosmo.comoving_distance(redshifts).value
for ell in ells:
    ks = np.append(ks, ell/chis) 
ks = np.sort(ks)
# ks = np.geomspace(0.0155, 37, Nk)               # wavenumbers
# Nl = 20
# ells = np.arange(150, 2000, Nl)
frequencies = np.array([[143],[217],[353],[545],[857],[3000]], dtype=np.double)     #Ghz

#Initialize Halo Model 
hcos = hm.HaloModel(redshifts, ks, ms=masses)

#Set CIB Parameters
hcos.set_cibParams('planck')

#Testing
hcos.get_power_2halo('cib', nu_obs=[545])

#Autocorrelation Spectra
for freqpair in frequencies:
    freqpair = np.array([freqpair])
    #Get 3D Power Spectra
    Pjj_tot = hcos.get_power("cib", "cib", nu_obs=freqpair)  # P(z,k)
    Pjj_1h = hcos.get_power_1halo("cib", "cib", nu_obs=freqpair)  # P(z,k)
    Pjj_2h = hcos.get_power_2halo("cib", "cib", nu_obs=freqpair)  # P(z,k)
    Pjj_cen = hcos.get_power_2halo("cib", "cib", nu_obs=freqpair, subhalos=False)  # P(z,k)

    #Limber Integrals
    C_tot, dcdz_tot = hcos.C_ii(ells, redshifts, ks, Pjj_tot, dcdzflag=True)
    C_1h, dcdz_1h = hcos.C_ii(ells, redshifts, ks, Pjj_1h, dcdzflag=True)
    C_2h, dcdz_2h = hcos.C_ii(ells, redshifts, ks, Pjj_2h, dcdzflag=True)
    C_cen, dcdz_cen = hcos.C_ii(ells, redshifts, ks, Pjj_cen, dcdzflag=True)

    #Save data
    np.save('pjj_tot_' + str(freqpair[0]) + 'X' + str(freqpair[0]), Pjj_tot)
    np.save('pjj_1h_' + str(freqpair[0]) + 'X' + str(freqpair[0]), Pjj_1h)
    np.save('pjj_2h_' + str(freqpair[0]) + 'X' + str(freqpair[0]), Pjj_2h)
    np.save('pjj_cen_' + str(freqpair[0]) + 'X' + str(freqpair[0]), Pjj_cen)
    np.save('C_tot_' + str(freqpair[0]) + 'X' + str(freqpair[0]), C_tot)
    np.save('C_1h_' + str(freqpair[0]) + 'X' + str(freqpair[0]), C_1h)
    np.save('C_2h_' + str(freqpair[0]) + 'X' + str(freqpair[0]), C_2h)
    np.save('C_cen_' + str(freqpair[0]) + 'X' + str(freqpair[0]), C_cen)

"""
Plots for a Single Frequency
"""

# #Plot Total C
# plt.loglog(ells[4:], C_tot[4:], label='total')
# plt.xlabel(r'$\ell$')
# plt.ylabel(rf'$C^{{ {frequencies[0,0]:0.0f} \;x\; {frequencies[0,0]:0.0f} }}_\ell$');
# plt.savefig('cii_tot.pdf', dpi=500, bbox_inches='tight')

# #Plot 1h and 2h C's
# plt.loglog(ells[4:], C_1h[4:], label='1 halo term')
# plt.loglog(ells[4:], C_2h[4:], label='2 halo term')
# plt.xlabel(r'$\ell$')
# plt.ylabel(rf'$C^{{ {frequencies[0,0]:0.0f} \;x\; {frequencies[0,0]:0.0f} }}_\ell$');
# plt.legend()
# plt.savefig('cii_1h2h.pdf', dpi=500, bbox_inches='tight')

# #Plot C without Satellites
# plt.clf()
# plt.loglog(ells[4:], C_cen[4:])
# plt.xlabel(r'$\ell$')
# plt.ylabel(rf'$C^{{ {frequencies[0,0]:0.0f} \;x\; {frequencies[0,0]:0.0f} }}_\ell$')
# plt.savefig('cii_tot_cen.pdf', dpi=500, bbox_inches='tight')

# #Plot dC/dz With Satellites
# test_ells = np.array([100, 300, 500, 1000])
# plt.figure(figsize=(10,7))
# for ell in test_ells:
#     #Get index
#     i = np.where(abs(ell - ells) <= 1)[0][0]

#     #Spectra
#     plt.semilogy(redshifts, dcdz_tot[:, i], label=rf"$\ell = {ells[i]}$, with satellites")

#     #Gravy
#     plt.xlabel(r'$z$')
#     plt.ylabel(r'$dC_{II} / dz$')
#     plt.title(rf'$\nu$ = {frequencies[0,0]}')
#     plt.legend()
# plt.savefig('dCdz_ii_tot.pdf', dpi=500, bbox_inches='tight');

# #Plot dC/dz Without Satellites
# test_ells = np.array([100, 300, 500, 1000])
# plt.figure(figsize=(10,7))
# for ell in test_ells:
#     #Get index
#     i = np.where(abs(ell - ells) <= 1)[0][0]

#     #Spectra
#     plt.semilogy(redshifts, dcdz_cen[:, i], label=rf"$\ell = {ells[i]}$, without satellites")

#     #Gravy
#     plt.xlabel(r'$z$')
#     plt.ylabel(r'$dC_{II} / dz$')
#     plt.title(rf'$\nu$ = {frequencies[0,0]}')
#     plt.legend()
# plt.savefig('dCdz_ii_cen.pdf', dpi=500, bbox_inches='tight');