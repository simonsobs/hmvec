import numpy as np
import hmvec as hm
import matplotlib.pyplot as plt
# import time

#Grid for Integration
Nz = 600                                 # num of redshifts
Nm = 500                                 # num of masses
Nk = 1000                                # num of wavenumbers
redshifts = np.linspace(0.01, 4.5, Nz)             # redshifts
masses = np.geomspace(1e10, 1e15, Nm)           # masses
ks = np.geomspace(1e-3, 100, Nk)               # wavenumbers
frequencies = np.array([[545.]])                  #Ghz

#Initialize Halo Model 
hcos = hm.HaloModel(redshifts, ks, ms=masses)

#Set CIB Parameters
hcos.set_cibParams('planck')

# #Testing
# hcos.testingCIB()

#Get 3D Power Spectra
Pjj_tot = hcos.get_power_1halo("cib", "cib", nu_obs=frequencies)  # P(z,k)
Pjj_1h = hcos.get_power_1halo("cib", "cib", nu_obs=frequencies)  # P(z,k)
Pjj_2h = hcos.get_power_2halo("cib", "cib", nu_obs=frequencies)  # P(z,k)
Pjj_cen = hcos.get_power_2halo("cib", "cib", nu_obs=frequencies, subhalos=False)  # P(z,k)

#Limber Integrals
Nl = 1000
ells = np.arange(Nl)
C_tot, dcdz_tot = hcos.C_ii(ells, redshifts, ks, Pjj_tot, dcdzflag=True)
C_1h, dcdz_1h = hcos.C_ii(ells, redshifts, ks, Pjj_1h, dcdzflag=True)
C_2h, dcdz_2h = hcos.C_ii(ells, redshifts, ks, Pjj_2h, dcdzflag=True)
C_cen, dcdz_cen = hcos.C_ii(ells, redshifts, ks, Pjj_cen, dcdzflag=True)


#Plot Total C's
plt.loglog(ells, C_tot, label='total')
plt.loglog(ells, C_1h, label='1 halo term')
plt.loglog(ells, C_2h, label='2 halo term')
plt.xlabel(r'$\ell$')
plt.ylabel(rf'$C^{{ {frequencies[0,0]:0.0f} \;x\; {frequencies[0,0]:0.0f} }}_\ell$');
plt.savefig('cii_tot.pdf', dpi=500, bbox_inches='tight')

#Plot Centrals' C's
plt.clf()
plt.loglog(ells, C_cen, label='total')
plt.xlabel(r'$\ell$')
plt.ylabel(rf'$C^{{ {frequencies[0,0]:0.0f} \;x\; {frequencies[0,0]:0.0f} }}_\ell$');
plt.savefig('cii_tot_cen.pdf', dpi=500, bbox_inches='tight')

#Plot dC/dz (z)
test_ells = np.array([100, 300, 500, 1000])
plt.figure(figsize=(10,7))
for ell in test_ells:
    #Get index
    i = np.where(abs(ell - ells) <= 1)[0][0]

    #Spectra
    plt.semilogy(redshifts, dcdz_tot[:, i], label=rf"$\ell = {ells[i]}$, with satellites")
    plt.semilogy(redshifts, dcdz_cen[:, i], label=rf"$\ell = {ells[i]}$, without satellites")

    #Gravy
    plt.xlabel(r'$z$')
    plt.ylabel(r'$dC_{II} / dz$')
    plt.title(rf'$\nu$ = {frequencies[0,0]}')
    plt.legend()
plt.savefig('dCdz_ii_tot.pdf', dpi=500, bbox_inches='tight');

