import numpy as np
import hmvec as hm
import matplotlib.pyplot as plt

#Grid for Integration
Nz = 600                                 # num of redshifts
Nm = 500                                 # num of masses
Nk = 1000                                # num of wavenumbers
redshifts = np.linspace(0.01, 4.5, Nz)             # redshifts
masses = np.geomspace(1e10, 1e15, Nm)           # masses
ks = np.geomspace(1e-3, 100, Nk)               # wavenumbers
frequencies = np.array([545.])

#Initialize Halo Model
hcos = hm.HaloModel(redshifts, ks, ms=masses)

#Get 3D Power Spectra
Pjj_2h = hcos.get_power_2halo("cib", "cib", nu_obs=frequencies)  # P(z,k)

#Limber Integrals
Nl = 1000
ells = np.arange(1000, num=Nl)
Cii, Cii_integrand = hcos.C_ii(ells, redshifts, ks, Pjj_2h, dcdzflag=True)

#Plot Cii
plt.loglog(ells, Cii)
plt.xlabel(r'$\ell$')
plt.ylabel(rf'$C^{{ {frequencies[0]:0.0f} \;x\; {frequencies[0]:0.0f} }}_\ell$');
plt.savefig('cii.pdf', dpi=500, bbox_inches='tight')

#Plot dC/dz (z)
test_ells = np.array([100,300, 500, 1000])
plt.figure(figsize=(10,7))
for ell in test_ells:
    #Get index
    i = np.where(abs(ell - ells) <= 1)[0][0]

    #Spectra
    plt.semilogy(redshifts, Cii_integrand[:, i], label=rf"$\ell = {ells[i]:0.0f}$")

    #Gravy
    plt.xlabel(r'$z$')
    plt.ylabel(r'$dC_{II} / dz$')
    plt.title(rf'$\nu$ = {frequencies[0]}')
    plt.legend()
plt.savefig('dCdz_ii.png', dpi=500, bbox_inches='tight');

