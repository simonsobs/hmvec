from hmvec import cosmology as hcos
import numpy as np
import matplotlib.pyplot as plt

bg = 2
ks = np.geomspace(1e-3,0.1,100) # Mpc^-1
z = 0.55
fnl = 20

h = hcos.Cosmology()
bgk = h.bias_fnl(bg,fnl,z,ks,deltac=1.42)

plt.plot(ks,bgk)
plt.axhline(y=bg,ls='--')
plt.xscale('log')
plt.xlabel(r'$k ({\rm Mpc}^{-1})$')
plt.ylabel('$b_g$')
plt.show()
