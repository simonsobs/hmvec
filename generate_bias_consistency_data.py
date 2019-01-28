import matplotlib.pyplot as plt
import numpy as np
import tinker

numzs = 1000
zmax = 3.
lognumin = -8
lognumax = 2
numnus = 10000
filename = "alpha_consistency.txt"

delta = 200.

zs = np.linspace(0.,zmax,numzs)
nus = np.logspace(lognumin,lognumax,numnus)

fnus = tinker.f_nu(nus[None],zs[:,None],delta=delta,norm_consistency=False,alpha=1)


bs = tinker.bias(nus,delta=delta)
alphas = 1./np.trapz(fnus*bs,nus,axis=-1)
np.savetxt(filename,np.vstack((zs,alphas)).T,header="zs,alphas")
print("alpha paramater at z=0 ", alphas[0],
      " should be close to Tinker et. al. 2010 parameter of ",
      tinker.default_params['tinker_f_nu_alpha_z0_delta_200'])
print("alpha paramater at z= ",zs[-1]," is ", alphas[-1])

plt.plot(zs,alphas)
plt.xlabel("$z$")
plt.ylabel("$\\alpha$")
plt.savefig("alpha_consistency.png")

print("Saved to ",filename, ".")
