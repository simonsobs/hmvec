from hmvec import cosmology as hcosmo
from orphics import io,cosmology
import numpy as np


fb = 0.15
H0 = 70.0
om = 1.0
omb = fb*om
omc = (1-fb)*om
h0 = H0/100
omch2 = omc*h0**2
ombh2 = omb*h0**2

models = []
#models.append(['EdS',{'omch2':omch2,'ombh2':ombh2,'H0':H0,'mnu':0.}]) # CLASS can't do this without specifying YHe
models.append(['LCDM',{'H0':75.0,'mnu':0.}])
models.append(['wCDM Phantom',{'w0':-1.2,'mnu':0.}])
models.append(['nuCDM',{'mnu':0.5}])
models.append(['wCDM',{'w0':-0.8,'mnu':0.}])
models.append(['wwaCDM',{'w0':-1.2,'wa':0.5,'mnu':0.}])
models.append(['okCDM open',{'omk':0.1,'mnu':0.}])
# models.append(['okCDM closed',{'omk':-0.05,'mnu':0.}]) # CAMB can't do this

zs = np.geomspace(0.1,1000,100)
avals = 1./(1+zs)
Dmd = avals

for engine in ['camb','class']:
    pl = io.Plotter('rCL',xlabel='$z$',ylabel=r'$\Delta D(z) / D(z)$',xyscale='loglin')

    for model in models:
        params = model[1]
        h = hcosmo.Cosmology(params,accuracy='low',engine=engine)
        label = model[0]
        dapprox = h.D_growth(avals,exact=False)
        dexact = h.D_growth(avals,exact=True)
        ddiff = (dapprox-dexact)/dexact
        pl.add(zs,ddiff,label=label)
        print(label)
    pl.hline(y=0)
    pl._ax.set_ylim(-0.2,0.2)
    pl.done(f'growth_{engine}.png')

              
