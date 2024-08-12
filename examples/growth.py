from hmvec import cosmology as hcosmo
from orphics import io,cosmology
import numpy as np


models = []
models.append(['EdS',hcosmo.get_eds_model()]) # CLASS can't do this without specifying YHe
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

pl = io.Plotter('rCL',xlabel='$z$',ylabel=r'$\Delta D(z) / D(z)$',xyscale='loglin')

for engine in ['camb','class']:

    for i,model in enumerate(models):
        params = model[1]
        h = hcosmo.Cosmology(params,accuracy='low',engine=engine)
        label = model[0]
        dapprox = h.D_growth(avals,exact=False)
        print(f'{label} \t\t {engine} \t {h.D_growth(1.0,exact=False):.2f}')
        dexact = h.D_growth(avals,exact=True)
        ddiff = (dapprox-dexact)/dexact
        pl.add(zs,ddiff,label=label if engine=='camb' else None,ls={'camb':'-','class':'--'}[engine],color=[f'C{x}' for x in range(len(models))][i])
        print(label)
pl.hline(y=0)
pl.legend('outside')
pl._ax.set_ylim(-0.25,0.6)
pl.done(f'growth.png')
