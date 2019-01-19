import hmvec as hm
import numpy as np
from orphics import io

def test_fft_integral():
    dx = 0.001
    xs = np.arange(dx,20.,dx)
    real = np.exp(-xs**2./2.)
    ks,uk = hm.fft_integral(xs,real)
    # pl = io.Plotter()
    # pl.add(xs,real)
    # pl.done()

    pl = io.Plotter(xyscale='loglin')
    pl.add(ks,uk)
    pl.add(ks,hm.analytic_fft_integral(ks),ls="--")
    pl.done()

    dr = 0.001
    rvir = 5.
    rmax = 10.*rvir
    r = np.arange(dr,rmax,dr)
    rs = 1
    rhos = 1e3
    rho = hm.rho_nfw(r,rhos,rs)

    # pl = io.Plotter(xyscale='loglog')
    # pl.add(r,rho)
    # pl.done()

    kmin = 1e-3
    kmax = 30
    ks = np.geomspace(kmin,kmax,2000)
    uk = hm.uk_brute_force(r,rho,rvir,ks)


    pl = io.Plotter(xyscale='loglog')
    pl.add(ks,uk)
    #for rmax in [10.,20.,50.,100.,1000.,1e4]:
    #for dr in [0.1,0.05,0.01,0.001]:
    for dr in [0.001]:
        rmax = 100
        fks,fuks = hm.uk_fft(lambda x: hm.rho_nfw(x,rhos,rs),rvir,dr=dr,rmax=rmax)
        pl.add(fks,fuks,ls="--")
    pl._ax.set_xlim(1e-3,20)
    pl._ax.set_ylim(1e-3,2)
    pl.done()

def test_cosmology():

    zs = np.linspace(0.5,3.,5)
    ms = np.geomspace(1e10,1e16,5000)

    # zs = zs[:,None]
    # ms = ms[None,:]
    # A = 2.1
    # sigma = ms**A+zs*0.
    # print(np.gradient(np.log(sigma),np.log(ms[0,:]),axis=-1))
    # sys.exit()


    ks = np.geomspace(1e-3,1,101)

    from enlib import bench
    with bench.show("init"):
        hcos = hm.HaloCosmology(zs,ks,ms=ms)


    print(hcos.nzm.shape,hcos.bh.shape)
    # pl = io.Plotter(xyscale='loglin')
    # for i in range(zs.size):
    #     pl.add(ms,hcos.bh[i,:],label="z=%.1f" % zs[i])
    # pl.done()

    # io.plot_img(hcos.nzm,flip=False,aspect='auto')

    bh = hcos.bh
    nzm = hcos.nzm


    
#test_fft_integral()
test_cosmology()
