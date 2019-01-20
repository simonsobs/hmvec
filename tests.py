import hmvec as hm
import numpy as np
from orphics import io
from enlib import bench

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

    zs = np.linspace(0.1,2.,2)
    ms = np.geomspace(2e9,1e13,10)

    # zs = zs[:,None]
    # ms = ms[None,:]
    # A = 2.1
    # sigma = ms**A+zs*0.
    # print(np.gradient(np.log(sigma),np.log(ms[0,:]),axis=-1))
    # sys.exit()


    ks = np.geomspace(1e-4,10,1001)

    from enlib import bench
    with bench.show("init"):
        hcos = hm.HaloCosmology(zs,ks,ms=ms)

    hcos.add_nfw_profile("matter",ms,dr=0.01)


def test_massfn():

    zs = np.linspace(0.1,2.,40)
    ms = np.geomspace(2e14,1e16,500)

    ks = np.geomspace(1e-3,1,101)

    from enlib import bench
    with bench.show("init"):
        hcos = hm.HaloCosmology(zs,ks,ms=ms)

    print(hcos.nzm.shape,hcos.bh.shape)
    bh = hcos.bh
    nzm = hcos.nzm

    # pl = io.Plotter(xyscale='loglog')
    # pl.add(ms,nzm[10,:])
    # pl.done()

    fsky = 0.4
    cSpeedKmPerSec = 299792.458
    nz = np.trapz(nzm,ms,axis=-1)*4.*np.pi*hcos.chis**2./hcos.Hzs*fsky * cSpeedKmPerSec 
    # print(nz.shape)
    # pl = io.Plotter()
    # pl.add(zs,nz)
    # pl.done()
    n = np.trapz(nz,zs)
    print(n)

def test_fft_transform():

    zs = np.linspace(0.1,4.,4)
    ms = np.geomspace(2e9,1e17,5)
    ks = np.geomspace(1e-4,100,1001)
    hcos = hm.HaloCosmology(zs,ks,ms=ms)
    with bench.show("nfw"):
        _,ouks = hcos.add_nfw_profile("matter",ms)
    cs = hcos.concentration(ms)
    rvirs = hcos.rvir(ms[None,:],hcos.zs[:,None])
    rscales = (rvirs/cs)[...,None]
    rhoscale = 1 # we do not care about normalization of profile
    rmax = 10*rvirs.max()
    dr = 0.001
    iks,iuk = hm.uk_fft(lambda rs: hm.rho_nfw(rs,rhoscale,rscales),rvirs,dr=dr,rmax=rmax)
    k = 0
    for i in range(iuk.shape[0]):
        for j in range(iuk.shape[1]):
            k += 1
            pl = io.Plotter(xyscale='loglog')
            pl.add(iks/(1.+hcos.zs[i]),iuk[i,j])
            pl.add(hcos.ks,ouks[i,j],ls='--')
            pl.hline(y=1)
            pl._ax.set_xlim(ks.min(),ks.max())
            pl.done("profile_test_%d.png" % k)
    print(iuk.shape)

def test_pmm():

    zs = np.linspace(0.1,3.,4)
    ms = np.geomspace(2e2,1e17,400)
    ks = np.geomspace(1e-4,100,1001)
    hcos = hm.HaloCosmology(zs,ks,ms=ms)
    with bench.show("nfw"):
        _,ouks = hcos.add_nfw_profile("matter",ms)
    pmm_1h = hcos.get_power_1halo_auto(name="matter")
    pmm_2h = hcos.get_power_2halo_auto(name="matter")
    print(pmm_1h.shape)
    pl = io.Plotter(xyscale='loglog')
    for i in range(zs.size):
        # pl.add(ks,pmm_1h[i],label="z=%.1f" % zs[i])
        pl.add(ks,pmm_2h[i],label="z=%.1f" % zs[i],ls="--",color="C%d" % i)
        pl.add(ks,pmm_2h[i]+pmm_1h[i],ls="-",color="C%d" % i)
    pl.done()
    
#test_fft_transform()    
test_pmm()    
# test_fft_integral()
#test_cosmology()
