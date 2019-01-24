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
    ms = np.geomspace(2e14,1e16,100)

    ks = np.geomspace(1e-3,10,101)

    from enlib import bench
    with bench.show("init"):
        hcos = hm.HaloCosmology(zs,ks,ms=ms,mass_function="tinker")

    print(hcos.nzm.shape,hcos.bh.shape)
    bh = hcos.bh
    nzm = hcos.nzm

    ims,ins = np.loadtxt("data/tinker2008Fig5.txt",unpack=True,delimiter=',')
    pl = io.Plotter(xyscale='linlin')
    pl.add(ims,ins,ls="--")
    pl.add(np.log10(ms*hcos.h),np.log10(nzm[0,:]*ms**2./hcos.rho_matter_z(0.)))
    pl.done()

    fsky = 0.4
    cSpeedKmPerSec = 299792.458
    nz = np.trapz(nzm,ms,axis=-1)*4.*np.pi*hcos.chis**2./hcos.Hzs*fsky * cSpeedKmPerSec 
    print(nz.shape)
    pl = io.Plotter()
    pl.add(zs,nz)
    pl.done()
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

    matt = True

    if matt:
        import halomodel as mmhm
        from orphics import cosmology
        cc = cosmology.Cosmology(hm.default_params,skipCls=True,low_acc=True)
        mmhmod = mmhm.HaloModel(cc)
    
    
    zs = np.array([0.,2.,3.])
    ms = np.geomspace(1e4,1e17,2000)
    ks = np.geomspace(1e-4,100,1001)


    if matt: mmP = mmhmod.P_mm_2h(ks,zs) + mmhmod.P_mm_1h(ks,zs)#,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    # print(mmhmod.halobias[:,0])
    
    #print(mmP2h.shape)
    
    
    hcos = hm.HaloCosmology(zs,ks,ms=ms,mass_function="sheth-torman")


    # pl = io.Plotter(xyscale='loglog')
    # pl.add(10**mmhmod.logm,mmhmod.halobias[:,1])
    # pl.add(ms,hcos.bh[1,:])
    # pl.done()
    
    
    _,ouks = hcos.add_nfw_profile("matter",ms)
    pmm_1h = hcos.get_power_1halo_auto(name="matter")
    pmm_2h = hcos.get_power_2halo_auto(name="matter")
    # sys.exit()
    print(pmm_1h.shape)
    pl = io.Plotter(xyscale='loglog')
    for i in range(zs.size):
        pl.add(ks,pmm_1h[i],label="z=%.1f" % zs[i],color="C%d" % i,ls="--",alpha=0.2)
        # pl.add(ks,pmm_2h[i],label="z=%.1f" % zs[i],ls="--",color="C%d" % i)
        pl.add(ks,pmm_2h[i]+pmm_1h[i],ls="--",color="C%d" % i)
        pl.add(ks,hcos.nPzk[i],ls="-",color="C%d" % i)
        if matt: pl.add(ks,mmP[i],ls="-.",color="C%d" % i)
    pl._ax.set_ylim(1e-1,1e5)
    pl.done()
    
#test_massfn()
#test_fft_transform()    
test_pmm()    
# test_fft_integral()
#test_cosmology()
