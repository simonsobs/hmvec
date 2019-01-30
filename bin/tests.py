import hmvec
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

    from szar import counts
    
    import hmf
    from cluster_toolkit import massfunction

    zs = np.linspace(0.,3.,20)
    ms = np.geomspace(1e14,1e17,200)

    ks = np.geomspace(1e-3,10,101)

    from enlib import bench
    with bench.show("init"):
        hcos = hm.HaloCosmology(zs,ks,ms=ms,mass_function="tinker")

    dndM_ct2 = np.zeros((zs.size,ms.size))
    for i,z in enumerate(zs):
        h = hmf.MassFunction(z=z,Mmin=np.log10(ms.min()*hcos.h),Mmax=np.log10(ms.max()*hcos.h))
        if i==0: dndM_ct = np.zeros((zs.size,h.dndm.size))
        dndM_ct[i,:] = h.dndm.copy()
        dndM_ct2[i,:] = massfunction.dndM_at_M(ms*hcos.h, hcos.ks_sigma2/hcos.h, hcos.sPzk[i]*hcos.h**3, hcos.om0)
        

    fsky = 0.4

    hmf = counts.Halo_MF(counts.ClusterCosmology(hcos.params,skipCls=True),np.log10(ms),zs)
    nz_szar = hmf.N_of_z()*fsky
    print(nz_szar,nz_szar.shape)
    # sys.exit()
        
    print(hcos.nzm.shape,hcos.bh.shape)
    bh = hcos.bh
    nzm = hcos.nzm

    # ims,ins = np.loadtxt("data/tinker2008Fig5.txt",unpack=True,delimiter=',')
    # pl = io.Plotter(xyscale='linlin')
    # pl.add(ims,ins,ls="--")
    # pl.add(np.log10(ms*hcos.h),np.log10(nzm[0,:]*ms**2./hcos.rho_matter_z(0.)))
    # pl.done()

    chis = hcos.results.angular_diameter_distance(hcos.zs) * (1+hcos.zs)
    nz = np.trapz(nzm,ms,axis=-1)*4.*np.pi*chis**2./hcos.results.h_of_z(hcos.zs)*fsky 
    nz_ct = np.trapz(dndM_ct,h.m,axis=-1)*4.*np.pi*chis**2./hcos.results.h_of_z(hcos.zs)*fsky  * hcos.h**3.
    nz_ct2 = np.trapz(dndM_ct2,ms,axis=-1)*4.*np.pi*chis**2./hcos.results.h_of_z(hcos.zs)*fsky * hcos.h**3.
    pl = io.Plotter()
    pl.add(zs,nz,label='hmvec')
    pl.add(hmf.zarr,nz_szar,ls='--',label='szar')
    pl.add(zs,nz_ct,ls='-.',label='hmf')
    pl.add(zs,nz_ct2,ls='-.',label='ct')
    pl.done()
    n = np.trapz(nz,zs)
    print(n)
    n = np.trapz(nz_szar,hmf.zarr)
    print(n)
    n = np.trapz(nz_ct,zs)
    print(n)
    n = np.trapz(nz_ct2,zs)
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

    matt = True # set to False if you don't have Matt and Moritz's halomodel.py for comparison

    if matt:
        import halomodel as mmhm
        from orphics import cosmology
        cc = cosmology.Cosmology(hmvec.default_params,skipCls=True,low_acc=False)
        mmhmod = mmhm.HaloModel(cc)
    
    
    zs = np.array([0.,2.,3.])#,1.,2.,3.])
    #zs = np.array([0.,2.,4.,6.])
    ms = np.geomspace(1e7,1e17,2000)
    #ks = np.geomspace(1e-4,100,1001)
    ks = np.geomspace(1e-5,100,10000)


    if matt: mmP = mmhmod.P_mm_2h(ks,zs) + mmhmod.P_mm_1h(ks,zs)#,logmlow=log10mlow,logmhigh=log10mhigh)[z_id,:]
    # print(mmhmod.halobias[:,0])
    
    #print(mmP2h.shape)
    
    
    hcos = hmvec.HaloCosmology(zs,ks,ms=ms,halofit='mead',mdef='vir',nfw_numeric=True)

    mmhb = mmhmod.halobias #np.load("mm_halobias.npy",)
    mmnfn = mmhmod.nfn #np.load("mm_nfn.npy")

    # pl = io.Plotter(xyscale='loglin')
    # for i in range(zs.size):
    #     pl.add(ks,(hcos.nPzk[i]-mmhmod.pknl[i])/hcos.nPzk[i])
    #     pl.add(ks,(hcos.Pzk[i]-mmhmod.pk[i])/hcos.Pzk[i])
    # pl.hline(y=0)
    # pl.done()
    
    

    # pl = io.Plotter(xyscale='loglog')
    # for i in range(3):
    #     pl.add(ms,hcos.nzm[i,:],color="C%d" % i)
    #     pl.add(ms,mmnfn[:,i],ls='--',color="C%d" % i)
    # pl.done()
    # pl = io.Plotter(xyscale='loglog')
    # for i in range(3):
    #     pl.add(ms,hcos.bh[i,:],color="C%d" % i)
    #     pl.add(ms,mmhb[:,i],ls='--',color="C%d" % i)
    # pl.done()
        
    

    # pl = io.Plotter(xyscale='loglog')
    # pl.add(10**mmhmod.logm,mmhmod.halobias[:,1])
    # pl.add(ms,hcos.bh[1,:])
    # pl.done()
    
    
    pmm_1h = hcos.get_power_1halo_auto(name="nfw")
    pmm_2h = hcos.get_power_2halo_auto(name="nfw")
    
    # sys.exit()
    print(pmm_1h.shape)
    for i in range(zs.size):
        pl = io.Plotter(xyscale='loglog',xlabel='$k$',ylabel='$P$')
        pl.add(ks,pmm_1h[i],label="z=%.1f" % zs[i],color="C%d" % i,ls="--",alpha=0.2)
        # pl.add(ks,pmm_2h[i],label="z=%.1f" % zs[i],ls="--",color="C%d" % i)
        pl.add(ks,pmm_2h[i]+pmm_1h[i],ls="--",color="C%d" % i)
        pl.add(ks,hcos.nPzk[i],ls="-",color="k",alpha=0.7)
        if matt: pl.add(ks,mmP[i],ls="-.",color="C%d" % i)
        pl.vline(x=10.)
        pl._ax.set_ylim(1e-1,1e5)
        pl.done("nonlincomp_z_%d.png" % i)

    for i in range(zs.size):
        pl = io.Plotter(xyscale='loglin',xlabel='$k$',ylabel='$P/P_{\\mathrm{NL}}$')
        pl.add(ks,(pmm_2h[i]+pmm_1h[i])/hcos.nPzk[i],ls="-",color="C%d" % i)
        if matt: pl.add(ks,mmP[i]/hcos.nPzk[i],ls="--",color="C%d" % i)
        pl.hline(y=1)
        pl.hline(y=0.9)
        pl.hline(y=1.1)
        pl.vline(x=10.)
        pl._ax.set_ylim(0.5,1.5)
        pl.done("nonlindiff_z_%d.png" % i)
    
    for i in range(zs.size):
        pl = io.Plotter(xyscale='loglin',xlabel='$k$',ylabel='$P/P_{\\mathrm{L}}$')
        pl.add(ks,(pmm_2h[i]+pmm_1h[i])/hcos.Pzk[i],ls="-",color="C%d" % i)
        if matt: pl.add(ks,mmP[i]/hcos.Pzk[i],ls="--",color="C%d" % i)
        pl.vline(x=10.)
        pl.hline(y=1)
        # pl.hline(y=0.9)
        # pl.hline(y=1.1)
        # pl._ax.set_ylim(0.5,1.5)
        pl.done("lindiff_z_%d.png" % i)

def test_battaglia():

    zs = np.array([0.])
    ks = np.geomspace(1e-4,1,10)
    hcos = hmvec.HaloCosmology(zs,ks,params = {'sigma2_numks':100}, skip_nfw=True)

    m200critz = 1.e13
    r = np.geomspace(1e-4,20.,10000)
    z = 1.
    omb = hcos.p['ombh2']/hcos.h**2.
    omm = (hcos.p['ombh2']/hcos.h**2.+hcos.p['omch2']/hcos.h**2.)
    rhocritz = hcos.rho_critical_z(z)
    
    rhos = hmvec.rho_gas(r,m200critz,z,omb,omm,rhocritz,
                            profile="AGN")

    r200 = hmvec.R_from_M(m200critz,rhocritz,delta=200)
    integrand = rhos.copy()*4.*np.pi*r**2.
    integrand[r>r200] = 0
    print(np.trapz(integrand,r)/(m200critz*(omb/omm)))

    # pl = io.Plotter(xyscale='loglog')
    # pl.add(r,rhos)
    # pl.done()

def test_mcon():

    zs = np.linspace(0.,1.,30)
    ks = np.geomspace(1e-4,1,10)
    hcos = hmvec.HaloCosmology(zs,ks,params = {'sigma2_numks':100}, skip_nfw=True)

    ms = np.geomspace(1e13,1e15,1000)
    cs = hmvec.duffy_concentration(ms[None,:],zs[:,None])

    rho1s = hcos.rho_matter_z(zs)
    rho2s = hcos.rho_critical_z(zs)

    with bench.show("vectorized"):
        mcritzs0 = hmvec.mdelta_from_mdelta(ms,cs,200.*rho1s,200.*rho2s,vectorized=True)
    with bench.show("unvectorized"):
        mcritzs1 = hmvec.mdelta_from_mdelta(ms,cs,200.*rho1s,200.*rho2s,vectorized=False)

    from orphics import io
    io.plot_img(np.log10(ms[None]+cs*0.),flip=False)
    io.plot_img(np.log10(mcritzs0),flip=False)
    io.plot_img((mcritzs0-mcritzs1)/mcritzs0,flip=False)

def test_gas_fft():

    zs = np.array([1.])
    ks = np.geomspace(1e-4,100,100)
    ms = np.geomspace(1e7,1e17,2000)
    hcos = hmvec.HaloCosmology(zs,ks,ms,nfw_numeric=False)
    
    hcos.add_battaglia_profile("electron",family="AGN",xmax=50,nxs=10000)

    pmm_1h = hcos.get_power_1halo_auto(name="nfw")
    pmm_2h = hcos.get_power_2halo_auto(name="nfw")

    pee_1h = hcos.get_power_1halo_auto(name="electron")
    pee_2h = hcos.get_power_2halo_auto(name="electron")

    pl = io.Plotter(xyscale='loglog')
    pl.add(ks,pee_1h[0]/pmm_1h[0])
    pl._ax.set_xlim(0.1,100)
    pl.done()
    
    pl = io.Plotter(xyscale='loglog')
    pl.add(ks,pmm_1h[0]+pmm_2h[0])
    pl.add(ks,pee_1h[0]+pee_2h[0],ls='--')
    pl.done()


test_gas_fft()
#test_mcon()
#test_battaglia()
#test_massfn()
#test_fft_transform()    
#test_pmm()    
# test_fft_integral()
#test_cosmology()
