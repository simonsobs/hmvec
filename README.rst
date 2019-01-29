=======
hmvec
=======

`hmvec` is a fast pure Python/numpy vectorized general halo model and HOD code.
Many great halo model codes exist. This one is meant to allow for quick
exploration and forecasting rather than allow for precision cosmological inference.

It calculates a vectorized FFT for a given profile over all points in mass and
redshift, but it currently does have one double loop over mass and redshift
to interpolate the profile Fourier transforms to the target wavenumbers. Every
other part of the code is vectorized.


* Free software: BSD license
* Documentation: in the works

Dependencies
------------

* Python>=2.7 or Python>=3.4
* numpy, scipy
* camb (Python package)

Usage
-----

One can quickly get the matter power spectrum for desired wavenumbers and
redshifts after specifying the mass grid to integrate over.

.. code-block:: python
		
    zs = np.linspace(0.1,3.,4)
    ms = np.geomspace(2e2,1e17,400)
    ks = np.geomspace(1e-4,100,1001)
    hcos = hm.HaloCosmology(zs,ks,ms=ms)
    hcos.add_nfw_profile("matter",ms)
    pmm_1h = hcos.get_power_1halo_auto(name="matter")
    pmm_2h = hcos.get_power_2halo_auto(name="matter")
   
