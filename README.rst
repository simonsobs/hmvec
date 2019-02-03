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
* numpy, scipy, matplotlib
* camb (Python package, recommend using `dev` branch)

Credits
-------

The theory used here follows the approach outlined in the appendix of
arxiv:1810.13423. This code has greatly benefited from comparisons with the
implementation written by Moritz
Munchmeyer and Matt Johnson for that paper. Some of the HOD functions are copied (and
modified) from there.

Installation
------------

Clone this repository and install with symbolic links as follows
so that changes you make to the code are immediately reflected.


.. code-block:: console

   pip install -e . --user

				

Usage
-----

One can quickly get the matter power spectrum for desired wavenumbers and
redshifts after specifying the mass grid to integrate over. Note that
the analytic NFW profile is initialized by default.

.. code-block:: python
		
	import hmvec as hm
	zs = np.linspace(0.1,3.,4)
	ms = np.geomspace(2e10,1e17,200)
	ks = np.geomspace(1e-4,100,1001)
	hcos = hm.HaloModel(zs,ks,ms=ms)
	pmm_1h = hcos.get_power_1halo(name="nfw")
	pmm_2h = hcos.get_power_2halo(name="nfw")

You can add functions that implement a profile of your choice. An electron
profile from Battaglia 2016 has also been implemented. It needs to
be FFTd numerically to get the electron power spectrum, which is done as follows:


.. code-block:: python
				
   hcos.add_battaglia_profile("electron",family="AGN",xmax=50,nxs=10000)
   pee_1h = hcos.get_power_1halo(name="electron")
   pee_2h = hcos.get_power_2halo(name="electron")
	
Cross-spectra can also be calculated:

.. code-block:: python
				
   pme_1h = hcos.get_power_1halo("nfw","electron")
   pme_2h = hcos.get_power_2halo("nfw","electron")
   
An HOD can be added as follows:

.. code-block:: python
				
   hcos.add_hod(name="g",mthresh=10**10.5+zs*0.)

and galaxy spectra and cross-spectra with matter and electrons can be
calculated just as above by specifying the chosen name for the HOD.
If the galaxy number density `ngal` is provided instead of `mthresh`,
the latter will be found iteratively.

Cosmic Shear / CMB Lensing autospectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`HaloModel` inherits from `cosmology.Cosmology` which contains some
convenient functions involving Limber integrals. To get a cosmic shear
power spectrum for example, you first build the total matter power
spectrum and pass it to the relevant member function of `cosmology.Cosmology`,

.. code-block:: python
				
   zs = np.linspace(0.,3.,30)
   ms = np.geomspace(2e10,1e17,200)
   ks = np.geomspace(1e-4,100,1001)
   hcos = hm.HaloModel(zs,ks,ms=ms)
   
   pmm_1h = hcos.get_power_1halo(name="nfw")
   pmm_2h = hcos.get_power_2halo(name="nfw")
   Pmm = pmm_1h + pmm_2h
   
   ells = np.linspace(100,600,10)
   Cls = hcos.C_kk(ells,ks,Pmm,lzs=2.5)


Galaxy-galaxy lensing / Galaxy-CMB lensing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, one can obtain cross-spectra for galaxy-galaxy lensing
and galaxy-CMB lensing,

.. code-block:: python
				
   hcos.add_hod(name="g",mthresh=10**10.5+zs*0.)
   pgm_1h = hcos.get_power_1halo("nfw","electron")
   pgm_2h = hcos.get_power_2halo("nfw","electron")
   Pgm = pgm_1h + pgm_2h
   
   ells = np.linspace(100,600,10)
   Cls = hcos.C_kg(ells,ks,Pgm,gzs=0.8,lzs=2.5)
