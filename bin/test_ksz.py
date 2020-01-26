from hmvec import ksz
import numpy as np

#print(ksz.kSZ([0.5],100).ksz_radial_function(0))

ells = np.arange(2,6000,40)
volume_gpc3 = 100.
z = 0.5
ngal_mpc3 = 0.0001

ksz.get_ksz_template_signal(ells,volume_gpc3,z,ngal_mpc3)
    
