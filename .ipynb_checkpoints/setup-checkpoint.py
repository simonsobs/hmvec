from distutils.core import setup, Extension
import os



setup(name='hmvec',
      version='0.1',
      description='Point source fitting',
      url='https://github.com/msyriac/hmvec',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['hmvec'],
      package_dir={'hmvec':'hmvec'},
      zip_safe=False)
