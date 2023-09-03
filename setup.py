from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


setup(
    version="0.0.1",
    name="price_impact",
    packages=["price_impact", "price_impact.c_utils","price_impact.modelling"],
    ext_modules=cythonize(["price_impact/c_utils/ewm.pyx", "price_impact/c_utils/linreg.pyx"]),
    zip_safe=True,
    install_requires=[
        'Cython',
        'numpy',  # Make sure NumPy is listed as a requirement
    ],
    include_dirs=[np.get_include()],
)
