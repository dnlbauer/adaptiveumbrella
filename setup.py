#!/usr/bin/env python3
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

LONG_DESCRIPTION = """`adaptiveumbrella` is a python implementation of the
[self-learning adaptive umbrella sampling algorithm (Wojtas-Niziurski†, Meng, Roux, Bernèche, 2013)]
(https://pubs.acs.org/doi/abs/10.1021/ct300978b) technique. It allows the calculation of a multidimensional potential
of mean force while automatically exploring the phase space.
"""

# Parse the version from the fiona module.
with open('adaptiveumbrella/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

setup(
    name='adaptiveumbrella',
    version=version,
    description='Adaptive umbrella sampling in Python',
    license='CC0',
    author='Daniel Bauer',
    author_email='bauer@cbs.tu-darmstadt.de',
    url='https://github.com/danijoo/adaptiveumbrella',
    long_description=LONG_DESCRIPTION,
    packages=['adaptiveumbrella'],
install_requires=['numpy'])