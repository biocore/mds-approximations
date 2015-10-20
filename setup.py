#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2015--, mdsa development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

# this setup.py file is heavlily based on scikit-bio's setup.py file

import re
import ast

from setuptools import find_packages, setup

# https://github.com/mitsuhiko/flask/blob/master/setup.py
_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('mdsa/__init__.py', 'rb') as f:
    hit = _version_re.search(f.read().decode('utf-8')).group(1)
    version = str(ast.literal_eval(hit))

classes = """
    Development Status :: 1 - Planning
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python
    Programming Language :: Python :: 2
    Programming Language :: Python :: 2.7
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.3
    Programming Language :: Python :: 3.4
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

setup(name='mdsa',
      version=version,
      license='BSD',
      description="Multi-dimensional scaling approximations",
      long_description="Benchmarking several multi-dimensional scaling "
                       "approximation methods",
      author="mdsa development team",
      author_email="antgonza@gmail.com",
      maintainer="mdsa development team",
      maintainer_email="antgonza@gmail.com",
      url='http://github.com/biocore/mds-approximations/',
      test_suite='nose.collector',
      packages=find_packages(),
      setup_requires=['numpy >= 1.9.2'],
      install_requires=[
          'IPython',
          'matplotlib >= 1.4.3',
          'numpy >= 1.9.2',
          'pandas >= 0.16.2',
          'scipy >= 0.15.1',
          'sklearn'
      ],
      extras_require={'test': ["nose", "pep8", "flake8"]},
      classifiers=classifiers)
