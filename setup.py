#!/usr/bin/env python3
"""entropix setup.py.

This file details modalities for packaging the entropix application.
"""

from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='entropix',
    description='Sampling SVD singular vectors for Distributional Semantics Models',
    author=' Alexandre Kabbach',
    author_email='akb@3azouz.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='2.0.1',
    url='https://github.com/akb89/entropix',
    download_url='https://github.com/akb89/entropix',
    license='MIT',
    keywords=['entropy', 'distributional semantics'],
    platforms=['any'],
    packages=['entropix', 'entropix.logging', 'entropix.exceptions',
              'entropix.utils', 'entropix.core'],
    package_data={'entropix': ['logging/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'entropix = entropix.main:main'
        ],
    },
    install_requires=['pyyaml>=4.2b1', 'numpy==1.19.0', 'embeddix==1.15.1'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    zip_safe=False,
)
