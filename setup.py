#!/usr/bin/env python3
"""matrixor setup.py.

This file details modalities for packaging the matrixor application.
"""

from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='entropix',
    description='Entropy, Language and Distributional Semantics',
    author=' Alexandre Kabbach',
    author_email='akb@3azouz.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.0',
    url='https://github.com/akb89/entropix',
    download_url='https://github.com/akb89/entropix',
    license='MIT',
    keywords=['entropy', 'distributional semantics'],
    platforms=['any'],
    packages=['entropix', 'entropix.logging', 'entropix.exceptions',
              'entropix.utils'],
    package_data={'matrixor': ['logging/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'entropix = entropix.main:main'
        ],
    },
    install_requires=['pyyaml>=4.2b1', 'tqdm==4.28.1'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    zip_safe=False,
)
