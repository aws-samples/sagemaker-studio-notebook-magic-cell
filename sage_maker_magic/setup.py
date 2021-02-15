# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from distutils.core import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='sage_maker_kernel',
    version='0.1',
    packages=['sage_maker_kernel'],
    url='https://github.nike.com/analytics-platform/provectus-sagemaker-image',
    author='Leonid Blokhin',
    author_email='leonid.blokhin@nike.com',
    description='SM kernel for Jupyter',
    long_description=readme,
    install_requires=[line for line in open('requirements.txt').readlines()],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)