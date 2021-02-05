# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from distutils.core import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='sagemaker_kernel',
    version='0.1',
    packages=['sagemaker_kernel'],
    url='https://github.com/aws-samples/sagemaker-studio-notebook-magic-cell',
    author='aws',
    description='SM kernel for Jupyter',
    long_description=readme,
    install_requires=[
        'jupyter_client', 'IPython', 'ipykernel'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
