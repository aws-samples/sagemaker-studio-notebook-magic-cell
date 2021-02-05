# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ipykernel.kernelapp import IPKernelApp
from . import SageMakerKernel

IPKernelApp.launch_instance(kernel_class=SageMakerKernel)
