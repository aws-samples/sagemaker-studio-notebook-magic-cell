FROM jupyter/minimal-notebook:latest


USER root
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    curl vim groff less &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the AWS CLI:
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip
RUN ./aws/install

COPY sagemaker_magic /usr/local/share/extensions/jupyter_ext
RUN chown -R $NB_UID /usr/local/share/extensions

USER $NB_UID

# Install various Python utilities for SageMaker:
# (Pinned to last tested major version for repeatability)
RUN pip install \
        'boto3>=1,<2' \
        'sagemaker>=2,<3' \
        'sagemaker-experiments>=0.1,<0.2' \
        'sagemaker-studio-image-build>=0.4,<0.5' \
        'smdebug>=0.9,<0.10' \
        'jupyter-console' \
        'pandas'

# Install custom kernels
#RUN pip install jupyter_contrib_nbextensions
RUN cd /usr/local/share/extensions/jupyter_ext && \
    python setup.py install && \
    python -m sage_maker_kernel.install --sys-prefix
