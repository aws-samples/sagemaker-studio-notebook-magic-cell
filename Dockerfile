FROM jupyter/minimal-notebook:latest

ENV DEFAULT_SM_CONFIG_PATH=/usr/local/share/extensions/jupyter_ext/sage_maker_kernel/default.conf

USER root
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    curl vim groff less &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the AWS CLI:
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip
RUN ./aws/install

COPY sage_maker_magic /usr/local/share/extensions/jupyter_ext
COPY config/default.conf ${DEFAULT_SM_CONFIG_PATH}
RUN chown -R $NB_UID /usr/local/share/extensions

USER $NB_UID

# Install various Python utilities for SageMaker:
# (Pinned to last tested major version for repeatability)
RUN pip install \
        'sagemaker-experiments>=0.1,<0.2' \
        'sagemaker-studio-image-build>=0.4,<0.5' \
        'smdebug>=0.9,<0.10' \
        'jupyter-console'

# Install scikit-learn
# For uploading a sample of the classic Iris dataset, which is included with Scikit-learn.
# Which we're using for the purposes as an example.
RUN pip install -U scikit-learn

# Install custom kernels
RUN cd /usr/local/share/extensions/jupyter_ext && \
    python setup.py install && \
    python -m sage_maker_kernel.install --sys-prefix
