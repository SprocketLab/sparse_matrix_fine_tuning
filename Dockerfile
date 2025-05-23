# Inspired by https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.10.0-cuda11.3-ubuntu20.04/Dockerfile
# ARG COMPAT=0
ARG PERSONAL=0
FROM nvcr.io/nvidia/pytorch:24.10-py3 as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod 777 /home/user
WORKDIR /home/user


# Set up personal environment
# FROM base-${COMPAT} as env-0
FROM base as env-0
FROM env-0 as env-1
# Use ONBUILD so that the dotfiles dir doesn't need to exist unless we're building a personal image
# https://stackoverflow.com/questions/31528384/conditional-copy-add-in-dockerfile
ONBUILD COPY dotfiles ./dotfiles
ONBUILD RUN cd ~/dotfiles && stow bash zsh tmux && sudo chsh -s /usr/bin/zsh $(whoami)

FROM env-${PERSONAL} as packages

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1

# General packages that we don't care about the version
# fs for reading tar files
RUN pip install pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops fs fvcore gsutil cmake pykeops \
    && python -m spacy download en_core_web_sm
# hydra
RUN pip install hydra-core==1.2.0 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 python-dotenv rich
# Core packages
RUN pip install datasets==2.12.0 wandb==0.12.21

# For MLPerf
RUN pip install git+https://github.com/mlcommons/logging.git@2.0.0-rc4

# This is for huggingface/examples and smyrf
RUN pip install tensorboard seqeval psutil sacrebleu rouge-score h5py

# ENV NVIDIA_REQUIRE_CUDA=cuda>=10.1

# added libs for PEFT
RUN pip install tqdm loralib
RUN pip install evaluate

# git for installing dependencies
# tzdata to set time zone
# wget and unzip to download data
# [2021-09-09] TD: zsh, stow, subversion, fasd are for setting up my personal environment.
# [2021-12-07] TD: openmpi-bin for MPI (multi-node training)
RUN apt-get update && apt-get install -y --no-install-recommends \
   build-essential \
   cmake \
   curl \
   ca-certificates \
   sudo \
   less \
   htop \
   git \
   tzdata \
   wget \
   tmux \
   zip \
   unzip \
   zsh stow subversion fasd \
   && rm -rf /var/lib/apt/lists/*  || true
   # openmpi-bin \

# m2 dependencies
RUN pip install einops omegaconf opt_einsum 
# Some how this fixes the pydantic__version__ bug...
RUN python -m pip install -U pydantic spacy==3.4.4
RUN pip install accelerate -U
RUN pip install jupyterlab==4.0.3
RUN pip install datasets==2.16.1
RUN pip install tensorboardX
RUN pip install bitsandbytes peft einops sentencepiece==0.1.99
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true
RUN pip install transformers
RUN pip install flash_attn==2.5.6
RUN pip install ray==2.37.0
# ENV PATH $CONDA_DIR/bin:$PATH
# RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
#     /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
#     rm -rf /tmp/*
RUN pip install pre-commit
RUN chmod -R 777 .
# build from the newest main branch with more potential fixes
RUN pip install triton==3.3.0
