# Inspired by https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.10.0-cuda11.3-ubuntu20.04/Dockerfile
# ARG COMPAT=0
ARG PERSONAL=0
# FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as base-0
FROM nvcr.io/nvidia/pytorch:22.06-py3 as base

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
# Need to put ARG in the build-stage where it is used, otherwise the ARG will be empty
# https://benkyriakou.com/posts/docker-args-empty
# https://github.com/distribution/distribution/issues/2459
ARG PYTHON_VERSION=3.8

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1


RUN git clone https://github.com/idiap/fast-transformers \
    && sed -i 's/\["-arch=compute_60"\]/\["-arch=compute_70"\]/' fast-transformers/setup.py \
    && pip install fast-transformers/ \
    && rm -rf fast-transformers

# xgboost conflicts with deepspeed
RUN pip uninstall -y xgboost && DS_BUILD_UTILS=1 DS_BUILD_FUSED_LAMB=1 pip install deepspeed==0.6.5

# General packages that we don't care about the version
# fs for reading tar files
RUN pip install pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops fs fvcore gsutil cmake pykeops \
    && python -m spacy download en_core_web_sm
# hydra
RUN pip install hydra-core==1.2.0 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 python-dotenv rich
# Core packages
RUN pip install datasets==2.12.0 pytorch-lightning==1.6.5 wandb==0.12.21 timm==0.6.5 torchmetrics==0.9.2

# For MLPerf
RUN pip install git+https://github.com/mlcommons/logging.git@2.0.0-rc4

# This is for huggingface/examples and smyrf
RUN pip install tensorboard seqeval psutil sacrebleu rouge-score tensorflow_datasets h5py

# ENV NVIDIA_REQUIRE_CUDA=cuda>=10.1

# added libs for PEFT
RUN pip install scikit-learn tqdm loralib 
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
RUN pip install "einops==0.5.0" "mosaicml[nlp,wandb]>=0.14.0,<0.15" "mosaicml-streaming==0.4.1" "omegaconf==2.2.3" "transformers==4.28.1" "opt_einsum" "triton==2.0.0.dev20221103"

