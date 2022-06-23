# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM tensorflow/tensorflow:2.6.0-gpu
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"

RUN apt-get update -y
RUN pip3 install --upgrade pip
# RUN apt-get install libgl1 -y

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

WORKDIR /ctfishpy/
COPY fish.yml .
# from https://stackoverflow.com/questions/55123637/activate-conda-environment-in-docker
RUN conda init bash \
    && . ~/.bashrc \
    && conda env create -f fish.yml \
    && conda activate colloids \
    && pip install ipython 

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /ctfishpy

# CMD ["python3", "train3dunet.py"]

# TODO use buildkit https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds