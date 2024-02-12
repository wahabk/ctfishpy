FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ENV PATH="/root/miniconda3/bin:$PATH"

RUN apt-get update
RUN apt-get update -y
RUN pip3 install --upgrade pip

ENV PYTHONDONTWRITEBYTECODE=1
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

WORKDIR /ctfishpy

# TODO use buildkit https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds