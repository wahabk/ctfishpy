# CTFishPy

My project to segment zebrafish bone from uCT data using deep learning.

# Installation

On windows:

```
# Download https://github.com/wahabk/ctfishpy from github
conda env create -f fish.yml # this will create an env using python 3.8 named 'fish'
conda activate fish
python3 -m pip install .
```

On Linux / Mac:

```
cd <installation_dir>
git clone https://github.com/wahabk/ctfishpy
conda env create -f fish.yaml # this will create an env using python 3.8 named 'fish'
conda activate fish
python3 -m pip install .
```

Quickest 

```
pip install git+https://github.com/wahabk/ctfishpy
```

# Usage

Check the jupyter notebooks in `examples/`

# Requirements

Make sure you have anaconda installed, for python dependencies check `setup.py`

# Dev

## Docker

I provide a docker image to make gpu training easier

Step 1 : Build image, this will build an image named ```zebrafish```

```
docker build . --tag=zebrafish 
docker build . --tag=zebrafish --network=host # if you're on vpn
```

Step 2 : Create container in interactive mode, this will start a shell inside the container

```
docker run -it \
	-v <data_dir>:<data_dir> \ # mount data directory as volume
	-v <repo_dir>:/colloidoscope \ # mount git repo directory as volume for interactive access
	--gpus all \ # allow the container to use all gpus
	--network=host \ # if you're on vpn
	zebrafish \ 
```

Note:
If you want to launch the container on a custom hard drive use:

```sudo dockerd --data-root <custom_dir>```
