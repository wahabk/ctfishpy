# CTFishPy

<a href=https://colab.research.google.com/github/wahabk/ctfishpy/blob/master/CTFishpy_Tutorial_segment.ipynb> 
<img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>
<img src="examples/Data/ctf_readme.gif" alt="readme_gif"/>

Automatic segmentation of zebrafish bone from uCT data using deep learning.

# Installation

```
pip install git+https://github.com/wahabk/ctfishpy
```

# Usage

Check the jupyter notebooks in `examples/`

# Dependencies

For python dependencies check `setup.py`

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
