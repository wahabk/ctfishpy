# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-get update -y
RUN pip3 install --upgrade pip
# RUN apt-get install libgl1 -y

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements-dev.txt .
RUN python -m pip install -r requirements-dev.txt

# CMD ["apt-get", "install", "libgl1", "-y"]

# COPY . /ctfishpy
# COPY .env /ctfishpy/

WORKDIR /ctfishpy

# CMD ["python3", "train3dunet.py"]

# docker run -it \
# 	-v /data/:/data/ \
# 	-v /home/mb16907/wahab/code/ctfishpy:/ctfishpy \
# 	--gpus all \
# 	fish \

# TODO use buildkit https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds