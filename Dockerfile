# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements-dev.txt .
RUN python -m pip install -r requirements-dev.txt

COPY . /ctfishpy
COPY .env /ctfishpy/.env

WORKDIR /ctfishpy

# CMD ["python3", "train3dunet.py"]

#docker run -it -v /data/:/data/ --gpus all test 