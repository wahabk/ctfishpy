# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:2.6.0-gpu

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . /src
WORKDIR /src
# docker run --rm -it --gpus all tensorflow/tensorflow:2.6.0-gpu
# source /home/ak18001/.virtualenvs/fish/bin/activate

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "train3dunet.py"]
