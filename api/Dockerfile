FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install python3.11 python3.11-dev gcc g++&& \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /usr/src/app

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./save_models.py .
RUN python3.11 save_models.py

COPY ./main.py .
