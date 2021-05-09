# syntax=docker/dockerfile:1

FROM python:3.6-slim-buster

WORKDIR /app

RUN apt-get update 
RUN apt-get install --reinstall build-essential -y
RUN apt-get install -y git
RUN apt-get install gcc -y


COPY . .
RUN pip3 install -r requirements.txt

RUN python -m nltk.downloader averaged_perceptron_tagger universal_tagset \
       punkt
