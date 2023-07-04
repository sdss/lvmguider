FROM python:3.11-slim-bookworm

MAINTAINER Jose Sanchez-Gallego, gallegoj@uw.edu
LABEL org.opencontainers.image.source https://github.com/albireox/lvmguider

WORKDIR /opt

COPY . lvmguider

RUN apt update -y
RUN apt install -y build-essential
RUN apt install -y astrometry.net

RUN pip3 install -U pip setuptools wheel
RUN cd lvmguider && pip3 install .
RUN rm -Rf lvmguider

ENTRYPOINT lvmguider actor start --debug
