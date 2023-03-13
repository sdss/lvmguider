FROM python:3.11-slim-bullseye

MAINTAINER Jose Sanchez-Gallego, gallegoj@uw.ed
LABEL org.opencontainers.image.source https://github.com/sdss/lvmguider

WORKDIR /opt

COPY . lvmguider

RUN pip3 install -U pip setuptools wheel
RUN cd lvmguider && pip3 install .
RUN rm -Rf lvmguider

ENTRYPOINT lvmguider actor start --debug
