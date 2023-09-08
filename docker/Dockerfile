FROM python:3.11-slim-bookworm

MAINTAINER Jose Sanchez-Gallego, gallegoj@uw.edu
LABEL org.opencontainers.image.source https://github.com/sdss/lvmguider

WORKDIR /opt

COPY . lvmguider

# Copy matplotlibrc
RUN mkdir -p /root/.config/matplotlib
COPY docker/matplotlibrc /root/.config/matplotlib/

# Install MS fonts
RUN apt-get update; apt-get install -y ttf-mscorefonts-installer fontconfig

# Refresh system font cache
RUN fc-cache -f -v

# Refresh matplotlib font cache
RUN rm -fr ~/.cache/matplotlib

RUN apt update -y
RUN apt install -y build-essential
RUN apt install -y astrometry.net

RUN pip3 install -U pip setuptools wheel
RUN cd lvmguider && pip3 install .
RUN rm -Rf lvmguider

# Set umask so that new files inherit the parent folder permissions.
RUN echo "umask 0002" >> /etc/bash.bashrc

ENTRYPOINT lvmguider actor start --debug
