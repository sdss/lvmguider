FROM python:3.11-slim-bookworm

MAINTAINER Jose Sanchez-Gallego, gallegoj@uw.edu
LABEL org.opencontainers.image.source https://github.com/sdss/lvmguider

WORKDIR /opt

COPY . lvmguider

# Copy matplotlibrc
RUN mkdir -p /root/.config/matplotlib
COPY docker/matplotlibrc /root/.config/matplotlib/

# Copy serif font
COPY docker/LiberationSerif-Regular.ttf /usr/share/fonts/

# Refresh system font cache
RUN apt update -y
RUN apt install -y fontconfig
RUN fc-cache -f -v

# Refresh matplotlib font cache
RUN rm -fr ~/.cache/matplotlib

RUN apt install -y build-essential
RUN apt install -y astrometry.net

# Ignore warnings about installing as packages with pip as root.
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip3 install -U pip setuptools wheel
RUN cd lvmguider && pip3 install .
RUN rm -Rf lvmguider

# Set umask so that new files inherit the parent folder permissions.
RUN echo "umask 0002" >> /etc/bash.bashrc

ENTRYPOINT pip3 install -U astropy-iers-data; lvmguider actor start --debug
