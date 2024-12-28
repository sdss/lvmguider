FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

LABEL org.opencontainers.image.authors="Jose Sanchez-Gallego, gallegoj@uw.edu"
LABEL org.opencontainers.image.source=https://github.com/sdss/lvmguider

WORKDIR /opt

COPY . lvmguider

# Copy matplotlibrc
RUN mkdir -p /root/.config/matplotlib
COPY docker/matplotlibrc /root/.config/matplotlib/

# Copy serif font
COPY docker/LiberationSerif-Regular.ttf /usr/share/fonts/

# Refresh system font cache
RUN apt-get update -y
RUN apt-get install -y fontconfig
RUN fc-cache -f -v

# Refresh matplotlib font cache
RUN rm -fr ~/.cache/matplotlib

RUN apt-get install -y build-essential
RUN apt-get install -y astrometry.net

# Ignore warnings about installing as packages with pip as root.
ENV PIP_ROOT_USER_ACTION=ignore

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="$PATH:/opt/lvmguider/.venv/bin"

# Sync the project
RUN cd lvmguider && uv sync --frozen --no-cache --no-dev

# Set umask so that new files inherit the parent folder permissions.
# Not sure this works so we are also setting the permissions in the entrypoint.
RUN echo "umask 0002" >> /etc/bash.bashrc

COPY ./docker/docker-entrypoint.sh /
RUN ["chmod", "+x", "/docker-entrypoint.sh"]

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["sh", "-c", "lvmguider actor start --debug"]
