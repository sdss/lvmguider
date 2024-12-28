#!/bin/bash
umask 0002

uv pip install -U astropy-iers-data
exec "$@"
