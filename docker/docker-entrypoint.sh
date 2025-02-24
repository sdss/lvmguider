#!/bin/bash
umask 0002

cd lvmguider && uv pip install -U astropy-iers-data
exec "$@"
