#!/bin/bash
umask 0002

pip3 install -U astropy-iers-data
exec "$@"
