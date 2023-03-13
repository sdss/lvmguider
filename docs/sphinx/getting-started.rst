Getting started with Trurl
==========================

Minimal example
---------------

The following is a minimum example of how to use the Trurl client to connect to the actor system and command the telescope to a science field, exposing during 15 minutes.

This code must be run in one of the LVM mountain servers with access to the RabbitMQ exchange.

.. code-block:: python

    >>> from lvmguider import Trurl

    >>> trurl = Trurl()
    >>> await trurl.init()
    >>> trurl.connected()
    True

    >>> await trurl.telescopes['sci'].update_status()
    {'is_tracking': False,
    'is_connected': True,
    'is_slewing': False,
    'is_enabled': False,
    'ra_j2000_hours': 9.31475952237581,
    'dec_j2000_degs': 26.2017449132552,
    'ra_apparent_hours': 9.33749451003047,
    'dec_apparent_degs': 26.1043875064315,
    'altitude_degs': -61.6417256185769,
    'azimuth_degs': 88.1701770599825,
    'field_angle_rate_at_target_degs_per_sec': 0.0,
    'field_angle_here_degs': -14.3931305251519,
    'field_angle_at_target_degs': 0.0,
    'axis0': {'dist_to_target_arcsec': 0.0,
    'is_enabled': False,
    'position_degs': 308.137461864407,
    'rms_error_arcsec': 0.0,
    'servo_error_arcsec': 0.0},
    'axis1': {'dist_to_target_arcsec': 0.0,
    'is_enabled': False,
    'position_degs': 308.137461864407,
    'rms_error_arcsec': 0.0,
    'servo_error_arcsec': 0.0,
    'position_timestamp': '2023-03-11 16:54:22.5626'},
    'model': {'filename': 'DefaultModel.pxp',
    'num_points_enabled': 99,
    'num_points_total': 111,
    'rms_error_arcsec': 18.1248458630523,
    'position_degs': 22.8931398305085,
    'position_timestamp': '2023-03-11 16:54:22.5626'},
    'geometry': 1}

Reference
---------

.. automodule:: trurl.core
   :members: Trurl
   :show-inheritance:
   :noindex:
