#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-01-17
# @Filename: conftest.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pytest

from clu.testing import setup_test_actor

from lvmguider.actor.actor import LVMGuiderActor


actor_config = {
    "telescope": "sci",
    "has_kmirror": True,
    "guide_in_rot": True,
    "focus": {
        "model": {
            "a": 0.1225,
            "b": 34.435,
        }
    },
    "actor": {
        "name": "lvm.sci.guider",
        "host": "10.8.38.21",
        "port": 5672,
    },
}


@pytest.fixture()
async def actor():
    _actor = LVMGuiderActor.from_config(actor_config)

    _actor = await setup_test_actor(_actor)  # type: ignore

    yield _actor

    _actor.mock_replies.clear()
    await _actor.stop()
