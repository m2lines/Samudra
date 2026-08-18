# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from samudra.utils.multiton import Multiton, MultitonScope


class DummyMultiton(Multiton):
    def _initialize(self, value=0):
        self.value = value


def test_direct_instantiation_raises():
    with pytest.raises(TypeError):
        DummyMultiton()


def test_get_instance_without_init():
    with pytest.raises(ValueError):
        DummyMultiton.get_instance()


def test_init_and_get_instance():
    with MultitonScope():
        instance = DummyMultiton.init_instance(42)
        got_instance = DummyMultiton.get_instance()
        assert instance is got_instance
        assert instance.value == 42


def test_duplicate_initialization():
    with MultitonScope():
        DummyMultiton.init_instance(10)
        with pytest.raises(ValueError):
            DummyMultiton.init_instance(20)


def test_scope_restoration():
    scope1 = MultitonScope()
    with scope1:
        _instance1 = DummyMultiton.init_instance(5)
        assert DummyMultiton.get_instance().value == 5

    with pytest.raises(ValueError):
        DummyMultiton.get_instance()

    with MultitonScope():
        _instance2 = DummyMultiton.init_instance(100)
        assert DummyMultiton.get_instance().value == 100

    with scope1:
        assert DummyMultiton.get_instance().value == 5
