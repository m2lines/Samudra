# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Strict JSON for the frozen dataclasses the tools write and read back.

`from_json` rejects missing or unknown fields and values of the wrong type, so a
hand-edited or truncated record fails loudly instead of being half-trusted. It
supports the field types the records use: str, int, float, bool, StrEnum,
`X | None`, `tuple[X, ...]`, `list[X]`, `dict[str, X]`, and nested records.
"""

import dataclasses
import json
import types
import typing
from enum import Enum
from typing import Any


def to_json(record) -> str:
    return json.dumps(dataclasses.asdict(record), indent=2) + "\n"


def from_json[T](cls: type[T], text: str) -> T:
    return decode(cls, json.loads(text), cls.__name__)


def decode(kind: Any, value: Any, where: str) -> Any:
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if isinstance(kind, type) and dataclasses.is_dataclass(kind):
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected an object, got {type(value).__name__}")
        hints = typing.get_type_hints(kind)
        names = {field.name for field in dataclasses.fields(kind)}
        if value.keys() != names:
            raise ValueError(
                f"{where}: missing {sorted(names - value.keys())}, "
                f"unknown {sorted(value.keys() - names)}"
            )
        return kind(
            **{
                name: decode(hints[name], value[name], f"{where}.{name}")
                for name in names
            }
        )
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in args:
            return None
        (inner,) = (arg for arg in args if arg is not type(None))
        return decode(inner, value, where)
    if origin in (tuple, list):
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected a list, got {type(value).__name__}")
        items = [decode(args[0], item, f"{where}[{i}]") for i, item in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected an object, got {type(value).__name__}")
        return {
            key: decode(args[1], item, f"{where}.{key}") for key, item in value.items()
        }
    if isinstance(kind, type) and issubclass(kind, Enum):
        return kind(value)
    # bool is an int in Python, but never a valid count, seed or duration here.
    valid = (int, float) if kind is float else (kind,)
    if isinstance(value, bool) != (kind is bool) or not isinstance(value, valid):
        raise TypeError(
            f"{where}: expected {kind.__name__}, got {type(value).__name__}"
        )
    return value
