"""Loading of the two functions teammates plug into the adapters. Pure Python.

A plug-in is named by a string ``"package.module:function"`` given as a ROS parameter, e.g.

    -p executor:=chessbot_arm.motion:pick_and_place      (arm_adapter_node)
    -p detector:=chessbot_vision.detector:detect         (vision_adapter_node)

The module only has to be importable by the node (i.e. installed in the sourced workspace).
"""

from __future__ import annotations

import importlib
from typing import Callable


def load_callable(spec: str) -> Callable:
    """Import ``"package.module:function"`` and return the function."""
    text = str(spec).strip()
    module_name, sep, attr = text.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"expected 'package.module:function', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import module {module_name!r}: {exc}") from exc
    try:
        obj = getattr(module, attr)
    except AttributeError:
        raise ValueError(f"module {module_name!r} has no attribute {attr!r}") from None
    if not callable(obj):
        raise ValueError(f"{spec!r} is not callable")
    return obj
