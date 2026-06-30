"""Compatibility implementations retained during power namespace adoption."""

from importlib import import_module
import sys
from types import ModuleType


def bind_module(public_name: str, impl_name: str, namespace: dict) -> None:
    impl = import_module(impl_name)
    for name, value in impl.__dict__.items():
        if name.startswith("__") and name not in {"__all__", "__doc__"}:
            continue
        namespace[name] = value
    namespace["_impl"] = impl

    class _MgrShim(ModuleType):
        def __getattr__(self, name: str):
            return getattr(impl, name)

        def __setattr__(self, name: str, value):
            setattr(impl, name, value)
            super().__setattr__(name, value)

        def __delattr__(self, name: str):
            if hasattr(impl, name):
                delattr(impl, name)
            super().__delattr__(name)

    sys.modules[public_name].__class__ = _MgrShim
