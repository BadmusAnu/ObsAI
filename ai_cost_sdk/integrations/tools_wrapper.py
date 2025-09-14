"""Generic tool wrapper."""

from __future__ import annotations

from functools import wraps

from .. import middleware


def priced_tool(name: str, unit_price: float | None = None, vendor: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with middleware.tool_call(name=name, vendor=vendor, unit_price=unit_price):
                return func(*args, **kwargs)

        return wrapper

    return decorator
