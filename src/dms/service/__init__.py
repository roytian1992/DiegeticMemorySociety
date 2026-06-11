"""HTTP service adapters for DMS."""

from typing import Any

__all__ = ["DMSServiceSettings", "create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from dms.service.fastapi_app import DMSServiceSettings, create_app

        return {"DMSServiceSettings": DMSServiceSettings, "create_app": create_app}[name]
    raise AttributeError(name)
