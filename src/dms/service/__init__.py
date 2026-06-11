"""HTTP service adapters for DMS."""

from typing import Any

__all__ = [
    "DMSServiceSettings",
    "create_app",
    "default_service_settings",
    "references_from_config",
    "references_from_settings",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from dms.service.fastapi_app import (
            DMSServiceSettings,
            create_app,
            default_service_settings,
            references_from_config,
            references_from_settings,
        )

        return {
            "DMSServiceSettings": DMSServiceSettings,
            "create_app": create_app,
            "default_service_settings": default_service_settings,
            "references_from_config": references_from_config,
            "references_from_settings": references_from_settings,
        }[name]
    raise AttributeError(name)
