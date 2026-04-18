"""Process- and cluster-wide GDAL configuration for pyramids.

``pyramids.configure(...)`` applies a dict of GDAL config options to
the current process via :func:`gdal.SetConfigOption`, and — when a
``dask.distributed.Client`` is passed — also registers a worker
plugin that replays the same config on every worker when they start.

Why: odc-stac's Pangeo benchmark showed that a single call to the
equivalent ``configure_rio`` cut a 68 s STAC load to 3.75 s (18×),
purely by setting cloud-friendly GDAL env options. pyramids users
shouldn't have to discover these individually — one call opts in.

The ``cloud_defaults=True`` preset mirrors odc-stac's
``GDAL_CLOUD_DEFAULTS`` plus two extras for HTTP range-request
performance. Individual keys can be overridden via ``**gdal_options``.

The ``dask.distributed`` import is gated — importing this module
does not pull dask. Only when ``client`` is non-None does the
plugin path engage.
"""

from __future__ import annotations

import logging
from typing import Any

from osgeo import gdal

logger = logging.getLogger(__name__)


GDAL_CLOUD_DEFAULTS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
}
"""Default GDAL config options for cloud-hosted COG / NetCDF reads.

Ported from odc-loader's ``GDAL_CLOUD_DEFAULTS`` with two additions
(``GDAL_HTTP_MULTIRANGE`` and ``VSI_CACHE``) that stackstac's
``DEFAULT_GDAL_ENV`` enables by default. Applied by
:func:`configure` when ``cloud_defaults=True``.
"""


def _expand_credentials(
    prefix: str, creds: dict[str, Any] | None
) -> dict[str, str]:
    """Expand a credentials dict into ``PREFIX_KEY=value`` entries.

    For example ``_expand_credentials("AWS", {"aws_unsigned": True})``
    returns ``{"AWS_NO_SIGN_REQUEST": "YES"}``.
    """
    out: dict[str, str] = {}
    if creds:
        for key, value in creds.items():
            if key == "aws_unsigned" and value:
                out["AWS_NO_SIGN_REQUEST"] = "YES"
                continue
            out[f"{prefix}_{key.upper()}"] = str(value)
    return out


def configure(
    *,
    cloud_defaults: bool = False,
    aws: dict[str, Any] | None = None,
    gs: dict[str, Any] | None = None,
    azure: dict[str, Any] | None = None,
    client: Any = None,
    **gdal_options: Any,
) -> dict[str, str]:
    """Apply GDAL config options to this process + (optionally) a dask cluster.

    Args:
        cloud_defaults: If True, apply :data:`GDAL_CLOUD_DEFAULTS`
            first. Individual keys can still be overridden via
            ``**gdal_options``.
        aws: AWS-namespaced credentials dict; keys are upcased and
            prefixed with ``AWS_``. The special key ``aws_unsigned``
            (truthy) expands to ``AWS_NO_SIGN_REQUEST=YES``.
        gs: Google Cloud Storage credentials dict, prefixed ``GS_``.
        azure: Azure credentials dict, prefixed ``AZURE_``.
        client: Optional :class:`dask.distributed.Client`. When given,
            a worker plugin is registered so every worker receives the
            same configuration on startup.
        **gdal_options: Any additional GDAL config options as
            keyword arguments (they override ``cloud_defaults``).

    Returns:
        dict[str, str]: The effective key->value config dict that was
        applied. Useful for logging and for round-tripping to workers.

    Examples:
        - Apply cloud defaults locally (no cluster):
            ```python
            >>> from pyramids import configure
            >>> applied = configure(cloud_defaults=True)
            >>> applied["GDAL_DISABLE_READDIR_ON_OPEN"]
            'EMPTY_DIR'

            ```
        - Override a key from cloud_defaults:
            ```python
            >>> from pyramids import configure
            >>> applied = configure(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="3")
            >>> applied["GDAL_HTTP_MAX_RETRY"]
            '3'

            ```
        - Expand AWS credentials via the shortcut:
            ```python
            >>> from pyramids import configure
            >>> applied = configure(aws={"aws_unsigned": True})
            >>> applied["AWS_NO_SIGN_REQUEST"]
            'YES'

            ```
    """
    env: dict[str, str] = {}
    if cloud_defaults:
        env.update(GDAL_CLOUD_DEFAULTS)
    env.update(_expand_credentials("AWS", aws))
    env.update(_expand_credentials("GS", gs))
    env.update(_expand_credentials("AZURE", azure))
    for key, value in gdal_options.items():
        env[key] = str(value)

    for key, value in env.items():
        gdal.SetConfigOption(key, value)
    logger.debug("pyramids.configure applied %d GDAL option(s)", len(env))

    if client is not None:
        _register_worker_plugin(client, env)

    return env


def _register_worker_plugin(client: Any, env: dict[str, str]) -> None:
    """Register a WorkerPlugin that replays ``env`` on each dask worker."""
    from dask.distributed import WorkerPlugin

    class PyramidsConfigPlugin(WorkerPlugin):
        """Apply pyramids.configure env on every dask worker."""

        name = "pyramids-configure"

        def __init__(self, env: dict[str, str]) -> None:
            self._env = env

        def setup(self, worker: Any) -> None:  # pragma: no cover - runs on workers
            for key, value in self._env.items():
                gdal.SetConfigOption(key, value)

    plugin = PyramidsConfigPlugin(env)
    try:
        client.register_plugin(plugin, name="pyramids-configure")
    except AttributeError:  # pragma: no cover - old dask API
        client.register_worker_plugin(plugin, name="pyramids-configure")
