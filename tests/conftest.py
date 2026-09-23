"""Keep the test suite independent from a developer's local ``.env``.

The application intentionally reads ``.env`` at runtime.  Tests, however,
should retain deterministic defaults even when a developer has configured a
remote embedding or generation provider for local development.  Individual
tests that need an env file continue to opt in with ``Settings(_env_file=...)``.
"""

from __future__ import annotations


def pytest_configure() -> None:
    """Disable implicit dotenv loading for the current pytest process only."""
    from app.core.config import Settings, get_settings

    # Pydantic Settings reads ``model_config`` when each Settings instance is
    # constructed.  Updating this process-local class configuration before
    # test collection prevents the developer's .env from changing globals that
    # modules create during import, while explicit ``_env_file=...`` calls are
    # unaffected.
    Settings.model_config = {**Settings.model_config, "env_file": None}
    get_settings.cache_clear()

