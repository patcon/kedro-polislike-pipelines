"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
# from {{cookiecutter.python_package}}.hooks import ProjectHooks

# Hooks are executed in a Last-In-First-Out (LIFO) order.
# HOOKS = (ProjectHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
from kedro.config import OmegaConfigLoader  # noqa: E402
from urllib.parse import urlparse


def extract_polis_id_from_url(
    polis_url: str, fallback: str | None = None
) -> str | None:
    """
    Extract polis_id from a polis_url for use in file paths.

    Args:
        polis_url: URL in format "https://polis.example.com/{polis_convo_id}"
                  or "https://polis.example.com/report/{polis_report_id}"
        fallback: Fallback value if URL extraction fails

    Returns:
        The polis_id extracted from the URL, fallback value, or None if both fail
    """
    if not polis_url:
        return fallback

    parsed = urlparse(polis_url)
    # Remove leading/trailing slashes and split path
    path_parts = parsed.path.strip("/").split("/")

    if not path_parts or not path_parts[-1]:
        return fallback

    # Check if it's a report URL
    if len(path_parts) >= 2 and path_parts[-2] == "report":
        polis_id = path_parts[-1]
        # Ensure report IDs start with 'r'
        if not polis_id.startswith("r"):
            polis_id = f"r{polis_id}"
    else:
        # Direct conversation URL
        polis_id = path_parts[-1]

    return polis_id


CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {
        "extract_polis_id": extract_polis_id_from_url,
    },
    #       "config_patterns": {
    #           "spark" : ["spark*/"],
    #           "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
    #       }
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
