from kedro.config import OmegaConfigLoader


def load_pipelines_config() -> dict:
    """
    Load pipeline configurations using OmegaConfigLoader.

    Returns:
        Dictionary containing all pipeline configurations from parameters_experimental.yml
    """
    config_loader = OmegaConfigLoader(
        conf_source="conf", base_env="base", default_run_env="local"
    )
    params = config_loader["parameters"]
    return params.get("pipelines", {})
