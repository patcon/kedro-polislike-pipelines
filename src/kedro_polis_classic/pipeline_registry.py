from kedro.pipeline import Pipeline
from kedro.config import OmegaConfigLoader
from .pipelines.polis import pipeline as polis_pipeline
from .pipelines.experimental import pipeline as experiment_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    # Load configuration to get experimental pipeline names dynamically
    config_loader = OmegaConfigLoader(
        conf_source="conf", base_env="base", default_run_env="local"
    )

    # Load parameters to get pipeline keys from parameters_experimental.yml
    params = config_loader["parameters"]
    experimental_pipeline_names = list(params.get("pipelines", {}).keys())
    # Create base pipelines
    pipelines = {
        "polis": polis_pipeline.create_pipeline(),
        "polis_classic": experiment_pipeline.create_pipeline("mean_pca_kmeans"),
        "__default__": polis_pipeline.create_pipeline(),
    }

    # Add experimental pipelines using iteration
    for name in experimental_pipeline_names:
        pipelines[name] = experiment_pipeline.create_pipeline(name)

    return pipelines
