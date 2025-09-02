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
    all_pipeline_keys = params.get("pipelines", {}).keys()
    # Filter out _defaults as it's not a real pipeline
    experimental_pipeline_names = [key for key in all_pipeline_keys if key != "_defaults"]

    pipelines = {}

    # Add shorthand name for original polis pipeline.
    pipelines["polis_classic"] = experiment_pipeline.create_pipeline("mean_pca_bestkmeans")

    # Add experimental pipelines using iteration
    for name in experimental_pipeline_names:
        pipelines[name] = experiment_pipeline.create_pipeline(name)

    # Add legacy pipelines to end of list.
    pipelines["polis_legacy"] = polis_pipeline.create_pipeline()

    return pipelines
