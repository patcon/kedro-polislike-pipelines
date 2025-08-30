from sklearn.pipeline import Pipeline
from .registry import ComponentRegistry

# Import components to ensure they are registered
from . import components


def build_pipeline_from_params(params: dict) -> Pipeline:
    steps = []
    for step_name in ["imputer", "reducer", "scaler", "clusterer"]:
        if step_name in params:
            step_config = params[step_name]
            name = step_config.pop("name")
            component = ComponentRegistry.get(name, **step_config)
            steps.append((step_name, component))
    return Pipeline(steps)
