from sklearn.pipeline import Pipeline
from .registry import ComponentRegistry


def build_pipeline_from_params(params):
    steps = []
    for step_name in ["imputer", "reducer", "scaler", "clusterer"]:
        step_config = params[step_name]
        name = step_config.pop("name")
        component = ComponentRegistry.get(name, **step_config)
        steps.append((step_name, component))
    return Pipeline(steps)
