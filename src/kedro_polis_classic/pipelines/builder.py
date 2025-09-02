from sklearn.pipeline import Pipeline
from ..estimators.registry import EstimatorRegistry

# Import estimators to ensure they are registered
from ..estimators import builtins


def build_pipeline_from_params(params: dict) -> Pipeline:
    steps = []
    for step_name in ["imputer", "reducer", "scaler", "filter", "clusterer"]:
        if step_name in params:
            step_config = params[step_name]
            name = step_config.pop("name")
            component = EstimatorRegistry.get(name, **step_config)
            steps.append((step_name, component))
    return Pipeline(steps)
