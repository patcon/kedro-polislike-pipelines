from sklearn.pipeline import Pipeline
from ..estimators.registry import EstimatorRegistry

# Import estimators to ensure they are registered
from ..estimators import builtins


def build_pipeline_from_params(params: dict) -> Pipeline:
    steps = []
    for step_name in ["imputer", "reducer", "scaler", "filter", "clusterer"]:
        if step_name in params:
            step_config = params[step_name].copy()  # Make a copy to avoid mutating original
            # Use 'estimator' if available, otherwise fall back to 'name' for backward compatibility
            estimator_name = step_config.pop("estimator", step_config.pop("name", None))
            if estimator_name is None:
                raise ValueError(f"No 'estimator' or 'name' field found in {step_name} config")
            component = EstimatorRegistry.get(estimator_name, **step_config)
            steps.append((step_name, component))
    return Pipeline(steps)
