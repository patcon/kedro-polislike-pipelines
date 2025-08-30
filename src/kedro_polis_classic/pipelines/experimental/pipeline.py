from kedro.pipeline import Pipeline, node
from ..registry import ComponentRegistry
import copy


def create_pipeline(params_key="pipeline_experiment") -> Pipeline:
    steps = ["imputer", "reducer", "scaler", "clusterer"]

    nodes = []
    input_dataset = "features_train"
    params = f"params:{params_key}"

    for step_name in steps:
        # Each step will get its config from params at runtime
        nodes.append(
            node(
                func=run_component_node,
                inputs=[input_dataset, params, step_name],
                outputs=f"{step_name}_output",
                name=f"{step_name}_node",
            )
        )
        input_dataset = f"{step_name}_output"

    # Optionally add a final node to wrap everything back into a Pipeline
    nodes.append(
        node(
            func=collect_pipeline_node,
            inputs=[params] + [f"{s}_output" for s in steps],
            outputs="trained_pipeline_experiment",
            name="final_pipeline_node",
        )
    )

    return Pipeline(nodes)


def run_component_node(X, params, step_name):
    """Runs a single component on X given params dict and step_name"""
    step_config = copy.deepcopy(params[step_name])
    name = step_config.pop("name")
    component = ComponentRegistry.get(name, **step_config)

    # Fit and transform (or just transform if already fitted)
    return component.fit_transform(X)


def collect_pipeline_node(params, *step_outputs):
    """Optionally collect the fitted pipeline objects into one sklearn Pipeline"""
    from sklearn.pipeline import Pipeline

    steps = []
    for step_name in ["imputer", "reducer", "scaler", "clusterer"]:
        step_config = copy.deepcopy(params[step_name])
        name = step_config.pop("name")
        component = ComponentRegistry.get(name, **step_config)
        steps.append((step_name, component))

    return Pipeline(steps)
