from ..builder import build_pipeline_from_params


def run_component_node(X, params, step_name):
    """
    Run a single pipeline step while passing full params dict.
    This allows Kedro-viz to show all sub-parameters.
    """
    step_config = params[step_name]
    pipeline = build_pipeline_from_params({step_name: step_config})
    return pipeline.fit_transform(X)
