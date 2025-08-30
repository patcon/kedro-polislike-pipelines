from ..builder import build_pipeline_from_params
import copy


def run_component_node(X, params, step_name):
    """
    Runs a single pipeline component.
    X: input features
    params: full nested pipeline parameters dict
    step_name: which step to build (imputer/reducer/scaler/clusterer)
    """
    # copy to avoid mutating params
    step_config = copy.deepcopy(params[step_name])
    pipeline = build_pipeline_from_params({step_name: step_config})

    return pipeline.fit_transform(X)
