from ..builder import build_pipeline_from_params


def run_pipeline_node(X, params):
    pipe = build_pipeline_from_params(params)
    pipe.fit(X)
    return pipe
