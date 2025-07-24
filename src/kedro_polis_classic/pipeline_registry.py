from kedro.pipeline import Pipeline, Node

def foo():
    return "dummy"

def register_pipelines():
    return {"__default__": Pipeline([ Node(foo, None, "dummy_output") ]) }
