import pandas as pd
import functools
import inspect


def process_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """Ensure x is a pandas Series (e.g., extract first column of a single-column DataFrame)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame")
        return x.iloc[:, 0]
    elif isinstance(x, pd.Series):
        return x
    else:
        raise TypeError("Expected Series or single-column DataFrame")


def ensure_series(argname: str):
    """Decorator to apply `process_series` to a named argument."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind args and kwargs to named parameters
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if argname not in bound.arguments:
                raise ValueError(
                    f"Argument '{argname}' not found when calling {func.__name__}"
                )

            bound.arguments[argname] = process_series(bound.arguments[argname])
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
