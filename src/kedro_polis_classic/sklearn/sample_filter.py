import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SampleMaskFilter(BaseEstimator, TransformerMixin):
    def __init__(self, mask=None):
        """
        Parameters
        ----------
        mask : array-like of shape (n_samples,), default=None
            Boolean mask specifying which samples to keep. If None, keep all samples.
        """
        self.mask = mask

    def fit(self, X, y=None):
        # nothing to learn, just check mask
        if self.mask is not None:
            # TODO: why this hack here? .T[0]
            self.mask_ = np.asarray(self.mask, dtype=bool).T[0]
            if self.mask_.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Mask length ({self.mask_.shape[0]}) must match n_samples in X ({X.shape[0]})"
                )
        else:
            self.mask_ = np.ones(X.shape[0], dtype=bool)
        return self

    def transform(self, X, y=None):
        # Apply mask to rows (samples/participants)
        if hasattr(X, "iloc"):
            # For pandas DataFrames
            X_filtered = X.iloc[self.mask_]
        else:
            # For numpy arrays, use boolean indexing on rows only
            X_filtered = X[self.mask_]

        if y is not None:
            y_filtered = np.asarray(y)[self.mask_]
            return X_filtered, y_filtered
        return X_filtered
