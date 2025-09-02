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
            self.mask_ = np.asarray(self.mask, dtype=bool)
            if self.mask_.shape[0] != X.shape[0]:
                raise ValueError("Mask length must match n_samples in X")
        else:
            self.mask_ = np.ones(X.shape[0], dtype=bool)
        return self

    def transform(self, X, y=None):
        X_filtered = X[self.mask_]
        if y is not None:
            y_filtered = np.asarray(y)[self.mask_]
            return X_filtered, y_filtered
        return X_filtered