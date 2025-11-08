from typing import Any, cast
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.metrics import silhouette_score


class BestClusterer(BaseEstimator, ClusterMixin):
    """
    Meta-estimator that selects the best number of clusters using silhouette score.
    Works with any estimator that has an `n_clusters` parameter.

    Parameters
    ----------
    base_estimator : estimator
        The clusterer to wrap (e.g. KMeans, PolisKMeans, HDBSCANFlat).
    k_bounds : (int, int), default=(2, 5)
        The inclusive range of k values to search [min_k, max_k].
    """

    def __init__(self, base_estimator, k_bounds=(2, 5)):
        self.base_estimator = base_estimator
        self.k_bounds = k_bounds

    def fit(self, X, y=None):
        self.best_score_ = -1
        self.best_k_ = None
        self.best_estimator_ = None

        min_k, max_k = self.k_bounds
        for k in range(min_k, max_k + 1):
            est = cast(Any, clone(self.base_estimator))
            est.set_params(n_clusters=k)
            labels = est.fit_predict(X)

            # silhouette requires >1 cluster
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X, labels)
            if score > self.best_score_:
                self.best_score_ = score
                self.best_k_ = k
                self.best_estimator_ = cast(Any, clone(est)).fit(X)

        if self.best_estimator_ is not None:
            self.labels_ = self.best_estimator_.labels_
        return self

    def fit_predict(self, X, y=None, **kwargs):
        return self.fit(X).labels_

    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("Must call fit before predict")
        return self.best_estimator_.predict(X)

    def transform(self, X):
        if self.best_estimator_ is None:
            raise ValueError("Must call fit before transform")
        return self.best_estimator_.transform(X)

    def score(self, X, y=None):
        if self.best_estimator_ is None:
            raise ValueError("Must call fit before score")
        labels = self.best_estimator_.predict(X)
        return silhouette_score(X, labels)
