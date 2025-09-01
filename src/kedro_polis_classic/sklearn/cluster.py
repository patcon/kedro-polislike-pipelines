from sklearn.cluster import KMeans
from .model_selection import BestClusterer
from sklearn.base import BaseEstimator, ClusterMixin
from hdbscan.flat import HDBSCAN_flat


class BestKMeans(BestClusterer):
    def __init__(self, k_bounds=(2, 5), **kmeans_params):
        super().__init__(base_estimator=KMeans(**kmeans_params), k_bounds=k_bounds)


class BestHDBSCANFlat(BestClusterer):
    def __init__(self, k_bounds=(2, 5), **hdbscan_params):
        # Create base estimator with default n_clusters, BestClusterer will override it
        base_estimator = HDBSCANFlat(**hdbscan_params)
        super().__init__(base_estimator=base_estimator, k_bounds=k_bounds)


class HDBSCANFlat(BaseEstimator, ClusterMixin):
    """
    A scikit-learn compatible estimator wrapper for HDBSCAN_flat.

    This wrapper allows `HDBSCAN_flat` to be used in sklearn-style pipelines
    with familiar methods (`fit`, `fit_predict`, `predict`) and parameter
    management (`get_params`, `set_params`).

    Parameters
    ----------
    n_clusters : int, default=None
        The number of clusters to extract from the HDBSCAN hierarchy.

    cluster_selection_method : {"leaf", "eom"}, default="eom"
        The method used to extract clusters from the condensed tree.

    min_cluster_size : int, default=5
        The minimum size of clusters; smaller clusters will be considered noise.

    kwargs : dict
        Additional keyword arguments passed directly to `HDBSCAN_flat`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to `fit`.

    clusterer_ : HDBSCAN_flat
        The underlying fitted HDBSCAN_flat object.
    """

    def __init__(
        self,
        n_clusters=None,
        cluster_selection_method="eom",
        min_cluster_size=5,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.cluster_selection_method = cluster_selection_method
        self.min_cluster_size = min_cluster_size
        self.kwargs = kwargs  # keep all extra params here

    def fit(self, X, y=None):
        """Fit HDBSCAN_flat to data."""
        self.clusterer_ = HDBSCAN_flat(
            X,
            n_clusters=self.n_clusters,
            cluster_selection_method=self.cluster_selection_method,
            min_cluster_size=self.min_cluster_size,
            **self.kwargs,
        )
        self.labels_ = self.clusterer_.labels_
        return self

    def fit_predict(self, X, y=None, **kwargs):
        """Fit to data and return cluster labels."""
        self.fit(X, y, **kwargs)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for X.

        Notes
        -----
        HDBSCAN is not naturally inductive; this method uses
        the approximate prediction available via the fitted clusterer.
        """
        if not hasattr(self, "clusterer_"):
            raise RuntimeError("You must fit the model before calling predict.")
        return self.clusterer_.approximate_predict(X)[0]
