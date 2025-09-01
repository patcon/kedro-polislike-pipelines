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
    A scikit-learn compatible estimator for flat HDBSCAN clustering.

    This wraps the functional :func:`HDBSCAN_flat` API into a proper
    estimator so that it can be used in scikit-learn pipelines and
    parameter search tools.

    Parameters
    ----------
    n_clusters : int, default=None
        Desired number of clusters. If ``None``, clustering is determined by
        the standard HDBSCAN procedure with ``cluster_selection_epsilon``.

    cluster_selection_epsilon : float, default=0.0
        Core-distance below which to stop splitting clusters. This parameter
        is ignored if ``n_clusters`` is supplied.

    clusterer : HDBSCAN, default=None
        If provided, reuse or copy this pre-trained HDBSCAN instance.
        Can be modified in-place or deep-copied depending on ``inplace``.

    inplace : bool, default=False
        If ``True`` and ``clusterer`` is supplied, modify the clusterer
        in-place. Otherwise, a copy will be used.

    **kwargs : dict
        Additional keyword arguments passed directly to :func:`HDBSCAN_flat`
        (and thus the underlying :class:`hdbscan.HDBSCAN`).

    Attributes
    ----------
    clusterer_ : HDBSCAN
        The trained HDBSCAN instance after applying flat clustering.

    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the training set.

    probabilities_ : ndarray of shape (n_samples,)
        Soft cluster membership strength for each point in the training set.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    >>> clusterer = HDBSCANFlat(n_clusters=3, min_cluster_size=5)
    >>> labels = clusterer.fit_predict(X)
    >>> set(labels)
    {0, 1, 2}
    """

    def __init__(
        self,
        n_clusters=None,
        cluster_selection_epsilon=0.0,
        clusterer=None,
        inplace=False,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.clusterer = clusterer
        self.inplace = inplace
        # keep all HDBSCAN init params as attributes so they show up in get_params/set_params
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y=None):
        """
        Fit HDBSCAN_flat with flat clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : HDBSCANFlat
            Fitted estimator.
        """
        # Gather all keyword args except the explicit params
        kwargs = {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in ["n_clusters", "cluster_selection_epsilon", "clusterer", "inplace"]
        }

        self.clusterer_ = HDBSCAN_flat(
            X,
            n_clusters=self.n_clusters,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            clusterer=self.clusterer,
            inplace=self.inplace,
            **kwargs,
        )
        return self

    def fit_predict(self, X, y=None, **kwargs):
        """Fit the model to data and return cluster labels."""
        self.fit(X, **kwargs)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted cluster labels.
        """
        if not hasattr(self, "clusterer_"):
            raise ValueError(
                "This HDBSCANFlat instance is not fitted yet. Call 'fit' first."
            )
        return self.clusterer_.approximate_predict(X)[0]

    @property
    def labels_(self):
        return self.clusterer_.labels_

    @property
    def probabilities_(self):
        return self.clusterer_.probabilities_
