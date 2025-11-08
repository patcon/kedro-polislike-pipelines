import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import pairwise_distances


class CenteredZeroImputer(BaseEstimator, TransformerMixin):
    """
    Imputer that fills NaN values with 0 and centers each user's votes.

    This approach:
    1. Fills missing values (NaN) with 0 (meaning "didn't vote")
    2. Centers each user's votes by subtracting their mean non-NaN value
    3. Results in 0 meaning "neutral" relative to user's own bias

    This keeps unvoted items from driving strong similarity while allowing
    PaCMAP's attraction/repulsion to act on voted items.

    Parameters
    ----------
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.
    """

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        """
        Fit the imputer on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # No fitting required for this imputer
        return self

    def transform(self, X):
        """
        Impute all missing values in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            The imputed dataset.
        """
        X = self._validate_input(X, copy=self.copy)

        # Initialize output array
        X_transformed = np.zeros_like(X)

        for i in range(X.shape[0]):
            row = X[i, :]
            nan_mask = np.isnan(row)

            if np.all(nan_mask):
                # User never voted - all zeros
                X_transformed[i, :] = 0
            else:
                # Step 1: Compute mean of observed votes only (ignoring NaNs)
                user_mean = np.nanmean(row)

                # Step 2: Center the observed votes by subtracting the mean
                centered_row = row - user_mean

                # Step 3: Fill remaining NaNs with 0 (neutral/unvoted)
                centered_row[nan_mask] = 0

                X_transformed[i, :] = centered_row

        return X_transformed

    def _validate_input(self, X, copy):
        """Validate input and optionally make a copy."""
        X = np.array(X, copy=copy, dtype=np.float64)
        return X


def masked_cosine_distance(u, v, missing_mask_u=None, missing_mask_v=None):
    """
    Compute cosine distance between two vectors, considering only co-rated dimensions.

    Parameters
    ----------
    u, v : array-like of shape (n_features,)
        Input vectors.
    missing_mask_u, missing_mask_v : array-like of shape (n_features,), optional
        Boolean masks indicating missing values. If None, assumes no missing values.

    Returns
    -------
    distance : float
        Cosine distance between u and v considering only co-rated dimensions.
        Returns 1.0 (maximum dissimilarity) if no co-rated dimensions exist.
    """
    u = np.asarray(u)
    v = np.asarray(v)

    if missing_mask_u is None:
        missing_mask_u = np.zeros(len(u), dtype=bool)
    if missing_mask_v is None:
        missing_mask_v = np.zeros(len(v), dtype=bool)

    # Create mask for co-rated dimensions (both users have non-missing values)
    co_rated_mask = ~missing_mask_u & ~missing_mask_v

    if co_rated_mask.sum() == 0:
        return 1.0  # Maximum dissimilarity if no overlap

    # Extract co-rated values
    u_corated = u[co_rated_mask]
    v_corated = v[co_rated_mask]

    # Compute cosine distance on co-rated dimensions
    dot_product = np.dot(u_corated, v_corated)
    norm_u = np.linalg.norm(u_corated)
    norm_v = np.linalg.norm(v_corated)

    if norm_u == 0 or norm_v == 0:
        return 1.0  # Maximum dissimilarity if either vector is zero

    cosine_similarity = dot_product / (norm_u * norm_v)
    # Clamp to [-1, 1] to handle numerical errors
    cosine_similarity = np.clip(cosine_similarity, -1, 1)

    return 1 - cosine_similarity


class PaCMAPWithMaskedDistance(BaseEstimator):
    """
    PaCMAP wrapper that supports masked distance computation for vote matrices.

    This wrapper:
    1. Fills NaN values with a sentinel value (default: 0)
    2. Computes nearest neighbors using masked distance (only co-rated dimensions)
    3. Uses PaCMAP's user-specified nearest neighbor functionality
    4. Provides the same interface as regular PaCMAP but handles missing values semantically

    This is the preferred approach for deliberation data as it handles missingness
    by only considering overlapping votes when computing similarity.

    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors to consider for nearest neighbor pairs.
    fill_value : float, default=0
        The value to use for filling missing values.
    distance_func : callable, default=masked_cosine_distance
        Distance function to use. Should accept two vectors and optional missing masks.
    pacmap_kwargs : dict, optional
        Additional keyword arguments to pass to the underlying PaCMAP instance.
    """

    def __init__(self, n_neighbors=10, fill_value=0, distance_func=masked_cosine_distance, **pacmap_kwargs):
        # Handle None value for n_neighbors (use PaCMAP's default of 10)
        self.n_neighbors = n_neighbors if n_neighbors is not None else 10
        self.fill_value = fill_value
        self.distance_func = distance_func
        self.pacmap_kwargs = pacmap_kwargs

    def fit(self, X, y=None):
        """
        Fit PaCMAP with masked distance computation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        import pacmap

        X = np.array(X, dtype=np.float32)  # PaCMAP expects float32

        # Store the missing value mask for distance computation
        self.missing_mask_ = np.isnan(X)

        # Fill NaN with the specified fill_value
        X_filled = np.where(np.isnan(X), self.fill_value, X)

        # Compute pairwise distance matrix using masked distances
        n_samples = X_filled.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    dist = self.distance_func(
                        X_filled[i], X_filled[j],
                        self.missing_mask_[i], self.missing_mask_[j]
                    )
                    distance_matrix[i, j] = dist

        # Find nearest neighbors based on masked distances
        nbrs = np.zeros((n_samples, self.n_neighbors), dtype=np.int32)
        for i in range(n_samples):
            # Get indices of nearest neighbors (excluding self)
            distances_i = distance_matrix[i, :]
            distances_i[i] = np.inf  # Exclude self
            nearest_indices = np.argsort(distances_i)[:self.n_neighbors]
            nbrs[i, :] = nearest_indices

        # Create scaled distances (set to 1.0 as no scaling needed)
        scaled_dist = np.ones((n_samples, self.n_neighbors), dtype=np.float32)

        # Generate neighbor pairs using PaCMAP's utility function
        pair_neighbors = pacmap.sample_neighbors_pair(
            X_filled, scaled_dist, nbrs, np.int32(self.n_neighbors)
        )

        # Create PaCMAP instance with user-specified neighbors
        pacmap_params = {
            'n_neighbors': self.n_neighbors,
            'pair_neighbors': pair_neighbors,
            **self.pacmap_kwargs
        }
        self.pacmap_ = pacmap.PaCMAP(**pacmap_params)

        # Fit and transform the data
        self.embedding_ = self.pacmap_.fit_transform(X_filled)

        return self

    def transform(self, X):
        """
        Transform new data using the fitted PaCMAP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data.
        """
        if not hasattr(self, 'pacmap_'):
            raise RuntimeError("You must fit the model before calling transform.")

        X = np.array(X, dtype=np.float32)
        X_filled = np.where(np.isnan(X), self.fill_value, X)

        return self.pacmap_.transform(X_filled)

    def fit_transform(self, X, y=None):
        """
        Fit the model and transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X, y).embedding_