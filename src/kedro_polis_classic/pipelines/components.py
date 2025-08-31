from .registry import ComponentRegistry


# Imputers
@ComponentRegistry.register("SimpleImputer")
def simple_imputer_factory(**kwargs):
    from sklearn.impute import SimpleImputer

    defaults: dict = dict()
    defaults.update(kwargs)
    return SimpleImputer(**kwargs)


@ComponentRegistry.register("KNNImputer")
def knn_imputer_factory(**kwargs):
    from sklearn.impute import KNNImputer

    defaults: dict = dict()
    defaults.update(kwargs)
    return KNNImputer(**kwargs)


# Reducers
@ComponentRegistry.register("PCA")
def pca_reducer_factory(**kwargs):
    from sklearn.decomposition import PCA

    defaults: dict = dict()
    defaults.update(kwargs)
    return PCA(**kwargs)


@ComponentRegistry.register("PaCMAP")
def pacmap_reducer_factory(**kwargs):
    from pacmap import PaCMAP

    defaults: dict = dict(n_neighbors=None)
    defaults.update(kwargs)
    return PaCMAP(**defaults)


@ComponentRegistry.register("LocalMAP")
def localmap_reducer_factory(**kwargs):
    from pacmap import LocalMAP

    defaults: dict = dict(n_neighbors=None)
    defaults.update(kwargs)
    return LocalMAP(**kwargs)


# Scalers
@ComponentRegistry.register("NoOpScaler")
def noop_scaler_factory(**kwargs):
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(**kwargs)


@ComponentRegistry.register("StandardScaler")
def standard_scaler_factory(**kwargs):
    from sklearn.preprocessing import StandardScaler

    defaults: dict = dict()
    defaults.update(kwargs)
    return StandardScaler(**kwargs)


# Clusterers
@ComponentRegistry.register("KMeans")
def kmeans_clusterer_factory(**kwargs):
    from sklearn.cluster import KMeans

    defaults: dict = dict()
    defaults.update(kwargs)
    return KMeans(**kwargs)


@ComponentRegistry.register("DBSCAN")
def hbscan_clusterer_factory(**kwargs):
    from hdbscan import HDBSCAN

    defaults: dict = dict()
    defaults.update(kwargs)
    return HDBSCAN(**kwargs)
