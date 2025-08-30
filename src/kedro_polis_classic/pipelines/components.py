from .registry import ComponentRegistry


# Imputers
@ComponentRegistry.register("SimpleImputer")
def simple_imputer_factory(**kwargs):
    from sklearn.impute import SimpleImputer

    return SimpleImputer(**kwargs)


@ComponentRegistry.register("KNNImputer")
def knn_imputer_factory(**kwargs):
    from sklearn.impute import KNNImputer

    return KNNImputer(**kwargs)


# Reducers
@ComponentRegistry.register("PCA")
def pca_reducer_factory(**kwargs):
    from sklearn.decomposition import PCA

    return PCA(**kwargs)


@ComponentRegistry.register("PaCMAP")
def pacmap_reducer_factory(**kwargs):
    from pacmap import PaCMAP

    return PaCMAP(**kwargs)


@ComponentRegistry.register("LocalMAP")
def localmap_reducer_factory(**kwargs):
    from pacmap import LocalMAP

    return LocalMAP(**kwargs)


# Scalers
@ComponentRegistry.register("NoOpScaler")
def noop_scaler_factory(**kwargs):
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(**kwargs)


@ComponentRegistry.register("StandardScaler")
def standard_scaler_factory(**kwargs):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler(**kwargs)


# Clusterers
@ComponentRegistry.register("KMeans")
def kmeans_clusterer_factory(**kwargs):
    from sklearn.cluster import KMeans

    return KMeans(**kwargs)


@ComponentRegistry.register("DBSCAN")
def hbscan_clusterer_factory(**kwargs):
    from hdbscan import HDBSCAN

    return HDBSCAN(**kwargs)
