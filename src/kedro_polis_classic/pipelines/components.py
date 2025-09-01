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


# Clusterers
@ComponentRegistry.register("KMeans")
def kmeans_clusterer_factory(**kwargs):
    from sklearn.cluster import KMeans

    defaults: dict = dict()
    defaults.update(kwargs)
    return KMeans(**kwargs)


@ComponentRegistry.register("BestKMeans")
def best_kmeans_clusterer_factory(**kwargs):
    from kedro_polis_classic.sklearn.cluster import BestKMeans

    defaults: dict = dict()
    defaults.update(kwargs)
    return BestKMeans(**kwargs)


@ComponentRegistry.register("HDBSCAN")
def hbscan_clusterer_factory(**kwargs):
    from hdbscan import HDBSCAN

    defaults: dict = dict()
    defaults.update(kwargs)
    return HDBSCAN(**kwargs)


@ComponentRegistry.register("HDBSCANFlat")
def hbscanflat_clusterer_factory(**kwargs):
    from ..sklearn.cluster import HDBSCANFlat

    defaults: dict = dict()
    defaults.update(kwargs)
    return HDBSCANFlat(**kwargs)


@ComponentRegistry.register("BestHDBSCANFlat")
def besthbscanflat_clusterer_factory(**kwargs):
    from ..sklearn.cluster import BestHDBSCANFlat

    defaults: dict = dict()
    defaults.update(kwargs)
    return BestHDBSCANFlat(**kwargs)
