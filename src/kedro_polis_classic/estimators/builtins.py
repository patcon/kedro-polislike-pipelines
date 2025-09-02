from .registry import EstimatorRegistry


# Imputers
@EstimatorRegistry.register("SimpleImputer")
def simple_imputer_factory(**kwargs):
    from sklearn.impute import SimpleImputer

    defaults: dict = dict()
    defaults.update(kwargs)
    return SimpleImputer(**kwargs)


@EstimatorRegistry.register("KNNImputer")
def knn_imputer_factory(**kwargs):
    from sklearn.impute import KNNImputer

    defaults: dict = dict()
    defaults.update(kwargs)
    return KNNImputer(**kwargs)


# Reducers
@EstimatorRegistry.register("PCA")
def pca_reducer_factory(**kwargs):
    from sklearn.decomposition import PCA

    defaults: dict = dict()
    defaults.update(kwargs)
    return PCA(**kwargs)


@EstimatorRegistry.register("UMAP")
def umap_reducer_factory(**kwargs):
    from umap import UMAP

    return UMAP(**kwargs)


@EstimatorRegistry.register("PaCMAP")
def pacmap_reducer_factory(**kwargs):
    from pacmap import PaCMAP

    defaults: dict = dict(n_neighbors=None)
    defaults.update(kwargs)
    return PaCMAP(**defaults)


@EstimatorRegistry.register("LocalMAP")
def localmap_reducer_factory(**kwargs):
    from pacmap import LocalMAP

    defaults: dict = dict(n_neighbors=None)
    defaults.update(kwargs)
    return LocalMAP(**kwargs)


# Scalers
@EstimatorRegistry.register("SparsityAwareScaler")
def sparsity_aware_scaler_factory(**kwargs):
    from reddwarf.sklearn.transformers import SparsityAwareScaler

    return SparsityAwareScaler(**kwargs)

# Clusterers
@EstimatorRegistry.register("KMeans")
def kmeans_clusterer_factory(**kwargs):
    from sklearn.cluster import KMeans

    defaults: dict = dict()
    defaults.update(kwargs)
    return KMeans(**kwargs)


@EstimatorRegistry.register("BestKMeans")
def best_kmeans_clusterer_factory(**kwargs):
    from kedro_polis_classic.sklearn.cluster import BestKMeans

    defaults: dict = dict()
    defaults.update(kwargs)
    return BestKMeans(**kwargs)


@EstimatorRegistry.register("HDBSCAN")
def hbscan_clusterer_factory(**kwargs):
    from hdbscan import HDBSCAN

    defaults: dict = dict()
    defaults.update(kwargs)
    return HDBSCAN(**kwargs)


@EstimatorRegistry.register("HDBSCANFlat")
def hbscanflat_clusterer_factory(**kwargs):
    from ..sklearn.cluster import HDBSCANFlat

    defaults: dict = dict()
    defaults.update(kwargs)
    return HDBSCANFlat(**kwargs)


@EstimatorRegistry.register("BestHDBSCANFlat")
def besthbscanflat_clusterer_factory(**kwargs):
    from ..sklearn.cluster import BestHDBSCANFlat

    defaults: dict = dict()
    defaults.update(kwargs)
    return BestHDBSCANFlat(**kwargs)


# Sample Filters
@EstimatorRegistry.register("SampleMaskFilter")
def sample_mask_filter_factory(**kwargs):
    from ..sklearn.sample_filter import SampleMaskFilter

    defaults: dict = dict()
    defaults.update(kwargs)
    return SampleMaskFilter(**kwargs)


@EstimatorRegistry.register("NoOpTransformer")
def noop_transformer_factory(**kwargs):
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(**kwargs)
