from sklearn.cluster import KMeans
from .model_selection import BestClusterer


class BestKMeans(BestClusterer):
    def __init__(self, k_bounds=(2, 5), **kmeans_params):
        super().__init__(base_estimator=KMeans(**kmeans_params), k_bounds=k_bounds)
