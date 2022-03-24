from scipy.spatial.distance import cdist


class QuarticKernel:
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, X, Y):
        dists = cdist(X / self.radius, Y / self.radius, metric='sqeuclidean')
        zero_idx = dists > 1
        K = 15 / 16 * (1 - dists) ** 2
        K[zero_idx] = 0
        return K
