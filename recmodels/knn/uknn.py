from recmodels.knn import Knn


class Uknn(Knn):
    def __init__(self, df, k_neighbors=10):
        super().__init__(df, user_based=True, k_neighbors=k_neighbors)
        self.similarity_matrix = self._jaccard_similarity_matrix()
