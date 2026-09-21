from torch import nn
from recmodels.knn import Knn


class Iknn(Knn):

    def score(self, users, items, k=10):
        return super().score(row=users, col=items, k=k)

    def __init__(self, df):
        super().__init__(df, user_based=False)

        self.similarity_matrix = self.jaccard_similarity_matrix()

    def recommend(self, users, items, k=10, top_n=None):
        return super().recommend(row=users, col=items, k=k, top_n=top_n)
