import abc
import torch
from torch import nn


import numpy as np
import pandas as pd


from recmodels.model import BaseModel
from tqdm import trange
import gc

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def idx_continous(df, col):
    min_idx = df[col].min()
    max_idx = df[col].max()
    n_unique_idx = df[col].nunique()
    if min_idx == 0 and n_unique_idx == max_idx + 1:
        return True
    else:
        return False


def preprocess_df(df):
    new_df = df.copy()
    if not idx_continous(df, "user"):
        new_df.loc[:, "user"] = df["user"].astype("category").cat.codes
    if not idx_continous(df, "item"):
        new_df.loc[:, "item"] = df["item"].astype("category").cat.codes
    new_df[["user", "item"]] = new_df[["user", "item"]].astype(int)
    return new_df


class Knn(BaseModel):
    def forward(self, users, items):
        pass

    def build_interaction_tensor_from_df(self):
        coordinates = self.df.values
        if self.user_based:
            matrix = torch.zeros(size=(self.n_users, self.n_items), device=self.device)
            rows = coordinates[:, 0]
            cols = coordinates[:, 1]
        else:
            matrix = torch.zeros(size=(self.n_items, self.n_users), device=self.device)
            rows = coordinates[:, 1]
            cols = coordinates[:, 0]

        matrix[rows, cols] = 1
        return matrix

    def fit(self, train_df, debug=False):
        new_df = preprocess_df(train_df)
        self.df = new_df[["user", "item"]]
        self.interaction_matrix = self.build_interaction_tensor_from_df()
        self.similarity_matrix = self._jaccard_similarity_matrix()

    def _jaccard_similarity_matrix(self):
        """
        Computes full pairwise Jaccard similarity matrix.
        Returns (n_users, n_users) if user_based, else (n_items, n_items).
        """
        M = self.interaction_matrix if self.user_based else self.interaction_matrix.T
        M = M.float()

        intersection = (
            M @ M.T
        )  # I[i,j] represents the ammount of shared interactions between users i and j (likewise for user_Based=false, for items)
        row_sums = M.sum(dim=1)  # (N,) items/users per row
        row_sums_i = row_sums.unsqueeze(1)  # (N, 1), broadcasts across columns
        row_sums_j = row_sums.unsqueeze(0)  # (1, N), broadcasts across rows
        union = (
            row_sums_i + row_sums_j - intersection
        )  # (N, 1) + (N,1) broadcasted to (N, N)

        similarity = torch.zeros_like(intersection)
        nonzero_mask = union > 0
        similarity[nonzero_mask] = intersection[nonzero_mask] / union[nonzero_mask]

        return similarity

    def _get_batch_neighborhoods(self, col, k):
        sims = self.similarity_matrix[col].clone()
        # ingore main diagonal
        sims[torch.arange(len(col)), col] = -1
        return torch.topk(sims, k).indices

    def score(self, row, col, k):
        # when user_based, row = user, col = items. elsewhise for item_based knn.
        # this documentation assumes user_based=True for simplicity.

        # (n_u, k)
        neighbors = self._get_batch_neighborhoods(row, k)
        # (n_u, k)
        sims = self.similarity_matrix[row.unsqueeze(1), neighbors]

        sum_all_sims = sims.sum(dim=1)  # (n_u,)
        sum_filtered_sims = torch.zeros(len(row), len(col))  # (n_u, n_i)

        for i in range(k):
            neighbor_i = neighbors[:, i]  # (n_u,)
            sim_i = sims[:, i]  # (n_u,)
            # check if each neighbor has interacted with the candidate col (n_u, n_i).
            interacted_i = self.interaction_matrix[neighbor_i][:, col].bool()
            # masks out neighbors without interaction for the particular item.
            sum_filtered_sims += sim_i.unsqueeze(1) * interacted_i

        scores = torch.where(
            sum_all_sims.unsqueeze(1) > 0,
            sum_filtered_sims / sum_all_sims.unsqueeze(1),
            torch.zeros_like(sum_filtered_sims),
        )
        return scores

    def __init__(self, df, user_based, k_neighbors=10):
        super().__init__()
        self.user_based = user_based
        self.k_neighbors = k_neighbors
        self.device = dev
        new_df = preprocess_df(df)
        self.df = new_df[["user", "item"]]
        self.n_users = self.df.user.max() + 1
        self.n_items = self.df.item.max() + 1
        self.interaction_matrix = self.build_interaction_tensor_from_df()
        self.similarity_matrix = self._jaccard_similarity_matrix()

    def recommend(self, users, k, candidates, mask=None):
        row, col = (users, candidates) if self.user_based else (candidates, users)
        scores = self.score(row, col, self.k_neighbors)
        if not self.user_based:
            scores = scores.T
        if mask is not None:
            scores = torch.where(
                mask == 1, scores, torch.full_like(scores, float("-inf"))
            )

        top_scores, top_positions = torch.topk(scores, k, dim=1)
        top_item_ids = candidates[top_positions]

        return top_item_ids, top_scores
