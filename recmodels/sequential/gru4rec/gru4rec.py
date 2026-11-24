import numpy as np
import torch


from recmodels.sequential.gru4rec.gru4rec_hidasi import GRU4RecHidasi
from recmodels.model import BaseModel

from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    ITEM_COL,
    TIMESTAMP_COL,
)


class Gru4rec(BaseModel):
    def __init__(
        self,
        params_dict,
        dev,
        user_col=USER_COL,
        item_col=ITEM_COL,
        timestamp_col=TIMESTAMP_COL,
        session_size=50,
    ):
        super().__init__()
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.predictor = GRU4RecHidasi(device=dev, **params_dict)
        self.session_size = session_size
    
    def forward(self, users, items):
        raise NotImplementedError(
            "Gru4rec scores from user histories; use predict/recommend instead."
        )

    def _set_train_df(self, train_df):
        df = train_df.sort_values([self.user_col, self.timestamp_col], kind="stable")
        chunk = df.groupby(self.user_col).cumcount() // self.session_size
        df["sessionId"] = df.groupby([self.user_col, chunk]).ngroup()
        df = df[df.groupby("sessionId")[self.item_col].transform("size") >= 2]
        return df

    def fit(self, train_df):
        self.session_based_df = self._set_train_df(train_df)
        self.predictor.fit(
            self.session_based_df,
            item_key=self.item_col,
            session_key="sessionId",
            time_key=self.timestamp_col,
        )

    def _to_model_idx(self, ids):
        """Raw ids -> (internal indices of the known ids, boolean mask of which ids are known)."""
        idx = self.predictor.data_iterator.itemidmap.reindex(
            ids
        ).values  # NaN for unknown ids
        known = ~np.isnan(idx)
        return idx[known].astype(np.int64), known

    def map_to_model_idx(self, items):
        return [self._to_model_idx(item)[0] for item in items]

    def recommend(self, users, k, candidates, history, mask=None, ):
        scores = self.predict(users, history)  # (n_users, n_model_items)
        dev = scores.device
        n_users, n_candidates = len(users), len(candidates)

        # model columns -> candidate positions (candidates unknown to the model stay -inf)
        model_cols, known = self._to_model_idx(candidates.cpu().numpy())
        cand_pos = torch.tensor(np.flatnonzero(known), device=dev)
        model_cols = torch.tensor(model_cols, device=dev)

        prediction_matrix = torch.full(
            (n_users, n_candidates), float("-inf"), device=dev
        )
        prediction_matrix[:, cand_pos] = scores[:, model_cols]

        if mask is not None:
            prediction_matrix = torch.where(
                mask.to(dev) == 1, prediction_matrix, float("-inf")
            )

        top_scores, top_idx = prediction_matrix.topk(min(k, n_candidates), dim=1)
        candidate_ids = candidates[top_idx.to(candidates.device)]

        return candidate_ids, top_scores

    def predict(self, users, history):
        """Score users from recent history.
        users and candidates are torch tensors, and history is a
        DataFrame of recent interactions. The history is filtered to the
        requested users and candidates, then grouped into sessions containing
        at most self.session_size interactions.
        """
        users_np = users.cpu().numpy()
        grouped = (
            history.groupby(self.user_col, sort=False)
            .tail(self.session_size)                    # last session_size interactions per user, order kept
            .groupby(self.user_col)[self.item_col]
            .agg(list)
            .reindex(users_np)                          # one row per user, in `users` order
            .apply(lambda x: x if isinstance(x, list) else [])  # no history -> empty list
            .to_frame(self.item_col)
        )
        return self._predict(grouped)

    @torch.no_grad()
    def _predict(self, histories, item_col="item"):
        """Full-catalog scores for each user, via left-padded batched forward passes.

        model:     trained GRU4Rec object
        histories: DataFrame, one row per session, `item_col` = list of raw item ids (chronological)
        returns:   tensor (n_users, n_items); row r = histories row r, column i = internal item index i
        """
        dev = self.predictor.device
        sequences = histories[item_col]
        # raw ids -> internal indices, unknown items dropped
        mapped_sequences = self.map_to_model_idx(sequences)
        num_items = len(mapped_sequences)
        max_sequence_size = max((len(s) for s in mapped_sequences), default=0)
        bias = self.predictor.model.By.weight[:, 0]  # fallback scores (empty history)
        if max_sequence_size == 0:
            # If no history is found, we just use the bias term as no hidden state is computed
            return bias.expand(num_items, -1).clone()

        # left padding: pad with 0s. because this is a real item id, we use a padding to help
        items = torch.zeros((num_items, max_sequence_size), dtype=torch.long)
        # padding mask: essentially used to understand which predictions to ignore.
        pad = torch.ones((num_items, max_sequence_size), dtype=torch.bool)
        for row_number, sequence in enumerate(mapped_sequences):
            # skips empty sequences
            if len(sequence):
                # Pastes the sequence to the len(sequence) rightmost entries of items tensor
                items[row_number, max_sequence_size - len(sequence) :] = torch.tensor(
                    sequence
                )
                pad[row_number, max_sequence_size - len(sequence) :] = False
        items, pad = items.to(dev), pad.to(dev)

        # Initialize hidden state
        hidden_state = [
            torch.zeros((num_items, layer_size), device=dev)
            for layer_size in self.predictor.layers
        ]
        dummy_item = torch.zeros(1, dtype=torch.long, device=dev)

        for step in range(max_sequence_size):
            if step == max_sequence_size - 1:
                Y = None  # Scores for the entire catalog
            else:
                Y = dummy_item  # scores the prediction for a dummy item. We only
                # use the output of the model in the last step, so it doesnt matter what
                # we're scoring before.
            scores = self.predictor.model.forward(
                items[:, step], hidden_state, Y, training=False
            )
            for h in hidden_state:
                h[pad[:, step]] = 0  # reset rows still in padding

        scores[pad[:, -1]] = bias  # users with empty history
        return scores
