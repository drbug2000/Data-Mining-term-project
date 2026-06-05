"""Click-tuned content head: bi-encoder MLP trained with class-balanced cross-entropy.

Architecture: Linear(384→256) → ReLU → Linear(256→128) → L2-norm → cosine × scale + bias
Trained on internal-train split; early-stopped on internal-val AUC.
Ensemble of N_SEEDS heads averages logits to reduce variance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils import binary_auc


class ContentHead(nn.Module):
    def __init__(self, d: int = 384, h: int = 256, p: int = 128):
        super().__init__()
        self.sq = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, p))
        self.aq = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, p))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        qp = nn.functional.normalize(self.sq(q), dim=1)
        ap = nn.functional.normalize(self.aq(a), dim=1)
        return self.scale * (qp * ap).sum(1) + self.b


def train_one_head(Q_tr: np.ndarray, A_tr: np.ndarray, Y_tr: np.ndarray,
                   Q_val: np.ndarray, A_val: np.ndarray, Y_val: np.ndarray,
                   seed: int, epochs: int = 55, batch_size: int = 8192,
                   lr: float = 1e-3, wd: float = 1e-3, patience: int = 8,
                   device: torch.device | None = None) -> ContentHead:
    """Train one head; early-stop on internal-val AUC. Return best checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = ContentHead().to(device)
    pos_weight = torch.tensor(
        (Y_tr == 0).sum() / max(1, (Y_tr == 1).sum()), device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    qt = torch.tensor(Q_tr, device=device)
    at = torch.tensor(A_tr, device=device)
    yt = torch.tensor(Y_tr.astype(np.float32), device=device)
    qv = torch.tensor(Q_val, device=device)
    av = torch.tensor(A_val, device=device)

    best_auc = -1.0
    best_state = None
    bad = 0
    n = len(yt)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            opt.zero_grad()
            loss_fn(model(qt[idx], at[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            auc = binary_auc(model(qv, av).cpu().numpy(), Y_val)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_ensemble(Q_tr: np.ndarray, A_tr: np.ndarray, Y_tr: np.ndarray,
                   Q_val: np.ndarray, A_val: np.ndarray, Y_val: np.ndarray,
                   *score_arrays: np.ndarray,
                   n_seeds: int = 30,
                   device: torch.device | None = None,
                   verbose: bool = True) -> list[np.ndarray]:
    """Train N_SEEDS heads; return ensemble-averaged logits for each score_array.

    score_arrays: arbitrary number of (N_i, 384) query-ad pairs to score.
                  Returns a list of (N_i,) averaged logit arrays, one per input.

    Usage:
        scores_val, scores_test = train_ensemble(
            Q_tr, A_tr, Y_tr, Q_val, A_val, Y_val,
            np.stack([q for q, _ in val_pairs]),
            np.stack([q for q, _ in test_pairs]),
        )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs = [np.zeros(len(qa[0]), dtype=np.float64) for qa in score_arrays]

    tensors = []
    for Q_i, A_i in score_arrays:
        tensors.append((
            torch.tensor(Q_i, dtype=torch.float32, device=device),
            torch.tensor(A_i, dtype=torch.float32, device=device),
        ))

    for seed in range(1, n_seeds + 1):
        model = train_one_head(Q_tr, A_tr, Y_tr, Q_val, A_val, Y_val,
                               seed=seed, device=device)
        model.eval()
        with torch.no_grad():
            for i, (qt_i, at_i) in enumerate(tensors):
                accs[i] += model(qt_i, at_i).cpu().numpy()
        if verbose and seed % 5 == 0:
            print(f"  content head seed {seed}/{n_seeds}")

    return [acc / n_seeds for acc in accs]
