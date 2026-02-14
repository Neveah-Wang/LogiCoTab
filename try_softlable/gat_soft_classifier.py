import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class LabelPropagationSoftLabelGenerator:
    """
    在隐空间 Z 上构建 kNN 图，用 Label Propagation 得到软标签 p in [0,1]。
    关键点：只用“高置信 seeds”做强监督，其余点作为未标注点由传播决定。
    """

    def __init__(
        self,
        n_neighbors: int = 30,
        alpha: float = 0.9,                 # propagation strength
        max_iter: int = 200,
        tol: float = 1e-6,
        seed_ratio: float = 0.2,            # per-class seed ratio
        min_seeds_per_class: int = 10,
        adaptive_bandwidth: bool = True,
        prior_from_data: bool = True,
        boundary_weight_lambda: float = 2.0, # sample weight strength for boundary points
        device: str = "cpu"
    ):
        self.n_neighbors = int(n_neighbors)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed_ratio = float(seed_ratio)
        self.min_seeds_per_class = int(min_seeds_per_class)
        self.adaptive_bandwidth = bool(adaptive_bandwidth)
        self.prior_from_data = bool(prior_from_data)
        self.boundary_weight_lambda = float(boundary_weight_lambda)
        self.device = device

    @staticmethod
    def _safe_minmax(x, eps=1e-12):
        x = np.asarray(x)
        return (x - x.min()) / (x.max() - x.min() + eps)

    def _build_knn_graph(self, Z: np.ndarray):
        """
        返回：
            neigh_idx: (n, k) int
            neigh_dist: (n, k) float
        """
        n = Z.shape[0]
        k = self.n_neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
        dist, idx = nbrs.kneighbors(Z)
        # drop self
        neigh_dist = dist[:, 1:]
        neigh_idx = idx[:, 1:]
        return neigh_idx, neigh_dist

    def _compute_weights(self, neigh_dist: np.ndarray):
        """
        RBF 权重：exp(-d^2 / (2*sigma_i^2))
        sigma_i 采用邻域平均距离（自适应带宽）
        """
        if self.adaptive_bandwidth:
            sigma = np.mean(neigh_dist, axis=1, keepdims=True)
            sigma = np.maximum(sigma, 1e-6)
        else:
            sigma = np.ones((neigh_dist.shape[0], 1)) * np.median(neigh_dist)

        w = np.exp(-(neigh_dist ** 2) / (2.0 * (sigma ** 2)))
        # 避免全 0
        w = np.maximum(w, 1e-12)
        return w

    def _select_seeds(self, y: np.ndarray, neigh_idx: np.ndarray):
        """
        按邻域纯度为每一类选择 seeds。
        purity(i) = mean( y[neighbors]==y[i] )
        每个类 选择 top n_seed 个 seeds，其中 n_seed = max(纯度为1的数量, 纯度最高的前seed_ratio个, 最小seeds数量)
        """
        n = y.shape[0]
        k = neigh_idx.shape[1]
        neigh_labels = y[neigh_idx]                  # (n, k)
        purity = (neigh_labels == y[:, None]).mean(axis=1)  # (n,)

        seeds = np.zeros(n, dtype=bool)
        classes = [0, 1]
        for c in classes:
            idx_c = np.where(y == c)[0]
            if idx_c.size == 0:
                continue

            # 选 top purity
            n_seed = max(int(np.ceil(idx_c.size * self.seed_ratio)), self.min_seeds_per_class)
            n_seed = max(n_seed, sum(purity[idx_c] == 1.0))
            n_seed = min(n_seed, idx_c.size)

            order = np.argsort(-purity[idx_c])   # descending
            chosen = idx_c[order[:n_seed]]
            seeds[chosen] = True

        return seeds, purity

    def _propagate_old(self, neigh_idx, weights, seeds_mask, y):
        """
        稀疏图上的迭代传播：F_{t+1} = alpha * S F_t + (1-alpha) * Y
        其中 S 为行归一化权重，Y 在 seeds 为 one-hot，否则为先验。
        """
        n, k = neigh_idx.shape
        eps = 1e-12
        """
        # prior
        if self.prior_from_data:
            pi = float((y == 1).mean())
        else:
            pi = 0.5
        prior = np.array([1.0 - pi, pi], dtype=np.float64)

        # Y: (n,2)
        Y = np.tile(prior[None, :], (n, 1))
        # clamp seeds to one-hot
        Y[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
        Y[seeds_mask & (y == 1)] = np.array([0.0, 1.0])
        """
        Y = np.ones((n, 2), dtype=np.float64)
        Y[y == 0] = np.array([1.0, 0.0])
        Y[y == 1] = np.array([0.0, 1.0])


        # init F
        F_cur = Y.copy()   # (n,2) = (7828, 2)

        # row-normalized weights => S
        row_sum = weights.sum(axis=1, keepdims=True) + eps
        S_w = weights / row_sum   # (n,k) = (7828, 30)
        # breakpoint()
        # iterative
        for _ in range(self.max_iter):
            neigh_F = F_cur[neigh_idx]                     # (n,k,2) = (7828, 30, 2)
            SF = (S_w[:, :, None] * neigh_F).sum(axis=1)   # (n,2)

            F_new = self.alpha * SF + (1.0 - self.alpha) * Y

            # clamp seeds (hard constraint)
            F_new[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
            F_new[seeds_mask & (y == 1)] = np.array([0.0, 1.0])

            diff = np.mean(np.abs(F_new - F_cur))
            F_cur = F_new
            if diff < self.tol:
                break

        # return prob of class 1
        p = F_cur[:, 1]
        p = np.clip(p, 0.0, 1.0)
        return p

    def _propagate(
        self, neigh_idx, weights, seeds_mask, y,
        # 超参数
        alpha_pos=0.80,
        alpha_neg=0.95,
        forbid_flip=True,
        flip_margin=1e-3
    ):
        """
        稀疏图上的迭代传播：F_{t+1} = alpha * S F_t + (1-alpha) * Y
        其中 S 为行归一化权重，Y 在 seeds 为 one-hot，否则为先验。
        """
        n, k = neigh_idx.shape
        eps = 1e-12
        """
        # prior
        if self.prior_from_data:
            pi = float((y == 1).mean())
        else:
            pi = 0.5
        prior = np.array([1.0 - pi, pi], dtype=np.float64)

        # Y: (n,2)
        Y = np.tile(prior[None, :], (n, 1))
        # clamp seeds to one-hot
        Y[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
        Y[seeds_mask & (y == 1)] = np.array([0.0, 1.0])
        """
        Y = np.ones((n, 2), dtype=np.float64)
        Y[y == 0] = np.array([1.0, 0.0])
        Y[y == 1] = np.array([0.0, 1.0])

        # --------------------------------------------------
        # 2) Build asymmetric transition matrix S
        #    (boost positive neighbors)
        # --------------------------------------------------
        neigh_labels = y[neigh_idx]  # (n, k)
        boost = np.ones_like(weights)
        boost[neigh_labels == 1] *= 1

        W = weights * boost
        W = np.maximum(W, eps)
        W = W / (W.sum(axis=1, keepdims=True) + eps)  # row-normalize

        # --------------------------------------------------
        # 3) Iterative propagation (per-node alpha)
        # --------------------------------------------------
        # init F
        F_cur = Y.copy()  # (n,2) = (7828, 2)

        pos_mask = (y == 1)
        neg_mask = (y == 0)
        alpha_i = np.zeros(n)
        alpha_i[pos_mask] = alpha_pos
        alpha_i[neg_mask] = alpha_neg

        for _ in range(self.max_iter):
            neigh_F = F_cur[neigh_idx]                    # (n,k,2) = (7828, 30, 2)
            SF = (W[:, :, None] * neigh_F).sum(axis=1)    # (n,2)

            F_new = alpha_i[:, None] * SF + (1.0 - alpha_i[:, None]) * Y

            # clamp seeds (hard constraint)
            F_new[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
            F_new[seeds_mask & (y == 1)] = np.array([0.0, 1.0])
            # breakpoint()
            # optional: forbid label flipping
            if forbid_flip:
                p = F_new[:, 1]
                p[pos_mask] = np.maximum(p[pos_mask], 0.5 + flip_margin)
                p[neg_mask] = np.minimum(p[neg_mask], 0.5 - flip_margin)
                F_new[:, 1] = p
                F_new[:, 0] = 1.0 - p

            diff = np.mean(np.abs(F_new - F_cur))
            F_cur = F_new
            if diff < self.tol:
                break

        # return prob of class 1
        p = F_cur[:, 1]
        p = np.clip(p, 0.0, 1.0)
        return p

    def generate(self, Z: np.ndarray, y: np.ndarray):
        """
        返回：
            soft_labels: (n,) in [0,1]
            sample_weights: (n,)  边界点权重（可用于训练加权）
            seeds_mask: (n,) bool
            purity: (n,) float
        """
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        neigh_idx, neigh_dist = self._build_knn_graph(Z)
        weights = self._compute_weights(neigh_dist)
        seeds_mask, purity = self._select_seeds(y, neigh_idx)
        soft_labels = self._propagate(neigh_idx, weights, seeds_mask, y)
        # soft_labels = self._propagate_old(neigh_idx, weights, seeds_mask, y)

        # boundary-aware sample weights: p near 0.5 => higher weight
        # w_i = 1 + lambda*(1 - |2p-1|)
        boundary_strength = 1.0 - np.abs(2.0 * soft_labels - 1.0)  # in [0,1]
        sample_weights = 1.0 + self.boundary_weight_lambda * boundary_strength

        return soft_labels, sample_weights, seeds_mask, purity

# ----------------------------
# Utils: build kNN edge list
# ----------------------------
def build_knn_edges(Z: np.ndarray, k: int = 30, include_self: bool = False):
    """
    Build directed kNN edges: i <- j (j is neighbor of i)
    Return:
        dst: (E,) target node indices i
        src: (E,) source node indices j
    """
    breakpoint()
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    nnbr = NearestNeighbors(n_neighbors=k + (1 if include_self else 0), algorithm="auto")
    nnbr.fit(Z)
    dist, idx = nnbr.kneighbors(Z)

    if include_self:
        neigh = idx
    else:
        neigh = idx[:, 1:]  # drop self

    dst = np.repeat(np.arange(n), neigh.shape[1])
    src = neigh.reshape(-1)
    return dst.astype(np.int64), src.astype(np.int64)


# ----------------------------
# Utils: segment softmax over edges grouped by dst
# ----------------------------
def segment_softmax(dst: torch.Tensor, e: torch.Tensor, num_nodes: int):
    """
    Compute softmax over edges for each destination node.
    dst: (E,) long, destination node id for each edge
    e:   (E,) float, unnormalized attention logits
    """
    # Use scatter_reduce if available (PyTorch >= 2.0)
    if hasattr(torch.Tensor, "scatter_reduce_"):
        # max per dst
        max_per = torch.full((num_nodes,), -1e30, device=e.device, dtype=e.dtype)
        max_per.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)

        e_exp = torch.exp(e - max_per[dst])

        sum_per = torch.zeros((num_nodes,), device=e.device, dtype=e.dtype)
        sum_per.scatter_add_(0, dst, e_exp)

        return e_exp / (sum_per[dst] + 1e-12)

    # Fallback (slower): loop over unique dst
    out = torch.empty_like(e)
    uniq = torch.unique(dst)
    for d in uniq:
        mask = (dst == d)
        out[mask] = torch.softmax(e[mask], dim=0)
    return out


# ----------------------------
# GAT Layer (edge list, multi-head)
# ----------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, concat=True, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # linear projection per head (implemented as one big weight)
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)

        # attention vectors a_src, a_dst per head
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))

        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, dst, src, num_nodes):
        """
        x: (N, Fin)
        dst, src: (E,)
        """
        N = x.size(0)
        h = self.W(x)  # (N, heads*out_dim)
        h = h.view(N, self.heads, self.out_dim)  # (N, H, D)

        h_dst = h[dst]  # (E, H, D)
        h_src = h[src]  # (E, H, D)

        # attention logits: e_ij = LeakyReLU( a_dst^T h_i + a_src^T h_j )
        e = (h_dst * self.a_dst[None, :, :]).sum(dim=-1) + (h_src * self.a_src[None, :, :]).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)  # (E, H)

        # edge softmax grouped by dst, per head
        alpha_list = []
        for head in range(self.heads):
            alpha_h = segment_softmax(dst, e[:, head], num_nodes)  # (E,)
            alpha_list.append(alpha_h)
        alpha = torch.stack(alpha_list, dim=1)  # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # message passing: sum_j alpha_ij * h_j
        out = torch.zeros((num_nodes, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        msg = alpha[:, :, None] * h_src  # (E, H, D)
        out.index_add_(0, dst, msg)      # aggregate into dst

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)  # (N, D)

        out = out + self.bias
        return out


# ----------------------------
# GAT Regressor (probability)
# ----------------------------
class GATSoftClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, dst, src):
        num_nodes = x.size(0)
        h = self.gat1(x, dst, src, num_nodes)
        h = F.elu(h)
        h = self.gat2(h, dst, src, num_nodes)
        logits = self.mlp(h).squeeze(-1)  # (N,)
        return logits


# ----------------------------
# Loss: cost-sensitive soft BCE + optional boundary weights
# ----------------------------
class CostSensitiveSoftBCELoss(nn.Module):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits, soft_targets, sample_weight=None):
        loss = F.binary_cross_entropy_with_logits(
            logits, soft_targets, pos_weight=self.pos_weight, reduction="none"
        )
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


# ----------------------------
# Training helper
# ----------------------------
def train_gat_soft(
    Z_all: np.ndarray,
    y_all: np.ndarray,
    soft_p_all: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray = None,
    k: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    boundary_lambda: float = 2.0,
    device: str = "cuda"
):
    """
    Transductive training on a single graph built from Z_all.
    Loss computed only on train_mask.
    soft_p_all is your LP-generated soft label.
    """
    Z_all = np.asarray(Z_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.int64)
    soft_p_all = np.asarray(soft_p_all, dtype=np.float32)

    # Build graph edges
    dst, src = build_knn_edges(Z_all, k=k, include_self=False)

    # Node features: [z, p, u]
    u_all = 1.0 - np.abs(2.0 * soft_p_all - 1.0)  # boundary uncertainty in [0,1]
    # X_all = np.concatenate([Z_all, soft_p_all[:, None], u_all[:, None]], axis=1)
    X_all = Z_all

    # pos_weight from hard labels (positive is minority)
    n_pos = int((y_all[train_mask] == 1).sum())
    n_neg = int((y_all[train_mask] == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    # boundary sample weights (optional)
    sample_w = 1.0 + boundary_lambda * u_all

    # Torch tensors
    X = torch.tensor(X_all, device=device)
    soft_t = torch.tensor(soft_p_all, device=device)
    w_t = torch.tensor(sample_w, device=device)
    dst_t = torch.tensor(dst, device=device, dtype=torch.long)
    src_t = torch.tensor(src, device=device, dtype=torch.long)
    train_mask_t = torch.tensor(train_mask, device=device, dtype=torch.bool)
    val_mask_t = None if val_mask is None else torch.tensor(val_mask, device=device, dtype=torch.bool)

    model = GATSoftClassifier(in_dim=X.size(1), hidden_dim=256, heads=4, dropout=0.2).to(device)
    crit = CostSensitiveSoftBCELoss(pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = -1.0

    def macro_f1_from_logits(logits, y_true, mask, thr=0.5):
        probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
        y = y_true[mask].cpu().numpy().astype(int)
        y_pred = (probs >= thr).astype(int)
        # macro F1
        from sklearn.metrics import f1_score
        return float(f1_score(y, y_pred, average="macro"))

    y_true_t = torch.tensor(y_all, device=device)

    for ep in range(1, epochs + 1):
        model.train()
        logits = model(X, dst_t, src_t)

        loss = crit(
            logits[train_mask_t],
            soft_t[train_mask_t],
            sample_weight=w_t[train_mask_t]
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if val_mask_t is not None:
            model.eval()
            with torch.no_grad():
                logits_val = model(X, dst_t, src_t)

            # threshold can be tuned; here keep 0.5 for monitoring
            val_f1 = macro_f1_from_logits(logits_val, y_true_t, val_mask_t, thr=0.5)

            if val_f1 > best_val:
                best_val = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if ep % 20 == 0:
                print(f"Epoch {ep:03d} | loss={loss.item():.4f} | val_macroF1@0.5={val_f1:.4f}")

        else:
            if ep % 20 == 0:
                print(f"Epoch {ep:03d} | loss={loss.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
