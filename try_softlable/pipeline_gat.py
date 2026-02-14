import os
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix
)

# ===== import your modules =====
import lib
from lib.make_dataset import make_dataset
from evaluate.t_SNE_onehot_soft_label import visualize_hard_vs_soft_labels
from gat_soft_classifier import train_gat_soft, LabelPropagationSoftLabelGenerator


# -------------------------------------------------------
# Utility: evaluation with threshold search (imbalanced)
# -------------------------------------------------------
def evaluate_with_threshold_search(y_true, y_score, metric="macro_f1"):
    best_t, best_s = 0.5, -1.0

    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_score >= t).astype(int)

        if metric == "macro_f1":
            s = f1_score(y_true, y_pred, average="macro")
        elif metric == "gmean":
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0
            s = np.sqrt(sen * spe)
        else:
            raise ValueError(metric)

        if s > best_s:
            best_s = s
            best_t = t

    return best_t, best_s

def evaluate(y, probs, mask, best_t, help="eval"):
    y_pred = (probs[mask] >= best_t).astype(int)
    y_score = probs[mask]

    f1 = f1_score(y, y_pred, average="macro")
    auc = roc_auc_score(y, y_score)
    mcc = matthews_corrcoef(y, y_pred)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = np.sqrt(sen * spe)

    print(f"\n================ {help} ================")
    print(f"Threshold: {best_t:.3f}")
    print(f"Macro-F1 : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print(f"G-Mean   : {gmean:.4f}")
    print("Classification report:")
    print(classification_report(y, y_pred, digits=4))

    return {
        "threshold": best_t,
        "macro_f1": f1,
        "auc": auc,
        "mcc": mcc,
        "gmean": gmean
    }

# -------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------
def run_lp_gat_pipeline(
    Z_train, y_train,
    Z_val, y_val,
    Z_test, y_test,
    device="cuda"
):
    """
    Full pipeline:
    1) LP soft labels on TRAIN set
    2) Build transductive graph (train+val+test)
    3) GAT soft-label classifier
    """

    # ---------------------------------------------------
    # Step 1: Label Propagation on TRAIN ONLY
    # ---------------------------------------------------
    print("\n[Step 1] Label Propagation (train set)")

    lp = LabelPropagationSoftLabelGenerator(
        n_neighbors=30,
        alpha=0.9,               # global alpha (used in seeds selection stage)
        seed_ratio=0.2,
        max_iter=200,
        boundary_weight_lambda=2.0
    )

    soft_p_train, sample_w, seeds_mask, purity = lp.generate(Z_train, y_train)

    # visualize_hard_vs_soft_labels(Z_train, y_train, soft_p_train)
    print(f"  Soft labels generated.")
    print(f"  Positive mean p: {soft_p_train[y_train==1].mean():.4f}")
    print(f"  Negative mean p: {soft_p_train[y_train==0].mean():.4f}")
    print(f"  Seeds ratio: {seeds_mask.mean():.3f}")

    # ---------------------------------------------------
    # Step 2: Prepare transductive graph data
    # ---------------------------------------------------
    print("\n[Step 2] Prepare transductive graph")

    # concatenate all splits
    Z_all = np.concatenate([Z_train, Z_val, Z_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)

    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)

    # soft labels:
    # - train: LP output
    # - val/test: initialize with HARD labels (no leakage)
    soft_p_all = np.zeros(len(y_all), dtype=np.float32)
    soft_p_all[:n_train] = soft_p_train
    soft_p_all[n_train:n_train+n_val] = y_val.astype(np.float32)
    soft_p_all[n_train+n_val:] = y_test.astype(np.float32)

    # masks
    train_mask = np.zeros(len(y_all), dtype=bool)
    val_mask = np.zeros(len(y_all), dtype=bool)
    test_mask = np.zeros(len(y_all), dtype=bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True

    # ---------------------------------------------------
    # Step 3: Train GAT classifier
    # ---------------------------------------------------
    print("\n[Step 3] Train GAT soft-label classifier")

    gat_model = train_gat_soft(
        Z_all=Z_all,
        y_all=y_all,
        soft_p_all=soft_p_all,
        train_mask=train_mask,
        val_mask=val_mask,
        k=30,
        epochs=200,
        boundary_lambda=2.0,
        device=device
    )

    # ---------------------------------------------------
    # Step 4: Evaluation
    # ---------------------------------------------------
    print("\n[Step 4] Evaluation")

    gat_model.eval()
    with torch.no_grad():
        from gat_soft_classifier import build_knn_edges
        dst, src = build_knn_edges(Z_all, k=30)

        # X = torch.tensor(
        #     np.concatenate([
        #         Z_all,
        #         soft_p_all[:, None],
        #         (1 - np.abs(2*soft_p_all - 1))[:, None]
        #     ], axis=1),
        #     device=device
        # )
        X = torch.tensor(Z_all, device=device)

        dst_t = torch.tensor(dst, device=device)
        src_t = torch.tensor(src, device=device)

        logits = gat_model(X, dst_t, src_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    # ---- validation threshold ----
    best_t, best_val = evaluate_with_threshold_search(
        y_all[val_mask], probs[val_mask], metric="macro_f1"
    )
    print(f"  Best threshold on VAL: {best_t:.3f}, macro-F1={best_val:.4f}")

    # ---- test metrics ----
    evaluate(y_test, probs, test_mask, best_t, help="test")
    evaluate(y_val, probs, val_mask, best_t, help="val")
    evaluate(y_train, probs, train_mask, best_t, help="train")

if __name__ == "__main__":

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成LogiCoTab-LP-GAT\exp/churn\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/adult\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/bean\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/page\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/buddy\CoTable\config.toml")
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/mammography\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/yeast_me2\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/obesity\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/shopper\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/magic\CoTable\config.toml")
    parent_dir = raw_config['parent_dir']

    # 1. 加载VAE生成的隐向量
    Z_train = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize.npy'))
    Z_test = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_val.npy'))
    Z_val = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_test.npy'))

    # 2. 展平(如果是3D)
    if Z_train.ndim == 3:
        Z_train = Z_train.reshape(Z_train.shape[0], -1)
        Z_test = Z_test.reshape(Z_test.shape[0], -1)
        Z_val = Z_val.reshape(Z_val.shape[0], -1)

    # 3. 加载标签(从原始数据)
    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y_train = dataset.y['train'].flatten()
    y_test = dataset.y['val'].flatten()
    y_val = dataset.y['test'].flatten()

    print(f"\n训练集: {Z_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"测试集: {Z_test.shape}, 类别分布: {np.bincount(y_test)}")
    print(f"验证集: {Z_val.shape}, 类别分布: {np.bincount(y_val)}")

    run_lp_gat_pipeline(Z_train, y_train, Z_val, y_val, Z_test, y_test, device="cuda")