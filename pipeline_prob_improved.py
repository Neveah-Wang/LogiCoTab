import os
import numpy as np
import torch
import argparse
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, accuracy_score
)

import lib
from lib.make_dataset import make_dataset
from evaluate.t_SNE_onehot_soft_label import visualize_hard_vs_soft_labels
from prob_soft_classifier_improved import (
    LabelPropagationSoftLabelGenerator,
    train_prob_model
)

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

def evaluate_split_old(y_true, y_score, best_t, help="eval"):
    y_pred = (y_score >= best_t).astype(int)

    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
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
    print(f"Sensitivity: {sen:.4f}")
    print(f"Specificity: {spe:.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    return {"macro_f1": f1, "auc": auc, "mcc": mcc, "gmean": gmean}


def evaluate_split(y_true, y_score, best_t, help="eval"):
    y_pred = (y_score >= best_t).astype(int)
    f1_macro = f1_score(y_true, y_pred, average="macro")  # 修复: 移除空格
    f1_0 = f1_score(y_true, y_pred, average=None)[0]  # 负样本F1
    f1_1 = f1_score(y_true, y_pred, average=None)[1]  # 正样本F1
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall for class 1)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    gmean = np.sqrt(sen * spe)

    print(f"\n================ {help} ================")
    print(f"Threshold: {best_t:.3f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1_macro:.4f} (Class0: {f1_0:.4f}, Class1: {f1_1:.4f})")
    print(f"AUC      : {auc:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print(f"G-Mean   : {gmean:.4f}")
    print(f"Sensitivity: {sen:.4f}")
    print(f"Specificity: {spe:.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # 返回完整指标字典
    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "f1_class0": f1_0,
        "f1_class1": f1_1,
        "auc": auc,
        "mcc": mcc,
        "gmean": gmean,
        "sensitivity": sen,
        "specificity": spe,
        "threshold": best_t
    }


# ===== 添加结果统计函数 =====
def summarize_results(all_results, splits=["train", "val", "test"]):
    """统计n次实验的结果"""
    metrics = ["accuracy", "macro_f1", "f1_class0", "f1_class1", "auc", "mcc", "gmean"]

    print("\n" + "=" * 80)
    print("实验结果统计 (平均值 ± 标准差 | 最小值 ~ 最大值)")
    print("=" * 80)

    for split in splits:
        print(f"\n[{split.upper()} SET]")
        print(f"{'Metric':<15} {'Mean ± Std':<25} {'Min ~ Max':<25}")
        print("-" * 60)

        for metric in metrics:
            values = [res[split][metric] for res in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            print(f"{metric:<15} {mean_val:.4f} ± {std_val:.4f}    {min_val:.4f} ~ {max_val:.4f}")

    # 保存详细结果到文件
    # import json
    # with open("evaluate/SLEGAT_mle_log/experiment_results.json", "w") as f:
    #     json.dump(all_results, f, indent=2)
    # print("\n✓ 详细结果已保存至: evaluate/SLEGAT_mle_log/experiment_results.json")

def run_pipeline_prob(
    raw_config,
    Z_train, y_train,
    Z_val, y_val,
    Z_test, y_test,
    # LP
    lp_k=30,
    lp_alpha=0.9,
    lp_seed_ratio=0.2,
    lp_max_iter=200,
    # lp_boundary_lambda=2.0,
    lp_boundary_lambda=1.0,
    # Train
    lr=1e-3,
    epochs=500,
    weight_decay=1e-5,
    focal_gamma=2.0,
    focal_alpha=0.25,
    use_kl=True,
    kl_weight=1.0,
    use_contrastive=True,
    contrastive_weight=0.5,
    contrastive_temp=0.07,
    minority_weight=2.0,
    device="cuda"
):
    print("=" * 80)
    print(f" Pipeline: LP soft labels + MLP prob + {kl_weight} * KL + Focal + {contrastive_weight} * SupCon")
    print("=" * 80)

    # -------- Step1: LP on TRAIN ONLY --------
    lp = LabelPropagationSoftLabelGenerator(
        n_neighbors=lp_k,
        alpha=lp_alpha,
        seed_ratio=lp_seed_ratio,
        max_iter=lp_max_iter,
        boundary_weight_lambda=lp_boundary_lambda
    )
    soft_p_tr, w_tr, seeds_mask, purity = lp.generate(raw_config, Z_train, y_train)

    print("\n[LP] soft label stats (train only):")
    print(f"  pos mean: {soft_p_tr[y_train==1].mean():.4f}")
    print(f"  neg mean: {soft_p_tr[y_train==0].mean():.4f}")
    print(f"  seeds ratio: {seeds_mask.mean():.3f}")
    # 绘制 t-SNE 可视化图像
    # visualize_hard_vs_soft_labels(Z_train, y_train, soft_p_tr)

    # -------- Step2: concat all (BUT training losses use train_mask only) --------
    """
    Z_all = np.concatenate([Z_train, Z_val, Z_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)

    n_tr, n_va, n_te = len(y_train), len(y_val), len(y_test)

    train_mask = np.zeros(len(y_all), dtype=bool); train_mask[:n_tr] = True
    val_mask   = np.zeros(len(y_all), dtype=bool); val_mask[n_tr:n_tr+n_va] = True
    test_mask  = np.zeros(len(y_all), dtype=bool); test_mask[n_tr+n_va:] = True

    # soft labels：只需要保证训练段是 LP 软标签即可；val/test 可用占位（不会进loss）
    soft_p_all = np.zeros(len(y_all), dtype=np.float32)
    soft_p_all[:n_tr] = soft_p_tr
    soft_p_all[n_tr:n_tr+n_va] = y_val.astype(np.float32)
    soft_p_all[n_tr+n_va:] = y_test.astype(np.float32)

    # boundary weights：训练段用 LP 的边界权重，其他段设 1
    w_all = np.ones(len(y_all), dtype=np.float32)
    w_all[:n_tr] = w_tr
    """

    # -------- Step3: train prob model (no leakage) --------
    model = train_prob_model(
        Z_train=Z_train,
        y_train=y_train,
        Z_val=Z_val,
        y_val=y_val,
        soft_p_all=soft_p_tr,
        sample_w_all=w_tr,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        use_kl=use_kl,
        kl_weight=kl_weight,
        use_contrastive=use_contrastive,
        contrastive_weight=contrastive_weight,
        contrastive_temp=contrastive_temp,
        minority_weight=minority_weight,
        device=device
    )

    # -------- Step4: inference + threshold search --------
    model.eval()
    with torch.no_grad():
        X_val = torch.tensor(Z_val, device=device)
        logits_val, _ = model(X_val)
        probs_val = torch.sigmoid(logits_val).detach().cpu().numpy()

        X_test = torch.tensor(Z_test, device=device)
        logits_test, _ = model(X_test)
        probs_test = torch.sigmoid(logits_test).detach().cpu().numpy()

        X_train = torch.tensor(Z_train, device=device)
        logits_train, _ = model(X_train)
        probs_train = torch.sigmoid(logits_train).detach().cpu().numpy()

    best_t, best_val = evaluate_with_threshold_search(y_val, probs_val, metric="macro_f1")
    print(f"\n[VAL] best threshold={best_t:.3f}, best macroF1={best_val:.4f}")

    res = {}
    res["train"] = evaluate_split(y_train, probs_train, best_t, help="TRAIN")
    res["val"]   = evaluate_split(y_val, probs_val, best_t, help="VAL")
    res["test"]  = evaluate_split(y_test, probs_test, best_t, help="TEST")
    return model, res


if __name__ == "__main__":

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT/exp/churn/CoTable/config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/adult\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/bean\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/page\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/buddy\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/mammography\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/yeast_me2\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/obesity\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/shopper\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/magic\CoTable\config.toml")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    raw_config = lib.util.load_config(args.config)

    parent_dir = raw_config['parent_dir']

    # 1. 加载VAE生成的隐向量
    Z_train = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize.npy'))
    Z_val = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_val.npy'))
    Z_test = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_test.npy'))

    # 2. 展平(如果是3D)
    if Z_train.ndim == 3:
        Z_train = Z_train.reshape(Z_train.shape[0], -1)
        Z_test = Z_test.reshape(Z_test.shape[0], -1)
        Z_val = Z_val.reshape(Z_val.shape[0], -1)

    # 3. 加载标签(从原始数据)
    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y_train = dataset.y['train'].flatten()
    y_val = dataset.y['val'].flatten()
    y_test = dataset.y['test'].flatten()

    print(f"\n数据统计:")
    print(f"  训练集: {Z_train.shape}, 正类比例: {y_train.mean():.2%}, 正类数量: {sum(y_train==1)}, 负类数量: {sum(y_train==0)}")
    print(f"  验证集: {Z_val.shape}, 正类比例: {y_val.mean():.2%}")
    print(f"  测试集: {Z_test.shape}, 正类比例: {y_test.mean():.2%}")

    # ===== 运行完整pipeline =====

    NUM_RUNS = 30
    all_results = []

    for run_idx in range(NUM_RUNS):
        print(f"\n{'=' * 80}")
        print(f" RUN {run_idx + 1}/{NUM_RUNS} (Seed={run_idx})")
        print(f"{'=' * 80}")

        model, results = run_pipeline_prob(
            raw_config,
            Z_train, y_train,
            Z_val, y_val,
            Z_test, y_test,
            # Z_val, y_val,  # test val 交换顺序
            # LP
            lp_k=raw_config['LP']['lp_k'],
            lp_alpha=raw_config['LP']['lp_alpha'],
            lp_seed_ratio=raw_config['LP']['lp_seed_ratio'],
            lp_max_iter=raw_config['LP']['lp_max_iter'],
            lp_boundary_lambda=raw_config['LP']['lp_boundary_lambda'],
            # Train
            lr=1e-3,
            epochs=raw_config['Train']['epochs'],
            weight_decay=1e-5,
            focal_gamma=2.0,
            focal_alpha=raw_config['Train']['focal_alpha'],
            use_kl=True,
            kl_weight=1.0,
            use_contrastive=True,
            contrastive_weight=0.0,
            contrastive_temp=0.07,
            minority_weight=2.0,
            device="cuda"
        )

        # 保存本次结果（包含train/val/test的所有指标）
        all_results.append({
            "run_id": run_idx,
            "train": results["train"],
            "val": results["val"],
            "test": results["test"]
        })

        # 释放GPU内存
        del model
        torch.cuda.empty_cache()

        # ===== 统计并输出最终结果 =====
    summarize_results(all_results)

    print("\n" + "=" * 80)
    print("n 次实验完成！")
    print("=" * 80)