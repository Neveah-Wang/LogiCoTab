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
from gat_soft_classifier_improved import (
    train_gat_soft_with_contrastive,
    LabelPropagationSoftLabelGenerator,
    build_knn_edges,
    GATSoftClassifierWithContrastive
)


# -------------------------------------------------------
# Utility: evaluation with threshold search (imbalanced)
# -------------------------------------------------------
def evaluate_with_threshold_search(y_true, y_score, metric="macro_f1"):
    """在验证集上搜索最优阈值"""
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
    """详细评估函数"""
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
    print(f"Sensitivity: {sen:.4f}")
    print(f"Specificity: {spe:.4f}")
    print("Classification report:")
    print(classification_report(y, y_pred, digits=4))

    return {
        "threshold": best_t,
        "macro_f1": f1,
        "auc": auc,
        "mcc": mcc,
        "gmean": gmean,
        "sensitivity": sen,
        "specificity": spe
    }


# -------------------------------------------------------
# Main Pipeline (优化版本)
# -------------------------------------------------------
def run_improved_pipeline(
    Z_train, y_train,
    Z_val, y_val,
    Z_test, y_test,
    # Label Propagation 参数
    lp_n_neighbors=30,
    lp_alpha=0.9,
    lp_seed_ratio=0.2,
    lp_max_iter=200,
    lp_boundary_lambda=2.0,
    # GAT 参数
    gat_k=30,
    gat_lr=1e-3,
    gat_epochs=100,
    gat_boundary_lambda=2.0,
    # Focal Loss 参数
    focal_gamma=2.0,
    focal_alpha=None,
    # 对比学习参数
    use_contrastive=True,
    contrastive_weight=0.5,
    contrastive_temp=0.07,
    minority_weight=2.0,
    # 其他
    device="cuda"
):
    """
    优化后的完整pipeline:
    
    1. Label Propagation 生成软标签 (标签增强)
    2. GAT特征学习 + Focal Loss + 对比学习 (特征学习 + 分类)
    3. 验证集阈值搜索
    
    关键改进:
    - Focal Loss: 自动关注难样本和边界样本
    - 对比学习: 基于硬标签增强类间区分度，给少数类更高影响力
    - 软标签: 提供更丰富的监督信号
    """
    
    print("="*80)
    print(" 优化版 Pipeline - 软标签 + GAT + Focal Loss + 对比学习")
    print("="*80)

    # ---------------------------------------------------
    # Step 1: Label Propagation on TRAIN ONLY
    # ---------------------------------------------------
    print("\n[Step 1] 标签传播生成软标签 (仅训练集)")
    print("-" * 80)

    lp = LabelPropagationSoftLabelGenerator(
        n_neighbors=lp_n_neighbors,
        alpha=lp_alpha,
        seed_ratio=lp_seed_ratio,
        max_iter=lp_max_iter,
        boundary_weight_lambda=lp_boundary_lambda
    )

    soft_p_train, sample_w, seeds_mask, purity = lp.generate(Z_train, y_train)

    print(f"\n软标签统计:")
    print(f"  正类软标签均值: {soft_p_train[y_train==1].mean():.4f}")
    print(f"  负类软标签均值: {soft_p_train[y_train==0].mean():.4f}")
    print(f"  Seeds比例: {seeds_mask.mean():.3f}")
    print(f"  边界样本 (0.3<p<0.7): {np.sum((soft_p_train>0.3) & (soft_p_train<0.7))} ({np.mean((soft_p_train>0.3) & (soft_p_train<0.7))*100:.1f}%)")
    # 绘制 t-SNE 可视化图像
    # visualize_hard_vs_soft_labels(Z_train, y_train, soft_p_train)

    # ---------------------------------------------------
    # Step 2: Prepare transductive graph data
    # ---------------------------------------------------
    print("\n[Step 2] 构建全局图 (train + val + test)")
    print("-" * 80)

    # Concatenate all splits
    Z_all = np.concatenate([Z_train, Z_val, Z_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)

    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)

    print(f"  训练集: {n_train}, 验证集: {n_val}, 测试集: {n_test}")

    # Soft labels: train用LP结果，val/test用硬标签初始化(无泄露)
    soft_p_all = np.zeros(len(y_all), dtype=np.float32)
    soft_p_all[:n_train] = soft_p_train
    soft_p_all[n_train:n_train+n_val] = y_val.astype(np.float32)
    soft_p_all[n_train+n_val:] = y_test.astype(np.float32)

    # Masks
    train_mask = np.zeros(len(y_all), dtype=bool)
    val_mask = np.zeros(len(y_all), dtype=bool)
    test_mask = np.zeros(len(y_all), dtype=bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True

    # ---------------------------------------------------
    # Step 3: Train GAT with Focal Loss + Contrastive
    # ---------------------------------------------------
    print("\n[Step 3] 训练GAT分类器 (Focal Loss + 对比学习)")
    print("-" * 80)

    gat_model, best_model = train_gat_soft_with_contrastive(
        Z_all=Z_all,
        y_all=y_all,
        soft_p_all=soft_p_all,
        train_mask=train_mask,
        val_mask=val_mask,
        k=gat_k,
        lr=gat_lr,
        epochs=gat_epochs,
        boundary_lambda=gat_boundary_lambda,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        use_contrastive=use_contrastive,
        contrastive_weight=contrastive_weight,
        contrastive_temp=contrastive_temp,
        minority_weight=minority_weight,
        device=device
    )

    # ---------------------------------------------------
    # Step 4: Evaluation
    # ---------------------------------------------------
    print("\n[Step 4] 模型评估")
    print("=" * 80)

    gat_model.eval()
    with torch.no_grad():
        dst, src = build_knn_edges(Z_all, k=gat_k)
        X = torch.tensor(Z_all, device=device)
        dst_t = torch.tensor(dst, device=device, dtype=torch.long)
        src_t = torch.tensor(src, device=device, dtype=torch.long)

        logits = gat_model(X, dst_t, src_t, return_features=False)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Validation threshold search
    best_t, best_val = evaluate_with_threshold_search(
        y_all[val_mask], probs[val_mask], metric="macro_f1"
    )
    print(f"\n最优阈值 (验证集): {best_t:.3f}, Macro-F1={best_val:.4f}")

    # Evaluate on all splits
    results = {}
    results['test'] = evaluate(y_test, probs, test_mask, best_t, help="测试集 (Test)")
    results['val'] = evaluate(y_val, probs, val_mask, best_t, help="验证集 (Validation)")
    results['train'] = evaluate(y_train, probs, train_mask, best_t, help="训练集 (Train)")

    print("\n" + "="*80)
    print(" 实验总结")
    print("="*80)
    print(f"配置:")
    print(f"  - 软标签传播: k={lp_n_neighbors}, alpha={lp_alpha}")
    print(f"  - Focal Loss: gamma={focal_gamma}, alpha={focal_alpha if focal_alpha else 'auto'}")
    print(f"  - 对比学习: {'启用' if use_contrastive else '禁用'}, weight={contrastive_weight}, minority_weight={minority_weight}")
    print(f"\n测试集结果:")
    print(f"  - Macro-F1: {results['test']['macro_f1']:.4f}")
    print(f"  - AUC: {results['test']['auc']:.4f}")
    print(f"  - MCC: {results['test']['mcc']:.4f}")
    print(f"  - G-Mean: {results['test']['gmean']:.4f}")
    print("="*80)

    return gat_model, results


# -------------------------------------------------------
# 消融实验: 对比不同配置
# -------------------------------------------------------
def ablation_study(Z_train, y_train, Z_val, y_val, Z_test, y_test, device="cuda"):
    """
    消融实验: 测试不同组件的贡献
    
    配置:
    1. Baseline: 仅GAT + BCE
    2. +Focal: GAT + Focal Loss
    3. +Contrastive: GAT + Focal + 对比学习
    4. Full: GAT + Focal + 对比学习 + 软标签
    """
    
    print("\n" + "="*80)
    print(" 消融实验")
    print("="*80)
    
    configs = [
        {
            "name": "1. Baseline (GAT + BCE)",
            "use_contrastive": False,
            "focal_gamma": 0.0,  # gamma=0退化为BCE
            "use_soft_label": False
        },
        {
            "name": "2. +Focal Loss",
            "use_contrastive": False,
            "focal_gamma": 2.0,
            "use_soft_label": False
        },
        {
            "name": "3. +Contrastive Learning",
            "use_contrastive": True,
            "focal_gamma": 2.0,
            "use_soft_label": False
        },
        {
            "name": "4. Full (Focal + Contrastive + Soft Label)",
            "use_contrastive": True,
            "focal_gamma": 2.0,
            "use_soft_label": True
        }
    ]
    
    results_summary = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f" {config['name']}")
        print(f"{'='*80}")
        
        # 准备数据
        if config['use_soft_label']:
            # 使用软标签
            lp = LabelPropagationSoftLabelGenerator(n_neighbors=30, alpha=0.9)
            soft_p_train, _, _, _ = lp.generate(Z_train, y_train)
        else:
            # 使用硬标签
            soft_p_train = y_train.astype(np.float32)
        
        # 构建全局数据
        Z_all = np.concatenate([Z_train, Z_val, Z_test], axis=0)
        y_all = np.concatenate([y_train, y_val, y_test], axis=0)
        
        soft_p_all = np.zeros(len(y_all), dtype=np.float32)
        soft_p_all[:len(y_train)] = soft_p_train
        soft_p_all[len(y_train):len(y_train)+len(y_val)] = y_val.astype(np.float32)
        soft_p_all[len(y_train)+len(y_val):] = y_test.astype(np.float32)
        
        train_mask = np.zeros(len(y_all), dtype=bool)
        val_mask = np.zeros(len(y_all), dtype=bool)
        test_mask = np.zeros(len(y_all), dtype=bool)
        
        train_mask[:len(y_train)] = True
        val_mask[len(y_train):len(y_train)+len(y_val)] = True
        test_mask[len(y_train)+len(y_val):] = True
        
        # 训练
        model = train_gat_soft_with_contrastive(
            Z_all=Z_all,
            y_all=y_all,
            soft_p_all=soft_p_all,
            train_mask=train_mask,
            val_mask=val_mask,
            k=30,
            lr=1e-3,
            epochs=100,  # 缩短epoch加速消融
            boundary_lambda=2.0,
            focal_gamma=config['focal_gamma'],
            use_contrastive=config['use_contrastive'],
            contrastive_weight=0.5,
            device=device
        )
        
        # 评估
        model.eval()
        with torch.no_grad():
            dst, src = build_knn_edges(Z_all, k=30)
            X = torch.tensor(Z_all, device=device)
            dst_t = torch.tensor(dst, device=device)
            src_t = torch.tensor(src, device=device)
            
            logits = model(X, dst_t, src_t, return_features=False)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # 验证集阈值搜索
        best_t, _ = evaluate_with_threshold_search(y_all[val_mask], probs[val_mask])
        
        # 测试集评估
        y_pred = (probs[test_mask] >= best_t).astype(int)
        f1 = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, probs[test_mask])
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        gmean = np.sqrt((tp/(tp+fn)) * (tn/(tn+fp)))
        
        results_summary.append({
            "name": config['name'],
            "f1": f1,
            "auc": auc,
            "gmean": gmean
        })
        
        print(f"\n测试集结果: F1={f1:.4f}, AUC={auc:.4f}, G-Mean={gmean:.4f}")
    
    # 打印汇总
    print("\n" + "="*80)
    print(" 消融实验汇总")
    print("="*80)
    print(f"{'配置':<50} {'Macro-F1':>10} {'AUC':>10} {'G-Mean':>10}")
    print("-"*80)
    for r in results_summary:
        print(f"{r['name']:<50} {r['f1']:>10.4f} {r['auc']:>10.4f} {r['gmean']:>10.4f}")
    print("="*80)


if __name__ == "__main__":

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/churn\CoTable\config.toml")
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/adult\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/bean\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/page\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/buddy\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/mammography\CoTable\config.toml")
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
    
    print(f"\n数据统计:")
    print(f"  训练集: {Z_train.shape}, 正类比例: {y_train.mean():.2%}")
    print(f"  验证集: {Z_val.shape}, 正类比例: {y_val.mean():.2%}")
    print(f"  测试集: {Z_test.shape}, 正类比例: {y_test.mean():.2%}")
    
    # ===== 运行完整pipeline =====
    model, results = run_improved_pipeline(
        Z_train, y_train,
        Z_val, y_val,
        Z_test, y_test,
        use_contrastive=True,
        focal_gamma=2.0,
        contrastive_weight=0.5,
        minority_weight=2.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # ===== 可选: 运行消融实验 =====
    # ablation_study(Z_train, y_train, Z_val, y_val, Z_test, y_test, 
    #                device="cuda" if torch.cuda.is_available() else "cpu")
