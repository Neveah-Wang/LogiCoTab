"""
AC-SVM训练和可视化脚本
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

import lib
from lib.make_dataset import make_dataset
from acsvm_classifier import ACSVMClassifier, gmean_score

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def visualize_ldc_distribution(ldc, y, save_path=None):
    """
    可视化局部密度对比(LDC)的分布
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. LDC直方图(按类别分开)
    axes[0].hist(ldc[y == 0], bins=50, alpha=0.6, label='Class 0', 
                color='blue', density=True)
    axes[0].hist(ldc[y == 1], bins=50, alpha=0.6, label='Class 1', 
                color='red', density=True)
    axes[0].axvline(x=1.0, color='black', linestyle='--', linewidth=2,
                   label='LDC=1 (边界)')
    axes[0].set_xlabel('Local Density Contrast (LDC)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('LDC Distribution by Class', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. LDC箱线图
    data_to_plot = [ldc[y == 0], ldc[y == 1]]
    bp = axes[1].boxplot(data_to_plot, labels=['Class 0', 'Class 1'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                   label='LDC=1')
    axes[1].set_ylabel('Local Density Contrast (LDC)', fontsize=12)
    axes[1].set_title('LDC Box Plot', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. LDC的CDF图
    sorted_ldc_0 = np.sort(ldc[y == 0])
    sorted_ldc_1 = np.sort(ldc[y == 1])
    cdf_0 = np.arange(1, len(sorted_ldc_0) + 1) / len(sorted_ldc_0)
    cdf_1 = np.arange(1, len(sorted_ldc_1) + 1) / len(sorted_ldc_1)
    
    axes[2].plot(sorted_ldc_0, cdf_0, label='Class 0', color='blue', linewidth=2)
    axes[2].plot(sorted_ldc_1, cdf_1, label='Class 1', color='red', linewidth=2)
    axes[2].axvline(x=1.0, color='black', linestyle='--', linewidth=2,
                   label='LDC=1')
    axes[2].set_xlabel('Local Density Contrast (LDC)', fontsize=12)
    axes[2].set_ylabel('Cumulative Probability', fontsize=12)
    axes[2].set_title('LDC Cumulative Distribution', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 LDC分布图保存至: {save_path}")
    
    plt.show()


def visualize_decision_boundary_2d(classifier, Z, y, save_path=None):
    """
    可视化决策边界(2D PCA投影)
    """
    # PCA降维到2D
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    
    # 创建网格
    h = 0.02  # 网格步长
    x_min, x_max = Z_2d[:, 0].min() - 1, Z_2d[:, 0].max() + 1
    y_min, y_max = Z_2d[:, 1].min() - 1, Z_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_2d)
    Z_pred = classifier.predict(grid_original)
    Z_pred = Z_pred.reshape(xx.shape)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 绘制决策边界
    ax.contourf(xx, yy, Z_pred, alpha=0.3, levels=[-0.5, 0.5, 1.5],
               colors=['blue', 'red'])
    ax.contour(xx, yy, Z_pred, levels=[0.5], colors='black', 
              linewidths=2, linestyles='--')
    
    # 绘制样本点
    scatter_0 = ax.scatter(Z_2d[y == 0, 0], Z_2d[y == 0, 1],
                          c='blue', alpha=0.6, s=50, edgecolors='black',
                          linewidths=0.5, label='Class 0')
    scatter_1 = ax.scatter(Z_2d[y == 1, 0], Z_2d[y == 1, 1],
                          c='red', alpha=0.6, s=50, edgecolors='black',
                          linewidths=0.5, label='Class 1')
    
    # 绘制支持向量
    if hasattr(classifier.final_svm, 'support_'):
        sv_indices = classifier.final_svm.support_
        sv_2d = Z_2d[sv_indices]
        ax.scatter(sv_2d[:, 0], sv_2d[:, 1], s=100, 
                  facecolors='none', edgecolors='green', linewidths=2,
                  label='Support Vectors')
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('AC-SVM Decision Boundary (2D PCA Projection)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 决策边界图保存至: {save_path}")
    
    plt.show()


def visualize_sample_difficulty(classifier, Z, y, ldc, save_path=None):
    """
    可视化样本难度分布
    """
    # 获取决策函数值
    decision_values = classifier.decision_function(Z)
    abs_decision = np.abs(decision_values)
    
    # PCA降维
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 1. 按LDC着色
    scatter1 = axes[0].scatter(Z_2d[:, 0], Z_2d[:, 1], c=ldc, 
                              cmap='RdYlGn_r', s=50, alpha=0.7,
                              edgecolors='black', linewidths=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('LDC (高=难)', fontsize=11)
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].set_title('Sample Difficulty by LDC', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 按到决策边界距离着色
    scatter2 = axes[1].scatter(Z_2d[:, 0], Z_2d[:, 1], c=abs_decision,
                              cmap='RdYlGn', s=50, alpha=0.7,
                              edgecolors='black', linewidths=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Distance to Boundary (高=易)', fontsize=11)
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].set_title('Sample Difficulty by Boundary Distance', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 样本难度图保存至: {save_path}")
    
    plt.show()


def visualize_stage_comparison(Z_train, y_train, svm_stage1, svm_stage2, svm_stage3):
    """
    可视化三个阶段的决策边界对比
    """
    # PCA降维
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z_train)
    
    # 创建网格
    h = 0.02
    x_min, x_max = Z_2d[:, 0].min() - 1, Z_2d[:, 0].max() + 1
    y_min, y_max = Z_2d[:, 1].min() - 1, Z_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_2d)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    stages = [
        (svm_stage1, 'Stage 1: Global Coarse Separation'),
        (svm_stage2, 'Stage 2: Local Refinement'),
        (svm_stage3, 'Stage 3: Boundary Polishing')
    ]
    
    for idx, (svm, title) in enumerate(stages):
        ax = axes[idx]
        
        # 预测网格
        Z_pred = svm.predict(grid_original).reshape(xx.shape)
        
        # 绘制决策边界
        ax.contourf(xx, yy, Z_pred, alpha=0.3, levels=[-0.5, 0.5, 1.5],
                   colors=['blue', 'red'])
        ax.contour(xx, yy, Z_pred, levels=[0.5], colors='black',
                  linewidths=2, linestyles='--')
        
        # 绘制样本
        ax.scatter(Z_2d[y_train == 0, 0], Z_2d[y_train == 0, 1],
                  c='blue', alpha=0.5, s=30, label='Class 0')
        ax.scatter(Z_2d[y_train == 1, 0], Z_2d[y_train == 1, 1],
                  c='red', alpha=0.5, s=30, label='Class 1')
        
        # 绘制支持向量
        if hasattr(svm, 'support_'):
            sv_2d = Z_2d[svm.support_]
            ax.scatter(sv_2d[:, 0], sv_2d[:, 1], s=80,
                      facecolors='none', edgecolors='green', linewidths=2,
                      label=f'SV ({len(svm.support_)})')
        
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - AC-SVM Classifier', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC曲线保存至: {save_path}")
    
    plt.show()


def main():
    """主函数:演示完整流程"""
    print("="*80)
    print("AC-SVM分类器训练与评估")
    print("="*80)
    
    # ==================== 1. 数据加载 ====================
    # 使用模拟数据演示
    """
    from sklearn.datasets import make_classification
    
    print("\n生成模拟不平衡数据...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],  # IR≈4
        flip_y=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    """

    # 使用真实数据
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/churn\CoTable\config.toml")
    parent_dir = raw_config['parent_dir']

    # 1. 加载VAE生成的隐向量
    X_train = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize.npy'))
    X_test = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_val.npy'))

    # 2. 展平(如果是3D)
    if X_train.ndim == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. 加载标签(从原始数据)
    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y_train = dataset.y['train'].flatten()
    y_test = dataset.y['val'].flatten()

    print(f"\n训练集: {X_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"测试集: {X_test.shape}, 类别分布: {np.bincount(y_test)}")
    
    # ==================== 2. 训练AC-SVM ====================
    classifier = ACSVMClassifier(
        k_neighbors=None,  # 自动选择
        alpha=0.7,
        beta=2.0,
        gamma=2.0,
        C_global_stage1=10.0,
        C_global_stage2=5.0,
        C_global_stage3=3.0,
        kernel="rbf",
        verbose=True
    )
    
    classifier.fit(X_train, y_train)
    
    # ==================== 3. 评估 ====================
    train_metrics = classifier.evaluate(X_train, y_train)
    classifier.print_metrics(train_metrics, dataset_name='(训练集)')
    
    test_metrics = classifier.evaluate(X_test, y_test)
    classifier.print_metrics(test_metrics, dataset_name='(测试集)')
    
    # ==================== 4. 可视化 ====================
    '''
    print("\n" + "="*80)
    print("生成可视化")
    print("="*80)
    
    # 计算LDC用于可视化
    ldc, _, _ = classifier.density_analyzer.compute_LDC(X_train, y_train)
    
    # LDC分布
    visualize_ldc_distribution(ldc, y_train)
    
    # 决策边界
    visualize_decision_boundary_2d(classifier, X_test, y_test)
    
    # 样本难度
    visualize_sample_difficulty(classifier, X_train, y_train, ldc)
    
    # 三阶段对比
    visualize_stage_comparison(
        X_train, y_train,
        classifier.svm_stage1,
        classifier.svm_stage2,
        classifier.svm_stage3
    )
    
    # ROC曲线
    y_proba = classifier.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_proba)
    
    print("\n" + "="*80)
    print("✅ 完成!")
    print("="*80)
    '''

if __name__ == '__main__':
    main()
