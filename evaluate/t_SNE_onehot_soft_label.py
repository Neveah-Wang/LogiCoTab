import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')




def visualize_hard_vs_soft_labels(Z, y_hard, y_soft, title_prefix="", save_path=None):
    """
    使用t-SNE可视化硬标签和软标签的区别
    """

    # 使用t-SNE降维到2D
    print("正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
    Z_2d = tsne.fit_transform(Z)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ===== 左图: 硬标签 =====
    ax1 = axes[0]

    # 分别绘制两类样本
    for label in [0, 1]:
        mask = y_hard == label
        ax1.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                    c='red' if label == 1 else 'blue',
                    label=f'Class {label} (n={mask.sum()})',
                    alpha=0.6,
                    s=3,
                    edgecolors='none',
                    linewidth=0)

    ax1.set_title(f'{title_prefix}Hard Labels (Binary)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ===== 右图: 软标签 (连续色谱) =====
    ax2 = axes[1]

    # 使用colormap显示软标签的连续性
    scatter = ax2.scatter(Z_2d[:, 0], Z_2d[:, 1],
                          c=y_soft,
                          cmap='RdYlBu_r',  # 红(1) -> 黄(0.5) -> 蓝(0)
                          vmin=0, vmax=1,
                          alpha=0.6,
                          s=3,
                          edgecolors='none',
                          linewidth=0)

    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Soft Label Probability', fontsize=12, rotation=270, labelpad=20)

    ax2.set_title(f'{title_prefix}Soft Labels (Continuous)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()
    plt.close()

    """
    # ===== 额外: 软标签分布直方图 =====
    fig2, ax = plt.subplots(1, 1, figsize=(10, 5))

    # 按硬标签分组绘制软标签分布
    ax.hist(y_soft[y_hard == 0], bins=50, alpha=0.6, label='Hard Label 0', color='blue', edgecolor='black')
    ax.hist(y_soft[y_hard == 1], bins=50, alpha=0.6, label='Hard Label 1', color='red', edgecolor='black')

    ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
    ax.set_xlabel('Soft Label Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{title_prefix}Soft Label Distribution by Hard Label', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        hist_path = save_path.replace('.png', '_distribution.png')
    else:
        hist_path = '/mnt/user-data/outputs/hard_vs_soft_labels_distribution.png'

    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"分布图已保存到: {hist_path}")
    plt.close()

    # ===== 打印统计信息 =====
    print("\n" + "=" * 60)
    print(f"软标签统计信息 ({title_prefix})")
    print("=" * 60)
    print(f"总样本数: {len(y_soft)}")
    print(f"\n硬标签为0的样本 (n={sum(y_hard == 0)}):")
    print(f"  软标签均值: {y_soft[y_hard == 0].mean():.4f}")
    print(f"  软标签标准差: {y_soft[y_hard == 0].std():.4f}")
    print(f"  软标签范围: [{y_soft[y_hard == 0].min():.4f}, {y_soft[y_hard == 0].max():.4f}]")

    print(f"\n硬标签为1的样本 (n={sum(y_hard == 1)}):")
    print(f"  软标签均值: {y_soft[y_hard == 1].mean():.4f}")
    print(f"  软标签标准差: {y_soft[y_hard == 1].std():.4f}")
    print(f"  软标签范围: [{y_soft[y_hard == 1].min():.4f}, {y_soft[y_hard == 1].max():.4f}]")

    print(f"\n全局软标签:")
    print(f"  均值: {y_soft.mean():.4f}")
    print(f"  标准差: {y_soft.std():.4f}")
    print(f"  中位数: {np.median(y_soft):.4f}")

    # 计算软标签的"置信度"
    confidence = np.abs(y_soft - 0.5) * 2  # 将[0,1]映射到[0,1],其中0.5最不确定
    print(f"\n软标签置信度(距离0.5的程度):")
    print(f"  平均置信度: {confidence.mean():.4f}")
    print(f"  低置信度样本(<0.3): {sum(confidence < 0.3)} ({sum(confidence < 0.3) / len(confidence) * 100:.1f}%)")

    print("=" * 60 + "\n")
    """
