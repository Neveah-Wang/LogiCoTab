import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

import lib
from lib.make_dataset import make_dataset
from TabClassifierfree.VAE import Model_VAE
from lib.bert_util import make_dataset_and_encode, get_bert_model


def calculate_metrics(data, labels):
    """
    计算类内紧密度(ICC)、类间分离度(ICS)、分离紧密度比(SCR)
    :param data: 样本矩阵 (n_samples, n_dims)
    :param labels: 标签向量 (n_samples,)
    :return: dict 包含icc0, icc1, ics, scr
    """
    # 分离正负样本
    data_0 = data[labels == 0]
    data_1 = data[labels == 1]

    # 计算类中心
    mu_0 = np.mean(data_0, axis=0) if len(data_0) > 0 else np.zeros(data.shape[1])
    mu_1 = np.mean(data_1, axis=0) if len(data_1) > 0 else np.zeros(data.shape[1])

    # 类内紧密度（ICC）：同类样本到中心的平均距离
    icc0 = np.mean(pairwise_distances(data_0, mu_0.reshape(1, -1), metric='euclidean')) if len(data_0) > 0 else 0
    icc1 = np.mean(pairwise_distances(data_1, mu_1.reshape(1, -1), metric='euclidean')) if len(data_1) > 0 else 0

    # 类间分离度（ICS）：类中心距离
    ics = np.linalg.norm(mu_0 - mu_1, ord=2)

    # 分离紧密度比（SCR）
    scr = ics / ((icc0 + icc1) / 2) if (icc0 + icc1) > 0 else 0

    return {
        "icc0": icc0, "icc1": icc1, "avg_icc": (icc0 + icc1) / 2,
        "ics": ics, "scr": scr
    }


def plot_metrics_comparison(metrics_original, metrics_latent):
    """
    绘制原始数据和隐空间数据的量化指标对比柱状图
    :param metrics_original: 原始数据的量化指标（calculate_metrics输出）
    :param metrics_latent: 隐空间数据的量化指标（calculate_metrics输出）
    """
    # 准备对比数据
    metrics_names = ["平均类内紧密度(ICC)", "类间分离度(ICS)", "分离紧密度比(SCR)"]
    original_values = [metrics_original["avg_icc"], metrics_original["ics"], metrics_original["scr"]]
    latent_values = [metrics_latent["avg_icc"], metrics_latent["ics"], metrics_latent["scr"]]

    # 创建柱状图
    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, original_values, width, label="original", color="lightcoral")
    rects2 = ax.bar(x + width / 2, latent_values, width, label="VAE latent", color="lightskyblue")

    # 添加标签和标题
    ax.set_xlabel("评价指标", fontsize=12)
    ax.set_ylabel("指标值", fontsize=12)
    ax.set_title("原始数据 vs VAE隐空间数据 量化指标对比", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()

    # 在柱状图上标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(f"D:\Study\自学\表格数据生成\LogiCoTab-vae\evaluate/t-SNE_LogicalVAE/\metrics_comparison_{raw_config['dataname']}.pdf",  format='pdf', dpi=300, bbox_inches="tight")
    plt.show()


def plot_tsne_comparison(original_data, latent_data, labels, title_suffix=""):
    """
    绘制原始数据和隐空间数据的t-SNE对比图
    :param original_data: 原始特征数据 (n_samples, n_feat)
    :param latent_data: 隐空间数据 (n_samples, n_latent)
    :param labels: 样本标签 (n_samples,)
    :param title_suffix: 图标题后缀（如“带对比约束VAE”）
    """
    # 初始化t-SNE模型（固定随机种子保证可复现）
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

    # 对原始数据和隐空间数据分别做t-SNE降维
    original_tsne = tsne.fit_transform(original_data)
    latent_tsne = tsne.fit_transform(latent_data)

    # 创建2行1列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制原始数据t-SNE
    sns.scatterplot(
        x=original_tsne[:, 0],
        y=original_tsne[:, 1],
        hue=labels,                 # 按标签着色
        palette=["red", "blue"],    # 颜色映射：标签0=红色，标签1=蓝色
        ax=ax1,                     # 绘制在第一个坐标轴上
        alpha=0.8,                  # 0~1：透明~完全不透明
        s=3,                        # 点的大小为5
        legend="full",              # 显示完整图例
        edgecolors='none',          # 去掉点的边框颜色
        linewidths=0,               # 边框线宽为0
    )
    ax1.set_title(f"Original data t-SNE{title_suffix}", fontsize=14)
    ax1.set_xlabel("t-SNE x")
    ax1.set_ylabel("t-SNE y")
    ax1.legend()

    # 绘制隐空间数据t-SNE
    sns.scatterplot(
        x=latent_tsne[:, 0],
        y=latent_tsne[:, 1],
        hue=labels,
        palette=["red", "blue"],
        ax=ax2,
        alpha=1,
        s=3,
        legend="full",
        edgecolors='none',
        linewidths=0,
    )
    ax2.set_title(f"VAE latent data t-SNE {title_suffix}", fontsize=14)
    ax2.set_xlabel("t-SNE x")
    ax2.set_ylabel("t-SNE y")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"D:\Study\自学\表格数据生成\LogiCoTab-vae\evaluate/t-SNE_LogicalVAE/tsne_{raw_config['dataname']}.pdf", format='pdf', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/winequality\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/yeast_me2\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/bean\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/magic\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/adult\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/LogiCoTab-vae\exp/churn\CoTable\config.toml")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', metavar='FILE')
    # args = parser.parse_args()
    # raw_config = lib.util.load_config(args.config)

    # ----------- Step1: 准备数据------------
    real_data_path = raw_config['real_data_path']
    parent_dir = raw_config['parent_dir']
    device = torch.device(raw_config['device'])

    dataset = make_dataset(real_data_path, raw_config)

    # 得到原始数据
    if dataset.X_cat is None:
        original_data = dataset.X_num['train']
    else:
        original_data = np.concatenate((dataset.X_num['train'], dataset.X_cat['train']), axis=1)

    # 得到隐空间数据
    """
    berttokenizer, bertmodel = get_bert_model(raw_config)
    all_pooler_outputs_train = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False,
                                                       data_path=real_data_path, split='train')
    model = Model_VAE(
        num_layers=raw_config['VAE']['num_layers'],
        d_numerical=raw_config['num_numerical_features'],
        categories=raw_config['num_categorical'],
        d_token=raw_config['VAE']['d_token'],
        n_head=raw_config['VAE']['n_head'],
        factor=raw_config['VAE']['factor'],
        bias=True,
        bert_name=raw_config['model_params']['bert']
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(parent_dir, 'vae_model.pth'), map_location=device))
    X_train_num = torch.from_numpy(dataset.X_num['train']).to(device)
    X_train_cat = torch.from_numpy(dataset.X_cat['train']).to(device) if dataset.X_cat else None
    _, _, _, latent_data = model.VAE(X_train_num, X_train_cat, all_pooler_outputs_train)
    latent_data = latent_data.detach().cpu().numpy()
    """
    latent_data = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize.npy'))
    latent_data = latent_data.reshape(latent_data.shape[0], -1)

    labels = dataset.y['train'].flatten()

    # ----------- Step2: 绘制t-SNE对比图 ------------
    plot_tsne_comparison(original_data, latent_data, labels, title_suffix="")

    # ----------- Step3: 计算量化指标 ------------
    metrics_original = calculate_metrics(original_data, labels)
    metrics_latent = calculate_metrics(latent_data, labels)

    # 打印量化指标对比
    print("=" * 50)
    print("原始数据量化指标：")
    for k, v in metrics_original.items():
        print(f"{k}: {v:.4f}")
    print("-" * 50)
    print("VAE隐空间数据量化指标：")
    for k, v in metrics_latent.items():
        print(f"{k}: {v:.4f}")
    print("=" * 50)

    plot_metrics_comparison(metrics_original, metrics_latent)

"""
python "evaluate/t-SNE(original and latent).py" --config D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\churn\CoTable\config.toml
"""