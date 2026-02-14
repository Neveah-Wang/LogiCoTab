import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix


import lib
from lib.make_dataset import make_dataset

class SoftLabelGenerator:
    """软标签生成器"""

    def __init__(self,
                 n_neighbors=15,
                 alpha_minority=0.6,
                 alpha_majority=0.3,
                 density_weight=True,
                 adaptive_bandwidth=True):
        """
        参数:
            n_neighbors: KNN邻居数量
            alpha_minority: 少数类硬标签保留系数
            alpha_majority: 多数类硬标签保留系数
            density_weight: 是否使用密度加权
            adaptive_bandwidth: 是否使用自适应带宽
        """
        self.n_neighbors = n_neighbors
        self.alpha_minority = alpha_minority
        self.alpha_majority = alpha_majority
        self.density_weight = density_weight
        self.adaptive_bandwidth = adaptive_bandwidth

    def compute_local_density(self, Z, neighbors_indices, neighbors_distances):
        """计算局部密度"""
        n_samples = Z.shape[0]
        density = np.zeros(n_samples)

        for i in range(n_samples):
            # 截断距离：第k个邻居的距离
            d_c = neighbors_distances[i, -1]
            if d_c == 0:
                d_c = 1e-6

            # 计算密度
            dist_to_neighbors = neighbors_distances[i]
            density[i] = np.mean(np.exp(-dist_to_neighbors / d_c))

        # 归一化密度到[0.5, 1.5]范围，避免过度惩罚或奖励
        density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        density = 0.5 + density  # 范围[0.5, 1.5]

        return density

    def compute_adaptive_bandwidth(self, neighbors_distances):
        """计算自适应带宽（基于局部邻域的平均距离）"""
        # 使用每个样本的k近邻平均距离作为带宽
        sigma = np.mean(neighbors_distances, axis=1, keepdims=True)
        sigma = np.maximum(sigma, 1e-6)  # 避免除零
        return sigma

    def generate_soft_labels(self, Z, y):
        """
        生成软标签

        参数:
            Z: 隐空间表示 (n_samples, latent_dim)
            y: 原始硬标签 (n_samples,)

        返回:
            soft_labels: 软标签 (n_samples,)
        """
        n_samples = Z.shape[0]

        # 1. 找到k近邻
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1,
                                algorithm='auto').fit(Z)
        distances, indices = nbrs.kneighbors(Z)

        # 去掉自己(第一个邻居)
        neighbors_distances = distances[:, 1:]
        neighbors_indices = indices[:, 1:]

        # 2. 计算局部密度
        if self.density_weight:
            density = self.compute_local_density(Z, neighbors_indices,
                                                 neighbors_distances)
        else:
            density = np.ones(n_samples)

        # 3. 计算自适应带宽
        if self.adaptive_bandwidth:
            sigma = self.compute_adaptive_bandwidth(neighbors_distances)
        else:
            sigma = np.ones((n_samples, 1)) * np.median(neighbors_distances)

        # 4. 计算邻域软化分数
        soft_scores = np.zeros(n_samples)

        for i in range(n_samples):
            neighbor_idx = neighbors_indices[i]
            neighbor_dist = neighbors_distances[i]
            neighbor_labels = y[neighbor_idx]
            neighbor_density = density[neighbor_idx]

            # 距离权重（高斯核）
            weights = np.exp(-neighbor_dist ** 2 / (2 * sigma[i] ** 2))

            # 密度权重
            if self.density_weight:
                weights = weights * neighbor_density

            # 加权平均邻居标签
            soft_scores[i] = np.sum(weights * neighbor_labels) / (np.sum(weights) + 1e-8)

        # 5. 确定每个样本的alpha系数
        minority_class = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
        alpha = np.where(y == minority_class,
                         self.alpha_minority,
                         self.alpha_majority)

        # 6. 混合硬标签和软化分数
        soft_labels = alpha * y + (1 - alpha) * soft_scores

        # 确保在[0, 1]范围内
        soft_labels = np.clip(soft_labels, 0, 1)

        return soft_labels


class SoftLabelClassifier(nn.Module):
    """基于软标签的分类器"""

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        """
        参数:
            input_dim: 输入维度（隐空间维度）
            hidden_dims: 隐藏层维度列表
            dropout: dropout比例
        """
        super(SoftLabelClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        """
        前向传播

        参数:
            z: 隐空间表示

        返回:
            logits: 分类logits
        """
        return self.network(z).squeeze(-1)


class SoftLabelLoss(nn.Module):
    """软标签损失函数（结合BCE和KL散度）"""

    def __init__(self, temperature=1.0, kl_weight=0.1):
        """
        参数:
            temperature: 温度参数，控制软标签的平滑程度
            kl_weight: KL散度损失权重
        """
        super(SoftLabelLoss, self).__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight

    def forward(self, logits, soft_labels, hard_labels=None):
        """
        计算损失

        参数:
            logits: 模型输出 (batch_size,)
            soft_labels: 软标签 (batch_size,)
            hard_labels: 硬标签（可选，用于辅助监督）

        返回:
            loss: 总损失
        """
        # 软标签BCE损失
        probs = torch.sigmoid(logits / self.temperature)
        soft_bce = F.binary_cross_entropy(probs, soft_labels, reduction='mean')

        # KL散度损失（鼓励预测分布接近软标签分布）
        kl_loss = 0
        if self.kl_weight > 0:
            # 将软标签视为目标分布
            p = torch.stack([1 - soft_labels, soft_labels], dim=1)
            q = torch.stack([1 - probs, probs], dim=1)
            kl_loss = F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')

        # 硬标签辅助损失（可选）
        hard_bce = 0
        if hard_labels is not None:
            hard_bce = F.binary_cross_entropy_with_logits(
                logits, hard_labels.float(), reduction='mean'
            )

        # 总损失
        total_loss = soft_bce + self.kl_weight * kl_loss + 0.1 * hard_bce

        return total_loss


class ImbalancedSoftLabelClassifier:
    """完整的非平衡软标签分类方案"""

    def __init__(self, latent_dim,
                 hidden_dims=[256, 1024, 256],
                 n_neighbors=15,
                 alpha_minority=0.6,
                 alpha_majority=0.3,
                 temperature=1.0,
                 kl_weight=0.1,
                 learning_rate=1e-3,
                 device='cuda'):
        """
        参数:
            latent_dim: VAE隐空间维度
            其他参数见各组件说明
        """
        self.device = device

        # 软标签生成器
        self.soft_label_gen = SoftLabelGenerator(
            n_neighbors=n_neighbors,
            alpha_minority=alpha_minority,
            alpha_majority=alpha_majority
        )

        # 分类器
        self.classifier = SoftLabelClassifier(
            latent_dim, hidden_dims
        ).to(device)

        # 损失函数
        self.criterion = SoftLabelLoss(
            temperature=temperature,
            kl_weight=kl_weight
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

    def fit(self, Z_train, y_train,
            epochs=100, batch_size=256,
            Z_val=None, y_val=None,
            verbose=True):
        """
        训练分类器

        参数:
            Z_train: 训练集隐空间表示 (numpy array)
            y_train: 训练集标签 (numpy array)
            epochs: 训练轮数
            batch_size: 批次大小
            Z_val: 验证集（可选）
            y_val: 验证集标签（可选）
            verbose: 是否打印训练信息
        """
        # 1. 生成软标签
        if verbose:
            print("Generating soft labels...")
        soft_labels_train = self.soft_label_gen.generate_soft_labels(Z_train, y_train)

        # 转换为tensor
        Z_train_tensor = torch.FloatTensor(Z_train).to(self.device)
        soft_labels_tensor = torch.FloatTensor(soft_labels_train).to(self.device)
        hard_labels_tensor = torch.FloatTensor(y_train).to(self.device)

        # 2. 训练循环
        n_samples = Z_train.shape[0]
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.classifier.train()

            # 随机打乱数据
            indices = torch.randperm(n_samples)
            epoch_loss = 0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # 获取batch数据
                Z_batch = Z_train_tensor[batch_indices]
                soft_labels_batch = soft_labels_tensor[batch_indices]
                hard_labels_batch = hard_labels_tensor[batch_indices]

                # 前向传播
                logits = self.classifier(Z_batch)
                loss = self.criterion(logits, soft_labels_batch, hard_labels_batch)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            # 验证
            if Z_val is not None and y_val is not None:
                val_loss = self.evaluate(Z_val, y_val)
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}")

    def evaluate_loss(self, Z, y):
        """评估模型"""
        self.classifier.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            logits = self.classifier(Z_tensor)
            loss = F.binary_cross_entropy_with_logits(
                logits, y_tensor, reduction='mean'
            )
        return loss.item()

    def predict_proba(self, Z):
        """预测概率"""
        self.classifier.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            logits = self.classifier(Z_tensor)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def predict(self, Z, threshold=0.5):
        """预测类别"""
        probs = self.predict_proba(Z)
        return (probs >= threshold).astype(int)

    def evaluate(self, Z, y, threshold=0.5, help='eval'):
        # 预测概率和类别
        y_score = self.predict_proba(Z)
        y_pred = (y_score >= threshold).astype(int)

        f1 = f1_score(y, y_pred, average='macro')
        auc = roc_auc_score(y, y_score)
        mcc = matthews_corrcoef(y, y_pred)
        report = classification_report(y, y_pred, digits=4)

        # 计算G-mean
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        G_mean = np.sqrt(sensitivity * specificity)

        print("")
        print('-' * 20, help, '-' * 20)
        print(f"F1 (macro): {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"mcc: {mcc:.4f}")
        print(f"G_mean: {G_mean:.4f}")
        print("Report:")
        print(report)

        return auc, f1


# 使用示例
def example_usage():
    """使用示例"""
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/churn\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/adult\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/bean\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/buddy\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/mammography\CoTable\config.toml")
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/yeast_me2\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/shopper\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/magic\CoTable\config.toml")
    parent_dir = raw_config['parent_dir']

    # 1. 加载VAE生成的隐向量
    Z_train = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize.npy'))
    Z_test = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_val.npy'))

    # 2. 展平(如果是3D)
    if Z_train.ndim == 3:
        Z_train = Z_train.reshape(Z_train.shape[0], -1)
        Z_test = Z_test.reshape(Z_test.shape[0], -1)

    # 3. 加载标签(从原始数据)
    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y_train = dataset.y['train'].flatten()
    y_test = dataset.y['val'].flatten()

    print(f"\n训练集: {Z_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"测试集: {Z_test.shape}, 类别分布: {np.bincount(y_test)}")


    # 初始化并训练分类器
    clf = ImbalancedSoftLabelClassifier(
        latent_dim=Z_train.shape[1],
        hidden_dims=[256, 1024, 256],
        n_neighbors=20,
        alpha_minority=0.7,  # 少数类保留更多硬标签
        alpha_majority=0.3,  # 多数类更软
        temperature=1.0,
        kl_weight=0.1,
        learning_rate=1e-3,
        device='cuda'
    )

    # 训练
    clf.fit(Z_train, y_train, epochs=500, batch_size=1024, verbose=True)

    # 预测
    clf.evaluate(Z_test, y_test, threshold=0.5, help='val')
    clf.evaluate(Z_train, y_train, threshold=0.5, help='train')


if __name__ == "__main__":
    example_usage()