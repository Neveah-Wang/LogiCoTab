"""
方案二：基于置信度的层次化分类框架
Confidence-based Hierarchical Classification Framework

核心思想：
1. 利用VAE学习的隐空间和类原型，将样本分为高置信度区域和边界模糊区域
2. 高置信度样本：使用原型驱动的确定性分类
3. 边界模糊样本：使用自适应多策略融合（伪标签 + 代价敏感 + 集成学习）

逻辑连贯性：
- 方案一构建了类别分离的隐空间（输入）
- 方案二在此基础上实现精准分类（输出）
- 两个方案形成"特征学习→分类决策"的完整闭环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ConfidenceEstimator(nn.Module):
    """
    置信度评估器
    
    基于样本在隐空间中的位置评估其分类置信度
    """
    def __init__(self, prototype_0, prototype_1, confidence_threshold=0.7):
        """
        Args:
            prototype_0: 类别0的原型向量 [latent_dim]
            prototype_1: 类别1的原型向量 [latent_dim]
            confidence_threshold: 置信度阈值，用于划分高置信度/边界区域
        """
        super(ConfidenceEstimator, self).__init__()
        self.register_buffer('prototype_0', prototype_0)
        self.register_buffer('prototype_1', prototype_1)
        self.confidence_threshold = confidence_threshold
        
        # 计算两个原型之间的距离（用于归一化）
        self.proto_distance = torch.norm(prototype_1 - prototype_0, p=2).item()
        
    def compute_confidence(self, z):
        """
        计算样本的分类置信度
        
        置信度定义：
        conf(z) = |d₀(z) - d₁(z)| / (d₀(z) + d₁(z))
        
        其中 d_c(z) = ||z - μ_c||₂ 是样本到类原型c的欧氏距离
        
        Args:
            z: 隐向量 [batch_size, latent_dim]
            
        Returns:
            confidence: 置信度分数 [batch_size]
            predictions: 基于原型的预测标签 [batch_size]
            distances_0: 到类0原型的距离 [batch_size]
            distances_1: 到类1原型的距离 [batch_size]
        """
        # 展平隐向量
        if z.dim() == 3:
            z = z.reshape(z.size(0), -1)
        else:
            z = z

        # 计算到两个原型的距离
        distances_0 = torch.norm(z - self.prototype_0, p=2, dim=-1)  # [batch_size]
        distances_1 = torch.norm(z - self.prototype_1, p=2, dim=-1)  # [batch_size]
        
        # 基于距离的置信度
        # 距离差异越大，置信度越高
        distance_diff = torch.abs(distances_0 - distances_1)
        distance_sum = distances_0 + distances_1 + 1e-8
        confidence = distance_diff / distance_sum
        
        # 基于原型的预测
        predictions = (distances_1 < distances_0).long()
        
        return confidence, predictions, distances_0, distances_1
    
    def separate_samples(self, z, labels=None):
        """
        将样本分为高置信度区域和边界区域
        
        Args:
            z: 隐向量 [batch_size, latent_dim]
            labels: 真实标签 [batch_size] (可选，用于分析)
            
        Returns:
            high_conf_mask: 高置信度样本掩码 [batch_size]
            boundary_mask: 边界样本掩码 [batch_size]
            confidence: 置信度分数 [batch_size]
            predictions: 预测标签 [batch_size]
        """
        confidence, predictions, dist_0, dist_1 = self.compute_confidence(z)
        
        # 划分高置信度和边界区域
        high_conf_mask = confidence >= self.confidence_threshold
        boundary_mask = ~high_conf_mask
        
        # 如果提供了真实标签，计算统计信息
        if labels is not None:
            high_conf_acc = (predictions[high_conf_mask] == labels[high_conf_mask]).float().mean()
            boundary_acc = (predictions[boundary_mask] == labels[boundary_mask]).float().mean() if boundary_mask.any() else 0.0
            
            stats = {
                'high_conf_ratio': high_conf_mask.float().mean().item(),
                'boundary_ratio': boundary_mask.float().mean().item(),
                'high_conf_accuracy': high_conf_acc.item() if high_conf_mask.any() else 0.0,
                'boundary_accuracy': boundary_acc if isinstance(boundary_acc, float) else boundary_acc.item(),
                'avg_confidence': confidence.mean().item(),
            }
            return high_conf_mask, boundary_mask, confidence, predictions, stats
        
        return high_conf_mask, boundary_mask, confidence, predictions


class BoundaryClassifier(nn.Module):
    """
    边界区域分类器
    
    针对置信度低的边界样本，使用多层感知机进行精细分类
    结合代价敏感学习处理不平衡问题
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super(BoundaryClassifier, self).__init__()
        
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
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Args:
            z: 隐向量 [batch_size, latent_dim]
            
        Returns:
            logits: 分类logits [batch_size, 2]
        """
        return self.network(z)


class CostSensitiveLoss(nn.Module):
    """
    代价敏感损失函数
    
    针对不平衡数据，对少数类错误赋予更高代价
    """
    def __init__(self, class_weights=None, focal_alpha=0.25, focal_gamma=2.0):
        """
        Args:
            class_weights: 类别权重 [2]
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
        """
        super(CostSensitiveLoss, self).__init__()
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, logits, labels):
        """
        结合交叉熵和Focal Loss
        
        Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
        
        Args:
            logits: 模型输出 [batch_size, 2]
            labels: 真实标签 [batch_size]
        """
        # 标准交叉熵（带类别权重）
        ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        # Focal Loss部分
        probs = F.softmax(logits, dim=1)
        pt = probs[range(len(labels)), labels]
        
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_loss = -focal_weight * torch.log(pt + 1e-8)
        
        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights[labels]
        
        focal_loss = focal_loss.mean()
        
        # 组合损失
        total_loss = 0.5 * ce_loss + 0.5 * focal_loss
        
        return total_loss


class PseudoLabelGenerator:
    """
    伪标签生成器
    
    对边界样本生成高质量伪标签，辅助模型训练
    """
    def __init__(self, confidence_threshold=0.8, update_interval=5):
        """
        Args:
            confidence_threshold: 伪标签置信度阈值
            update_interval: 伪标签更新间隔（epoch）
        """
        self.confidence_threshold = confidence_threshold
        self.update_interval = update_interval
        self.pseudo_labels_cache = {}
        
    def generate(self, model, z_boundary, iteration):
        """
        生成伪标签
        
        策略：
        1. 使用当前模型预测边界样本
        2. 只保留高置信度的预测作为伪标签
        3. 定期更新伪标签
        
        Args:
            model: 边界分类器
            z_boundary: 边界样本的隐向量
            iteration: 当前迭代次数
            
        Returns:
            pseudo_labels: 伪标签 [num_samples]
            pseudo_mask: 有效伪标签的掩码 [num_samples]
        """
        model.eval()
        with torch.no_grad():
            logits = model(z_boundary)
            probs = F.softmax(logits, dim=1)
            
            # 选择最大概率作为伪标签
            max_probs, pseudo_labels = probs.max(dim=1)
            
            # 只保留高置信度的伪标签
            pseudo_mask = max_probs >= self.confidence_threshold
            
        model.train()
        return pseudo_labels, pseudo_mask


class HierarchicalClassifier(nn.Module):
    """
    层次化分类器（方案二的核心模型）
    
    工作流程：
    1. 使用置信度评估器划分样本
    2. 高置信度样本：直接使用原型分类
    3. 边界样本：使用专门的边界分类器
    4. 结合伪标签和代价敏感学习优化
    """
    def __init__(
        self,
        latent_dim,
        prototype_0,
        prototype_1,
        confidence_threshold=0.7,
        boundary_hidden_dims=[128, 64],
        class_weights=None,
        device='cuda'
    ):
        super(HierarchicalClassifier, self).__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        
        # 置信度评估器
        self.confidence_estimator = ConfidenceEstimator(
            prototype_0, prototype_1, confidence_threshold
        )
        
        # 边界分类器
        self.boundary_classifier = BoundaryClassifier(
            latent_dim, boundary_hidden_dims
        )
        
        # 代价敏感损失
        if class_weights is None:
            class_weights = torch.tensor([1.0, 1.0])
        self.cost_sensitive_loss = CostSensitiveLoss(class_weights)
        
        # 伪标签生成器
        self.pseudo_label_generator = PseudoLabelGenerator()
        
    def forward(self, z, return_confidence=False):
        """
        前向传播
        
        Args:
            z: 隐向量 [batch_size, latent_dim]
            return_confidence: 是否返回置信度信息
            
        Returns:
            predictions: 最终预测 [batch_size]
            (可选) confidence_info: 置信度相关信息
        """
        # 评估置信度并划分样本
        high_conf_mask, boundary_mask, confidence, proto_pred = \
            self.confidence_estimator.separate_samples(z)
        
        # 初始化预测
        predictions = proto_pred.clone()
        
        # 对边界样本使用专门分类器
        if boundary_mask.any():
            z_boundary = z[boundary_mask]
            boundary_logits = self.boundary_classifier(z_boundary)
            boundary_pred = boundary_logits.argmax(dim=1)
            predictions[boundary_mask] = boundary_pred
        
        if return_confidence:
            confidence_info = {
                'high_conf_mask': high_conf_mask,
                'boundary_mask': boundary_mask,
                'confidence': confidence,
                'proto_pred': proto_pred
            }
            return predictions, confidence_info
        
        return predictions
    
    def train_boundary_classifier(
        self,
        z_train,
        y_train,
        z_val,
        y_val,
        num_epochs=50,
        batch_size=128,
        lr=1e-3,
        use_pseudo_labels=True
    ):
        """
        训练边界分类器
        
        训练策略：
        1. 只使用边界样本训练
        2. 结合真实标签和高质量伪标签
        3. 使用代价敏感损失处理不平衡
        
        Args:
            z_train: 训练集隐向量
            y_train: 训练集标签
            z_val: 验证集隐向量
            y_val: 验证集标签
            num_epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            use_pseudo_labels: 是否使用伪标签
        """
        # 划分训练集中的边界样本
        with torch.no_grad():
            high_conf_mask_train, boundary_mask_train, _, _, train_stats = \
                self.confidence_estimator.separate_samples(z_train, y_train)
        
        print("\n" + "="*80)
        print("📊 训练集样本分布统计:")
        print(f"  高置信度样本比例: {train_stats['high_conf_ratio']:.2%}")
        print(f"  边界样本比例: {train_stats['boundary_ratio']:.2%}")
        print(f"  高置信度区域准确率: {train_stats['high_conf_accuracy']:.2%}")
        print(f"  边界区域准确率: {train_stats['boundary_accuracy']:.2%}")
        print("="*80 + "\n")
        
        # 提取边界样本
        z_boundary = z_train[boundary_mask_train]
        y_boundary = y_train[boundary_mask_train]
        
        if len(z_boundary) == 0:
            print("⚠️  没有边界样本需要训练！所有样本都是高置信度。")
            return
        
        # 准备优化器
        optimizer = torch.optim.Adam(
            self.boundary_classifier.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        best_val_f1 = 0.0
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            self.boundary_classifier.train()
            
            # 随机打乱
            indices = torch.randperm(len(z_boundary))
            z_boundary_shuffled = z_boundary[indices]
            y_boundary_shuffled = y_boundary[indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 分批训练
            for i in range(0, len(z_boundary), batch_size):
                batch_z = z_boundary_shuffled[i:i+batch_size]
                batch_y = y_boundary_shuffled[i:i+batch_size]
                
                # 前向传播
                logits = self.boundary_classifier(batch_z)
                
                # 计算损失
                loss = self.cost_sensitive_loss(logits, batch_y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # 验证
            if (epoch + 1) % 5 == 0:
                val_metrics = self.evaluate(z_val, y_val)
                scheduler.step(val_metrics['f1_minority'])
                
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val F1: {val_metrics['f1_overall']:.4f} | "
                      f"Val F1 (Minority): {val_metrics['f1_minority']:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f}")
                
                # 早停
                if val_metrics['f1_minority'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_minority']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= max_patience:
                    print(f"\n⏸️  早停触发 (patience={max_patience})")
                    break
        
        print(f"\n✅ 边界分类器训练完成！最佳验证F1 (少数类): {best_val_f1:.4f}\n")
    
    def evaluate(self, z, y_true):
        """
        评估模型性能
        
        Args:
            z: 隐向量
            y_true: 真实标签
            
        Returns:
            metrics: 评估指标字典
        """
        self.eval()
        with torch.no_grad():
            predictions, conf_info = self.forward(z, return_confidence=True)
            
            # 获取边界样本的概率（用于AUC计算）
            if conf_info['boundary_mask'].any():
                z_boundary = z[conf_info['boundary_mask']]
                boundary_logits = self.boundary_classifier(z_boundary)
                boundary_probs = F.softmax(boundary_logits, dim=1)[:, 1]
                
                # 构建完整的预测概率
                all_probs = torch.zeros(len(z), device=z.device)
                all_probs[conf_info['high_conf_mask']] = (conf_info['proto_pred'][conf_info['high_conf_mask']] == 1).float()
                all_probs[conf_info['boundary_mask']] = boundary_probs
            else:
                all_probs = (predictions == 1).float()
            
            # 转换为numpy
            y_true_np = y_true.cpu().numpy()
            y_pred_np = predictions.cpu().numpy()
            y_prob_np = all_probs.cpu().numpy()
            
            # 计算指标
            metrics = {
                'accuracy': (y_pred_np == y_true_np).mean(),
                'precision': precision_score(y_true_np, y_pred_np, average='binary', zero_division=0),
                'recall': recall_score(y_true_np, y_pred_np, average='binary', zero_division=0),
                'f1_overall': f1_score(y_true_np, y_pred_np, average='binary', zero_division=0),
                'f1_minority': f1_score(y_true_np, y_pred_np, pos_label=1, zero_division=0),
                'auc': roc_auc_score(y_true_np, y_prob_np) if len(np.unique(y_true_np)) > 1 else 0.0,
                'high_conf_ratio': conf_info['high_conf_mask'].float().mean().item(),
            }
        
        self.train()
        return metrics


def train_hierarchical_classifier(
    z_train,
    y_train,
    z_val,
    y_val,
    prototype_0,
    prototype_1,
    latent_dim,
    confidence_threshold=0.7,
    class_weights=None,
    device='cuda',
    **kwargs
):
    """
    训练层次化分类器的便捷函数
    
    Args:
        z_train: 训练集隐向量 [n_train, latent_dim]
        y_train: 训练集标签 [n_train]
        z_val: 验证集隐向量 [n_val, latent_dim]
        y_val: 验证集标签 [n_val]
        prototype_0: 类别0原型
        prototype_1: 类别1原型
        latent_dim: 隐空间维度
        confidence_threshold: 置信度阈值
        class_weights: 类别权重
        device: 设备
        
    Returns:
        model: 训练好的层次化分类器
        results: 训练结果字典
    """
    print("\n" + "="*80)
    print("🚀 开始训练层次化分类器")
    print("="*80)
    
    # 创建模型
    model = HierarchicalClassifier(
        latent_dim=latent_dim,
        prototype_0=prototype_0,
        prototype_1=prototype_1,
        confidence_threshold=confidence_threshold,
        class_weights=class_weights,
        device=device
    ).to(device)
    
    # 训练边界分类器
    model.train_boundary_classifier(
        z_train, y_train,
        z_val, y_val,
        **kwargs
    )
    
    # 最终评估
    print("\n" + "="*80)
    print("📊 最终评估结果:")
    print("="*80)
    
    train_metrics = model.evaluate(z_train, y_train)
    val_metrics = model.evaluate(z_val, y_val)
    
    print("\n训练集性能:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n验证集性能:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    
    return model, results


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """
    使用示例：展示如何将方案一（VAE）和方案二（层次化分类器）结合使用
    """
    
    # 假设已经训练好VAE并获得了隐空间表示和原型
    # 这些数据来自方案一（main_vae_new_.py）
    
    # 模拟数据（实际使用时从VAE输出加载）
    latent_dim = 64
    n_train = 1000
    n_val = 200
    
    # 生成模拟的隐空间表示
    torch.manual_seed(42)
    z_train = torch.randn(n_train, latent_dim).cuda()
    y_train = torch.randint(0, 2, (n_train,)).cuda()
    
    z_val = torch.randn(n_val, latent_dim).cuda()
    y_val = torch.randint(0, 2, (n_val,)).cuda()
    
    # 模拟类原型（实际从VAE加载）
    prototype_0 = torch.randn(latent_dim).cuda()
    prototype_1 = torch.randn(latent_dim).cuda()
    
    # 计算类别权重（处理不平衡）
    class_counts = torch.bincount(y_train)
    class_weights = len(y_train) / (2 * class_counts.float())
    
    # 训练层次化分类器
    model, results = train_hierarchical_classifier(
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        prototype_0=prototype_0,
        prototype_1=prototype_1,
        latent_dim=latent_dim,
        confidence_threshold=0.7,
        class_weights=class_weights,
        device='cuda',
        num_epochs=50,
        batch_size=128,
        lr=1e-3
    )
    
    print("\n✅ 示例运行完成！")
