"""
完整的隐空间分类训练流程
结合VAE隐向量进行非平衡数据分类
"""
import lib
from lib.make_dataset import make_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import os


# ==================== 数据集 ====================
class LatentDataset(Dataset):
    """隐空间数据集"""
    
    def __init__(self, latent_vectors, labels):
        """
        Args:
            latent_vectors: numpy array [N, seq_len, d_token] 或 [N, d_token]
            labels: numpy array [N]
        """
        self.latent_vectors = torch.from_numpy(latent_vectors).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.latent_vectors[idx], self.labels[idx]



class PrototypeRefinementClassifier(nn.Module):
    """基于原型精炼的度量学习分类器"""
    
    def __init__(self, latent_dim, n_prototypes_per_class=3, 
                 n_classes=2, beta=10.0, margin=2.0):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_prototypes = n_prototypes_per_class
        self.n_classes = n_classes
        self.beta = beta
        self.margin = margin
        
        # 可学习的原型参数
        self.prototypes = nn.Parameter(
            torch.randn(n_classes, n_prototypes_per_class, latent_dim)
        )
        
        # 距离度量网络
        self.metric_transform = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def compute_distances(self, z, prototypes):
        """计算样本到原型的距离"""
        z_transformed = self.metric_transform(z)
        
        z_expanded = z_transformed.unsqueeze(1).unsqueeze(2)
        proto_expanded = prototypes.unsqueeze(0)
        
        distances = torch.norm(z_expanded - proto_expanded, dim=-1)
        
        return distances
    
    def compute_soft_assignments(self, distances):
        """计算软分配权重"""
        assignments = F.softmax(-self.beta * distances, dim=-1)
        return assignments
    
    def prototype_loss(self, z, labels):
        """计算原型损失"""
        distances = self.compute_distances(z, self.prototypes)
        assignments = self.compute_soft_assignments(distances)
        weighted_distances = (assignments * distances).sum(dim=-1)
        logits = -weighted_distances
        loss = F.cross_entropy(logits, labels)
        
        return loss, logits
    
    def separation_loss(self):
        """原型分离损失"""
        loss = 0
        count = 0
        
        for c1 in range(self.n_classes):
            for c2 in range(c1 + 1, self.n_classes):
                for k1 in range(self.n_prototypes):
                    for k2 in range(self.n_prototypes):
                        proto1 = self.prototypes[c1, k1]
                        proto2 = self.prototypes[c2, k2]
                        
                        dist = torch.norm(proto1 - proto2)
                        loss += F.relu(self.margin - dist)
                        count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, device=self.prototypes.device)
    
    def forward(self, z, labels=None):
        """前向传播"""
        if z.dim() == 3:
            z = z[:, 0, :]
        
        if labels is None:
            distances = self.compute_distances(z, self.prototypes)
            assignments = self.compute_soft_assignments(distances)
            weighted_distances = (assignments * distances).sum(dim=-1)
            logits = -weighted_distances
            return logits
        
        proto_loss, logits = self.prototype_loss(z, labels)
        sep_loss = self.separation_loss()
        
        total_loss = proto_loss + 0.1 * sep_loss
        
        return logits, total_loss


# ==================== 训练器 ====================
class LatentClassifierTrainer:
    """隐空间分类器训练器"""
    
    def __init__(self, model, device, lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.max_patience = 2000
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_difficulties = []
        
        for batch_z, batch_labels in train_loader:
            batch_z = batch_z.to(self.device)
            batch_labels = batch_labels.to(self.device)

            logits, loss = self.model(batch_z, batch_labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        
        results = {
            'loss': avg_loss,
            'accuracy': acc,
        }
        
        if all_difficulties:
            results['avg_difficulty'] = np.mean(all_difficulties)
        
        return results
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_z, batch_labels in val_loader:
                batch_z = batch_z.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 前向传播
                logits, loss = self.model(batch_z, batch_labels)
                
                # 统计
                total_loss += loss.item()
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        all_probs = np.array(all_probs)
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # G-mean (对不平衡数据重要)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = np.sqrt(sensitivity * specificity)
        
        results = {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'gmean': gmean,
            'confusion_matrix': cm,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        return results
    
    def train(self, train_loader, val_loader, epochs=100, save_dir='./checkpoints'):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 80)
        print("开始训练隐空间分类器")
        print("=" * 80)

        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:
            # print(f"\nEpoch {epoch + 1}/{epochs}")
            # print("-" * 80)
            
            # 训练
            train_results = self.train_epoch(train_loader)
            # print(f"Train Loss: {train_results['loss']:.4f}, "
            #       f"Acc: {train_results['accuracy']:.4f}")
            
            if 'avg_difficulty' in train_results:
                print(f"Avg Difficulty: {train_results['avg_difficulty']:.4f}")
            
            # 验证
            val_results = self.evaluate(val_loader)
            # print(f"Val Loss: {val_results['loss']:.4f}, "
            #       f"Acc: {val_results['accuracy']:.4f}")
            # print(f"Precision: {val_results['precision']:.4f}, "
            #       f"Recall: {val_results['recall']:.4f}, "
            #       f"F1: {val_results['f1']:.4f}")
            # print(f"AUC: {val_results['auc']:.4f}, "
            #       f"G-mean: {val_results['gmean']:.4f}")
            # print(f"Sensitivity: {val_results['sensitivity']:.4f}, "
            #       f"Specificity: {val_results['specificity']:.4f}")

            pbar.set_postfix({
                'epoch': epoch,
                'train_loss': train_results['loss'],
                'train_accuracy': train_results['accuracy'],
                'val_loss': val_results['loss'],
                'val_accuracy': val_results['accuracy'],
                'val_f1': val_results['f1'],
                'val_auc': val_results['auc'],
                'val_gmean': val_results['gmean'],
                'val_precision': val_results['precision'],
                'val_recall': val_results['recall'],
            })

            # 学习率调整
            self.scheduler.step(val_results['loss'])
            
            # 早停和模型保存
            if val_results['f1'] > self.best_val_f1:
                self.best_val_f1 = val_results['f1']
                self.best_val_loss = val_results['loss']
                self.patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_results': val_results
                }, os.path.join(save_dir, 'best_model.pth'))
                
                # print(f"✅ 保存最佳模型 (F1: {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.max_patience:
                    print(f"\n早停! 在epoch {epoch + 1}停止训练")
                    break
        
        print("\n" + "=" * 80)
        print(f"训练完成! 最佳验证F1: {self.best_val_f1:.4f}")
        print("=" * 80)
        
        return self.best_val_f1


# ==================== 主函数 ====================
def main():
    """
    主函数: 加载VAE隐向量并训练分类器
    """
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/churn\CoTable\config.toml")
    # ==================== 配置 ====================
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'latent_dir': f"{raw_config['parent_dir']}/latent_data",  # VAE保存的隐向量目录
        'data_dir': raw_config['real_data_path'],  # 原始数据目录(获取标签)
        'batch_size': 1200,
        'epochs': 1200,
        'lr': 1e-3,
        'save_dir': f"{raw_config['parent_dir']}/classification_checkpoints"
    }
    
    device = torch.device(raw_config['device'])
    print(f"使用设备: {device}")
    
    # ==================== 加载数据 ====================
    print("\n加载隐空间数据...")
    
    # 加载VAE生成的隐向量
    train_z = np.load(os.path.join(config['latent_dir'], 'latent_z_after_reparameterize.npy'))
    val_z = np.load(os.path.join(config['latent_dir'], 'latent_z_after_reparameterize_val.npy'))
    train_z = train_z.reshape(train_z.shape[0], -1)
    val_z = val_z.reshape(val_z.shape[0], -1)

    
    # 加载标签(从原始数据)
    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    train_labels = dataset.y['train'].flatten()
    val_labels = dataset.y['val'].flatten()

    print(f"训练集: {train_z.shape}, 标签: {train_labels.shape}")
    print(f"验证集: {val_z.shape}, 标签: {val_labels.shape}")
    print(f"类别分布 - 训练集: {np.bincount(train_labels)}")
    print(f"类别分布 - 验证集: {np.bincount(val_labels)}")
    
    # ==================== 创建数据集 ====================
    train_dataset = LatentDataset(train_z, train_labels)
    val_dataset = LatentDataset(val_z, val_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    # ==================== 创建模型 ====================
    # 推断latent_dim
    if train_z.ndim == 3:
        latent_dim = train_z.shape[1] * train_z.shape[2]
    else:
        latent_dim = train_z.shape[1]
    
    print(f"\n隐空间维度: {latent_dim}")

    model = PrototypeRefinementClassifier(
        latent_dim=latent_dim,
        n_prototypes_per_class=6,
        beta=10.0,
        margin=2.0
    )

    # ==================== 训练 ====================
    trainer = LatentClassifierTrainer(
        model=model,
        device=device,
        lr=config['lr']
    )
    
    best_f1 = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        save_dir=config['save_dir']
    )
    
    print(f"\n最终最佳F1分数: {best_f1:.4f}")


if __name__ == '__main__':
    main()

