"""
完整实验对比脚本
对比多种隐空间分类方案的性能
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from try_Prototype_Refinement.latent_classifier_training import (
    HardSampleMiningClassifier,
    PrototypeRefinementClassifier,
    LatentDataset,
    LatentClassifierTrainer
)


class BaselineMLPClassifier(nn.Module):
    """Baseline: 标准MLP分类器"""
    
    def __init__(self, latent_dim, hidden_dim=256, n_classes=2):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, z, labels=None):
        if z.dim() == 3:
            z = z[:, 0, :]
        
        logits = self.classifier(z)
        
        if labels is None:
            return logits
        
        loss = nn.functional.cross_entropy(logits, labels)
        return logits, loss


class FocalLossClassifier(nn.Module):
    """使用Focal Loss的分类器"""
    
    def __init__(self, latent_dim, hidden_dim=256, n_classes=2, gamma=2.0):
        super().__init__()
        
        self.gamma = gamma
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def focal_loss(self, logits, labels):
        probs = nn.functional.softmax(logits, dim=-1)
        target_probs = probs[range(len(labels)), labels]
        focal_weight = (1 - target_probs) ** self.gamma
        ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
        return (focal_weight * ce_loss).mean()
    
    def forward(self, z, labels=None):
        if z.dim() == 3:
            z = z[:, 0, :]
        
        logits = self.classifier(z)
        
        if labels is None:
            return logits
        
        loss = self.focal_loss(logits, labels)
        return logits, loss


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.results = {}
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
    
    def load_data(self):
        """加载数据"""
        print("\n" + "=" * 80)
        print("加载数据")
        print("=" * 80)
        
        # 加载隐向量
        train_z = np.load(os.path.join(
            self.config['latent_dir'], 'train_z.npy'
        ))
        val_z = np.load(os.path.join(
            self.config['latent_dir'], 'val_z.npy'
        ))
        
        # 加载标签
        train_data = pd.read_csv(os.path.join(
            self.config['data_dir'], 'train.csv'
        ))
        val_data = pd.read_csv(os.path.join(
            self.config['data_dir'], 'val.csv'
        ))
        
        train_labels = train_data['label'].values
        val_labels = val_data['label'].values
        
        print(f"训练集: {train_z.shape}, 标签: {train_labels.shape}")
        print(f"验证集: {val_z.shape}, 标签: {val_labels.shape}")
        print(f"训练集类别分布: {np.bincount(train_labels)}")
        print(f"验证集类别分布: {np.bincount(val_labels)}")
        
        # 计算不平衡率
        imbalance_ratio = np.bincount(train_labels).max() / np.bincount(train_labels).min()
        print(f"不平衡率: {imbalance_ratio:.2f}")
        
        return train_z, val_z, train_labels, val_labels
    
    def create_dataloaders(self, train_z, val_z, train_labels, val_labels):
        """创建数据加载器"""
        train_dataset = LatentDataset(train_z, train_labels)
        val_dataset = LatentDataset(val_z, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def get_latent_dim(self, z):
        """获取隐空间维度"""
        if z.ndim == 3:
            return z.shape[1] * z.shape[2]
        else:
            return z.shape[1]
    
    def train_and_evaluate(self, model, train_loader, val_loader, model_name):
        """训练并评估模型"""
        print(f"\n{'=' * 80}")
        print(f"训练 {model_name}")
        print(f"{'=' * 80}")
        
        trainer = LatentClassifierTrainer(
            model=model,
            device=self.device,
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # 训练
        save_dir = os.path.join(self.config['save_dir'], model_name)
        best_f1 = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['epochs'],
            save_dir=save_dir
        )
        
        # 加载最佳模型
        checkpoint = torch.load(
            os.path.join(save_dir, 'best_model.pth'),
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估
        val_results = checkpoint['val_results']
        
        print(f"\n{model_name} 最终结果:")
        print(f"  Accuracy: {val_results['accuracy']:.4f}")
        print(f"  Precision: {val_results['precision']:.4f}")
        print(f"  Recall: {val_results['recall']:.4f}")
        print(f"  F1: {val_results['f1']:.4f}")
        print(f"  AUC: {val_results['auc']:.4f}")
        print(f"  G-mean: {val_results['gmean']:.4f}")
        
        self.results[model_name] = val_results
        
        return model, val_results
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("\n" + "=" * 80)
        print("开始完整实验对比")
        print("=" * 80)
        
        # 加载数据
        train_z, val_z, train_labels, val_labels = self.load_data()
        train_loader, val_loader = self.create_dataloaders(
            train_z, val_z, train_labels, val_labels
        )
        
        latent_dim = self.get_latent_dim(train_z)
        print(f"\n隐空间维度: {latent_dim}")
        
        # 1. Baseline MLP
        print("\n" + "=" * 80)
        print("实验1: Baseline MLP")
        print("=" * 80)
        baseline = BaselineMLPClassifier(latent_dim).to(self.device)
        self.train_and_evaluate(baseline, train_loader, val_loader, "baseline_mlp")
        
        # 2. Focal Loss
        print("\n" + "=" * 80)
        print("实验2: Focal Loss")
        print("=" * 80)
        focal = FocalLossClassifier(latent_dim, gamma=2.0).to(self.device)
        self.train_and_evaluate(focal, train_loader, val_loader, "focal_loss")
        
        # 3. Hard Sample Mining
        print("\n" + "=" * 80)
        print("实验3: Hard Sample Mining")
        print("=" * 80)
        hsm = HardSampleMiningClassifier(
            latent_dim, gamma=2.0, alpha=0.7
        ).to(self.device)
        self.train_and_evaluate(hsm, train_loader, val_loader, "hard_sample_mining")
        
        # 4. Prototype Refinement
        print("\n" + "=" * 80)
        print("实验4: Prototype Refinement")
        print("=" * 80)
        proto = PrototypeRefinementClassifier(
            latent_dim, n_prototypes_per_class=3, beta=10.0
        ).to(self.device)
        self.train_and_evaluate(proto, train_loader, val_loader, "prototype_refinement")
        
        # 保存结果
        self.save_results()
        
        # 生成对比图
        self.plot_comparison()
        
        return self.results
    
    def save_results(self):
        """保存实验结果"""
        # 转换为DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # 保存CSV
        results_df.to_csv(
            os.path.join(self.config['save_dir'], 'experiment_results.csv')
        )
        
        # 保存JSON
        results_json = {}
        for model_name, results in self.results.items():
            results_json[model_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v.tolist()
                for k, v in results.items()
                if k != 'confusion_matrix'
            }
        
        with open(os.path.join(self.config['save_dir'], 'experiment_results.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✅ 结果已保存到: {self.config['save_dir']}")
    
    def plot_comparison(self):
        """绘制对比图"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'gmean']
        model_names = list(self.results.keys())
        
        # 提取数据
        data = {metric: [] for metric in metrics}
        for model_name in model_names:
            for metric in metrics:
                data[metric].append(self.results[model_name][metric])
        
        # 绘图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            bars = ax.bar(range(len(model_names)), data[metric], 
                         color=plt.cm.Set3(range(len(model_names))), alpha=0.8)
            
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1])
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config['save_dir'], 'metrics_comparison.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # 绘制混淆矩阵对比
        fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 4))
        
        for idx, model_name in enumerate(model_names):
            cm = self.results[model_name]['confusion_matrix']
            
            ax = axes[idx] if len(model_names) > 1 else axes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config['save_dir'], 'confusion_matrices.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print(f"✅ 对比图已保存")
    
    def generate_latex_table(self):
        """生成LaTeX表格"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'gmean']
        model_names = list(self.results.keys())
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Classification Performance Comparison}\n"
        latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex += "\\hline\n"
        latex += "Model & " + " & ".join([m.upper() for m in metrics]) + " \\\\\n"
        latex += "\\hline\n"
        
        for model_name in model_names:
            row = model_name.replace('_', ' ').title()
            for metric in metrics:
                value = self.results[model_name][metric]
                row += f" & {value:.3f}"
            row += " \\\\\n"
            latex += row
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"
        
        with open(os.path.join(self.config['save_dir'], 'results_table.tex'), 'w') as f:
            f.write(latex)
        
        print(f"✅ LaTeX表格已生成")
        
        return latex


def main():
    """主函数"""
    
    # 配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'latent_dir': './save_model/latent_data',
        'data_dir': './exp/mammography/CoTable',
        'save_dir': './experiment_results',
        'batch_size': 128,
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-4
    }
    
    # 创建实验运行器
    runner = ExperimentRunner(config)
    
    # 运行实验
    results = runner.run_all_experiments()
    
    # 生成LaTeX表格
    latex_table = runner.generate_latex_table()
    
    # 打印最终汇总
    print("\n" + "=" * 80)
    print("实验汇总")
    print("=" * 80)
    
    results_df = pd.DataFrame(results).T
    print(results_df[['accuracy', 'precision', 'recall', 'f1', 'auc', 'gmean']])
    
    # 找出最佳模型
    best_f1_model = max(results.items(), key=lambda x: x[1]['f1'])
    best_gmean_model = max(results.items(), key=lambda x: x[1]['gmean'])
    
    print(f"\n🏆 最佳F1模型: {best_f1_model[0]} (F1={best_f1_model[1]['f1']:.4f})")
    print(f"🏆 最佳G-mean模型: {best_gmean_model[0]} (G-mean={best_gmean_model[1]['gmean']:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ 所有实验完成!")
    print(f"📊 结果保存在: {config['save_dir']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
