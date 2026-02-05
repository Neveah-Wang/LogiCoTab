"""
方案一（VAE）+ 方案二（层次化分类器）集成示例
Integration Example: VAE (Solution 1) + Hierarchical Classifier (Solution 2)

使用流程：
1. 运行方案一（main_vae_new_.py）训练VAE，生成隐空间表示和类原型
2. 加载方案一的输出
3. 训练方案二的层次化分类器
4. 使用层次化分类器进行预测和评估
"""
import lib
from lib.make_dataset import make_dataset

import torch
import numpy as np
import os
from try_Hierarchical_classifier.hierarchical_classifier import (
    train_hierarchical_classifier
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置参数 ====================
class Config:
    """配置类"""
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/churn\CoTable\config.toml")

    # 路径配置
    vae_output_dir = raw_config['parent_dir']  # 方案一的输出目录
    latent_data_dir = os.path.join(vae_output_dir, "latent_data")
    
    # 设备配置
    device = raw_config['device']
    
    # 层次化分类器配置
    confidence_threshold = 0.7  # 置信度阈值
    boundary_hidden_dims = [128, 64]  # 边界分类器隐藏层
    
    # 训练配置
    num_epochs = 50
    batch_size = 128
    learning_rate = 1e-3
    patience = 10
    
    # 伪标签配置
    use_pseudo_labels = True
    pseudo_confidence = 0.8


def load_vae_outputs(raw_config, latent_data_dir, device):
    """
    加载方案一（VAE）的输出
    
    Args:
        latent_data_dir: 隐空间数据目录
        device: 设备
        
    Returns:
        data_dict: 包含隐向量、标签和原型的字典
    """
    print("\n" + "="*80)
    print("📂 加载方案一（VAE）的输出")
    print("="*80)
    
    # 加载隐向量
    z_train = torch.from_numpy(
        np.load(os.path.join(latent_data_dir, 'latent_z_after_reparameterize.npy'))
    ).float().to(device)

    z_val = torch.from_numpy(
        np.load(os.path.join(latent_data_dir, 'latent_z_after_reparameterize_val.npy'))
    ).float().to(device)

    if z_train.dim() == 3:
        z_train = z_train.reshape(z_train.size(0), -1)
    else:
        z_train = z_train

    if z_val.dim() == 3:
        z_val = z_val.reshape(z_val.size(0), -1)
    else:
        z_val = z_val
    
    # 加载类原型
    prototypes = np.load(
        os.path.join(latent_data_dir, 'prototypes.npy'),
        allow_pickle=True
    ).item()
    
    prototype_0 = torch.from_numpy(prototypes['prototype_0']).float().to(device)
    prototype_1 = torch.from_numpy(prototypes['prototype_1']).float().to(device)
    proto_distance = prototypes['proto_distance']
    
    print(f"✅ 隐向量加载完成:")
    print(f"   - 训练集: {z_train.shape}")
    print(f"   - 验证集: {z_val.shape}")
    print(f"   - 类原型距离: {proto_distance:.4f}")
    print(f"   - 隐空间维度: {z_train.shape[1]}")
    
    # 加载标签（需要从原始数据集获取）
    # 这里假设标签也被保存了，实际使用时根据情况调整
    # try:
    #     y_train = torch.from_numpy(
    #         np.load(os.path.join(latent_data_dir, 'train_labels.npy'))
    #     ).long().to(device)
    #     y_val = torch.from_numpy(
    #         np.load(os.path.join(latent_data_dir, 'val_labels.npy'))
    #     ).long().to(device)
    # except FileNotFoundError:
    #     print("⚠️  未找到保存的标签，需要手动加载原始数据集的标签")
    #     print("提示：可以在方案一的代码中添加标签保存逻辑")
    #     raise

    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y_train = torch.from_numpy(dataset.y['train'].flatten()).long().to(device)
    y_val = torch.from_numpy(dataset.y['val'].flatten()).long().to(device)

    # 计算类别权重（用于代价敏感学习）
    class_counts = torch.bincount(y_train)
    class_weights = len(y_train) / (2 * class_counts.float())
    
    print(f"\n📊 数据集统计:")
    print(f"   - 类别0数量: {class_counts[0].item()}")
    print(f"   - 类别1数量: {class_counts[1].item()}")
    print(f"   - 不平衡比: {max(class_counts).item() / min(class_counts).item():.2f}:1")
    print(f"   - 类别权重: {class_weights.cpu().numpy()}")
    
    data_dict = {
        'z_train': z_train,
        'y_train': y_train,
        'z_val': z_val,
        'y_val': y_val,
        'prototype_0': prototype_0,
        'prototype_1': prototype_1,
        'proto_distance': proto_distance,
        'class_weights': class_weights,
        'latent_dim': z_train.shape[1]
    }
    
    return data_dict


def train_solution2(data_dict, config):
    """
    训练方案二的层次化分类器
    
    Args:
        data_dict: 方案一的输出数据
        config: 配置对象
        
    Returns:
        model: 训练好的层次化分类器
        results: 训练结果
    """
    print("\n" + "="*80)
    print("🚀 开始训练方案二：层次化分类器")
    print("="*80)
    
    # 训练模型
    model, results = train_hierarchical_classifier(
        z_train=data_dict['z_train'],
        y_train=data_dict['y_train'],
        z_val=data_dict['z_val'],
        y_val=data_dict['y_val'],
        prototype_0=data_dict['prototype_0'],
        prototype_1=data_dict['prototype_1'],
        latent_dim=data_dict['latent_dim'],
        confidence_threshold=config.confidence_threshold,
        class_weights=data_dict['class_weights'],
        device=config.device,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        use_pseudo_labels=config.use_pseudo_labels
    )
    
    return model, results


def evaluate_and_visualize(model, data_dict, config, save_dir="visualizations"):
    """
    评估模型并生成可视化结果
    
    Args:
        model: 训练好的模型
        data_dict: 数据字典
        config: 配置对象
        save_dir: 可视化结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 详细评估与可视化")
    print("="*80)
    
    # ==================== 1. 在验证集上评估 ====================
    model.eval()
    with torch.no_grad():
        predictions, conf_info = model.forward(
            data_dict['z_val'], 
            return_confidence=True
        )
    
    y_true = data_dict['y_val'].cpu().numpy()
    y_pred = predictions.cpu().numpy()
    
    # 打印分类报告
    print("\n📋 分类报告:")
    print(classification_report(
        y_true, y_pred, 
        target_names=['Class 0', 'Class 1'],
        digits=4
    ))
    
    # ==================== 2. 混淆矩阵可视化 ====================
    plt.figure(figsize=(15, 5))
    
    # 总体混淆矩阵
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 高置信度区域混淆矩阵
    plt.subplot(1, 3, 2)
    high_conf_mask = conf_info['high_conf_mask'].cpu().numpy()
    if high_conf_mask.any():
        cm_high = confusion_matrix(
            y_true[high_conf_mask], 
            y_pred[high_conf_mask]
        )
        sns.heatmap(cm_high, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title(f'High Confidence Region\n({high_conf_mask.sum()} samples)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 边界区域混淆矩阵
    plt.subplot(1, 3, 3)
    boundary_mask = conf_info['boundary_mask'].cpu().numpy()
    if boundary_mask.any():
        cm_boundary = confusion_matrix(
            y_true[boundary_mask], 
            y_pred[boundary_mask]
        )
        sns.heatmap(cm_boundary, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title(f'Boundary Region\n({boundary_mask.sum()} samples)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存至: {os.path.join(save_dir, 'confusion_matrices.png')}")
    
    # ==================== 3. 置信度分布可视化 ====================
    plt.figure(figsize=(12, 5))
    
    confidence = conf_info['confidence'].cpu().numpy()
    
    # 按真实标签分组的置信度分布
    plt.subplot(1, 2, 1)
    plt.hist(confidence[y_true == 0], bins=50, alpha=0.6, label='Class 0', color='blue')
    plt.hist(confidence[y_true == 1], bins=50, alpha=0.6, label='Class 1', color='red')
    plt.axvline(config.confidence_threshold, color='green', linestyle='--', 
                linewidth=2, label=f'Threshold={config.confidence_threshold}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 按预测正确性分组的置信度分布
    plt.subplot(1, 2, 2)
    correct = (y_true == y_pred)
    plt.hist(confidence[correct], bins=50, alpha=0.6, label='Correct', color='green')
    plt.hist(confidence[~correct], bins=50, alpha=0.6, label='Incorrect', color='orange')
    plt.axvline(config.confidence_threshold, color='black', linestyle='--', 
                linewidth=2, label=f'Threshold={config.confidence_threshold}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Prediction Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 置信度分布已保存至: {os.path.join(save_dir, 'confidence_distribution.png')}")
    
    # ==================== 4. 区域性能对比 ====================
    print("\n📈 区域性能对比:")
    print("-" * 60)
    
    # 高置信度区域
    if high_conf_mask.any():
        high_acc = (y_true[high_conf_mask] == y_pred[high_conf_mask]).mean()
        print(f"高置信度区域:")
        print(f"  样本数量: {high_conf_mask.sum()}")
        print(f"  样本比例: {high_conf_mask.mean():.2%}")
        print(f"  准确率: {high_acc:.4f}")
    
    # 边界区域
    if boundary_mask.any():
        boundary_acc = (y_true[boundary_mask] == y_pred[boundary_mask]).mean()
        print(f"\n边界区域:")
        print(f"  样本数量: {boundary_mask.sum()}")
        print(f"  样本比例: {boundary_mask.mean():.2%}")
        print(f"  准确率: {boundary_acc:.4f}")
    
    print("-" * 60)


def save_model(model, save_path):
    """
    保存训练好的模型
    
    Args:
        model: 训练好的模型
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'confidence_threshold': model.confidence_estimator.confidence_threshold,
        'proto_distance': model.confidence_estimator.proto_distance,
    }, save_path)
    print(f"\n💾 模型已保存至: {save_path}")


def main():
    """主函数"""
    # 初始化配置
    config = Config()
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/churn\CoTable\config.toml")
    
    print("\n" + "="*80)
    print("🎯 方案一（VAE）+ 方案二（层次化分类器）集成示例")
    print("="*80)
    print(f"设备: {config.device}")
    print(f"置信度阈值: {config.confidence_threshold}")
    print(f"训练轮数: {config.num_epochs}")
    print("="*80)
    
    # 步骤1：加载方案一的输出
    data_dict = load_vae_outputs(raw_config, config.latent_data_dir, config.device)
    
    # 步骤2：训练方案二的层次化分类器
    model, results = train_solution2(data_dict, config)
    
    # 步骤3：详细评估与可视化
    evaluate_and_visualize(model, data_dict, config)
    
    # 步骤4：保存模型
    save_model(model, os.path.join(config.vae_output_dir, "hierarchical_classifier.pth"))
    
    print("\n" + "="*80)
    print("✅ 全部流程完成！")
    print("="*80)
    print("\n最终结果总结:")
    print(f"  训练集F1 (minority): {results['train_metrics']['f1_minority']:.4f}")
    print(f"  验证集F1 (minority): {results['val_metrics']['f1_minority']:.4f}")
    print(f"  验证集AUC: {results['val_metrics']['auc']:.4f}")
    print(f"  高置信度样本比例: {results['val_metrics']['high_conf_ratio']:.2%}")
    print("\n建议:")
    print("  1. 查看可视化结果以分析模型性能")
    print("  2. 根据置信度分布调整阈值")
    print("  3. 对比不同配置下的性能")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


"""
使用说明：

1. 确保已经运行方案一（main_vae_new_.py）并生成了隐空间数据
2. 修改Config类中的路径配置，指向方案一的输出目录
3. 在方案一的代码中添加标签保存逻辑（如果尚未保存）：
   
   # 在main_vae_new_.py的训练完成后添加：
   np.save(os.path.join(latent_dir, 'train_labels.npy'), dataset.y['train'])
   np.save(os.path.join(latent_dir, 'val_labels.npy'), dataset.y['val'])

4. 运行本脚本：
   python integration_example.py

5. 查看生成的可视化结果和模型文件

注意事项：
- 确保GPU可用（如果使用CUDA）
- 根据数据集大小调整batch_size和内存使用
- 置信度阈值可能需要根据具体数据集调整
- 可以通过消融实验优化超参数配置
"""
