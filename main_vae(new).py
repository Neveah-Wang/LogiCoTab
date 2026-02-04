import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
import os

import lib
from lib import util
from lib.make_dataset import make_dataset, prepare_dataloader_respectively
from lib.data_preprocess import inverse_transformer
from lib.tensorboard_logger import TensorBoardLogger
from TabClassifierfree.VAE import Model_VAE, Encoder_model, Decoder_model
from lib.bert_util import make_dataset_and_encode, get_bert_model


class AdaptiveVAELoss(nn.Module):
    """
    自适应VAE损失函数，结合重构、KL散度和有监督对比学习
    """

    def __init__(self, latent_dim, device, temperature=0.07, base_temperature=0.07,
                 momentum=0.9, margin=1.0):
        super(AdaptiveVAELoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.momentum = momentum
        self.margin = margin

        # 注册类原型为buffer（不参与梯度更新，但会保存到模型中）
        self.register_buffer('prototype_0', torch.zeros(latent_dim).to(self.device))
        self.register_buffer('prototype_1', torch.zeros(latent_dim).to(self.device))
        self.register_buffer('prototype_initialized', torch.tensor(False))

        # 用于统计类别分布
        self.register_buffer('class_counts', torch.zeros(2))

    def update_prototypes(self, z, labels):
        """
        使用动量更新类原型

        Args:
            z: 隐向量 [batch_size, seq_len, d_token] 或 [batch_size, latent_dim]
            labels: 标签 [batch_size]
        """

        with torch.no_grad():
            # 展平隐向量
            if z.dim() == 3:
                z_flat = z.reshape(z.size(0), -1)
            else:
                z_flat = z

            # 分离不同类别的样本
            mask_0 = (labels == 0)
            mask_1 = (labels == 1)

            if mask_0.any():
                batch_proto_0 = z_flat[mask_0].mean(dim=0)
                if not self.prototype_initialized:
                    self.prototype_0.copy_(batch_proto_0)
                else:
                    self.prototype_0.mul_(self.momentum).add_(
                        batch_proto_0, alpha=1 - self.momentum
                    )
                self.class_counts[0] += mask_0.sum().item()

            if mask_1.any():
                batch_proto_1 = z_flat[mask_1].mean(dim=0)
                if not self.prototype_initialized:
                    self.prototype_1.copy_(batch_proto_1)
                else:
                    self.prototype_1.mul_(self.momentum).add_(
                        batch_proto_1, alpha=1 - self.momentum
                    )
                self.class_counts[1] += mask_1.sum().item()

            if not self.prototype_initialized and mask_0.any() and mask_1.any():
                self.prototype_initialized.fill_(True)

    def compute_reconstruction_loss(self, X_num, X_cat, Recon_X_num, Recon_X_cat):
        """
        计算重构损失
        """
        # 数值特征：MSE
        mse_loss = F.mse_loss(Recon_X_num, X_num, reduction='mean')

        # 类别特征：交叉熵
        ce_loss = 0
        acc = 0
        total_num = 0

        if X_cat is not None and Recon_X_cat is not None:
            for idx, x_cat_logits in enumerate(Recon_X_cat):
                if x_cat_logits is not None:
                    ce_loss += F.cross_entropy(x_cat_logits, X_cat[:, idx],
                                               reduction='mean')
                    pred = x_cat_logits.argmax(dim=-1)
                    acc += (pred == X_cat[:, idx]).float().sum()
                    total_num += pred.shape[0]

            if total_num > 0:
                ce_loss /= (idx + 1)
                acc /= total_num
            else:
                ce_loss = torch.tensor(0.0, device=self.device)
                acc = torch.tensor(0.0, device=self.device)
        else:
            ce_loss = torch.tensor(0.0, device=self.device)
            acc = torch.tensor(0.0, device=self.device)

        return mse_loss + ce_loss, mse_loss, ce_loss, acc

    def compute_adaptive_kl_loss(self, mu, logvar, labels):
        """
        自适应KL散度损失

        核心思想：不同类别有不同的目标分布中心
        L_KL = Σ_c w_c · KL(q(z|x_c) || N(μ_c, I))

        其中 μ_c 是类别c的原型，w_c是类别权重
        """
        # 处理多维mu和logvar
        if mu.dim() == 3:
            batch_size, seq_len, d_token = mu.shape
            mu_flat = mu.reshape(batch_size, -1)
            logvar_flat = logvar.reshape(batch_size, -1)
        else:
            mu_flat = mu
            logvar_flat = logvar

        kl_loss = 0
        valid_classes = 0

        for class_id in [0, 1]:
            mask = (labels == class_id)
            if not mask.any():
                continue

            # 获取该类别的样本
            mu_c = mu_flat[mask]
            logvar_c = logvar_flat[mask]

            # 获取类原型作为目标均值
            if self.prototype_initialized:
                target_mu = self.prototype_0 if class_id == 0 else self.prototype_1
                target_mu = target_mu.to(self.device)

                # 计算到类原型的KL散度
                # KL(N(μ, σ²) || N(μ_proto, I))
                kl_c = -0.5 * torch.sum(
                    1 + logvar_c
                    - (mu_c - target_mu).pow(2)
                    - logvar_c.exp(),
                    dim=-1
                )
            else:
                # 初始阶段使用标准KL散度
                kl_c = -0.5 * torch.sum(
                    1 + logvar_c - mu_c.pow(2) - logvar_c.exp(),
                    dim=-1
                )

            # 类别权重（处理不平衡）
            weight = 1.0 / (self.class_counts[class_id] + 1e-6)
            kl_loss += weight * kl_c.mean()
            valid_classes += 1

        # 归一化
        if valid_classes > 0:
            kl_loss = kl_loss / valid_classes

        return kl_loss

    def compute_supervised_contrastive_loss(self, z, labels):
        """
        有监督对比学习损失

        公式：
        L_supcon = Σ_{i∈I} -1/|P(i)| Σ_{p∈P(i)} log[
            exp(z_i·z_p / τ) /
            Σ_{a∈A(i)} exp(z_i·z_a / τ)
        ]
        """
        # 展平隐向量
        if z.dim() == 3:
            z_flat = z.reshape(z.size(0), -1)
        else:
            z_flat = z

        batch_size = z_flat.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=self.device)

        # 归一化特征向量
        z_norm = F.normalize(z_flat, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature

        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(self.device)
        mask_negative = 1 - mask_positive

        # 移除对角线（自身）
        logits_mask = torch.ones_like(mask_positive).scatter_(
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask_positive = mask_positive * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        # 计算每个样本的正样本平均对数概率
        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / (mask_positive.sum(dim=1) + 1e-9)

        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def compute_prototype_separation_loss(self):
        """
        显式约束类原型分离

        L_sep = max(0, margin - ||μ_0 - μ_1||²)
        """
        if not self.prototype_initialized:
            return torch.tensor(0.0, device=self.device)

        proto_dist = torch.norm(self.prototype_0 - self.prototype_1, p=2)
        separation_loss = torch.clamp(self.margin - proto_dist, min=0.0)

        return separation_loss

    def get_prototype_distance(self):
        """
        获取类原型之间的距离（用于监控）
        """
        if not self.prototype_initialized:
            return 0.0
        return torch.norm(self.prototype_0 - self.prototype_1, p=2).item()

    def compute_intra_class_variance(self, z, labels):
        """
        计算类内方差 (Intra-class Variance)

        对每个类别c, 计算:
        Var_c = 1/N_c Σ_{i∈C_c} ||z_i - μ_c^proto||²

        总体类内方差 (加权平均):
        Var_intra = Σ_c (N_c/N) * Var_c

        Args:
            z: 隐向量 [batch_size, seq_len, d_token] 或 [batch_size, latent_dim]
            labels: 标签 [batch_size]

        Returns:
            intra_var_dict: 包含各类别方差和总体方差的字典
        """
        # 展平隐向量
        if z.dim() == 3:
            z_flat = z.reshape(z.size(0), -1)
        else:
            z_flat = z

        if not self.prototype_initialized:
            return {
                'intra_var_class_0': 0.0,
                'intra_var_class_1': 0.0,
                'intra_var_total': 0.0,
                'intra_var_weighted': 0.0
            }

        intra_var_dict = {}
        total_samples = z_flat.shape[0]
        weighted_var = 0.0

        for class_id in [0, 1]:
            mask = (labels == class_id)
            n_samples = mask.sum().item()

            if n_samples > 0:
                # 获取该类别的隐向量
                z_c = z_flat[mask]

                # 获取类原型
                proto_c = self.prototype_0 if class_id == 0 else self.prototype_1
                proto_c = proto_c.to(self.device)

                # 计算类内方差: 1/N_c Σ ||z_i - μ_proto||²
                distances_sq = torch.norm(z_c - proto_c, p=2, dim=1) ** 2
                var_c = distances_sq.mean().item()

                intra_var_dict[f'intra_var_class_{class_id}'] = var_c

                # 加权累积
                weighted_var += (n_samples / total_samples) * var_c
            else:
                intra_var_dict[f'intra_var_class_{class_id}'] = 0.0

        # 简单平均(不考虑样本数量)
        intra_var_dict['intra_var_total'] = np.mean([
            intra_var_dict['intra_var_class_0'],
            intra_var_dict['intra_var_class_1']
        ])

        # 加权平均(考虑样本数量,推荐使用)
        intra_var_dict['intra_var_weighted'] = weighted_var

        return intra_var_dict

    def forward(self, X_num, X_cat, Recon_X_num, Recon_X_cat,
                mu, logvar, z, labels, update_prototypes=True):
        """
        计算总损失

        Args:
            update_prototypes: 是否更新原型（训练时True，验证时False）

        Returns:
            total_loss, loss_dict
        """
        # 更新类原型（仅在训练时）
        if update_prototypes:
            self.update_prototypes(z.detach(), labels)

        # 1. 重构损失
        recon_loss, mse_loss, ce_loss, acc = self.compute_reconstruction_loss(
            X_num, X_cat, Recon_X_num, Recon_X_cat
        )

        # 2. 自适应KL散度
        kl_loss = self.compute_adaptive_kl_loss(mu, logvar, labels)

        # 3. 有监督对比损失
        contrast_loss = self.compute_supervised_contrastive_loss(z, labels)

        # 4. 类原型分离损失
        separation_loss = self.compute_prototype_separation_loss()

        # 5. 计算类内方差 (不参与反向传播,仅用于监控)
        with torch.no_grad():
            intra_var_dict = self.compute_intra_class_variance(z, labels)

        # 组合损失字典
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'mse_loss': mse_loss.item(),
            'ce_loss': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            'kl_loss': kl_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'separation_loss': separation_loss.item(),
            'acc': acc.item() if isinstance(acc, torch.Tensor) else acc,
            'proto_dist': self.get_prototype_distance(),
            # 新增: 类内方差指标
            'intra_var_class_0': intra_var_dict['intra_var_class_0'],
            'intra_var_class_1': intra_var_dict['intra_var_class_1'],
            'intra_var_total': intra_var_dict['intra_var_total'],
            'intra_var_weighted': intra_var_dict['intra_var_weighted'],
        }

        return recon_loss, kl_loss, contrast_loss, separation_loss, loss_dict


def evaluate_model(model, criterion, dataset, all_pooler_outputs_val, device, split='val'):
    """
    评估模型性能

    Args:
        model: VAE模型
        criterion: 损失函数
        dataset: 数据集
        all_pooler_outputs_val: BERT编码
        device: 设备
        split: 'val' 或 'test'

    Returns:
        eval_losses: 评估损失字典
    """
    model.eval()

    with torch.no_grad():
        # 加载验证集数据
        X_num = torch.from_numpy(dataset.X_num[split]).to(device)
        X_cat = torch.from_numpy(dataset.X_cat[split]).to(device) if dataset.X_cat else None
        y = torch.from_numpy(dataset.y[split]).to(device).flatten()

        # 前向传播
        Recon_X_num, Recon_X_cat, mu, logvar, z = model(
            X_num, X_cat, all_pooler_outputs_val
        )

        # 计算损失（不更新原型）
        recon_loss, kl_loss, contrast_loss, separation_loss, loss_dict = criterion(
            X_num, X_cat,
            Recon_X_num, Recon_X_cat,
            mu, logvar, z, y,
            update_prototypes=False  # 验证时不更新原型
        )

        # 添加前缀
        eval_losses = {f'val_{k}': v for k, v in loss_dict.items()}

        # 计算总损失（用于调度器）
        eval_losses['val_total_loss'] = (
            loss_dict['recon_loss'] +
            loss_dict['kl_loss'] +
            loss_dict['contrast_loss'] +
            loss_dict['separation_loss']
        )

    return eval_losses



def main_train(train_loader, all_pooler_outputs_val, raw_config, dataset, all_pooler_outputs_train=None):
    """
    主训练函数（整合你的原始评估逻辑）
    """
    device = torch.device(raw_config['device'])
    save_dir = raw_config['parent_dir']
    # 初始化TensorBoard日志记录器
    tb_logger = TensorBoardLogger(save_dir, enabled=True)

    # ==================== 模型初始化 ====================
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

    pre_encoder = Encoder_model(
        num_layers=raw_config['VAE']['num_layers'],
        d_numerical=raw_config['num_numerical_features'],
        categories=raw_config['num_categorical'],
        d_token=raw_config['VAE']['d_token'],
        n_head=raw_config['VAE']['n_head'],
        factor=raw_config['VAE']['factor'],
        bert_name=raw_config['model_params']['bert']
    ).to(device)

    pre_decoder = Decoder_model(
        num_layers=raw_config['VAE']['num_layers'],
        d_numerical=raw_config['num_numerical_features'],
        categories=raw_config['num_categorical'],
        d_token=raw_config['VAE']['d_token'],
        n_head=raw_config['VAE']['n_head'],
        factor=raw_config['VAE']['factor']
    ).to(device)

    # model.load_state_dict(torch.load(os.path.join(save_dir, 'vae_model.pth'), map_location=device))
    # pre_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'vae_encoder_model.pth'), map_location=device))
    # pre_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'vae_decoder_model.pth'), map_location=device))

    pre_encoder.eval()
    pre_decoder.eval()

    # ==================== 损失函数初始化 ====================
    # 计算隐空间维度
    latent_dim = (
         raw_config['num_numerical_features'] +
         raw_config['num_categorical_features'] + 1
    ) * raw_config['VAE']['d_token']

    criterion = AdaptiveVAELoss(
        latent_dim=latent_dim,
        device=device,
        temperature=raw_config['VAE'].get('temperature', 0.07),
        base_temperature=raw_config['VAE'].get('base_temperature', 0.07),
        momentum=raw_config['VAE'].get('momentum', 0.9),
        margin=raw_config['VAE'].get('margin', 2.0)
    )

    # ==================== 优化器设置 ====================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=raw_config['VAE']['lr'],
        weight_decay=raw_config['VAE']['wd']
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=raw_config['VAE']['scheduler_fac'],
        patience=raw_config['VAE']['scheduler_patience'],
        verbose=False
    )

    # ==================== 训练超参数 ====================
    num_epochs = raw_config['VAE']['epoch']

    # Beta调度策略
    max_beta = raw_config['VAE'].get('beta_max', 1e-2)
    min_beta = raw_config['VAE'].get('beta_min', 1e-5)
    lambd = raw_config['VAE'].get('beta_decay', 0.7)

    # 其他损失权重
    gamma = raw_config['VAE'].get('gamma', 0.3)  # 对比损失权重
    lambda_sep = raw_config['VAE'].get('lambda_sep', 0.2)  # 分离损失权重

    beta = max_beta
    patience = 0
    best_train_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']

    # ==================== CSV日志初始化 ====================
    csv_columns = [
        'epoch', 'beta', 'lr',
        # 训练指标
        'train_total_loss', 'train_recon_loss', 'train_mse_loss', 'train_ce_loss',
        'train_kl_loss', 'train_contrast_loss', 'train_separation_loss',
        'train_acc', 'train_proto_dist',
        'train_intra_var_class_0', 'train_intra_var_class_1',
        'train_intra_var_total', 'train_intra_var_weighted',
        # 验证指标
        'val_total_loss', 'val_recon_loss', 'val_mse_loss', 'val_ce_loss',
        'val_kl_loss', 'val_contrast_loss', 'val_separation_loss',
        'val_acc', 'val_proto_dist',
        'val_intra_var_class_0', 'val_intra_var_class_1',
        'val_intra_var_total', 'val_intra_var_weighted',
    ]

    df_init = pd.DataFrame(columns=csv_columns)
    csv_path = os.path.join(save_dir, "vae_loss_detailed.csv")
    df_init.to_csv(csv_path, index=False)

    print("=" * 80)
    print("🚀 开始训练改进的VAE模型...")
    print(f"📊 隐空间维度: {latent_dim}")
    print(f"🎯 损失权重 - Beta: {beta:.6f}, Gamma: {gamma}, Lambda_sep: {lambda_sep}")
    print("=" * 80)

    start_time = time.time()
    pbar = tqdm(range(num_epochs), desc="Training", file=sys.stdout)

    for ep in pbar:
        # ==================== 训练阶段 ====================
        model.train()

        epoch_train_losses = {
            'total': 0.0, 'recon': 0.0, 'mse': 0.0, 'ce': 0.0,
            'kl': 0.0, 'contrast': 0.0, 'separation': 0.0,
            'acc': 0.0, 'proto_dist': 0.0,
            'intra_var_class_0': 0.0, 'intra_var_class_1': 0.0,
            'intra_var_total': 0.0, 'intra_var_weighted': 0.0,
        }
        total_samples = 0

        for batch_num, batch_cat, batch_y, batch_cls_heads in train_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device) if batch_cat is not None else None
            batch_y = batch_y.to(device).flatten()

            optimizer.zero_grad()

            # 前向传播
            Recon_X_num, Recon_X_cat, mu, logvar, z = model(
                batch_num, batch_cat, batch_cls_heads
            )

            # 计算损失
            recon_loss, kl_loss, contrast_loss, separation_loss, loss_dict = criterion(
                batch_num, batch_cat,
                Recon_X_num, Recon_X_cat,
                mu, logvar, z, batch_y,
                update_prototypes=True  # 训练时更新原型
            )

            # 总损失
            total_loss = (
                recon_loss +
                beta * kl_loss +
                gamma * contrast_loss +
                lambda_sep * separation_loss
            )

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 累计损失
            batch_size = batch_num.shape[0]
            epoch_train_losses['total'] += total_loss.item() * batch_size
            epoch_train_losses['recon'] += loss_dict['recon_loss'] * batch_size
            epoch_train_losses['mse'] += loss_dict['mse_loss'] * batch_size
            epoch_train_losses['ce'] += loss_dict['ce_loss'] * batch_size
            epoch_train_losses['kl'] += loss_dict['kl_loss'] * batch_size
            epoch_train_losses['contrast'] += loss_dict['contrast_loss'] * batch_size
            epoch_train_losses['separation'] += loss_dict['separation_loss'] * batch_size
            epoch_train_losses['acc'] += loss_dict['acc'] * batch_size
            epoch_train_losses['proto_dist'] += loss_dict['proto_dist'] * batch_size
            epoch_train_losses['intra_var_class_0'] += loss_dict['intra_var_class_0'] * batch_size
            epoch_train_losses['intra_var_class_1'] += loss_dict['intra_var_class_1'] * batch_size
            epoch_train_losses['intra_var_total'] += loss_dict['intra_var_total'] * batch_size
            epoch_train_losses['intra_var_weighted'] += loss_dict['intra_var_weighted'] * batch_size
            total_samples += batch_size

        # 计算训练集平均损失
        avg_train_losses = {k: v / total_samples for k, v in epoch_train_losses.items()}

        # ==================== 验证阶段 ====================
        eval_losses = evaluate_model(
            model, criterion, dataset,
            all_pooler_outputs_val, device, split='val'
        )

        val_total_loss = eval_losses['val_total_loss']

        # ==================== 学习率调度 ====================
        scheduler.step(val_total_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != current_lr:
            current_lr = new_lr
            # print(f"\n📉 学习率更新: {current_lr:.8f}")

        # ==================== 模型保存逻辑 ====================
        if val_total_loss < best_train_loss:
            best_train_loss = val_total_loss
            patience = 0

            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(save_dir, 'vae_model.pth'))
            torch.save(criterion.state_dict(), os.path.join(save_dir, 'vae_criterion.pth'))

            # print(f"\n✅ 保存最佳模型! Val Loss: {val_total_loss:.6f}")
        else:
            patience += 1

            # Beta衰减策略
            if patience >= 10:
                if beta > min_beta:
                    old_beta = beta
                    beta = max(beta * lambd, min_beta)
                    # print(f"\n⚠️ Beta衰减: {old_beta:.6f} -> {beta:.6f}")
                    patience = 0

        # ==================== 进度条更新 ====================
        pbar.set_description(
            f"Epoch {ep + 1}/{num_epochs} | "
            f"β={beta:.1e} | "
            f"Train Loss: {avg_train_losses['total']:.4f} | "
            f"Val Loss: {val_total_loss:.4f} | "
            f"Train Acc: {avg_train_losses['acc']:.4f} | "
            f"Val Acc: {eval_losses['val_acc']:.4f} | "
            f"Proto Dist: {eval_losses['val_proto_dist']:.4f} |"
            f"Intra Var: {eval_losses['val_intra_var_weighted']:.4f}"
        )

        # ==================== 保存日志 ====================
        log_data = pd.DataFrame([{
            'epoch': ep,
            'beta': beta,
            'lr': current_lr,
            # 训练指标
            'train_total_loss': avg_train_losses['total'],
            'train_recon_loss': avg_train_losses['recon'],
            'train_mse_loss': avg_train_losses['mse'],
            'train_ce_loss': avg_train_losses['ce'],
            'train_kl_loss': avg_train_losses['kl'],
            'train_contrast_loss': avg_train_losses['contrast'],
            'train_separation_loss': avg_train_losses['separation'],
            'train_acc': avg_train_losses['acc'],
            'train_proto_dist': avg_train_losses['proto_dist'],
            'train_intra_var_class_0': avg_train_losses['intra_var_class_0'],
            'train_intra_var_class_1': avg_train_losses['intra_var_class_1'],
            'train_intra_var_total': avg_train_losses['intra_var_total'],
            'train_intra_var_weighted': avg_train_losses['intra_var_weighted'],
            # 验证指标
            'val_total_loss': eval_losses['val_total_loss'],
            'val_recon_loss': eval_losses['val_recon_loss'],
            'val_mse_loss': eval_losses['val_mse_loss'],
            'val_ce_loss': eval_losses['val_ce_loss'],
            'val_kl_loss': eval_losses['val_kl_loss'],
            'val_contrast_loss': eval_losses['val_contrast_loss'],
            'val_separation_loss': eval_losses['val_separation_loss'],
            'val_acc': eval_losses['val_acc'],
            'val_proto_dist': eval_losses['val_proto_dist'],
            'val_intra_var_class_0': eval_losses['val_intra_var_class_0'],
            'val_intra_var_class_1': eval_losses['val_intra_var_class_1'],
            'val_intra_var_total': eval_losses['val_intra_var_total'],
            'val_intra_var_weighted': eval_losses['val_intra_var_weighted'],
        }])

        # log 保存到文件
        log_data.to_csv(csv_path, mode='a', header=False, index=False)
        # log 添加到 TensorBoard
        tb_logger.log_epoch(
            epoch=ep,
            train_losses=avg_train_losses,
            val_losses=eval_losses,
            beta=beta,
            lr=current_lr
        )

    # ==================== 训练完成 ====================
    end_time = time.time()
    training_time = (end_time - start_time) / 60

    print("\n" + "=" * 80)
    print(f"✅ 训练完成! 总耗时: {training_time:.2f} 分钟")
    print(f"📊 最佳验证损失: {best_train_loss:.6f}")
    print(f"💾 模型已保存至: {save_dir}")
    print("=" * 80)

    # ==================== 保存隐空间嵌入 ====================
    print("\n🔄 正在保存隐空间嵌入...")

    with torch.no_grad():
        # 加载最佳模型权重
        model.load_state_dict(torch.load(os.path.join(save_dir, 'vae_model.pth'),
                                         map_location=device))
        model.eval()

        # 加载权重到编码器和解码器
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        # 保存编码器和解码器
        torch.save(pre_encoder.state_dict(), os.path.join(save_dir, 'vae_encoder_model.pth'))
        torch.save(pre_decoder.state_dict(), os.path.join(save_dir, 'vae_decoder_model.pth'))

        # 编码训练数据
        X_train_num = torch.from_numpy(dataset.X_num['train']).to(device)
        X_train_cat = torch.from_numpy(dataset.X_cat['train']).to(device) if dataset.X_cat else None
        X_val_num = torch.from_numpy(dataset.X_num['val']).to(device)
        X_val_cat = torch.from_numpy(dataset.X_cat['val']).to(device) if dataset.X_cat else None

        # 获取隐空间表示
        train_z = pre_encoder(X_train_num, X_train_cat, all_pooler_outputs_train).detach().cpu().numpy()
        val_z = pre_encoder(X_val_num, X_val_cat, all_pooler_outputs_val).detach().cpu().numpy()

        # 获取重参数化后的隐向量
        _, _, _, latent_z_after_reparameterize = model.VAE(
            X_train_num, X_train_cat, all_pooler_outputs_train
        )
        latent_z_after_reparameterize = latent_z_after_reparameterize.detach().cpu().numpy()

        _, _, _, latent_z_after_reparameterize_val = model.VAE(
            X_val_num, X_val_cat, all_pooler_outputs_val
        )
        latent_z_after_reparameterize_val = latent_z_after_reparameterize_val.detach().cpu().numpy()

        # 保存隐向量
        latent_dir = os.path.join(save_dir, 'latent_data')
        os.makedirs(latent_dir, exist_ok=True)

        np.save(os.path.join(latent_dir, 'train_z.npy'), train_z)
        np.save(os.path.join(latent_dir, 'val_z.npy'), val_z)
        np.save(os.path.join(latent_dir, 'latent_z_after_reparameterize.npy'), latent_z_after_reparameterize)
        np.save(os.path.join(latent_dir, 'latent_z_after_reparameterize_val.npy'), latent_z_after_reparameterize_val)

        # 额外保存类原型（用于可视化分析）
        prototype_dict = {
            'prototype_0': criterion.prototype_0.cpu().numpy(),
            'prototype_1': criterion.prototype_1.cpu().numpy(),
            'proto_distance': criterion.get_prototype_distance()
        }
        np.save(os.path.join(latent_dir, 'prototypes.npy'), prototype_dict)

        print(f"✅ 隐空间嵌入已保存至: {latent_dir}")
        print(f"   - train_z.npy: {train_z.shape}")
        print(f"   - latent_z_after_reparameterize.npy: {latent_z_after_reparameterize.shape}")
        print(f"   - prototypes.npy (类原型距离: {prototype_dict['proto_distance']:.4f})")


    tb_logger.close()
    return model, criterion

def main(raw_config):
    real_data_path = raw_config['real_data_path']
    device = torch.device(raw_config['device'])

    # 创建数据集
    dataset = make_dataset(real_data_path, raw_config)

    # ==================== 准备BERT编码 ====================
    all_pooler_outputs_train = None
    all_pooler_outputs_val = None

    if raw_config['num_categorical_features']:
        # 1. 第一种方式，Bert编码
        berttokenizer, bertmodel = get_bert_model(raw_config)
        all_pooler_outputs_train = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device,
                                                           with_label=False, data_path=real_data_path, split='train')
        all_pooler_outputs_val = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False,
                                                         data_path=real_data_path, split='val')
        # all_pooler_outputs_all = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False, split='all')
        torch.save(all_pooler_outputs_train, f"{raw_config['parent_dir']}/all_pooler_outputs_train.pth")
        torch.save(all_pooler_outputs_val, f"{raw_config['parent_dir']}/all_pooler_outputs_val.pth")
        # torch.save(all_pooler_outputs_all, f"{raw_config['parent_dir']}/all_pooler_outputs_all.pth")

        # 2. 第二种方式，直接加载
        """
        all_pooler_outputs_train = torch.load(f"{raw_config['parent_dir']}/all_pooler_outputs_train.pth")
        all_pooler_outputs_val = torch.load(f"{raw_config['parent_dir']}/all_pooler_outputs_val.pth")
        # all_pooler_outputs_all = torch.load(f"{raw_config['parent_dir']}/all_pooler_outputs_all.pth")
        """

        print("all_pooler_outputs_train.shape: ", all_pooler_outputs_train.shape)
        print("all_pooler_outputs_val.shape: ", all_pooler_outputs_val.shape)
        # print(all_pooler_outputs_all.shape)

    # ==================== 创建DataLoader ====================
    train_loader = prepare_dataloader_respectively(
        dataset,
        all_pooler_outputs_train,
        split='train',
        batch_size=raw_config['VAE']['batch_size']
    )

    # ==================== 开始训练 ====================
    model, criterion = main_train(
        train_loader,
        all_pooler_outputs_val,
        raw_config,
        dataset,
        all_pooler_outputs_train
    )

if __name__ == '__main__':
    raw_config_list = []
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/buddy\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\shopper\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\churn\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/adult\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/magic\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/bean\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/yeast_me2\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/winequality1\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/pageblocks\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/mammography\CoTable\config.toml"))

    for raw_config in raw_config_list:
        main(raw_config)

"""
python main_vae(new).py
"""