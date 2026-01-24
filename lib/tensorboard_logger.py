from torch.utils.tensorboard import SummaryWriter
import os


class TensorBoardLogger:
    """
    TensorBoard日志记录器，用于VAE训练过程的可视化监控
    """

    def __init__(self, log_dir, enabled=True):
        """
        初始化TensorBoard日志记录器

        Args:
            log_dir: 日志保存目录
            enabled: 是否启用TensorBoard（默认True）
        """
        self.enabled = enabled
        if self.enabled:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'runs'))
            print(f"📊 TensorBoard has been started. Log directory: {os.path.join(log_dir, 'runs')}")
            print(f"💡 Use the command to view: tensorboard --logdir={os.path.join(log_dir, 'runs')}")
        else:
            self.writer = None

    def log_epoch(self, epoch, train_losses, val_losses, beta, lr):
        """
        记录一个epoch的所有指标

        Args:
            epoch: 当前epoch数
            train_losses: 训练损失字典 {'total', 'recon', 'mse', 'ce', 'kl', 'contrast', 'separation', 'acc', 'proto_dist'}
            val_losses: 验证损失字典 (同上，带'val_'前缀)
            beta: 当前beta值
            lr: 当前学习率
        """
        if not self.enabled:
            return

        # 1. 总损失对比
        self.writer.add_scalars('Loss/Total', {
            'train': train_losses['total'],
            'val': val_losses['val_total_loss']
        }, epoch)

        # 2. 重构损失对比
        self.writer.add_scalars('Loss/Reconstruction', {
            'train': train_losses['recon'],
            'val': val_losses['val_recon_loss']
        }, epoch)

        # 3. MSE损失对比
        self.writer.add_scalars('Loss/MSE', {
            'train': train_losses['mse'],
            'val': val_losses['val_mse_loss']
        }, epoch)

        # 4. 交叉熵损失对比
        self.writer.add_scalars('Loss/CrossEntropy', {
            'train': train_losses['ce'],
            'val': val_losses['val_ce_loss']
        }, epoch)

        # 5. KL散度对比
        self.writer.add_scalars('Loss/KL_Divergence', {
            'train': train_losses['kl'],
            'val': val_losses['val_kl_loss']
        }, epoch)

        # 6. 对比损失对比
        self.writer.add_scalars('Loss/Contrastive', {
            'train': train_losses['contrast'],
            'val': val_losses['val_contrast_loss']
        }, epoch)

        # 7. 分离损失对比
        self.writer.add_scalars('Loss/Separation', {
            'train': train_losses['separation'],
            'val': val_losses['val_separation_loss']
        }, epoch)

        # 8. 准确率对比
        self.writer.add_scalars('Metrics/Accuracy', {
            'train': train_losses['acc'],
            'val': val_losses['val_acc']
        }, epoch)

        # 9. 原型距离对比
        self.writer.add_scalars('Metrics/Prototype_Distance', {
            'train': train_losses['proto_dist'],
            'val': val_losses['val_proto_dist']
        }, epoch)

        # 10. 超参数
        self.writer.add_scalar('Hyperparameters/Beta', beta, epoch)
        self.writer.add_scalar('Hyperparameters/Learning_Rate', lr, epoch)

        # 11. 损失组成比例（堆叠图）
        self.writer.add_scalars('Loss/Train_Components', {
            'recon': train_losses['recon'],
            'kl_weighted': train_losses['kl'] * beta,
            'contrast': train_losses['contrast'],
            'separation': train_losses['separation']
        }, epoch)

    def log_batch(self, global_step, batch_loss):
        """
        记录单个batch的损失（可选，用于更细粒度监控）

        Args:
            global_step: 全局步数
            batch_loss: batch损失值
        """
        if not self.enabled:
            return

        self.writer.add_scalar('Loss/Batch', batch_loss, global_step)

    def log_histogram(self, epoch, model):
        """
        记录模型参数和梯度的分布（可选）

        Args:
            epoch: 当前epoch数
            model: 模型
        """
        if not self.enabled:
            return

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def log_embeddings(self, epoch, embeddings, labels, tag='latent_space'):
        """
        记录潜在空间嵌入（可选，用于可视化）

        Args:
            epoch: 当前epoch数
            embeddings: 嵌入向量 [N, D]
            labels: 标签 [N]
            tag: 标签名
        """
        if not self.enabled:
            return

        self.writer.add_embedding(
            embeddings,
            metadata=labels,
            tag=tag,
            global_step=epoch
        )

    def close(self):
        """关闭TensorBoard writer"""
        if self.enabled and self.writer is not None:
            self.writer.close()
            # print("✅ TensorBoard日志已保存并关闭")


# ===== 使用示例 =====
if __name__ == '__main__':
    """
    在你的main函数中这样使用：

    def main(train_loader, all_pooler_outputs_val, raw_config, dataset, all_pooler_outputs_train=None):
        device = torch.device(raw_config['device'])
        save_dir = raw_config['parent_dir']

        # 初始化TensorBoard日志记录器（只需这一行！）
        tb_logger = TensorBoardLogger(save_dir, enabled=True)

        # ... 原来的模型初始化代码 ...

        for ep in pbar:
            # ... 训练代码 ...

            # 记录epoch指标（只需这一行！）
            tb_logger.log_epoch(
                epoch=ep,
                train_losses=avg_train_losses,
                val_losses=eval_losses,
                beta=beta,
                lr=current_lr
            )

            # 原来的CSV保存代码保持不变
            log_data.to_csv(csv_path, mode='a', header=False, index=False)

        # 训练结束时关闭（只需这一行！）
        tb_logger.close()

        return model, criterion


    然后在终端运行：
    tensorboard --logdir=你的保存目录/runs

    在浏览器打开 http://localhost:6006 即可查看实时曲线！
    """
    pass