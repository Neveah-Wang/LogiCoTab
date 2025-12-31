# 关于数据

`Dataset-all` 是 LogiCoTab 使用的数据

`Dataset` 中只有少数类
- `Dataset\chrun\` 中只有 label 为 1 （少数类别）的数据。
- `Dataset\data_preprocess.ipynb` 数据预处理，用于获取少数类别

# 关于模型 & 训练

`exp/churn/CoTable`
- `vae_decoder_model.pth` 只用少数类数据训练得到
- `vae_encoder_model.pth` 只用少数类数据训练得到
- `vae_model.pth` 只用少数类数据训练得到
- `model_{epoch}` 只用少数类数据训练得到

只用少数类数据，从头训练

### DDPM
不做多条件组合引导。

当前的 Attention：
```python
# Path：LogiCoTab-oversampling\TabClassifierfree\Transformer_noise_prediction.py
x_residual = x
x = self.norm(x)
x = self.attention1(x, x, x)  # 只用一个 Self-Attention
x = x_residual + x
x = self.feedforward(x) + x

```

LogiCoTab 的 Attention：
```python
# Path：v11\TabClassifierfree\Transformer_noise_prediction.py
x_residual = x
x = self.norm(x)
x = self.attention1(cls_sum, x, x)  # 第一个Attention 使用组合条件作为输入
x = x_residual + x
x = self.feedforward(x) + x

x_residual = x
x = self.norm(x)
x = self.attention2(cls_label, x, x) # 第二个 Attention 使用 lable 条件作为输入
x = x_residual + x
x = self.feedforward(x) + x
```