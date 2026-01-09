本文件主要解释：本版本的代码都做了哪些修改：

# 关于数据

`Dataset-all` 是 LogiCoTab 使用的数据
- `Dataset-all\bean` 本来是7分类。现在修改为2分类（两个类别：CALI 和 Aother）
- `Dataset-all\bean(BOMBAYandOther)` 本来是7分类。现在修改为2分类（两个类别：BOMBAY 和 Other）
- `Dataset-all\bean(SIRAandOther)` 本来是7分类。现在修改为2分类（两个类别： SIRA 和 Other）

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

# 关于评估 evaluate
- `mle_catboost` 
    - 修改了原始版本的打印指标的方式
    - 改成：打印分类过程的所有指标。


# 关于该版本中的测试实验
- `detect_overlap.py` 检测重叠样本。
    - k-折叠原理，将原始训练集中的 多数类 分成k份
    - 用剩余的 k-1 个多数类 and 少数类，训练一个分类器
    - 用分类器找到分类错误 and 低置信度 的 重叠样本
    - 删除低置信度样本，只保留高置信度样本

- `filier.py` 过滤生成低质量的样本
    - 用原始训练集训练分类器
    - 合成样本（只生成少数类样本）
    - 用分类器过滤 低置信度 的 合成样本
    - 将合成样本 加入 原始训练集
    - 再次训练分类器，观察是否有提升
    - 如果有提升，就更新最佳分类器为当前分类器
    - 重复以上步骤，直到数据平衡，或者分数不再上涨
