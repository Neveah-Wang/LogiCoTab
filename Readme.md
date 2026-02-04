本文件主要解释：本版本的代码都做了哪些修改：

# 关于数据
- 所有使用到的数据全都存放在`Dataset-all`中，
- 所有数据都更改为二分类

具体参考以下信息：

| 数据集      | 领域               | 样本数量 | 特征维度 | 离散型 | 不平衡比                     |来源    |
|-------------|--------------------|----------|----------|--------|------------------------------|-----|
| Magic       | 物理学             | 19020    | 10       | 0      | 1.84                         | https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
| Adult       | 社会学             | 48842    | 6        | 8      | 3.18                         |https://archive.ics.uci.edu/dataset/2/adult
| Churn       | 金融               | 10000    | 5        | 5      | 3.91                         |https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
| Shopping    | 商业               | 12330    | 10       | 7      | 5.46                         |https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
| Obesity     | 医学               | 2110     | 8        | 6      | 2.51                         |https://github.com/shaecodes/Obesity-Level-Prediction
| Wine Quality | 物理化学/商业      | 6497     | 11       | 25.41  | 多分类，Normal:Other = 6.36  |https://archive.ics.uci.edu/dataset/186/wine+quality
| Bean        | 生物学             | 13611    | 16       | 0      | 25.07                        |https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
| yeast_me2   | 生物学             | 1484     | 8        | 0      | 28.1                         |https://archive.ics.uci.edu/dataset/110/yeast
| Page        | 计算机             | 5473     | 10       | 0      | 194.46                       |https://archive.ics.uci.edu/dataset/78/page+blocks+classification
| Buddy       | 生物学             | 18834    | 4        | 5      | 213.02                       |https://www.kaggle.com/datasets/akash14/adopt-a-buddy


# 关于模型 & 训练
## VAE
提出一种新的原型对比vae，可以实现类内聚拢和类间分离。代码位于`main_vae(new).py`

## DDPM
取消多条件组合引导，主干网络只保留一层transformer
```python
x_residual = x
x = self.norm(x)
x = self.attention1(x, x, x)  # 不用其他特征。试一试效果
x = x_residual + x
x = self.feedforward(x) + x
```

# 关于评估分类效果

GOSC 使用 `filer.py` 评估。
- 用原始训练集训练分类器
- 合成样本（只生成少数类样本）
- 用分类器过滤 低置信度 的 合成样本
- 将合成样本 加入 原始训练集
- 再次训练分类器，观察是否有提升
- 如果有提升，就更新最佳分类器为当前分类器
- 重复以上步骤，直到数据平衡，或者分数不再上涨
- 最终返回最好的分类结果
- 结果都保存在` evaluate/mle_log(AucF1AccGmeanMcc)/[dataset].log`

其他 baseline 的过采样方法的评估方法:
- 代码在 `baseline/[SMOTE,TabDDPM,TabSyn, ....]/eval.py`