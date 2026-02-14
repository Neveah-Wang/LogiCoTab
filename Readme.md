当前分支为:

研究内容二：的全部代码

# 关于数据
- 延用内容一的所有数据
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
- 延用内容一的vae，实现类内聚拢和类间分离。代码位于`main_vae(new).py`
- train test val 的隐空间数据 全都保存在 `exp\[dataset_name]\CoTable\latent_data`

## LP + MLP(KL + Focal Loss)
- 关于软标签 + MLP 的算法，直接执行 `pipeline_prob_improved.py` 即可。
- 模型和算法 都在 `prob_soft_classifier_improved.py` 中

# 关于评估
- `evaluate\t_SNE_onehot_soft_label.py` 用于绘制 硬标签 和 软标签 的 t-SNE 可视化图像
-  `plot_focal_loss.py` 和 `plot_focal_loss_parameter.py` 用于绘制 Focal Loss 的曲线
