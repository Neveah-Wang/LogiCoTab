核心思想
借鉴原型网络(Prototypic Networks)和有监督对比学习(Supervised Contrastive Learning)，设计一个三阶段的损失函数：

```python
L_total = L_recon + β·L_KL_adaptive + γ·L_supervised_contrast
```

关键创新点
1. 自适应KL散度：根据类别分布调整KL约束
2. 有监督对比损失：同时考虑正负样本对
3. 动量更新的类原型：稳定类中心学习
4. 温度缩放：控制分布的集中程度

---

## 数学公式详解

### 1. 总损失函数

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta(t) \cdot \mathcal{L}_{\text{KL}}^{\text{adaptive}} + \gamma \cdot \mathcal{L}_{\text{supcon}} + \lambda \cdot \mathcal{L}_{\text{sep}}
$$

### 2. 自适应KL散度

$$
\mathcal{L}_{\text{KL}}^{\text{adaptive}} = \sum_{c \in \{0,1\}} w_c \cdot \mathbb{E}_{x \in C_c}\left[ \text{KL}\left( q_\phi(z|x) \| \mathcal{N}(\mu_c^{\text{proto}}, \mathbf{I}) \right) \right]
$$

其中：
- $\mu_c^{\text{proto}}$ 是类别 $c$ 的原型（通过动量更新）
- $w_c = \frac{1}{N_c + \epsilon}$ 是类别权重

### 3. 有监督对比损失

$$
\mathcal{L}_{\text{supcon}} = \sum_{i=1}^{N} -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p) / \tau)}{\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a) / \tau)}
$$

其中：
- $P(i) = \{p \in [N] : y_p = y_i, p \neq i\}$ （同类正样本）
- $A(i) = [N] \setminus \{i\}$ （所有其他样本）
- $\text{sim}(u,v) = u^T v / (\|u\| \|v\|)$ （余弦相似度）

### 4. 类原型分离损失

$$
\mathcal{L}_{\text{sep}} = \max(0, m - \|\mu_0^{\text{proto}} - \mu_1^{\text{proto}}\|_2^2)
$$

### 5. 动量更新原型

$$
\mu_c^{\text{proto}} \leftarrow \alpha \cdot \mu_c^{\text{proto}} + (1-\alpha) \cdot \frac{1}{|B_c|}\sum_{i \in B_c} z_i
$$


# 有监督对比损失 (Supervised Contrastive Loss) 详解

## 一、核心思想

有监督对比损失的本质是：**在隐空间中，让同类样本尽可能相似，让不同类样本尽可能不同**。

与传统交叉熵损失相比，它的优势在于：
- ✅ **利用批次内所有样本关系**（而非仅考虑样本与标签）
- ✅ **学习更具判别性的特征表示**（距离度量更直观）
- ✅ **对噪声标签更鲁棒**（多个正样本对平均）

---

## 二、数学原理推导

### 2.1 基础概念

给定一个batch的样本 $\{x_1, x_2, ..., x_N\}$ 及其标签 $\{y_1, y_2, ..., y_N\}$，经过编码器得到隐向量 $\{z_1, z_2, ..., z_N\}$。

对于样本 $i$，定义：
- **正样本集合**：$P(i) = \{p \in [N] : y_p = y_i, p \neq i\}$（同类且不是自己）
- **负样本集合**：$N(i) = \{n \in [N] : y_n \neq y_i\}$（不同类）
- **所有其他样本**：$A(i) = [N] \setminus \{i\}$（除了自己）

### 2.2 相似度计算

首先，将隐向量**归一化**到单位球面：
$$
\hat{z}_i = \frac{z_i}{\|z_i\|_2}
$$

然后计算余弦相似度（带温度参数 $\tau$）：
$$
\text{sim}(z_i, z_j) = \frac{\hat{z}_i^\top \hat{z}_j}{\tau}
$$

**温度参数 $\tau$ 的作用**：
- $\tau \downarrow$：相似度差异被放大，模型更注重难负样本（hard negatives）
- $\tau \uparrow$：相似度差异被平滑，训练更稳定但可能欠拟合

典型取值：$\tau \in [0.05, 0.1]$

### 2.3 损失函数完整形式

$$
\mathcal{L}_{\text{supcon}}^i = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p))}{\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a))}
$$

**分解理解**：

1. **分子**：$\exp(\text{sim}(z_i, z_p))$ 
   - 当前样本与某个正样本的相似度
   
2. **分母**：$\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a))$
   - 当前样本与所有其他样本的相似度之和（归一化项）
   - 包含正样本 + 负样本
   
3. **对数比值**：
   $$
   \log \frac{\exp(\text{sim}(z_i, z_p))}{\sum_{a} \exp(\text{sim}(z_i, z_a))} = \text{sim}(z_i, z_p) - \log \sum_{a} \exp(\text{sim}(z_i, z_a))
   $$
   - 最大化正样本相似度
   - 最小化与所有样本的平均相似度

4. **多个正样本平均**：$\frac{1}{|P(i)|} \sum_{p \in P(i)}$
   - 充分利用batch内所有同类样本
   - 提高对噪声标签的鲁棒性

---

## 三、直观解释

### 3.1 用概率视角理解

可以将损失改写为：
$$
\mathcal{L}_{\text{supcon}}^i = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log P(a=p | i)
$$

其中：
$$
P(a=p | i) = \frac{\exp(\text{sim}(z_i, z_p))}{\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a))}
$$

**含义**：最大化"从样本 $i$ 的邻域中采样到正样本 $p$ 的概率"。

### 3.2 与其他损失的对比

| 损失类型 | 优化目标 | 样本关系 |
|---------|---------|---------|
| **交叉熵** | 最大化 $P(y_i|x_i)$ | 样本 → 标签 |
| **三元组损失** | $\|z_i - z_p\| < \|z_i - z_n\| + m$ | 1个正样本 + 1个负样本 |
| **N-Pair损失** | 扩展三元组到多个负样本 | 1个正样本 + N个负样本 |
| **监督对比** | 最大化所有正样本对的相似度 | **多个正样本 + 多个负样本** |

---

## 四、代码实现的关键步骤

### 4.1 归一化隐向量

```python
# 展平 [batch_size, seq_len, d_token] → [batch_size, latent_dim]
z_flat = z.reshape(z.size(0), -1)

# L2归一化到单位球面
z_norm = F.normalize(z_flat, dim=1)
# z_norm[i] 的范数 = 1
```

**为什么要归一化？**
- 消除向量模长的影响，只关注方向
- 余弦相似度 = 归一化向量的点积
- 数值稳定性更好

### 4.2 计算相似度矩阵

```python
# 相似度矩阵 [batch_size, batch_size]
similarity_matrix = torch.matmul(z_norm, z_norm.T) / temperature
# similarity_matrix[i, j] = cos(z_i, z_j) / τ
```

示例（4个样本，温度=0.1）：
```
样本标签: [0, 0, 1, 1]

相似度矩阵（/τ前）:
        [1.00,  0.85,  0.20,  0.15]
        [0.85,  1.00,  0.18,  0.22]
        [0.20,  0.18,  1.00,  0.90]
        [0.15,  0.22,  0.90,  1.00]

除以τ=0.1后:
        [10.0,  8.5,  2.0,  1.5]
        [8.5,  10.0, 1.8,  2.2]
        [2.0,  1.8,  10.0, 9.0]
        [1.5,  2.2,  9.0,  10.0]
```

### 4.3 构建掩码矩阵

```python
# 标签掩码 [batch_size, batch_size]
labels = labels.view(-1, 1)  # [batch_size, 1]
mask_positive = torch.eq(labels, labels.T).float()
# mask_positive[i, j] = 1 if y_i == y_j else 0

# 移除对角线（自己不是正样本）
logits_mask = torch.ones_like(mask_positive)
logits_mask.scatter_(
    1, 
    torch.arange(batch_size).view(-1, 1), 
    0
)
mask_positive = mask_positive * logits_mask
```

示例：
```
标签: [0, 0, 1, 1]

原始正样本掩码:
[1, 1, 0, 0]
[1, 1, 0, 0]
[0, 0, 1, 1]
[0, 0, 1, 1]

移除对角线后:
[0, 1, 0, 0]  ← 样本0的正样本是样本1
[1, 0, 0, 0]  ← 样本1的正样本是样本0
[0, 0, 0, 1]  ← 样本2的正样本是样本3
[0, 0, 1, 0]  ← 样本3的正样本是样本2
```

### 4.4 计算对数概率

```python
# 计算分母（LogSumExp技巧）
exp_logits = torch.exp(similarity_matrix) * logits_mask
log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
# log_prob[i, j] = sim(i,j) - log(Σ_k exp(sim(i,k)))
```

**数值稳定性技巧**：
$$
\log \sum_k \exp(x_k) = \max(x) + \log \sum_k \exp(x_k - \max(x))



### 4.5 聚合损失
```python
# 对每个样本，计算其与所有正样本的平均对数概率
mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / (mask_positive.sum(dim=1) + 1e-9)

# 总损失（负对数似然）
loss = -(temperature / base_temperature) * mean_log_prob_pos.mean()
```
#### 逐步拆解：

Step 1: 提取正样本对的对数概率
```
# mask_positive * log_prob
        [0,     -1.70,  0,      0    ]  ← 只保留正样本对
        [-1.70, 0,      0,      0    ]
        [0,     0,      0,      -1.20]
        [0,     0,      -1.20,  0    ]
```

Step 2: 对每个样本求和并平均
```
# 样本0: -1.70 / 1 = -1.70
# 样本1: -1.70 / 1 = -1.70
# 样本2: -1.20 / 1 = -1.20
# 样本3: -1.20 / 1 = -1.20
mean_log_prob_pos = [-1.70, -1.70, -1.20, -1.20]
```
Step 3: 取负号并批次平均

```loss = -mean([-1.70, -1.70, -1.20, -1.20]) = 1.45```

温度缩放因子的作用：

```loss = -(temperature / base_temperature) * mean_log_prob_pos.mean()```

当 temperature = base_temperature 时，系数为1（标准损失）

当 temperature < base_temperature 时，损失被缩小（梯度更温和）

通常设置为相等，即系数为1