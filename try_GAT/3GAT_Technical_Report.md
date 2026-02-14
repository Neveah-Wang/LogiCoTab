# 基于图注意力网络的不平衡数据分类方法：完整技术报告

## 摘要

本文提出一种面向不平衡表格数据分类的图注意力网络(Graph Attention Network, GAT)方法。该方法通过标签传播生成软标签进行标签增强，利用GAT学习隐空间的拓扑结构特征，并引入Focal Loss和监督对比学习处理类别不平衡问题。整个框架包含三个核心模块：(1) 基于k近邻图的标签传播软标签生成；(2) 多头图注意力特征学习；(3) 双任务优化策略（Focal Loss分类 + 对比学习特征增强）。

---

## 1. 问题定义与数学符号

### 1.1 问题设定

给定不平衡二分类数据集 $\mathcal{D} = \{(z_i, y_i)\}_{i=1}^{n}$，其中：
- $z_i \in \mathbb{R}^d$ 为样本在VAE隐空间中的表示
- $y_i \in \{0, 1\}$ 为硬标签，$y_i=1$ 表示少数类（正类），$y_i=0$ 表示多数类（负类）
- 类别不平衡比例 $\rho = \frac{|\{i: y_i=0\}|}{|\{i: y_i=1\}|} \gg 1$

数据集划分为训练集 $\mathcal{D}_{train}$、验证集 $\mathcal{D}_{val}$ 和测试集 $\mathcal{D}_{test}$。

**目标**：学习分类器 $f: \mathbb{R}^d \to [0,1]$，使其在测试集上最大化宏平均F1分数(Macro-F1)或G-Mean等不平衡指标。

### 1.2 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $n$ | 样本总数 | 标量 |
| $d$ | 隐空间特征维度 | 标量 |
| $Z \in \mathbb{R}^{n \times d}$ | 隐空间特征矩阵 | $n \times d$ |
| $y \in \{0,1\}^n$ | 硬标签向量 | $n$ |
| $p \in [0,1]^n$ | 软标签向量 | $n$ |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | k近邻图 | - |
| $\mathcal{N}(i)$ | 节点$i$的邻居集合 | - |
| $k$ | 近邻数量 | 标量 |
| $H$ | 注意力头数 | 标量 |
| $D$ | 隐藏层维度 | 标量 |

---

## 2. 软标签生成：基于标签传播的标签增强

### 2.1 动机与理论基础

**核心问题**：在不平衡数据中，类间重叠区域的边界样本存在标注不确定性，硬标签(0/1)无法表达这种不确定性，容易导致过拟合。

**解决方案**：通过标签传播(Label Propagation, LP)算法在k近邻图上扩散标签信息，生成软标签 $p_i \in [0,1]$，使得：
- 簇内核心样本: $p_i \approx y_i$ (接近硬标签)
- 边界噪声样本: $p_i \approx 0.5$ (高不确定性)

**理论依据**：
1. **流形假设**(Manifold Assumption)：同类样本在隐空间中位于同一流形上
2. **平滑假设**(Smoothness Assumption)：相近样本应具有相似标签
3. **聚类假设**(Cluster Assumption)：决策边界应位于低密度区域

### 2.2 k近邻图构建

#### 2.2.1 邻域选择

对每个样本 $z_i$，计算其k个最近邻：

$$
\mathcal{N}(i) = \underset{j \neq i}{\text{argmin}_{k}} \|z_i - z_j\|_2
$$

构造有向图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中：
- 节点集合 $\mathcal{V} = \{1, 2, \ldots, n\}$
- 边集合 $\mathcal{E} = \{(i, j) : j \in \mathcal{N}(i)\}$

#### 2.2.2 边权重计算（RBF核）

使用自适应带宽的RBF核计算边权重：

$$
w_{ij} = \exp\left(-\frac{\|z_i - z_j\|_2^2}{2\sigma_i^2}\right), \quad j \in \mathcal{N}(i)
$$

其中自适应带宽 $\sigma_i$ 由邻域平均距离确定：

$$
\sigma_i = \frac{1}{k} \sum_{j \in \mathcal{N}(i)} \|z_i - z_j\|_2
$$

**物理意义**：
- $\sigma_i$ 大：样本$i$处于稀疏区域，邻域影响范围广
- $\sigma_i$ 小：样本$i$处于稠密区域，邻域影响范围窄

#### 2.2.3 少数类增强权重

为增强少数类的影响力，对正类邻居的权重进行放大：

$$
\tilde{w}_{ij} = w_{ij} \cdot \eta^{\mathbb{1}[y_j = 1]}
$$

其中：
- $\eta \in [1, 3]$ 为少数类增强系数
- $\mathbb{1}[\cdot]$ 为指示函数

**设计原则**：$\eta$ 应与不平衡比例 $\rho$ 正相关，推荐 $\eta = 1 + \log(1+\rho)$

#### 2.2.4 转移矩阵构造

对增强后的权重进行行归一化，得到概率转移矩阵 $S$：

$$
S_{ij} = \frac{\tilde{w}_{ij}}{\sum_{j' \in \mathcal{N}(i)} \tilde{w}_{ij'}}
$$

满足：$\sum_{j \in \mathcal{N}(i)} S_{ij} = 1$

### 2.3 种子节点选择

**核心思想**：仅选择高置信度样本作为种子(seeds)，允许其他样本通过传播获得标签。

#### 2.3.1 邻域纯度定义

定义节点$i$的邻域纯度(purity)为：

$$
\text{purity}(i) = \frac{1}{k} \sum_{j \in \mathcal{N}(i)} \mathbb{1}[y_j = y_i]
$$

**物理意义**：
- $\text{purity}(i) = 1$：节点$i$的所有邻居与其同类，高置信度
- $\text{purity}(i) \approx 0.5$：节点$i$处于类边界，低置信度

#### 2.3.2 分层种子选择

对每个类别 $c \in \{0, 1\}$ 分别选择种子：

$$
\mathcal{S}_c = \underset{i: y_i=c}{\text{argtop}_{n_c}} \text{purity}(i)
$$

其中：
$$
n_c = \max\left(\lceil r \cdot |\{i: y_i=c\}| \rceil, \, n_{\min}, \, |\{i: y_i=c, \text{purity}(i)=1\}|\right)
$$

参数说明：
- $r$：种子比例，推荐 $r=0.2$
- $n_{\min}$：最小种子数，推荐 $n_{\min}=10$

**设计原则**：
1. 确保每个类至少有 $n_{\min}$ 个种子
2. 所有完美纯度($=1$)的样本必须作为种子
3. 少数类和多数类种子数量平衡

### 2.4 标签传播算法

#### 2.4.1 标签矩阵初始化

定义二分类标签矩阵 $F \in \mathbb{R}^{n \times 2}$，其中 $F_{i,c}$ 表示节点$i$属于类别$c$的概率：

$$
Y_{i,c} = \mathbb{1}[y_i = c], \quad c \in \{0, 1\}
$$

初始化 $F^{(0)} = Y$

#### 2.4.2 差异化传播系数

为不同类别设置不同的传播系数 $\alpha_i$：

$$
\alpha_i = \begin{cases}
\alpha^+ & \text{if } y_i = 1 \text{ (少数类)} \\
\alpha^- & \text{if } y_i = 0 \text{ (多数类)}
\end{cases}
$$

**推荐配置**：
- $\alpha^+ = 0.80$（少数类更依赖原始标签，抗干扰）
- $\alpha^- = 0.95$（多数类更依赖邻居扩散，易受影响）

**理论解释**：
- 少数类样本稀疏，邻居可能包含多数类，需保护原始标签
- 多数类样本充足，邻居多为同类，可充分传播

#### 2.4.3 迭代传播更新

在第 $t$ 次迭代中，更新规则为：

$$
F_i^{(t+1)} = \alpha_i \sum_{j \in \mathcal{N}(i)} S_{ij} F_j^{(t)} + (1 - \alpha_i) Y_i
$$

**矩阵形式**（更紧凑）：

$$
F^{(t+1)} = \text{diag}(\alpha) \cdot S \cdot F^{(t)} + \text{diag}(1-\alpha) \cdot Y
$$

#### 2.4.4 种子约束（Clamping）

每次迭代后，强制种子节点回到原始标签：

$$
F_i^{(t+1)} = Y_i, \quad \forall i \in \mathcal{S}_0 \cup \mathcal{S}_1
$$

**作用**：
1. 防止种子标签漂移
2. 保证传播稳定性
3. 注入强监督信号

#### 2.4.5 标签翻转抑制（可选）

为防止正类样本被误传播为负类（或反之），添加翻转约束：

$$
p_i^{(t+1)} = \begin{cases}
\max(p_i^{(t+1)}, 0.5 + \epsilon) & \text{if } y_i = 1 \\
\min(p_i^{(t+1)}, 0.5 - \epsilon) & \text{if } y_i = 0
\end{cases}
$$

其中 $p_i = F_{i,1}$ 为正类概率，$\epsilon > 0$ 为安全边界（推荐 $\epsilon = 0.01$）

#### 2.4.6 收敛判据

当相邻两次迭代的变化量小于阈值时停止：

$$
\frac{1}{n} \sum_{i=1}^n \|F_i^{(t+1)} - F_i^{(t)}\|_1 < \tau
$$

推荐 $\tau = 10^{-6}$，最大迭代次数 $T_{\max} = 200$

### 2.5 软标签提取与归一化

#### 2.5.1 概率归一化

确保每行和为1：

$$
F_i^{(\text{final})} = \frac{F_i^{(T)}}{\sum_{c=0}^1 F_{i,c}^{(T)}}
$$

#### 2.5.2 软标签定义

提取正类概率作为软标签：

$$
p_i = F_{i,1}^{(\text{final})} \in [0, 1]
$$

**物理意义**：
- $p_i \approx 1$：高置信度正类
- $p_i \approx 0$：高置信度负类
- $p_i \approx 0.5$：边界样本，高不确定性

### 2.6 边界样本权重

#### 2.6.1 不确定性度量

定义样本$i$的不确定性为：

$$
u_i = 1 - |2p_i - 1| = 1 - |F_{i,1} - F_{i,0}|
$$

性质：
- $u_i = 0$：完全确定（$p_i \in \{0, 1\}$）
- $u_i = 1$：完全不确定（$p_i = 0.5$）

#### 2.6.2 样本权重设计

为边界样本赋予更高权重：

$$
w_i = 1 + \lambda \cdot u_i
$$

其中 $\lambda \geq 0$ 为边界强调系数（推荐 $\lambda = 2.0$）

**作用**：
1. 让模型更关注难分样本
2. 符合Focal Loss的难样本挖掘思想
3. 与课程学习(Curriculum Learning)思想一致

---

## 3. 图注意力网络：特征学习

### 3.1 动机与架构设计

**核心思想**：在隐空间中，样本之间的拓扑结构蕴含丰富的类别信息。GAT通过可学习的注意力机制自适应聚合邻居特征。

**与标签传播的区别**：
- 标签传播：无参数，基于距离的固定权重
- GAT：有参数，基于特征的自适应权重

### 3.2 图构建

#### 3.2.1 全局k近邻图

对所有样本（训练集+验证集+测试集）构建统一的k近邻图：

$$
\mathcal{G}' = (\mathcal{V}', \mathcal{E}')
$$

其中：
- $\mathcal{V}' = \{1, \ldots, n_{train}, \ldots, n_{train}+n_{val}, \ldots, n_{total}\}$
- $\mathcal{E}' = \{(i,j) : j \in \mathcal{N}'(i)\}$

**重要说明**（Transductive设定）：
- 图结构包含所有节点（包括测试集）
- 但仅训练集节点的标签用于监督
- 验证集/测试集节点的标签不参与训练

#### 3.2.2 边列表表示

将图表示为两个向量 $(dst, src)$：

$$
\begin{aligned}
dst &= [\underbrace{1, \ldots, 1}_{k\text{次}}, \underbrace{2, \ldots, 2}_{k\text{次}}, \ldots, \underbrace{n, \ldots, n}_{k\text{次}}] \\
src &= [\mathcal{N}'(1), \mathcal{N}'(2), \ldots, \mathcal{N}'(n)]
\end{aligned}
$$

每条边 $(dst[e], src[e])$ 表示从节点 $src[e]$ 向节点 $dst[e]$ 传递信息。

### 3.3 单头注意力机制

#### 3.3.1 特征线性变换

对节点特征进行线性投影：

$$
h_i = W z_i, \quad W \in \mathbb{R}^{D \times d}
$$

其中 $D$ 为隐藏层维度。

#### 3.3.2 注意力系数计算

对每条边 $(i, j)$ 计算注意力logit：

$$
e_{ij} = \text{LeakyReLU}\left(a^T [h_i \| h_j]\right)
$$

其中：
- $a \in \mathbb{R}^{2D}$ 为可学习的注意力向量
- $\|$ 表示拼接操作
- LeakyReLU 为激活函数：$\text{LeakyReLU}(x) = \max(0.2x, x)$

**等价形式**（分解式）：

$$
e_{ij} = \text{LeakyReLU}\left(a_{dst}^T h_i + a_{src}^T h_j\right)
$$

其中 $a_{dst}, a_{src} \in \mathbb{R}^D$ 分别为目标节点和源节点的注意力向量。

#### 3.3.3 注意力归一化（Softmax）

对每个节点$i$，对其所有入边进行softmax归一化：

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}'(i)} \exp(e_{ik})}
$$

**分组Softmax的技术实现**（GPU加速）：

$$
\alpha_{ij} = \frac{\exp(e_{ij} - \max_{k \in \mathcal{N}'(i)} e_{ik})}{\sum_{k \in \mathcal{N}'(i)} \exp(e_{ik} - \max_{k \in \mathcal{N}'(i)} e_{ik})}
$$

减去最大值防止数值溢出。

#### 3.3.4 消息传递与聚合

节点$i$的更新表示为：

$$
h_i' = \sum_{j \in \mathcal{N}'(i)} \alpha_{ij} h_j
$$

**物理意义**：
- $\alpha_{ij}$ 高：节点$j$对$i$的影响大（特征相关性强）
- $\alpha_{ij}$ 低：节点$j$对$i$的影响小（特征相关性弱）

### 3.4 多头注意力机制

#### 3.4.1 动机

单头注意力可能陷入局部最优，多头机制通过多个独立的注意力头增强表达能力。

#### 3.4.2 并行计算

对每个头 $h \in \{1, \ldots, H\}$ 分别计算：

$$
h_i^{(h)} = \sum_{j \in \mathcal{N}'(i)} \alpha_{ij}^{(h)} \left(W^{(h)} z_j\right)
$$

其中：
- $W^{(h)} \in \mathbb{R}^{D \times d}$ 为第$h$个头的投影矩阵
- $\alpha_{ij}^{(h)}$ 为第$h$个头的注意力系数

#### 3.4.3 多头聚合

**拼接策略**（用于中间层）：

$$
h_i' = \text{Concat}(h_i^{(1)}, h_i^{(2)}, \ldots, h_i^{(H)}) \in \mathbb{R}^{H \cdot D}
$$

**平均策略**（用于输出层）：

$$
h_i' = \frac{1}{H} \sum_{h=1}^H h_i^{(h)} \in \mathbb{R}^D
$$

### 3.5 GAT层堆叠

#### 3.5.1 第一层GAT

$$
H^{(1)} = \text{Concat}_{h=1}^H \left[\sum_{j \in \mathcal{N}'(i)} \alpha_{ij}^{(h)} W^{(h)} z_j\right]
$$

输出维度：$H^{(1)} \in \mathbb{R}^{n \times (H \cdot D)}$

#### 3.5.2 中间激活

$$
H^{(1)} \leftarrow \text{ELU}(H^{(1)})
$$

其中 ELU 为指数线性单元：

$$
\text{ELU}(x) = \begin{cases}
x & \text{if } x > 0 \\
e^x - 1 & \text{if } x \leq 0
\end{cases}
$$

**优势**：
1. 负值时有非零梯度（避免"死神经元"）
2. 均值接近0（加速收敛）
3. 平滑的非线性

#### 3.5.3 第二层GAT

$$
H^{(2)} = \frac{1}{H} \sum_{h=1}^H \left[\sum_{j \in \mathcal{N}'(i)} \beta_{ij}^{(h)} W'^{(h)} H_j^{(1)}\right]
$$

输出维度：$H^{(2)} \in \mathbb{R}^{n \times D}$

### 3.6 双任务输出头

#### 3.6.1 分类头（Classification Head）

对第二层输出应用MLP：

$$
\text{logit}_i = \text{Linear}(\text{ReLU}(\text{Dropout}(H_i^{(2)})))
$$

具体展开：

$$
\begin{aligned}
\tilde{h}_i &= \text{Dropout}(H_i^{(2)}, p=0.2) \\
\bar{h}_i &= \text{ReLU}(\tilde{h}_i) \\
\text{logit}_i &= w_{cls}^T \bar{h}_i + b_{cls}
\end{aligned}
$$

其中 $w_{cls} \in \mathbb{R}^D, b_{cls} \in \mathbb{R}$

**输出**：分类logit $\in \mathbb{R}^n$

#### 3.6.2 投影头（Projection Head for Contrastive Learning）

对第二层输出应用两层MLP：

$$
\begin{aligned}
\hat{h}_i &= \text{ReLU}(W_{proj1} H_i^{(2)} + b_{proj1}) \\
v_i &= W_{proj2} \hat{h}_i + b_{proj2}
\end{aligned}
$$

其中：
- $W_{proj1} \in \mathbb{R}^{D \times D}, W_{proj2} \in \mathbb{R}^{D_{proj} \times D}$
- $D_{proj}$ 为投影维度（推荐64）

**输出**：投影特征 $v_i \in \mathbb{R}^{D_{proj}}$，用于对比学习

---

## 4. 损失函数设计

### 4.1 Focal Loss：难样本自适应加权

#### 4.1.1 动机

**传统BCE的问题**：

$$
\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^n \left[p_i \log(\hat{p}_i) + (1-p_i)\log(1-\hat{p}_i)\right]
$$

其中 $\hat{p}_i = \sigma(\text{logit}_i)$ 为预测概率。

**问题**：
1. 易分样本（$\hat{p}_i \approx p_i$）仍贡献大量损失
2. 类别不平衡时，多数类主导梯度
3. 无法自动关注边界样本

#### 4.1.2 Focal Loss定义

$$
\mathcal{L}_{Focal} = -\frac{1}{n} \sum_{i=1}^n \alpha_t^{(i)} (1-p_t^{(i)})^\gamma \log(p_t^{(i)})
$$

其中：
- $p_t^{(i)} = p_i \hat{p}_i + (1-p_i)(1-\hat{p}_i)$ 为"正确类别的预测概率"
- $\gamma \geq 0$ 为聚焦参数（focusing parameter）
- $\alpha_t^{(i)} = p_i \alpha + (1-p_i)(1-\alpha)$ 为类别平衡权重

#### 4.1.3 调制因子分析

调制因子 $(1-p_t)^\gamma$ 的作用：

| 样本类型 | $p_t$ | $(1-p_t)^\gamma$ (γ=2) | 损失权重 |
|----------|-------|------------------------|----------|
| 易分样本 | 0.9 | 0.01 | 极低 ↓ |
| 中等样本 | 0.7 | 0.09 | 低 |
| 难分样本 | 0.5 | 0.25 | 中 |
| 边界样本 | 0.3 | 0.49 | 高 ↑ |
| 误分样本 | 0.1 | 0.81 | 极高 ↑↑ |

**关键洞察**：
- $\gamma = 0$：退化为BCE
- $\gamma = 2$（推荐）：标准Focal Loss
- $\gamma = 5$：极度关注难样本（适合极端不平衡）

#### 4.1.4 类别平衡权重

$$
\alpha = \frac{n_{neg}}{n_{neg} + n_{pos}}
$$

**示例**：
- 如果 $\rho = 9:1$，则 $\alpha = 0.9$
- 正类损失乘以 $\alpha=0.9$（高权重）
- 负类损失乘以 $1-\alpha=0.1$（低权重）

#### 4.1.5 边界样本加权（可选）

结合软标签的不确定性：

$$
\mathcal{L}_{Focal}^{weighted} = \frac{1}{\sum_i w_i} \sum_{i=1}^n w_i \cdot \alpha_t^{(i)} (1-p_t^{(i)})^\gamma \log(p_t^{(i)})
$$

其中 $w_i = 1 + \lambda u_i$（见2.6节）

**效果叠加**：
- Focal Loss：自动关注预测难的样本
- 边界权重：手动强调标签不确定的样本
- 两者互补：预测难 ∩ 标签不确定 = 真正的边界样本

### 4.2 监督对比学习损失

#### 4.2.1 动机

**核心思想**：通过对比学习增强特征空间的判别性，使同类样本在特征空间中聚合，异类样本分离。

**与分类的互补性**：
- 分类损失：优化决策边界（what to predict）
- 对比损失：优化特征表示（how to represent）

#### 4.2.2 对比损失定义

对于节点$i$，定义监督对比损失为：

$$
\mathcal{L}_{contrast}^{(i)} = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(v_i \cdot v_p / \tau)}{\sum_{a \in A(i)} \exp(v_i \cdot v_a / \tau)}
$$

其中：
- $v_i$ 为节点$i$的L2归一化投影特征：$v_i = \frac{\text{proj}(H_i^{(2)})}{\|\text{proj}(H_i^{(2)})\|}$
- $P(i) = \{p : y_p = y_i, p \neq i\}$ 为正样本集（同类样本）
- $A(i) = \{a : a \neq i\}$ 为所有样本集（除自己外）
- $\tau > 0$ 为温度参数（推荐0.07）

#### 4.2.3 温度参数的作用

温度 $\tau$ 控制分布的"尖锐度"：

$$
\text{sim}(i, j) = \frac{\exp(v_i \cdot v_j / \tau)}{\sum_k \exp(v_i \cdot v_k / \tau)}
$$

- $\tau \to 0$：分布极度尖锐（hard assignment）
- $\tau \to \infty$：分布均匀（忽略相似度差异）
- $\tau = 0.07$：平衡区分度和泛化性

#### 4.2.4 少数类增强策略

为少数类样本赋予更高权重：

$$
\mathcal{L}_{contrast} = \frac{1}{\sum_i c_i} \sum_{i=1}^n c_i \cdot \mathcal{L}_{contrast}^{(i)}
$$

其中：
$$
c_i = \begin{cases}
\lambda_{minority} & \text{if } y_i = 1 \\
1 & \text{if } y_i = 0
\end{cases}
$$

**推荐配置**：
- $\rho < 5$: $\lambda_{minority} = 1.5$
- $5 \leq \rho < 10$: $\lambda_{minority} = 2.0$
- $\rho \geq 10$: $\lambda_{minority} = 3.0$

**理论依据**：
1. 少数类样本稀少，每个样本更宝贵
2. 增大权重相当于过采样，但无需生成合成样本
3. 在特征空间中强化少数类簇的紧致性

#### 4.2.5 对比损失的梯度分析

对 $v_i$ 求导：

$$
\frac{\partial \mathcal{L}_{contrast}^{(i)}}{\partial v_i} = \frac{1}{\tau |P(i)|} \sum_{p \in P(i)} \left[\sum_{a \in A(i)} \text{sim}(i,a) v_a - v_p\right]
$$

**物理意义**：
- 第一项：将$v_i$推向所有样本的加权平均（repulsion from all）
- 第二项：将$v_i$拉向正样本$v_p$（attraction to positives）
- 净效应：同类聚合 + 异类分离

### 4.3 总损失函数

$$
\mathcal{L}_{total} = \mathcal{L}_{Focal} + \beta \cdot \mathcal{L}_{contrast}
$$

其中 $\beta \in [0.3, 0.7]$ 为对比损失权重。

**权重选择原则**：
1. $\beta$ 过大：特征学习主导，可能牺牲分类性能
2. $\beta$ 过小：对比学习作用微弱
3. 推荐从 $\beta=0.5$ 开始，通过验证集调优

**训练策略**（可选）：
- Warm-up：前10 epochs $\beta=0$（仅训练分类）
- Ramp-up：10-30 epochs 线性增加 $\beta$ 至目标值
- Stable：30+ epochs 固定 $\beta$

---

## 5. 优化算法

### 5.1 优化器选择

使用 **AdamW** (Adam with Weight Decay)：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
\end{aligned}
$$

**参数配置**：
- 学习率 $\eta = 10^{-3}$
- $\beta_1 = 0.9, \beta_2 = 0.999$
- Weight decay $\lambda = 10^{-5}$
- $\epsilon = 10^{-8}$

**为何选择AdamW**：
1. 对学习率不敏感（相比SGD）
2. 自适应学习率（适合不同参数规模）
3. Weight decay解耦（更好的正则化）

### 5.2 早停策略

在验证集上监控宏平均F1：

$$
F1_{macro} = \frac{1}{2}\left(\frac{2 \cdot TP_0}{2 \cdot TP_0 + FP_0 + FN_0} + \frac{2 \cdot TP_1}{2 \cdot TP_1 + FP_1 + FN_1}\right)
$$

**早停规则**：
- 记录最佳验证F1: $F1_{best}$
- 如果连续20 epochs无改进，停止训练
- 加载 $F1_{best}$ 对应的模型参数

### 5.3 学习率调度（可选）

**余弦退火**（Cosine Annealing）：

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)
$$

其中：
- $\eta_{max} = 10^{-3}$（初始学习率）
- $\eta_{min} = 10^{-5}$（最小学习率）
- $T_{max} = 200$（总epoch数）

---

## 6. 推理与阈值优化

### 6.1 模型推理

#### 6.1.1 Logits计算

$$
\text{logit}_i = f_{GAT}(Z, \mathcal{E}'; \theta^*)
$$

其中 $\theta^*$ 为训练得到的最优参数。

#### 6.1.2 概率预测

$$
\hat{p}_i = \sigma(\text{logit}_i) = \frac{1}{1 + e^{-\text{logit}_i}}
$$

### 6.2 决策阈值优化

#### 6.2.1 阈值搜索空间

在验证集上搜索最优阈值 $t^*$：

$$
t^* = \underset{t \in [0.05, 0.95]}{\text{argmax}} \, F1_{macro}(t)
$$

搜索步长：0.01（共91个候选值）

#### 6.2.2 阈值相关指标

对于给定阈值 $t$：

$$
\hat{y}_i = \begin{cases}
1 & \text{if } \hat{p}_i \geq t \\
0 & \text{if } \hat{p}_i < t
\end{cases}
$$

**Macro F1**：

$$
F1_{macro}(t) = \frac{1}{2}\left(F1_0(t) + F1_1(t)\right)
$$

**G-Mean** (几何平均)：

$$
\text{G-Mean}(t) = \sqrt{\text{Sensitivity}(t) \times \text{Specificity}(t)}
$$

其中：
- $\text{Sensitivity} = \frac{TP}{TP + FN}$ (召回率)
- $\text{Specificity} = \frac{TN}{TN + FP}$ (特异性)

#### 6.2.3 为何不固定 $t=0.5$

在不平衡数据中，$t=0.5$ 通常**不是最优**：

**示例**：假设 $\rho = 9:1$
- 模型倾向于预测多数类，导致 $\hat{p}_i$ 整体偏小
- 固定 $t=0.5$ 会导致 FN（假阴性）过高
- 最优阈值可能在 $t^* \approx 0.3$ 附近

**阈值优化的本质**：
$$
t^* = \underset{t}{\text{argmin}} \left[\text{Cost}_{FP}(t) + \text{Cost}_{FN}(t)\right]
$$

在Macro-F1下，两类错误代价相等。

### 6.3 测试集评估

使用验证集上的最优阈值 $t^*$ 在测试集上评估：

$$
\hat{y}_i^{test} = \mathbb{1}[\hat{p}_i^{test} \geq t^*]
$$

**关键指标**：
1. **Macro-F1**：平衡两类性能
2. **AUC**：阈值无关指标，衡量排序能力
3. **MCC** (Matthews相关系数)：综合考虑TP/TN/FP/FN
4. **G-Mean**：几何平均敏感性和特异性

---

## 7. 理论分析

### 7.1 为何软标签有效？

#### 7.1.1 信息论视角

硬标签熵：
$$
H(y) = -\sum_{c \in \{0,1\}} \mathbb{1}[y=c] \log \mathbb{1}[y=c] = 0
$$

软标签熵：
$$
H(p) = -p \log p - (1-p) \log(1-p) > 0
$$

**结论**：软标签提供更丰富的信息，缓解过拟合。

#### 7.1.2 正则化视角

软标签等价于标签平滑(Label Smoothing)：

$$
p_i^{smooth} = (1-\epsilon) y_i + \epsilon \cdot \text{uniform}
$$

但软标签更智能：$\epsilon$ 依赖于邻域结构，而非全局常数。

### 7.2 为何GAT优于GCN？

#### 7.2.1 GCN的更新规则

$$
H_i^{GCN} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} W H_j\right)
$$

权重 $\frac{1}{\sqrt{d_i d_j}}$ 仅依赖度数，与特征无关。

#### 7.2.2 GAT的优势

$$
H_i^{GAT} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}(H_i, H_j) W H_j\right)
$$

权重 $\alpha_{ij}$ 依赖特征，自适应调整。

**适用场景**：
- 图结构噪声大 → GAT更鲁棒
- 节点重要性差异大 → GAT可自动区分

### 7.3 对比学习为何增强少数类？

#### 7.3.1 特征空间分析

对比损失的效果：

$$
\min_{v_i} \mathcal{L}_{contrast} \Leftrightarrow \begin{cases}
\max \text{intra-class similarity} \\
\min \text{inter-class similarity}
\end{cases}
$$

#### 7.3.2 少数类加权的几何意义

权重 $\lambda_{minority}$ 相当于：
1. 增大少数类簇的"引力"
2. 减小少数类与多数类的"斥力差异"
3. 让少数类形成更紧致的簇

**数学表达**：

$$
\frac{\partial \mathcal{L}}{\partial v_i} \propto \lambda_{minority} \cdot \text{(gradients)}
$$

等价于梯度上升步长放大 $\lambda_{minority}$ 倍。

---

## 8. 算法复杂度分析

### 8.1 时间复杂度

#### 8.1.1 标签传播阶段

- kNN图构建：$O(n d \log n)$ (使用KD树)
- 每次迭代：$O(n k)$
- 总计：$O(n d \log n + T \cdot nk)$

#### 8.1.2 GAT训练阶段

单次前向传播：
- GAT层1：$O(|E| \cdot H \cdot D) = O(nk \cdot H \cdot D)$
- GAT层2：$O(nk \cdot D^2)$
- 分类头：$O(n \cdot D)$
- 对比损失：$O(n^2 \cdot D_{proj})$ （瓶颈！）

单个epoch：$O(nk \cdot H \cdot D + n^2 \cdot D_{proj})$

**瓶颈**：对比学习需要计算 $n \times n$ 相似度矩阵。

**优化方案**：
1. 仅对训练集计算对比损失（$n \to n_{train}$）
2. Batch sampling（随机采样子集）
3. 负样本采样（不计算所有负样本对）

### 8.2 空间复杂度

- kNN图：$O(nk)$
- GAT参数：$O(d \cdot H \cdot D + H \cdot D^2)$
- 中间激活：$O(n \cdot H \cdot D)$
- 对比学习相似度矩阵：$O(n^2)$ （峰值）

**总计**：$O(n^2 + nHD)$

---

## 9. 实验设计指南

### 9.1 消融实验

| 配置 | 软标签 | Focal Loss | 对比学习 |
|------|--------|------------|----------|
| Baseline | ✗ | ✗ | ✗ |
| +Soft | ✓ | ✗ | ✗ |
| +Focal | ✓ | ✓ | ✗ |
| Full | ✓ | ✓ | ✓ |

**目的**：验证每个模块的边际贡献。

### 9.2 超参数敏感性分析

| 参数 | 候选值 | 建议 |
|------|--------|------|
| $\gamma$ | {0, 1, 2, 3, 5} | 2 |
| $\beta$ | {0.3, 0.5, 0.7, 1.0} | 0.5 |
| $\lambda_{minority}$ | {1.0, 1.5, 2.0, 3.0} | 2.0 |
| $k$ | {10, 20, 30, 50} | 30 |
| $\tau$ | {0.05, 0.07, 0.1, 0.2} | 0.07 |

### 9.3 对比基线

1. **传统方法**：SMOTE, ADASYN, RUSBoost
2. **深度学习**：MLP + Class Weight, Focal Loss
3. **图方法**：GCN, GraphSAINT
4. **集成方法**：BalancedRandomForest, EasyEnsemble

---

## 10. 结论

### 10.1 方法总结

本方法通过三个核心组件协同工作：

1. **标签传播**：在隐空间中传播标签信息，生成软标签，为边界样本提供不确定性建模
2. **图注意力网络**：利用可学习的注意力机制聚合邻域特征，自适应学习拓扑结构
3. **双任务优化**：Focal Loss关注难样本 + 对比学习增强特征判别性，共同应对类别不平衡

### 10.2 理论贡献

1. 将软标签生成与图神经网络有机结合
2. 分离软标签（分类）和硬标签（对比）的使用场景
3. 提出少数类感知的对比学习策略

### 10.3 实践价值

1. 端到端可训练，无需复杂的预处理
2. 理论有据，易于调参
3. 可扩展性强，适用于各类不平衡图数据

---

## 参考文献

[1] Veličković et al. "Graph Attention Networks" (ICLR 2018)

[2] Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

[3] Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)

[4] Zhu & Ghahramani "Learning from Labeled and Unlabeled Data with Label Propagation" (CMU-CALD-02)

[5] He et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)

[6] Chawla et al. "SMOTE: Synthetic Minority Over-sampling Technique" (JAIR 2002)
