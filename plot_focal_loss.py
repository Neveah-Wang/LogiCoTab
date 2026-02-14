"""
绘制不同软标签下的Focal损失曲面
展示混合Focal损失在不同软标签值下的行为特征
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号

def hybrid_focal_loss(s, p_hat, y, gamma=2.0, alpha=None):
    """
    计算混合Focal损失

    参数:
        s: 软标签
        p_hat: 预测概率
        y: 硬标签
        gamma: 聚焦参数
        alpha: 类别平衡权重（可选）

    返回:
        focal_loss: Focal损失值
    """
    # 一致性度量 p_t（基于软标签）
    p_t = 1 - np.abs(s - p_hat)

    # 调制因子
    focal_weight = (1 - p_t) ** gamma

    # BCE损失（基于硬标签）
    eps = 1e-7  # 数值稳定性
    bce = -(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))

    # Focal损失
    focal_loss = focal_weight * bce

    # 类别平衡权重（可选）
    if alpha is not None:
        alpha_t = s * alpha + (1 - s) * (1 - alpha)
        focal_loss = alpha_t * focal_loss

    return focal_loss

# ============================================================================
# y=1 (正类) 的损失曲线
# ============================================================================

# 创建图形
fig, ax = plt.subplots(figsize=(10, 7))

# 设置参数
y_fixed =1  # 固定硬标签为正类
gamma = 2.0  # 聚焦参数
p_range = np.linspace(0.00001, 0.99, 200)  # 预测概率范围

# 不同的软标签值
s_values = [0.5, 0.6, 0.8, 1.0]
# s_values = [0.1, 0.3, 0.4, 0.5]
colors = plt.cm.rainbow(np.linspace(0, 1, len(s_values))) # 颜色方案
# linestyles = ['-', '--', '-.', ':']

# 绘制每个软标签对应的损失曲线
for i, s_val in enumerate(s_values):
    focal_curve = []
    for p in p_range:
        loss = hybrid_focal_loss(s_val, p, y_fixed, gamma)
        focal_curve.append(loss)

    # 绘制曲线
    ax.plot(p_range, focal_curve,
            color=colors[i],
            linewidth=2.5,
            label=f'软标签 p = {s_val}',
            alpha=0.8)

    # 标记软标签位置（预测与软标签一致的点）
    loss_at_s = hybrid_focal_loss(s_val, s_val, y_fixed, gamma)
    ax.plot(s_val, loss_at_s, 'o',
            color=colors[i],
            markersize=10,
            markeredgecolor='white',
            markeredgewidth=1,
            zorder=5)

# 添加决策阈值线
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5,
           alpha=0.6, label='决策阈值 = 0.5')

# 添加零损失参考线
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# 设置标签和标题
ax.set_xlabel('预测概率 $\hat{p}$', fontsize=24, fontweight='bold')
ax.set_ylabel('焦点损失', fontsize=24, fontweight='bold')
# ax.set_title('原始标签 y=1（正样本）', fontsize=14, fontweight='bold', pad=15)

# 设置图例
ax.legend(fontsize=20, loc='upper right', framealpha=0.95, edgecolor='gray', fancybox=True)

# 网格
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# 设置坐标轴范围
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 2.0)

ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()

# 保存图形
plt.savefig('不同软标签的焦点损失曲线y=1.pdf',)

# 显示图形
plt.show()

# ============================================================================
# y=0 (负类) 的损失曲线
# ============================================================================

# 创建图形
fig, ax = plt.subplots(figsize=(10, 7))

# 设置参数
y_fixed = 0  # 固定硬标签为正类
gamma = 2.0  # 聚焦参数
p_range = np.linspace(0.00001, 0.9999, 200)  # 预测概率范围

# 不同的软标签值
s_values = [0.5, 0.4, 0.2, 0.0]
colors = plt.cm.rainbow(np.linspace(0, 1, len(s_values))) # 颜色方案
# linestyles = ['-', '--', '-.', ':']

# 绘制每个软标签对应的损失曲线
for i, s_val in enumerate(s_values):
    focal_curve = []
    for p in p_range:
        loss = hybrid_focal_loss(s_val, p, y_fixed, gamma)
        focal_curve.append(loss)

    # 绘制曲线
    ax.plot(p_range, focal_curve,
            color=colors[i],
            linewidth=2.5,
            label=f'软标签 p = {s_val}',
            alpha=0.8)

    # 标记软标签位置（预测与软标签一致的点）
    loss_at_s = hybrid_focal_loss(s_val, s_val, y_fixed, gamma)
    ax.plot(s_val, loss_at_s, 'o',
            color=colors[i],
            markersize=10,
            markeredgecolor='white',
            markeredgewidth=1,
            zorder=5)

# 添加决策阈值线
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='决策阈值 = 0.5')

# 添加零损失参考线
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# 设置标签和标题
ax.set_xlabel('预测概率 $\hat{p}$', fontsize=24, fontweight='bold')
ax.set_ylabel('焦点损失', fontsize=24, fontweight='bold')
# ax.set_title('原始标签 y=0（负样本）', fontsize=16, fontweight='bold', pad=15)

# 设置图例
ax.legend(fontsize=20, loc='upper left', framealpha=0.95,
          edgecolor='gray', fancybox=True)

# 网格
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# 设置坐标轴范围
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 2.0)

ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()

# 保存图形
plt.savefig('不同软标签的焦点损失曲线y=0.pdf',)

# 显示图形
plt.show()


# ============================================================================
# 额外分析：打印关键数值
# ============================================================================
'''
print("\n" + "=" * 70)
print("关键数值分析")
print("=" * 70)

print(f"\n硬标签: y = {y_fixed}")
print(f"聚焦参数: γ = {gamma}")

for s_val in s_values:
    print(f"\n软标签 s = {s_val}:")

    # 预测与软标签一致时的损失
    loss_at_s = hybrid_focal_loss(s_val, s_val, y_fixed, gamma)
    print(f"  当 p̂ = s = {s_val}: Loss = {loss_at_s:.4f} (p_t = 1.0)")

    # 预测在决策阈值时的损失
    loss_at_threshold = hybrid_focal_loss(s_val, 0.5, y_fixed, gamma)
    p_t_at_threshold = 1 - np.abs(s_val - 0.5)
    print(f"  当 p̂ = 0.5 (阈值): Loss = {loss_at_threshold:.4f} (p_t = {p_t_at_threshold:.2f})")

    # 预测接近硬标签时的损失
    loss_near_y = hybrid_focal_loss(s_val, 0.9, y_fixed, gamma)
    p_t_near_y = 1 - np.abs(s_val - 0.9)
    print(f"  当 p̂ = 0.9 (接近y): Loss = {loss_near_y:.4f} (p_t = {p_t_near_y:.2f})")

print("\n" + "=" * 70)
print("核心观察")
print("=" * 70)
print("""
1. 所有曲线在 p̂ = s 处损失为0（调制因子消失）
2. 软标签越接近0.5，曲线整体越"平坦"（边界样本）
3. 软标签越接近1，曲线在p̂<0.5区域越"陡峭"（高置信样本）
4. 即使s=0.5，当p̂>0.5时损失仍然较低（BCE引导正确分类）
5. 最低损失点并非总在s处，而是在s和y之间的某个位置
""")

print("\n" + "=" * 70)
print("设计合理性验证")
print("=" * 70)
print("""
✓ 防止标签翻转：即使s=0.5，模型被鼓励预测p̂>0.5（因为y=1）
✓ 软标签对齐：当p̂=s时，调制因子为0，损失最小
✓ 难度自适应：不同s值产生不同的损失曲面，适应样本难度
✓ 平滑过渡：从软标签引导（调制因子）到硬标签引导（BCE）的平滑切换
""")

'''