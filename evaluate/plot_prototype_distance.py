import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ================= 1. 环境配置 (解决中文显示问题) =================
# 尝试加载常用中文字体，确保在 Windows/Mac/Linux 环境下的兼容性
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ================= 2. 数据读取 =================
# 建议使用绝对路径或确保脚本与CSV在同一目录
# df = pd.read_csv('vae_loss_detailed.csv')
# 此处使用您原本的逻辑，仅保留一个有效路径
df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\churn\CoTable/vae_loss_detailed.csv')
df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/adult\CoTable/vae_loss_detailed.csv')
df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable/vae_loss_detailed.csv')
df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable/vae_loss_detailed.csv')
df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/shopper\CoTable/vae_loss_detailed.csv')

epochs = df['epoch']
train_proto_dist = df['train_proto_dist']
val_proto_dist = df['val_proto_dist']

# ================= 3. 绘图与美化 =================
plt.style.use('seaborn-v0_8-paper')  # 使用更简洁的风格基础
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# 配色方案
color_train = '#2c3e50'  # 深灰蓝
color_val = '#e74c3c'    # 珊瑚红

# 绘制折线
ax.plot(epochs, train_proto_dist,
        label='训练集',
        color=color_train,
        linewidth=2,
        alpha=0.85,
        linestyle='-')

ax.plot(epochs, val_proto_dist,
        label='测试集',
        color=color_val,
        linewidth=2,
        alpha=0.85,
        linestyle='--')

ax.set_xlim(left=0, right=max(epochs))
# 坐标轴标签
ax.set_xlabel('训练轮次', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('类原型距离', fontsize=14, fontweight='bold', labelpad=10)

# 细节微调
ax.grid(True, linestyle='--', alpha=0.4, which='both') # 柔化网格线
ax.spines['top'].set_visible(False)    # 去掉顶部边框
ax.spines['right'].set_visible(False)  # 去掉右侧边框
ax.tick_params(axis='both', which='major', labelsize=12)

# 图例优化
ax.legend(
    loc='lower right',
    frameon=True,
    shadow=False,  # 取消阴影
    fontsize=12,
)

# 自动调整布局并保存
plt.tight_layout()
plt.savefig('prototype_distance_comparison_v2.png', bbox_inches='tight', dpi=300)
plt.show()