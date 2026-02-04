import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

# ================= 1. 环境配置 (解决中文显示问题) =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 数据读取 =================
# df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\churn\CoTable/vae_loss_detailed.csv')
# df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/adult\CoTable/vae_loss_detailed.csv')
# df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable/vae_loss_detailed.csv')
# df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable/vae_loss_detailed.csv')
# df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/shopper\CoTable/vae_loss_detailed.csv')
df = pd.read_csv('D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/buddy\CoTable/vae_loss_detailed.csv')

epochs = df['epoch'].values
train_proto_dist = df['train_proto_dist'].values
val_proto_dist = df['val_proto_dist'].values


# ================= 3. 平滑处理 =================
# 方法1: 移动平均 (推荐,简单有效)
def moving_average(data, window_size=10):
    """移动平均平滑 """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# 方法2: Savitzky-Golay滤波 (推荐,保留峰值特征)
def savgol_smooth(data, window_length=11, polyorder=3):
    """Savitzky-Golay滤波平滑
    window_length: 窗口大小(必须为奇数),越大越平滑
    polyorder: 多项式阶数,通常为 2-5
    """
    if len(data) < window_length:
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    return savgol_filter(data, window_length, polyorder)


# 方法3: 样条插值 (最平滑,但可能过度拟合)
def spline_smooth(x, y, num_points=300):
    """B样条插值平滑"""
    spl = make_interp_spline(x, y, k=3)
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth


# ========== 选择平滑方法 ==========
# 推荐使用 Savitzky-Golay 滤波,调整 window_length 控制平滑程度
SMOOTH_METHOD = 'savgol'  # 可选: 'moving_avg', 'savgol', 'spline', 'none'

if SMOOTH_METHOD == 'moving_avg':
    window = 15  # 窗口越大越平滑,建议10-20
    train_smooth = moving_average(train_proto_dist, window)
    val_smooth = moving_average(val_proto_dist, window)
    epochs_smooth = epochs[window - 1:]  # 调整epoch对齐

elif SMOOTH_METHOD == 'savgol':
    window = 21  # 必须为奇数,越大越平滑,建议11-31
    poly = 3  # 多项式阶数,2-5之间
    train_smooth = savgol_smooth(train_proto_dist, window, poly)
    val_smooth = savgol_smooth(val_proto_dist, window, poly)
    epochs_smooth = epochs

elif SMOOTH_METHOD == 'spline':
    epochs_train_smooth, train_smooth = spline_smooth(epochs, train_proto_dist, 300)
    epochs_val_smooth, val_smooth = spline_smooth(epochs, val_proto_dist, 300)
    epochs_smooth = epochs_train_smooth

else:  # 'none' - 不平滑
    train_smooth = train_proto_dist
    val_smooth = val_proto_dist
    epochs_smooth = epochs

# ================= 4. 绘图与美化 =================
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

color_train = '#2c3e50'
color_val = '#e74c3c'

# 可选: 先绘制原始数据的淡化版本作为背景参考
ax.plot(epochs, train_proto_dist, color=color_train, linewidth=0.5, alpha=0.2, linestyle='-')
ax.plot(epochs, val_proto_dist, color=color_val, linewidth=0.5, alpha=0.2, linestyle='-')

# 绘制平滑后的曲线
ax.plot(epochs_smooth, train_smooth,
        label='训练集',
        color=color_train,
        linewidth=2.5,
        alpha=0.9,
        linestyle='-')

if SMOOTH_METHOD == 'spline':
    ax.plot(epochs_val_smooth, val_smooth,
            label='测试集',
            color=color_val,
            linewidth=2.5,
            alpha=0.9,
            linestyle='--')
else:
    ax.plot(epochs_smooth, val_smooth,
            label='测试集',
            color=color_val,
            linewidth=2.5,
            alpha=0.9,
            linestyle='--')

ax.set_xlim(left=0, right=max(epochs))
ax.set_xlabel('训练轮次', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('类原型距离', fontsize=14, fontweight='bold', labelpad=10)

ax.grid(True, linestyle='--', alpha=0.4, which='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

ax.legend(
    loc='best',
    frameon=True,
    shadow=False,
    fontsize=12,
)

plt.tight_layout()
plt.savefig('prototype_distance_smooth.png', bbox_inches='tight', dpi=300)
plt.show()