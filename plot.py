import os
import json
import re
import matplotlib.pyplot as plt

# 设置你的 JSON 文件所在目录
json_dir = "D:\Study\自学\表格数据生成\LogiCoTab-oversampling\exp\churn\CoTable\synthesis_null"

# 用于存储横纵坐标数据
n_sample = []
recall_val_1 = []
recall_test_1 = []
recall_train_real_1 = []
recall_train_1 = []
precision_val_1 = []
precision_test_1 = []
precision_train_real_1 = []
precision_train_1 = []
f1_val_1 = []
f1_test_1 = []
f1_train_real_1 = []
f1_train_1 = []

recall_val_0 = []
recall_test_0 = []
recall_train_real_0 = []
recall_train_0 = []
precision_val_0 = []
precision_test_0 = []
precision_train_real_0 = []
precision_train_0 = []
f1_val_0 = []
f1_test_0 = []
f1_train_real_0 = []
f1_train_0 = []

auc_val = []
auc_test = []
auc_train_real = []
auc_train = []

# 遍历目录中的所有文件
for filename in os.listdir(json_dir):
    if filename.startswith("eval_catboost_nsample") and filename.endswith(".json"):
        # 用正则提取 nsample 后的数字
        match = re.search(r'nsample(\d+)', filename)
        if match:
            nsample = int(match.group(1))
            filepath = os.path.join(json_dir, filename)

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                # 提取值
                recall_val_1.append(data["merged"]["val"]["1"]["recall"]["mean"])
                recall_test_1.append(data["merged"]["test"]["1"]["recall"]["mean"])
                recall_train_real_1.append(data["merged"]["train_real"]["1"]["recall"]["mean"])
                recall_train_1.append(data["merged"]["train"]["1"]["recall"]["mean"])
                precision_val_1.append(data["merged"]["val"]["1"]["precision"]["mean"])
                precision_test_1.append(data["merged"]["test"]["1"]["precision"]["mean"])
                precision_train_real_1.append(data["merged"]["train_real"]["1"]["precision"]["mean"])
                precision_train_1.append(data["merged"]["train"]["1"]["precision"]["mean"])
                f1_val_1.append(data["merged"]["val"]["1"]["f1-score"]["mean"])
                f1_test_1.append(data["merged"]["test"]["1"]["f1-score"]["mean"])
                f1_train_real_1.append(data["merged"]["train_real"]["1"]["f1-score"]["mean"])
                f1_train_1.append(data["merged"]["train"]["1"]["f1-score"]["mean"])

                recall_val_0.append(data["merged"]["val"]["0"]["recall"]["mean"])
                recall_test_0.append(data["merged"]["test"]["0"]["recall"]["mean"])
                recall_train_real_0.append(data["merged"]["train_real"]["0"]["recall"]["mean"])
                recall_train_0.append(data["merged"]["train"]["0"]["recall"]["mean"])
                precision_val_0.append(data["merged"]["val"]["0"]["precision"]["mean"])
                precision_test_0.append(data["merged"]["test"]["0"]["precision"]["mean"])
                precision_train_real_0.append(data["merged"]["train_real"]["0"]["precision"]["mean"])
                precision_train_0.append(data["merged"]["train"]["0"]["precision"]["mean"])
                f1_val_0.append(data["merged"]["val"]["0"]["f1-score"]["mean"])
                f1_test_0.append(data["merged"]["test"]["0"]["f1-score"]["mean"])
                f1_train_real_0.append(data["merged"]["train_real"]["0"]["f1-score"]["mean"])
                f1_train_0.append(data["merged"]["train"]["0"]["f1-score"]["mean"])

                auc_val.append(data["merged"]["val"]["roc_auc"]["mean"])
                auc_test.append(data["merged"]["test"]["roc_auc"]["mean"])
                auc_train_real.append(data["merged"]["train_real"]["roc_auc"]["mean"])
                auc_train.append(data["merged"]["train"]["roc_auc"]["mean"])
                n_sample.append(nsample)
            except (KeyError, json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: 跳过文件 {filename}，原因: {e}")
        else:
            print(f"Warning: 无法从文件名 {filename} 提取 nsample")

# %%
plt.figure()

# 按横坐标排序（可选，确保折线顺序正确）
sorted_pairs = sorted(zip(n_sample, recall_val_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="val")

sorted_pairs = sorted(zip(n_sample, recall_test_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="test")

sorted_pairs = sorted(zip(n_sample, recall_train_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train")

sorted_pairs = sorted(zip(n_sample, recall_train_real_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train_real")


plt.xlabel("n_sample")
plt.ylabel("Recall")
plt.title("Recall of class 1")
plt.legend()
plt.xticks(range(min(x), max(x) + 1, 400))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# %%
plt.figure()

# 按横坐标排序（可选，确保折线顺序正确）
sorted_pairs = sorted(zip(n_sample, precision_val_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="val")

sorted_pairs = sorted(zip(n_sample, precision_test_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="test")

sorted_pairs = sorted(zip(n_sample, precision_train_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train")

sorted_pairs = sorted(zip(n_sample, precision_train_real_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train_real")


plt.xlabel("n_sample")
plt.ylabel("Precision")
plt.title("Precision of class 1")
plt.legend()
plt.xticks(range(min(x), max(x) + 1, 400))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# %%
plt.figure()

# 按横坐标排序（可选，确保折线顺序正确）
sorted_pairs = sorted(zip(n_sample, f1_val_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="val")

sorted_pairs = sorted(zip(n_sample, f1_test_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="test")

sorted_pairs = sorted(zip(n_sample, f1_train_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train")

sorted_pairs = sorted(zip(n_sample, f1_train_real_1))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train_real")


plt.xlabel("n_sample")
plt.ylabel("F1")
plt.title("F1 of class 1")
plt.legend()
plt.xticks(range(min(x), max(x) + 1, 400))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# %%
plt.figure()

# 按横坐标排序（可选，确保折线顺序正确）
sorted_pairs = sorted(zip(n_sample, recall_val_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="val")

sorted_pairs = sorted(zip(n_sample, recall_test_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="test")

sorted_pairs = sorted(zip(n_sample, recall_train_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train")

# sorted_pairs = sorted(zip(n_sample, recall_train_real_0))
# x, y = zip(*sorted_pairs)
# plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train_real")


plt.xlabel("n_sample")
plt.ylabel("Recall")
plt.title("Recall of class 0")
plt.legend()
plt.xticks(range(min(x), max(x) + 1, 400))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# %%
plt.figure()

# 按横坐标排序（可选，确保折线顺序正确）
sorted_pairs = sorted(zip(n_sample, precision_val_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="val")

sorted_pairs = sorted(zip(n_sample, precision_test_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="test")

sorted_pairs = sorted(zip(n_sample, precision_train_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train")

sorted_pairs = sorted(zip(n_sample, precision_train_real_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train_real")


plt.xlabel("n_sample")
plt.ylabel("Precision")
plt.title("Precision of class 0")
plt.legend()
plt.xticks(range(min(x), max(x) + 1, 400))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# %%
plt.figure()

# 按横坐标排序（可选，确保折线顺序正确）
sorted_pairs = sorted(zip(n_sample, f1_val_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="val")

sorted_pairs = sorted(zip(n_sample, f1_test_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="test")

sorted_pairs = sorted(zip(n_sample, f1_train_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train")

sorted_pairs = sorted(zip(n_sample, f1_train_real_0))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train_real")


plt.xlabel("n_sample")
plt.ylabel("F1")
plt.title("F1 of class 0")
plt.legend()
plt.xticks(range(min(x), max(x) + 1, 400))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# %%
plt.figure()

# 按横坐标排序（可选，确保折线顺序正确）
sorted_pairs = sorted(zip(n_sample, auc_val))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="val")

sorted_pairs = sorted(zip(n_sample, auc_test))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="test")

sorted_pairs = sorted(zip(n_sample, auc_train))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train")

sorted_pairs = sorted(zip(n_sample, auc_train_real))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label="train_real")


plt.xlabel("n_sample")
plt.ylabel("roc_auc")
plt.title("roc_auc")
plt.legend()
plt.xticks(range(min(x), max(x) + 1, 400))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()