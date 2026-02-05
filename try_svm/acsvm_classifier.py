"""
自适应代价敏感软边界SVM分类器 (AC-SVM)
Adaptive Cost-Sensitive Soft-Margin SVM Classifier

基于局部密度对比的样本难度评估
针对VAE隐空间优化的线性分类器
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix,
                             classification_report, recall_score)
import warnings
warnings.filterwarnings('ignore')


class LocalDensityAnalyzer:
    """
    局部密度分析器
    用于计算局部密度对比(LDC)和边界样本识别
    """
    
    def __init__(self, k_neighbors=30, algorithm='ball_tree'):
        """
        参数:
            k_neighbors: k近邻数量
            algorithm: 'ball_tree', 'kd_tree', 'brute'
        """
        self.k = k_neighbors
        self.algorithm = algorithm
        self.nbrs = None
    
    def fit(self, Z):
        """
        拟合k近邻模型
        """
        self.nbrs = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1因为要排除自身
            algorithm=self.algorithm,
            n_jobs=-1
        )
        self.nbrs.fit(Z)
        return self
    
    def compute_LDC(self, Z, y):
        """
        计算局部密度对比 (Local Density Contrast)
        
        LDC(z_i) = r_diff / (r_same + ε)
        
        其中:
            r_same: k近邻中同类样本的比例
            r_diff: k近邻中异类样本的比例
        
        返回:
            ldc: 局部密度对比数组, shape [N]
        """
        if self.nbrs is None:
            self.fit(Z)
        
        # 找k近邻
        distances, indices = self.nbrs.kneighbors(Z)
        
        N = len(Z)
        ldc = np.zeros(N)
        r_same = np.zeros(N)
        r_diff = np.zeros(N)
        
        for i in range(N):
            # 排除自身(第一个邻居)
            neighbor_indices = indices[i, 1:]
            neighbor_labels = y[neighbor_indices]
            
            # 计算同类和异类比例
            same_class = np.sum(neighbor_labels == y[i])
            diff_class = np.sum(neighbor_labels != y[i])
            
            r_same[i] = same_class / self.k
            r_diff[i] = diff_class / self.k
            
            # 计算LDC
            ldc[i] = r_diff[i] / (r_same[i] + 1e-6)
        
        return ldc, r_same, r_diff
    
    def identify_boundary_samples(self, Z, decision_values, threshold=0.5):
        """
        识别边界样本
        
        参数:
            Z: 特征矩阵
            decision_values: SVM决策函数值
            threshold: 边界阈值
        
        返回:
            boundary_mask: 布尔数组,True表示边界样本
        """
        # 归一化的决策值(近似到边界的距离)
        abs_decision = np.abs(decision_values)
        
        # 边界样本:决策值接近0
        boundary_mask = abs_decision < threshold
        
        return boundary_mask
    
    def get_neighborhood_purity(self, Z, y):
        """
        计算邻域纯度
        
        返回:
            purity: 邻域纯度 (r_same), shape [N]
        """
        _, r_same, _ = self.compute_LDC(Z, y)
        return r_same


class AdaptiveCostComputer:
    """
    自适应代价计算器
    """
    
    def __init__(self, alpha=0.7, beta=2.0, gamma=2.0):
        """
        参数:
            alpha: 不平衡惩罚指数
            beta: 局部权重最大放大倍数
            gamma: tanh增长速率
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def compute_base_cost(self, y, imbalance_ratio):
        """
        计算基础代价
        
        C_base(y) = ρ^α for minority class, 1 for majority class
        
        返回:
            base_costs: shape [N]
        """
        N = len(y)
        base_costs = np.ones(N)
        
        # 识别少数类
        class_counts = np.bincount(y)
        minority_class = np.argmin(class_counts)
        
        # 设置少数类的基础代价
        minority_mask = (y == minority_class)
        base_costs[minority_mask] = imbalance_ratio ** self.alpha
        
        return base_costs
    
    def compute_local_weight(self, ldc):
        """
        计算局部难度权重
        
        ω_local(z) = 1 + β * tanh(γ * LDC(z))
        
        参数:
            ldc: 局部密度对比
        
        返回:
            weights: 局部权重
        """
        weights = 1.0 + self.beta * np.tanh(self.gamma * ldc)
        return weights
    
    def compute_adaptive_costs(self, y, ldc, imbalance_ratio, stage='stage1'):
        """
        计算自适应代价
        
        参数:
            y: 标签
            ldc: 局部密度对比
            imbalance_ratio: 不平衡比
            stage: 训练阶段
        
        返回:
            costs: 自适应代价, shape [N]
        """
        # 基础代价
        base_costs = self.compute_base_cost(y, imbalance_ratio)
        
        if stage == 'stage1':
            # 阶段1: 仅使用基础代价
            return base_costs
        
        elif stage == 'stage2' or stage == 'stage3':
            # 阶段2/3: 添加局部难度权重
            local_weights = self.compute_local_weight(ldc)
            costs = base_costs * local_weights
            return costs
        
        else:
            raise ValueError(f"Unknown stage: {stage}")


class ACSVMClassifier:
    """
    自适应代价敏感软边界SVM分类器
    Adaptive Cost-Sensitive Soft-Margin SVM Classifier
    """
    
    def __init__(self, k_neighbors=None, alpha=0.7, beta=2.0, gamma=2.0,
                 C_global_stage1=10.0, C_global_stage2=5.0, C_global_stage3=3.0,
                 kernel='linear', verbose=True):
        """
        参数:
            k_neighbors: k近邻数量 (None表示自动选择)
            alpha: 不平衡惩罚指数
            beta: 局部权重最大放大倍数
            gamma: tanh增长速率
            C_global_stage1/2/3: 各阶段的正则化参数
            kernel: SVM核函数 ('linear' 推荐)
            verbose: 是否打印信息
        """
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.C_global = {
            'stage1': C_global_stage1,
            'stage2': C_global_stage2,
            'stage3': C_global_stage3
        }
        self.kernel = kernel
        self.verbose = verbose
        
        # 组件
        self.density_analyzer = None
        self.cost_computer = AdaptiveCostComputer(alpha, beta, gamma)
        
        # 模型
        self.svm_stage1 = None
        self.svm_stage2 = None
        self.svm_stage3 = None
        self.final_svm = None
        
        # 数据统计
        self.imbalance_ratio = 2.0
        self.n_samples = 0
        self.n_features = 0
    
    def _auto_select_k(self, N):
        """
        自动选择k近邻数量
        k = min(30, sqrt(N))
        """
        k = min(30, int(np.sqrt(N)))
        return max(10, k)  # 至少10
    
    def _compute_imbalance_ratio(self, y):
        """
        计算不平衡比
        """
        class_counts = np.bincount(y)
        return max(class_counts) / min(class_counts)
    
    def fit_stage(self, Z, y, ldc, stage='stage1', warm_start_svm=None):
        """
        单阶段训练
        
        参数:
            Z: 特征矩阵
            y: 标签
            ldc: 局部密度对比
            stage: 训练阶段
            warm_start_svm: 用于初始化的SVM
        
        返回:
            svm: 训练好的SVM模型
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"训练阶段: {stage}")
            print(f"{'='*60}")
        
        # 计算自适应代价
        sample_weights = self.cost_computer.compute_adaptive_costs(
            y, ldc, self.imbalance_ratio, stage=stage
        )
        
        if self.verbose:
            print(f"样本权重统计:")
            print(f"  最小值: {sample_weights.min():.4f}")
            print(f"  最大值: {sample_weights.max():.4f}")
            print(f"  均值: {sample_weights.mean():.4f}")
            print(f"  中位数: {np.median(sample_weights):.4f}")
        
        # 创建SVM
        svm = SVC(
            kernel=self.kernel,
            C=self.C_global[stage],
            class_weight=None,  # 使用sample_weight代替
            probability=True,   # 启用概率预测
            random_state=42
        )
        
        # 训练
        if self.verbose:
            print(f"训练SVM (C={self.C_global[stage]})...")
        
        svm.fit(Z, y, sample_weight=sample_weights)
        
        # 评估
        if self.verbose:
            y_pred = svm.predict(Z)
            train_acc = np.mean(y_pred == y)
            
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
            g_mean = np.sqrt(recall_0 * recall_1)
            
            print(f"训练集性能:")
            print(f"  准确率: {train_acc:.4f}")
            print(f"  G-Mean: {g_mean:.4f}")
            print(f"  支持向量数: {len(svm.support_)}")
            print(f"  支持向量比例: {len(svm.support_)/len(Z):.2%}")
        
        return svm
    
    def fit(self, Z_train, y_train):
        """
        三阶段渐进式训练
        
        参数:
            Z_train: 训练集特征
            y_train: 训练集标签
        """
        self.n_samples, self.n_features = Z_train.shape
        self.imbalance_ratio = self._compute_imbalance_ratio(y_train)
        
        if self.verbose:
            print("="*80)
            print("AC-SVM 三阶段渐进式训练")
            print("="*80)
            print(f"\n数据集统计:")
            print(f"  样本数: {self.n_samples}")
            print(f"  特征维度: {self.n_features}")
            print(f"  类别分布: {np.bincount(y_train)}")
            print(f"  不平衡比: {self.imbalance_ratio:.2f}")
        
        # 自动选择k
        if self.k_neighbors is None:
            self.k_neighbors = self._auto_select_k(self.n_samples)
        
        if self.verbose:
            print(f"  k近邻数: {self.k_neighbors}")
        
        # 初始化局部密度分析器
        if self.verbose:
            print(f"\n计算局部密度对比 (LDC)...")
        
        self.density_analyzer = LocalDensityAnalyzer(
            k_neighbors=self.k_neighbors,
            algorithm='ball_tree'
        )
        self.density_analyzer.fit(Z_train)
        ldc, r_same, r_diff = self.density_analyzer.compute_LDC(Z_train, y_train)
        
        if self.verbose:
            print(f"LDC统计:")
            print(f"  最小值: {ldc.min():.4f}")
            print(f"  最大值: {ldc.max():.4f}")
            print(f"  均值: {ldc.mean():.4f}")
            print(f"  中位数: {np.median(ldc):.4f}")
            print(f"  LDC>1的样本比例: {np.mean(ldc > 1.0):.2%}")
        
        # ==================== 阶段1: 全局粗分 ====================
        self.svm_stage1 = self.fit_stage(Z_train, y_train, ldc, stage='stage1')
        
        # ==================== 阶段2: 局部精细化 ====================
        self.svm_stage2 = self.fit_stage(
            Z_train, y_train, ldc, stage='stage2',
            warm_start_svm=self.svm_stage1
        )
        
        # ==================== 阶段3: 边界打磨 ====================
        # 识别边界样本
        decision_values = self.svm_stage2.decision_function(Z_train)
        boundary_mask = self.density_analyzer.identify_boundary_samples(
            Z_train, decision_values, threshold=0.5
        )
        
        if self.verbose:
            print(f"\n边界样本识别:")
            print(f"  边界样本数: {np.sum(boundary_mask)}")
            print(f"  边界样本比例: {np.mean(boundary_mask):.2%}")
        
        # 对边界样本使用更高代价
        ldc_stage3 = ldc.copy()
        ldc_stage3[boundary_mask] *= 1.5  # 边界样本的LDC放大1.5倍
        
        self.svm_stage3 = self.fit_stage(
            Z_train, y_train, ldc_stage3, stage='stage3',
            warm_start_svm=self.svm_stage2
        )
        
        # 最终模型
        self.final_svm = self.svm_stage3
        
        if self.verbose:
            print("\n" + "="*80)
            print("✅ 三阶段训练完成!")
            print("="*80)
        
        return self
    
    def predict(self, Z):
        """预测类别"""
        if self.final_svm is None:
            raise ValueError("模型未训练,请先调用fit()")
        return self.final_svm.predict(Z)
    
    def predict_proba(self, Z):
        """预测概率"""
        if self.final_svm is None:
            raise ValueError("模型未训练,请先调用fit()")
        return self.final_svm.predict_proba(Z)
    
    def decision_function(self, Z):
        """决策函数值"""
        if self.final_svm is None:
            raise ValueError("模型未训练,请先调用fit()")
        return self.final_svm.decision_function(Z)
    
    def evaluate(self, Z, y_true):
        """
        评估性能
        
        返回:
            metrics: 性能指标字典
        """
        y_pred = self.predict(Z)
        y_proba = self.predict_proba(Z)[:, 1]
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算指标
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        g_mean = np.sqrt(recall_0 * recall_1)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.5
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'G-Mean': g_mean,
            'F1-Macro': f1_macro,
            'F1-Weighted': f1_weighted,
            'AUC': auc,
            'Accuracy': accuracy,
            'Recall-0': recall_0,
            'Recall-1': recall_1,
            'Precision-0': precision_0,
            'Precision-1': precision_1,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Confusion Matrix': cm,
            'N_Support_Vectors': len(self.final_svm.support_)
        }
        
        return metrics
    
    def print_metrics(self, metrics, dataset_name=''):
        """打印评估指标"""
        print(f"\n{'='*60}")
        print(f"评估结果 {dataset_name}")
        print(f"{'='*60}")
        print(f"分类性能:")
        print(f"  G-Mean:      {metrics['G-Mean']:.4f}")
        print(f"  F1-Macro:    {metrics['F1-Macro']:.4f}")
        print(f"  F1-Weighted: {metrics['F1-Weighted']:.4f}")
        print(f"  AUC:         {metrics['AUC']:.4f}")
        print(f"  Accuracy:    {metrics['Accuracy']:.4f}")
        print(f"\n各类性能:")
        print(f"  Recall-0:    {metrics['Recall-0']:.4f}")
        print(f"  Recall-1:    {metrics['Recall-1']:.4f}")
        print(f"  Precision-0: {metrics['Precision-0']:.4f}")
        print(f"  Precision-1: {metrics['Precision-1']:.4f}")
        print(f"\n模型信息:")
        print(f"  支持向量数: {metrics['N_Support_Vectors']}")
        print(f"\n混淆矩阵:")
        print(metrics['Confusion Matrix'])
        print(f"{'='*60}\n")


def gmean_score(y_true, y_pred):
    """计算G-Mean分数"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    return np.sqrt(recall_0 * recall_1)
