"""
用 Tokenizer 提取语义信息
将语义信息作为分类模型的输入
"""
import torch
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.utils import resample
from collections import Counter
import warnings
import subprocess

import lib
from lib.make_dataset import read_pure_data, make_dataset, make_dataset_for_evaluation, Dataset
from lib.bert_util import get_bert_model, make_dataset_and_encode
from evaluate.mle_catboost import get_catboost_config
from TabClassifierfree.VAE import Tokenizer, Model_VAE

warnings.filterwarnings('ignore')


def build_CatBoostClassifier(seed=0, tokenizer_=False):
    catboost_config = get_catboost_config(raw_config['dataname'], is_cv=True)
    if tokenizer_:
        catboost_config["cat_features"] = []

    model = CatBoostClassifier(
        loss_function="MultiClass" if dataset.is_multiclass() else "Logloss",
        **catboost_config,
        eval_metric='TotalF1',
        random_seed=seed,
    )
    return model

def build_XGBClassifier(seed=0, scale_pos_weight=1.0):
    return XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        use_label_encoder=False,
        verbosity=0,
        # 可根据需要调整以下参数
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=200
    )

def build_MLPClassifier(seed):
    return MLPClassifier(
        hidden_layer_sizes=(64, 128, 64),
        max_iter=500,
        random_state=seed
    )

def evaluate(classifier, X, y, help:str):
    y_pred = classifier.predict(X)
    y_score = classifier.predict_proba(X)[:, 1]

    f1 = f1_score(y, y_pred, average='macro')
    auc = roc_auc_score(y, y_score)
    mcc = matthews_corrcoef(y, y_pred)
    report = classification_report(y, y_pred, digits=4)

    # 计算G-mean
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    G_mean = np.sqrt(sensitivity * specificity)

    print("")
    print('-' * 20, help, '-' * 20)
    print(f"F1 (macro): {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"mcc: {mcc:.4f}")
    print(f"G_mean: {G_mean:.4f}")
    print("Report:")
    print(report)

    return auc, f1

def oversampling(minority_class, sampling_method='CoTable', n_sample=1000):
    print('-' * 20, sampling_method, 'is sampling... ','-' * 20)
    pipeline = {
        'CoTable': 'main_sample.py',
        'TabDDPM': 'baselines/TabDDPM/main.py',
        'TabSyn': 'baselines/TabSyn/main.py',
        'CoDi': 'baselines/CoDi/main.py',
        'STaSy': 'baselines/STaSy/main.py',
        'GReaT': 'baselines/GReaT/main.py',
        'SMOTE': 'baselines/SMOTE/main.py',
        'TVAE': 'baselines/CTGAN_TVAE/main_tvae.py',
        'CTGAN': 'baselines/CTGAN_TVAE/main_ctgan.py',
    }
    parent_dir = raw_config['parent_dir']
    subprocess.run(['python', f'{pipeline[sampling_method]}', '--config', f'{parent_dir}/config.toml', '--n_sample', f'{n_sample}', '--sample'],  check=True)

    berttokenizer, bertmodel = get_bert_model(raw_config)

    synthetic_data_path = f'{parent_dir}/synthesis_null'
    X_num_synthesis, X_cat_synthesis, y_synthesis = read_pure_data(synthetic_data_path, raw_config, split='synthesis')

    # 对数值型数据进行标准化
    if dataset_semantic.num_transformer is not None and X_num_synthesis is not None:
        X_num_synthesis = dataset_semantic.num_transformer.transform(X_num_synthesis)

    # 对离散型数据进行编码
    if X_cat_synthesis is not None and dataset_semantic.cat_transformer is not None:
        X_cat_synthesis = dataset_semantic.cat_transformer.transform(X_cat_synthesis)

    # 对 Lable 进行encode
    if y_synthesis is not None and dataset_semantic.y_transformer is not None:
        y_synthesis = dataset_semantic.y_transformer.transform(y_synthesis)
        assert (y_synthesis == minority_class).all()

    X_num_synthesis = torch.from_numpy(X_num_synthesis).to(device)
    X_cat_synthesis = torch.from_numpy(X_cat_synthesis).to(device)
    y_synthesis = torch.from_numpy(y_synthesis).to(device)
    all_pooler_outputs_synthesis = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False, data_path=synthetic_data_path, split='synthesis')

    X_synthesis = tokenizer(X_num_synthesis, X_cat_synthesis, all_pooler_outputs_synthesis)
    X_synthesis = X_synthesis.reshape(X_synthesis.shape[0], -1).cpu().detach().numpy()
    X_synthesis = pd.DataFrame(X_synthesis)
    return X_synthesis


class IterativeFilteredOversampling:
    def __init__(self,
                 generator_func=oversampling,
                 generator_params=None,
                 max_iter=10,
                 patience=3,
                 n_synthetic_per_iter=200,
                 confidence_threshold=0.7,
                 scoring='f1',
                 random_state=42):
        self.generator_func = generator_func
        self.generator_params = generator_params or {'noise_scale': 0.1}
        self.max_iter = max_iter
        self.patience = patience
        self.n_synthetic_per_iter = n_synthetic_per_iter
        self.confidence_threshold = confidence_threshold
        self.scoring = scoring
        self.random_state = random_state
        np.random.seed(random_state)

        # 存储最终模型和历史
        self.best_auc = -np.inf
        self.best_auc_model = None
        self.best_f1 = -np.inf
        self.best_f1_model = None
        self.history = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # 若未提供验证集，则从训练集划分
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=self.random_state
            )

        X_current = X_train.copy()
        y_current = y_train.copy()

        minority_class = min(Counter(y_current).items(), key=lambda x: x[1])[0]
        X_minority = X_current[y_current == minority_class]
        X_majority = X_current[y_current != minority_class]
        print("minority_class: ", minority_class)
        print("len(X_minority): ", len(X_minority))
        print("len(X_majority): ", len(X_majority))

        assert len(X_minority) > 0

        no_improve_count = 0

        for iteration in range(self.max_iter):
            print("")
            print('=' * 40, f' Iter {iteration} ', '=' * 40)
            print("The number of training sets: ", len(y_current))
            print("The number of X_minority: ", len(X_current[y_current == minority_class]))
            print("The number of X_majority : ", len(X_current[y_current != minority_class]))

            tuples = []
            for seed in range(10):
                # 步骤1: 训练当前分类器
                # model = build_CatBoostClassifier(seed, tokenizer_=True)
                # model.fit(X_current, y_current, eval_set=(X_val, y_val), verbose=100)
                model = build_MLPClassifier(seed)
                model.fit(X_current, y_current)

                # 步骤2: 在验证集上评估
                score_val_auc, score_val_f1 = evaluate(model, X_val, y_val, help=f"iteration={iteration}, val, seed={seed}")
                score_train_auc, score_train_f1 = evaluate(model, X_train, y_train, help=f"iteration={iteration}, train, seed={seed}")

                tuples.append((score_val_auc, score_val_f1, model))

            (best_auc_local, f1_local, best_auc_model_local) = max(tuples, key=lambda x: x[0])
            (auc_local, best_f1_local, best_f1_model_local) = max(tuples, key=lambda x: x[1])

            if iteration == 0:
                (worst_auc_baseline, f1_baseline, worst_auc_model_baseline) = min(tuples, key=lambda x: x[0])
                (auc_baseline, worst_f1_baseline, worst_f1_model_baseline) = min(tuples, key=lambda x: x[1])
                self.worst_auc_model_baseline = worst_auc_model_baseline
                self.worst_f1_model_baseline = worst_f1_model_baseline

            # 检查是否提升
            if best_auc_local > self.best_auc or best_f1_local > self.best_f1:
                no_improve_count = 0

                if best_auc_local > self.best_auc:
                    print("!"*15, "AUC有提升", "!"*15)
                    self.best_auc = best_auc_local
                    self.best_auc_model = best_auc_model_local
                if best_f1_local > self.best_f1:
                    print("!" * 15, "F1有提升", "!" * 15)
                    self.best_f1 = best_f1_local
                    self.best_f1_model = best_f1_model_local
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                print("!!!Early stopping triggered!!!")
                break

            # 步骤3: 生成合成样本（仅针对少数类）
            synthetic_samples = self.generator_func(
                minority_class,
                sampling_method='CoTable',
                n_sample=len(X_minority) * 5
            )
            if len(X_current[y_current == minority_class]) > len(X_majority) + len(X_minority):
                print("!!!Already balanced!!!")
                break


            # 步骤4: 用当前模型对合成样本打分
            try:
                proba = self.best_f1_model.predict_proba(synthetic_samples)
                # max_proba = np.max(proba, axis=1)
                proba_minority_class = proba[:, minority_class]
                pred = self.best_f1_model.predict(synthetic_samples)
                # 保留高置信度且预测为少数类的样本
                mask = (proba_minority_class >= self.confidence_threshold) & (pred == minority_class)
                print("lalalalalalal")
            except:
                # 如果模型不支持 predict_proba，回退到硬预测
                pred = self.best_f1_model.predict(synthetic_samples)
                mask = (pred == minority_class)

            high_quality_synthetic = synthetic_samples[mask]

            if not isinstance(X_current, pd.DataFrame):
                X_current = pd.DataFrame(X_current)
            if not isinstance(high_quality_synthetic, pd.DataFrame):
                high_quality_synthetic = pd.DataFrame(high_quality_synthetic)

            y_synthetic = pd.Series([minority_class] * len(high_quality_synthetic))

            print(f"Generated {len(synthetic_samples)} → Kept {len(high_quality_synthetic)} high-quality samples")

            # 步骤5: 合并到当前训练集
            X_current = pd.concat([X_current, high_quality_synthetic], ignore_index=True)
            y_current = pd.concat([y_current, y_synthetic], ignore_index=True)
            # 随机打乱合并后的数据集
            shuffled_idx = np.random.permutation(len(X_current))
            X_current = X_current.iloc[shuffled_idx].reset_index(drop=True)
            y_current = y_current.iloc[shuffled_idx].reset_index(drop=True)

        # 最终用最佳模型作为结果
        # self.final_model = self.best_model
        return self



if __name__ == "__main__":
    """
    # 创建模拟不平衡数据集（含类重叠）
    X, y = make_classification(
        n_samples=3000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # 严重不平衡
        flip_y=0.1,  # 引入噪声/重叠
        random_state=0
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    """

    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-oversampling\exp\churn\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-oversampling\exp/adult\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-oversampling\exp/shopper\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-oversampling\exp/bean\CoTable\config.toml")


    """ Step1: 计算 Baseline（原始训练集的） 的效果"""
    T_dict = raw_config['eval']['Transform']
    T_dict['normalization'] = "None"
    dataset, X = make_dataset_for_evaluation(
        raw_config,
        synthetic_data_path=None,
        real_data_path=raw_config['all_data_path'],
        eval_type='real',
        T_dict=T_dict,
        change_val=False,
        sampling_method=None
    )

    X_train = X['train']
    y_train = pd.Series(dataset.y['train'].ravel())
    X_val = X['val']
    y_val = pd.Series(dataset.y['val'].ravel())

    # 基线：原始决策树
    # base_clf = build_CatBoostClassifier(seed=0)
    # base_clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    # base_clf = RandomForestClassifier(random_state=42)
    # base_clf = def build_XGBClassifier(seed=0, scale_pos_weight=1.0)
    base_clf = build_MLPClassifier(seed=0)
    base_clf.fit(X_train, y_train)

    # 评估
    evaluate(base_clf, X_val, y_val, 'Baseline（原始训练集）')


    """ Step2: 提取数据的语义信息，使用语言信息训练分类模型 """
    device = torch.device(raw_config['device'])
    # save_dir = raw_config['parent_dir']
    save_dir = "D:\Study\自学\表格数据生成/v11\exp\churn\CoTable"

    Pretrained_VAE = Model_VAE(
        num_layers=raw_config['VAE']['num_layers'],
        d_numerical=raw_config['num_numerical_features'],
        categories=raw_config['num_categorical'],
        d_token=raw_config['VAE']['d_token'],
        n_head=raw_config['VAE']['n_head'],
        factor=raw_config['VAE']['factor'],
        bias=True,
        bert_name=raw_config['model_params']['bert']
    ).to(device)

    tokenizer = Tokenizer(
        d_numerical=raw_config['num_numerical_features'],
        categories=raw_config['num_categorical'],
        d_token=raw_config['VAE']['d_token'],
        bias=True,
        bert_name=raw_config['model_params']['bert']
    ).to(device)

    Pretrained_VAE.load_state_dict(torch.load(os.path.join(save_dir, 'vae_model.pth'), map_location=device))
    tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())

    dataset_semantic = make_dataset(raw_config['all_data_path'], raw_config)
    all_pooler_outputs_train = torch.load(f"{save_dir}/all_pooler_outputs_train.pth", map_location=device)
    all_pooler_outputs_val = torch.load(f"{save_dir}/all_pooler_outputs_val.pth", map_location=device)

    def prepare_data_respectively(D: Dataset, all_pooler_outputs):
        # if split != 'all':
        X_num_train = torch.from_numpy(D.X_num['train']).to(device)
        X_num_val = torch.from_numpy(D.X_num['val']).to(device)
        if all_pooler_outputs is not None:
            X_num_train = X_num_train.type_as(all_pooler_outputs)
            X_num_val = X_num_val.type_as(all_pooler_outputs)
        X_cat_train = torch.from_numpy(D.X_cat['train']).to(device) if D.X_cat else None
        X_cat_val = torch.from_numpy(D.X_cat['val']).to(device) if D.X_cat else None
        # y_tain = torch.from_numpy(D.y['train'])
        # y_val = torch.from_numpy(D.y['val'])

        return (X_num_train, X_cat_train), (X_num_val, X_cat_val)

    (X_num_train, X_cat_train), (X_num_val, X_cat_val) = prepare_data_respectively(dataset_semantic, all_pooler_outputs_train)
    X_train = tokenizer(X_num_train, X_cat_train, all_pooler_outputs_train)
    X_val = tokenizer(X_num_val, X_cat_val, all_pooler_outputs_val)

    X_train = X_train.reshape(X_train.size(0), -1).cpu().detach().numpy()
    X_val = X_val.reshape(X_val.size(0), -1).cpu().detach().numpy()

    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)


    """ Step3: 使用提出的迭代过滤过采样方法 """
    model = IterativeFilteredOversampling(
        # base_classifier=RandomForestClassifier(random_state=42),
        # base_classifier=XGBClassifier(
        #     objective='binary:logistic',
        #     eval_metric='logloss',
        #     scale_pos_weight=1.0,
        #     random_state=0,
        #     use_label_encoder=False,
        #     verbosity=0,
        #     # 可根据需要调整以下参数
        #     max_depth=8,
        #     learning_rate=0.1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     n_estimators=200
        # ),
        n_synthetic_per_iter=150,
        confidence_threshold=0.8,
        max_iter=20,
        patience=5,
        scoring='auc'
    )
    model.fit(X_train, y_train, X_val, y_val)

    """ Step4: 打印最后的评估结果 """
    # 评估
    evaluate(model.best_f1_model, X_val, y_val, "final_best_f1_model")
    evaluate(model.best_auc_model, X_val, y_val, "final_best_auc_model")
    evaluate(model.worst_auc_model_baseline, X_val, y_val, "baseline_worst_auc_model")
    evaluate(model.worst_f1_model_baseline, X_val, y_val, "baseline_worst_f1_model")
