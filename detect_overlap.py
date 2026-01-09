import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import lib
from lib.make_dataset import read_pure_data, make_dataset_for_evaluation
from evaluate.mle_catboost import get_catboost_config

def boundary_function_helper(X_train, y_train, X_test, y_test, test, THRESHOLD=0.4, classifier='CatBoostClassifier'):
    EPOCHS = 200

    if classifier == 'CatBoostClassifier':
        catboost_config = get_catboost_config(raw_config['dataname'], is_cv=True)
        model = CatBoostClassifier(
            loss_function="MultiClass" if dataset.is_multiclass() else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=0,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        proba_test = model.predict_proba(X_test)
        proba_train = model.predict_proba(X_train)

    else:
        #hyperparemeter tuning
        param_dist = {'n_estimators': [50]}
        rf = RandomForestClassifier()
        rand_search = RandomizedSearchCV(rf,
                                        param_distributions = param_dist,
                                        n_iter=5,
                                        cv=5)
        rand_search.fit(X_train, y_train)
        best_rf = rand_search.best_estimator_

        y_pred = best_rf.predict(X_test)
        y_pred_train = best_rf.predict(X_train)
        proba_test = best_rf.predict_proba(X_test)
        proba_train = best_rf.predict_proba(X_train)
    
    # 统计“高不确定性”样本数量
    uncertain_train = 0
    for prob in proba_train:
        if prob[0] >= THRESHOLD and prob[1] >= THRESHOLD:
            uncertain_train += 1 
    uncertain_test = 0
    for prob in proba_test:
        if prob[0] >= THRESHOLD and prob[1] >= THRESHOLD:
            uncertain_test += 1 
    print(f"{uncertain_train} out of {len(proba_train)} samples have higher uncertainity from the training set")
    print(f"{uncertain_test} out of {len(proba_test)} samples have higher uncertainity from the test set")

    # 识别“边界样本”
    boundary = []
    for i in range(len(y_test)):
        if (y_test[i] != y_pred[i]) or (proba_test[i][0] >= THRESHOLD and proba_test[i][1] >= THRESHOLD):
            boundary.append(i)
    print(f"There are {len(boundary)} points that have been wrongly predicted or are at the boundary")
        # Finding boundary samples from the test set by thresholding on the prediction probabilities or checking if a sample has been incorrectly predicted
    
    # creating a temporary dataframe that contains isBoundary=1 for the samples in the boundary list and 0 for others
    df1 = test
    df1['isBoundary'] = 0
    for wrong in boundary:
        df1.loc[wrong, "isBoundary"] = 1

    return df1

def find_boundary(df, TARGET,  RANDOM_STATE=42, threshold=0.4):
        
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    df_class1 = df[df[TARGET] == 1]
    df_class0 = df[df[TARGET] == 0]
    df = pd.concat([df_class0, df_class1], axis=0, ignore_index=True)

    k = 9
    step = len(df_class0)//k
    start = [step*i for i in range(k)]
    end = [step*(i+1) for i in range(k-1)]
    end.append(len(df_class0))
    bnd = []

    for i in range(k):
        print(f"Split {i+1}")
        train = df.drop([k for k in range(start[i], end[i])], axis=0)
        test = df.drop(train.index, axis=0)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        X_train = train.drop(TARGET, axis=1)
        y_train = train[TARGET]
        X_test = test.drop(TARGET, axis=1)
        y_test = test[TARGET]

        df_i = boundary_function_helper(X_train, y_train, X_test, y_test, test, threshold, classifier='CatBoostClassifier')
        bnd.append(df_i)
        """
        # get boundary dataframe
        if i == 0:
            df1 = boundary_function_helper(X_train, y_train, X_test, y_test, test, threshold, classifier='CatBoostClassifier')
        else:
            df2 = boundary_function_helper(X_train, y_train, X_test, y_test, test, threshold, classifier='CatBoostClassifier')
        """
    
    # bnd =  pd.concat([df1, df2], axis=0)
    bnd =  pd.concat(bnd, axis=0)
    condition = (bnd['isBoundary'] == 0) | (bnd[TARGET] == 1)
    cleaned = bnd[condition]
    cleaned = cleaned.drop(['isBoundary'], axis=1)
    cleaned = pd.concat([cleaned, df_class1], axis=0)
    """
    # print isBoundary=1 and target=0 count
    print("Majority Boundary = ", bnd[(bnd['isBoundary'] == 1) & (bnd[TARGET] == 0)].shape[0])
    print('-'*100)

    bnd['cond'] = 0
    bnd.loc[(bnd['isBoundary'] == 1) & (bnd[TARGET] == 0), 'cond'] = 1
    # remove isBoundary, target column
    bnd = bnd.drop(['isBoundary', TARGET], axis=1)

    # add class 1 as cond 2
    df_class1 = df_class1.copy()
    df_class1.loc[:, 'cond'] = 2
    df_class1 = df_class1.drop(TARGET, axis=1)

    bnd = pd.concat([bnd, df_class1], axis=0)
    """
    return bnd, cleaned

if __name__ == "__main__":

    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-oversampling\exp\churn\CoTable\config.toml")
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

    DATANAME = raw_config['dataname']
    TARGET = raw_config['y_column'][0]
    THRESHOLD = 0.5


    X_train = X['train']
    y_train = pd.DataFrame(dataset.y['train'], columns=raw_config['y_column'])
    df = pd.concat([X_train, y_train], axis=1)

    bnd, cleaned = find_boundary(df, TARGET, threshold=THRESHOLD)

    bnd.to_csv(f"{raw_config['parent_dir']}/boundary.csv", index=False)
    cleaned.to_csv(f"{raw_config['parent_dir']}/cleaned.csv", index=False)

    print("Saved Ternary Target successfully")


