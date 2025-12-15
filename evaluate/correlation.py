import lib
import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, entropy
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# dataname = 'shopper'
# X_num_columns = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", ]
# X_cat_columns = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", ]
# y_columns = ["Revenue"]

# dataname = 'page'
# X_num_columns = ["height", "lenght", "area", "eccen", "p_black", "p_and", "mean_tr", "blackpix", "blackand", "wb_trans",]
# X_cat_columns = []
# y_columns = ["Class",]

# dataname = 'magic'
# X_num_columns = ["Length", "Width", "Size", "Conc", "Conc1", "Asym", "M3Long", "M3Trans", "Alpha", "Dist",]
# X_cat_columns = []
# y_columns = ["class",]

dataname = 'bean'
X_num_columns = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
                 "Extent", "Solidity", "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",]
X_cat_columns = []
y_columns = ["Class",]

# dataname = 'adult'
# X_num_columns = ["age", "fnlwgt", "education_num", "capital-gain", "capital-loss", "hours-per-week",]
# X_cat_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country",]
# y_columns = ["salary",]

# dataname = 'obesity'
# X_num_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE",]
# X_cat_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",]
# y_columns = ["NObeyesdad",]

# dataname = 'buddy'
# X_num_columns = ["length(m)", "height(cm)",]
# X_cat_columns = ["condition", "color_type", "X1", "X2", "breed_category",]
# y_columns = ["pet_category",]

# dataname = 'churn'
# X_num_columns = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",]
# X_cat_columns = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember",]
# y_columns = ["Exited",]


# dataname = 'bike'
# dataname = 'insurance'
# dataname = 'productivity'
# dataname = 'abalone'
raw_config = lib.util.load_config(f"D:\Study\自学\表格数据生成/v11\exp/{dataname}\CoTable\config.toml")

'''creat save folder'''
save_folder = f'D:\Study\自学\表格数据生成/v11\exp\{dataname}\correlation'
os.makedirs(save_folder, exist_ok=True)

'''Read original data'''
# X_cat_train = np.load(f"D:\Study\自学\表格数据生成/v11\Dataset/{dataname}\X_cat_train.npy", allow_pickle=True).astype(str)
X_num_train = np.load(f"D:\Study\自学\表格数据生成/v11\Dataset/{dataname}\X_num_train.npy", allow_pickle=True)
y_train = np.load(f"D:\Study\自学\表格数据生成/v11\Dataset/{dataname}\Y_train.npy", allow_pickle=True).astype(str)
# y_train = np.load(f"D:\Study\自学\表格数据生成/v11\Dataset/{dataname}/real\Y_train.npy", allow_pickle=True)
y_train = y_train.reshape(y_train.shape[0], 1)
print(y_train)

# cat_encoder = OrdinalEncoder(dtype='int64', handle_unknown="use_encoded_value", unknown_value=-1)
# cat_encoder.fit(X_cat_train)
# X_cat_train = cat_encoder.transform(X_cat_train)

y_encoder = OrdinalEncoder(dtype='int64')
y_encoder.fit(y_train)
y_train = y_encoder.transform(y_train)



# for method in ['CoTable', 'TabSyn','TabDDPM','STaSy', 'CTGAN', 'TVAE', 'CoDi', 'GReaT','SMOTE',]:
# for method in ['TabSyn','TabDDPM','STaSy', 'CTGAN', 'TVAE', 'CoDi', 'GReaT','SMOTE',]:
# for method in ['CoTable', 'TabSyn','TabDDPM','STaSy', 'CTGAN', 'CoDi', 'GReaT','SMOTE',]:
# for method in ['CoTable', 'TabSyn', 'TabDDPM', 'STaSy', 'CTGAN', 'TVAE', 'CoDi']:
# for method in ['CoTable', 'TabSyn','TabDDPM','STaSy', 'CoDi', 'GReaT',]:
# for method in ['GReaT', 'SMOTE']:
# for method in ['CTGAN', 'TVAE']:
# for method in ['SMOTE']:
# for method in ['GReaT']:
for method in ['CoTable']:
# for method in ['TabDDPM']:
# for method in ['TabSyn']:
    ''' Read generated data'''
    file = f'D:\Study\自学\表格数据生成/v11\exp/{dataname}/{method}\synthesis_null'
    # X_cat_synthesis = np.load(f"{file}\X_cat_synthesis.npy", allow_pickle=True).astype(str)
    # X_cat_synthesis[X_cat_synthesis == 'None'] = 'nan'
    X_num_synthesis = np.load(f"{file}\X_num_synthesis.npy", allow_pickle=True)
    y_synthesis = np.load(f"{file}\Y_synthesis.npy", allow_pickle=True).astype(str)
    # y_synthesis = np.load(f"{file}\Y_synthesis.npy", allow_pickle=True)
    y_synthesis = y_synthesis.reshape(y_synthesis.shape[0], 1)

    if method == 'CoTable' and raw_config.get('X_num_columns_real', False):
        # generated_data = pd.DataFrame(np.concatenate((X_num_synthesis, X_cat_synthesis, y_synthesis), axis=1), columns=raw_config['X_num_columns'] + raw_config['X_cat_columns'] + raw_config['y_column'])
        generated_data = pd.DataFrame(np.concatenate((X_num_synthesis, y_synthesis), axis=1), columns=raw_config['X_num_columns'] + raw_config['X_cat_columns'] + raw_config['y_column'])
        X_num_synthesis = generated_data[raw_config['X_num_columns_real']].values.astype(np.float32)
        X_cat_synthesis = generated_data[raw_config['X_cat_columns_real']].values
        y_synthesis = generated_data[raw_config['y_column_real']].values.astype(np.float32)

    # X_cat_synthesis = cat_encoder.transform(X_cat_synthesis)
    y_synthesis = y_encoder.transform(y_synthesis)

    # if raw_config.get('X_num_columns_real', False):
    #     original_data = pd.DataFrame(np.concatenate((X_num_train, X_cat_train, y_train), axis=1), columns=raw_config['X_num_columns_real'] + raw_config['X_cat_columns_real'] + raw_config['y_column_real'])
    #     generated_data = pd.DataFrame(np.concatenate((X_num_synthesis, X_cat_synthesis, y_synthesis), axis=1), columns=raw_config['X_num_columns_real'] + raw_config['X_cat_columns_real'] + raw_config['y_column_real'])
    # else:
    #     original_data = pd.DataFrame(np.concatenate((X_num_train, X_cat_train, y_train), axis=1),columns=raw_config['X_num_columns'] + raw_config['X_cat_columns'] + raw_config['y_column'])
    #     generated_data = pd.DataFrame(np.concatenate((X_num_synthesis, X_cat_synthesis, y_synthesis), axis=1),columns=raw_config['X_num_columns'] + raw_config['X_cat_columns'] + raw_config['y_column'])


    original_data = pd.DataFrame(np.concatenate((X_num_train, y_train), axis=1), columns=X_num_columns + X_cat_columns + y_columns)
    generated_data = pd.DataFrame(np.concatenate((X_num_synthesis, y_synthesis), axis=1), columns=X_num_columns + X_cat_columns + y_columns)

    # %%
    '''Calculate correlation matrices and their absolute differences'''
    real_corr = original_data.corr()
    generated_corr = generated_data.corr()
    diff_corr = np.abs(real_corr - generated_corr)

    '''Replace NaN values with 0'''
    diff_corr = diff_corr.fillna(0)

    '''Plot heatmap of the absolute difference'''
    plt.figure(figsize=(10, 8))
    # sns.heatmap(diff_corr, cmap='Reds', cbar=False, vmin=0, vmax=2, annot=True, fmt=".2f")
    sns.heatmap(diff_corr, cmap='Reds', cbar=False, vmin=0, vmax=0.8, annot=False, xticklabels=False, yticklabels=False)
    # plt.title(f'{mathod}', fontsize=20)
    # plt.title('Absolute difference between correlation matrices', fontsize=20)
    # plt.xlabel('Features', fontsize=16)
    plt.ylabel(f'{method}', fontsize=25)
    '''设置刻度值的大小'''
    plt.tick_params(axis='both', which='major', labelsize=16)

    '''Save plot'''
    heatmap_filename = os.path.join(save_folder, f'{method}_correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_filename)
    plt.close()

    '''Calculate and print the sum of all values in the difference correlation matrix'''
    total_sum = diff_corr.values.sum()
    # print(f'Sum of all values in the correlation heatmap: {total_sum}')
    # print(f'max value: {np.max(diff_corr)}')
    # print(f'min value: {np.min(diff_corr)}')

    '''Save the sum to a file'''
    sum_filename = os.path.join(save_folder, 'metrics.txt')
    with open(sum_filename, 'a') as f:
        f.write(f'{method}:\n')
        f.write(f'Sum of all values in the correlation heatmap: {total_sum}' + '\n')
        f.write(f'max value: {np.max(diff_corr)}' + '\n')
        f.write(f'min value: {np.min(diff_corr)}' + '\n')
