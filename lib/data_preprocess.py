from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List

import numpy as np
import pandas as pd
import os
import torch
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OrdinalEncoder, OneHotEncoder

from lib import util
from lib.util import ArrayDict, Normalization, CatEncoding, YEncoding

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def normalize(raw_config, X: ArrayDict, normalization: Normalization, seed=0, return_normalizer: bool = False) -> ArrayDict:
    """
    对连续型的数据进行标准化
    :param X: Dict[str, np.ndarray], 待标准化的数据
    :param normalization: str, 标准化的方式
    :param seed: 随机种子
    :param return_normalizer:  是否返回 normalizer
    :return: Dict[str, np.ndarray], 标准化后的数据
    """
    # if raw_config['task_type'] != 'regression':
    #     real_data_path = raw_config['real_data_path']
    #     X_train = np.load(os.path.join(real_data_path, f'X_num_train.npy'), allow_pickle=True)
    # else:
    #     X_train = X['train']
    real_data_path = raw_config['real_data_path']
    X_train = np.load(os.path.join(real_data_path, f'X_num_train.npy'), allow_pickle=True)

    # if not raw_config['model_params']['is_y_cond'] and raw_config['task_type'] == 'regression':
    #     y_train = np.load(os.path.join(real_data_path, f'y_train.npy'), allow_pickle=True)
    #     X_train = concat_y_to_X(X_train, y_train)

    if normalization == 'standard':
        normalizer = StandardScaler()
    elif normalization == 'minmax':
        normalizer = MinMaxScaler(feature_range=(-1, 1))
    elif normalization == 'quantile':
        normalizer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        util.raise_unknown('normalization', normalization)

    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_encoder(raw_config, X: ArrayDict, cat_encode_policy: CatEncoding, return_encoder: bool = False) -> ArrayDict:
    """
    对离散型的数据进行编码
    :param X:
    :param cat_encode_policy:
    :param return_encoder:
    :return:
    """
    # if raw_config['task_type'] != 'regression':
    #     real_data_path = raw_config['real_data_path']
    #     X_train = np.load(os.path.join(real_data_path, f'X_cat_train.npy'), allow_pickle=True).astype(str)
    # else:
    #     X_train = X['train']
    real_data_path = raw_config['real_data_path']
    X_train = np.load(os.path.join(real_data_path, f'X_cat_train.npy'), allow_pickle=True).astype(str)

    # if not raw_config['model_params']['is_y_cond'] and raw_config['task_type'] != 'regression':
    #     y_train = np.load(os.path.join(real_data_path, f'y_train.npy'), allow_pickle=True)
    #     X_train = concat_y_to_X(X_train, y_train)

    if cat_encode_policy == 'one-hot':
        cat_encoder = OneHotEncoder(sparse_output=False)

    elif cat_encode_policy == 'Ordinal':
        cat_encoder = OrdinalEncoder(dtype='int64')

    else:
        util.raise_unknown('cat_encode_policy', cat_encode_policy)


    cat_encoder.fit(X_train)
    if return_encoder:
        return {k: cat_encoder.transform(v) for k, v in X.items()}, cat_encoder
    return {k: cat_encoder.transform(v) for k, v in X.items()}



def lable_encoder(raw_config, Y:ArrayDict, policy: YEncoding, plus1: bool = True, return_encoder: bool = False) -> ArrayDict:
    """
    将标签 Y 编码为 1,2,3,...,N
    """
    y_info = {}
    # if raw_config['task_type'] != 'regression':
    #     real_data_path = raw_config['real_data_path']
    #     Y_train = np.load(os.path.join(real_data_path, f'Y_train.npy'), allow_pickle=True).astype(str).reshape(-1,1)
    # else:
    #     Y_train = Y['train']
    real_data_path = raw_config['real_data_path']
    Y_train = np.load(os.path.join(real_data_path, f'Y_train.npy'), allow_pickle=True).astype(str).reshape(-1, 1)

    if policy == 'Ordinal':
        Encoder = OrdinalEncoder(dtype='int64')

    elif policy == 'one-hot':
        Encoder = OneHotEncoder(sparse_output=False)

    elif policy == 'None':
        if plus1:
            Y = {k: v + 1 for k, v in Y.items()}
        else:
            Y = {k: v for k, v in Y.items()}
        return Y, y_info, None

    elif policy == 'Standard':
        Encoder = StandardScaler()
        mean, std = float(Y['train'].mean()), float(Y['train'].std())
        y_info['police'] = policy
        y_info['mean'] = mean
        y_info['std'] = std

    elif policy == 'quantile':
        Encoder = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(Y['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=0,
        )

    elif policy == 'minmax':
        Encoder = MinMaxScaler(feature_range=(-1, 1))

    else:
        util.raise_unknown('y_policy', policy)

    Encoder.fit(Y_train)
    if plus1:
        Y = {k: Encoder.transform(v)+1 for k, v in Y.items()}
        """
        在这里 +1 是因为： 
        使用sklearn.preprocessing.OrdinalEncoder 可以将多个字符映分别射成 0~N-1 中的数字
        这里我希望将这些字符分别对应 1~N 中的数字，所以+1。
        数字 0 将在之后的操作中对应给空字符。
        注意：在使用 .inverse_transform() 方法时，要记得-1
        """
    else:
        Y = {k: Encoder.transform(v) for k, v in Y.items()}

    if return_encoder:
        return Y, y_info, Encoder
    return Y, y_info, None



def inverse_transformer(x: torch.Tensor, y: torch.Tensor, dataset, plus1: bool = True) -> (np.ndarray, np.ndarray):
    """
    将编码的数据进行解码
    """
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    x_num = x[:, :dataset.info['n_num_features']]
    x_cat = x[:, dataset.info['n_num_features']:]
    x_num_restored = dataset.num_transformer.inverse_transform(x_num) if dataset.num_transformer is not None else x_num
    x_cat_restored = dataset.cat_transformer.inverse_transform(x_cat) if dataset.cat_transformer is not None else x_cat
    if plus1:
        y_restored = dataset.y_transformer.inverse_transform(y-1) if dataset.y_transformer is not None else y-1
    else:
        y_restored = dataset.y_transformer.inverse_transform(y) if dataset.y_transformer is not None else y
    return x_num_restored, x_cat_restored, y_restored


def inverse_transformer_respectively(x_num: torch.Tensor, x_cat: torch.Tensor, y: torch.Tensor, dataset, plus1: bool) -> (np.ndarray, np.ndarray):
    """
    将编码的数据进行解码
    """
    x_num = x_num.cpu().detach().numpy()
    x_cat = x_cat.cpu().detach().numpy() if x_cat is not None else None
    y = y.cpu().detach().numpy()

    x_num_restored = dataset.num_transformer.inverse_transform(x_num) if dataset.num_transformer is not None else x_num
    x_cat_restored = dataset.cat_transformer.inverse_transform(x_cat) if dataset.cat_transformer is not None else x_cat
    if plus1:
        y_restored = dataset.y_transformer.inverse_transform(y-1) if dataset.y_transformer is not None else y-1
    else:
        y_restored = dataset.y_transformer.inverse_transform(y) if dataset.y_transformer is not None else y
    return x_num_restored, x_cat_restored, y_restored



if __name__ == '__main__':
    # 用一个主函数对上面的方法进行正确性测试
    from lib import read_pure_data

    X_cat = {}
    X_num = {}
    y = {}

    for split in ['train', 'val', 'test']:
        X_num_t, X_cat_t, y_t = read_pure_data('D:/Study/自学/表格数据生成/v11/Dataset/obesity/', split)
        if X_num is not None:
            X_num[split] = X_num_t
        if X_cat is not None:
            X_cat[split] = X_cat_t
        y[split] = y_t

    raw_config = {'real_data_path': 'D:/Study/自学/表格数据生成/v11/Dataset/obesity/'}

    # 标准化 X
    # print(pd.DataFrame(X_num['train'][0:, 0:]))
    #
    # X_num, num_transformer = normalize(raw_config, X_num, 'quantile', return_normalizer=True)
    # print(pd.DataFrame(X_num['train'][0:, 0:]))
    #
    # restored_data = num_transformer.inverse_transform(X_num['train'])
    # print(pd.DataFrame(restored_data[0:, 0:]))

    # encoder X_cat
    # print(pd.DataFrame(X_cat['train'][0:, 0:]))
    #
    # X_cat, cat_transformer = cat_encoder(raw_config, X_cat, 'one-hot', return_encoder=True)
    # print(pd.DataFrame(X_cat['train'][0:, 0:]))
    #
    # restored_data = cat_transformer.inverse_transform(X_cat['train'])
    # print(pd.DataFrame(restored_data[0:, 0:]))
    """
    from sklearn.pipeline import make_pipeline

    print(pd.DataFrame(X_cat['train'][0:, 0:]))

    unknown_value = np.iinfo('int64').max - 3
    oe = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X_cat['train'])

    encoder = make_pipeline(oe)
    encoder.fit(X_cat['train'])

    X = {k: encoder.transform(v) for k, v in X_cat.items()}

    max_values = X['train'].max(axis=0)
    for part in X.keys():
        if part == 'train': continue
        for column_idx in range(X[part].shape[1]):
            X[part][X[part][:, column_idx] == unknown_value, column_idx] = (max_values[column_idx] + 1)

    print(pd.DataFrame(X['train'][0:, 0:]))
    """

    # 编码 Y
    print("y['test']:")
    print(y['test'])

    y_encoded, y_info, Encoder = lable_encoder(raw_config, y, 'Ordinal', plus1=False, return_encoder=True)
    print("y_encoded['test']:")
    print(y_encoded['test'])

    y_restored = Encoder.inverse_transform(y_encoded['test'])
    print("y_restored['test']:")
    print(y_restored)

    categories = Encoder.categories_
    print(categories)
    print(type(categories))

    # np.save('E:\study\自学\表格数据生成\Tab-ClassifierFree\Dataset\CICIDS2017\DataIntermediate\y_test.npy', y_encoded['test']-1)
    # np.save('E:\study\自学\表格数据生成\Tab-ClassifierFree\Dataset\CICIDS2017\DataIntermediate\y_train.npy', y_encoded['train']-1)
    # np.save('E:\study\自学\表格数据生成\Tab-ClassifierFree\Dataset\CICIDS2017\DataIntermediate\y_val.npy', y_encoded['val']-1)

    # 测试 context_mask
    """
    context_mask = torch.bernoulli(torch.zeros_like(torch.tensor(y_encoded['test'])) + 0.5)
    context_mask = context_mask.type(torch.int)
    print("context_mask:")
    print(context_mask)
    y_mask = torch.tensor(y_encoded['test']) * context_mask
    print("y_mask:")
    print(y_mask)
    """

    # 测试 nn.embedding 嵌入空字符
    """
    import torch.nn as nn
    import torch
    embedding = nn.Embedding(16, 5)

    empty_char_index = 0  # 空字符的索引
    embedding.weight.data[empty_char_index] = torch.zeros(5)

    y_mask = y_mask.squeeze()
    print("y_mask_squeen")
    print(y_mask)
    embedded_sentence = embedding(y_mask)
    print(embedded_sentence)
    """