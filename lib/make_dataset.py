import numpy as np
import os
import time
import sys

import pandas as pd
import tomli
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List


from lib import util, data_preprocess
from lib.metrics import calculate_metrics as calculate_metrics_
from lib.util import ArrayDict, TensorDict, Normalization, CatEncoding, YEncoding, TaskType



class Dataset():
    def __init__(self, X_num: Optional[ArrayDict], X_cat: Optional[ArrayDict], y: ArrayDict,
                 info: Dict[str, Any], y_info: Dict[str, Any],
                 task_type: TaskType, n_classes: Optional[int]):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y
        self.info = info          # 数据集的 info.json 中的信息
        self.y_info = y_info      # 这里的 y_info 是用于存放 y 的 sta 和 mean，当处理回归任务时，tabddpm会把y进行标准化，就需要y的sta和mean。目前还没有测试过回归任务，回归对y的处理还没有看。
        self.task_type = task_type
        self.n_classes = n_classes
        self.num_transformer = None
        self.cat_transformer = None
        self.y_transformer = None

    def is_binclass(self) -> bool:
        return self.task_type == 'binclass'

    def is_multiclass(self) -> bool:
        return self.task_type == 'multiclass'

    def is_regression(self) -> bool:
        return self.task_type == 'regression'

    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    def n_features(self) -> int:
        return self.n_num_features() + self.n_cat_features()

    def get_category_sizes(self, part: str) -> List[int]:
        if self.X_cat is None:
            return []
        else:
            # X = self.cat_transformer.inverse_transform(self.X_cat[part]) if self.cat_transformer else self.X_cat[part]
            # XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
            XT = self.X_cat[part].T.cpu().tolist() if isinstance(self.X_cat[part], torch.Tensor) else self.X_cat[part].T.tolist()
            return [len(set(x)) for x in XT]

    def calculate_metrics(self, predictions: Dict[str, np.ndarray], prediction_type: Optional[str],) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(self.y[x], predictions[x], self.task_type, prediction_type, self.y_info)
            for x in predictions
        }
        print("prediction_type: ", prediction_type)
        print("task_type: ", self.task_type)
        if self.task_type == 'regression':
            score_key = 'rmse'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_sign = 1

        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]

        return metrics


def read_pure_data(path, raw_config, split='train'):
    Y = np.load(os.path.join(path, f'Y_{split}.npy'), allow_pickle=True)
    if (raw_config['task_type'] == "regression") and (not raw_config.get('X_num_columns_real', False)):
        Y = Y.astype(np.float32)
    else:
        Y = Y.astype(str)

    if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=1)

    X_num = None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True).astype(np.float32)

    X_cat = None
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True).astype(str)

    return X_num, X_cat, Y


def read_data(data_path: str, raw_config):
    X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
    X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
    y = {}
    splits = ['train', 'val', 'test', 'all'] if os.path.exists(os.path.join(data_path, 'Y_all.npy')) else ['train', 'val', 'test']
    for split in splits:
        X_num_t, X_cat_t, y_t = read_pure_data(data_path, raw_config, split)
        if X_num is not None:
            X_num[split] = X_num_t
        if X_cat is not None:
            X_cat[split] = X_cat_t
        y[split] = y_t

    return X_num, X_cat, y


def read_changed_val(path, val_size=0.2):
    path = Path(path)
    X_num_train, X_cat_train, y_train = read_pure_data(path, 'train')
    X_num_val, X_cat_val, y_val = read_pure_data(path, 'val')
    is_regression = util.load_json(path / 'info.json')['task_type'] == 'regression'

    y = np.concatenate([y_train, y_val], axis=0)

    ixs = np.arange(y.shape[0])
    if is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)
    y_train = y[train_ixs]
    y_val = y[val_ixs]

    if X_num_train is not None:
        X_num = np.concatenate([X_num_train, X_num_val], axis=0)
        X_num_train = X_num[train_ixs]
        X_num_val = X_num[val_ixs]

    if X_cat_train is not None:
        X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
        X_cat_train = X_cat[train_ixs]
        X_cat_val = X_cat[val_ixs]

    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val


def transform_dataset(
        raw_config,
        X_num: Optional[ArrayDict],
        X_cat: Optional[ArrayDict],
        y: Optional[ArrayDict],
        normalization: Normalization,
        cat_encode_policy: CatEncoding,
        y_policy: YEncoding,
        y_plus1: bool
):
    num_transformer = None
    cat_transformer = None
    y_transformer = None
    y_info = None

    # 对数值型数据进行标准化
    if X_num is not None and normalization != "None":
        X_num, num_transformer = data_preprocess.normalize(raw_config, X_num, normalization, return_normalizer=True)

    # 对离散型数据进行编码
    if X_cat is not None and cat_encode_policy != "None":
        X_cat, cat_transformer = data_preprocess.cat_encoder(raw_config, X_cat, cat_encode_policy, return_encoder=True)

    # 对 Lable 进行encode
    if y is not None:
        y, y_info, y_transformer = data_preprocess.lable_encoder(raw_config, y, y_policy, y_plus1, return_encoder=True)

    return X_num, X_cat, y, num_transformer, cat_transformer, y_transformer, y_info


def make_dataset(data_path, raw_config: dict) -> Dataset:
    """
    将数据集读入，并进行预处理
    """

    # 读入数据
    X_num, X_cat, y = read_data(data_path, raw_config)
    # print(X_num)
    # print(X_cat)
    # print(y)
    # 处理数据
    y_plus1 = raw_config['Transform'].get('y_plus1', False)
    X_num, X_cat, y, num_transformer, cat_transformer, y_transformer, y_info = transform_dataset(raw_config, X_num, X_cat, y,
                                                                                         raw_config['Transform']['normalization'],
                                                                                         raw_config['Transform']['cat_encode_policy'],
                                                                                         raw_config['Transform']['y_policy'],
                                                                                         y_plus1)
    # make dataset
    info = util.load_json(os.path.join(data_path, 'info.json'))
    dataset = Dataset(X_num, X_cat, y, info=info, y_info=y_info, task_type=info.get('task_type'), n_classes=info.get('n_classes'))
    dataset.num_transformer = num_transformer
    dataset.cat_transformer = cat_transformer
    dataset.y_transformer = y_transformer

    return dataset


def make_dataset_for_evaluation(raw_config, synthetic_data_path, real_data_path, eval_type, T_dict, change_val, sampling_method) -> (Dataset, dict):
    # 1. read data

    # T_dict["normalization"] = "minmax"
    # T_dict["cat_encoding"] = None

    if change_val:  # 把 train 和 val 合并打乱，再划分一遍
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2)

    if eval_type == 'synthetic':
        print(f'loading synthetic data: {synthetic_data_path}')
        X_num_train, X_cat_train, y_train = read_pure_data(synthetic_data_path, raw_config, split='synthesis')

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num_train, X_cat_train, y_train = read_pure_data(real_data_path, raw_config, split='train')

    elif eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(real_data_path, raw_config, split='train')
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, raw_config, split='synthesis')
        y = np.concatenate([y_real, y_fake], axis=0)

        X_num_train = None
        if X_num_real is not None:
            X_num_train = np.concatenate([X_num_real, X_num_fake], axis=0)
        X_cat_train = None
        if X_cat_real is not None:
            X_cat_train = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, raw_config, 'val')
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, raw_config, 'test')

    if raw_config.get('X_num_columns_real', False):
        train = concat_to_pd(raw_config, X_num_train, X_cat_train, y_train)
        X_num_train = train[raw_config['X_num_columns_real']].values.astype(np.float32)
        X_cat_train = train[raw_config['X_cat_columns_real']].values
        y_train = train[raw_config['y_column_real']].values.astype(np.float32)
        test = concat_to_pd(raw_config, X_num_test, X_cat_test, y_test)
        X_num_test = test[raw_config['X_num_columns_real']].values.astype(np.float32)
        X_cat_test = test[raw_config['X_cat_columns_real']].values
        y_test = test[raw_config['y_column_real']].values.astype(np.float32)
        val = concat_to_pd(raw_config, X_num_val, X_cat_val, y_val)
        X_num_val = val[raw_config['X_num_columns_real']].values.astype(np.float32)
        X_cat_val = val[raw_config['X_cat_columns_real']].values
        y_val = val[raw_config['y_column_real']].values.astype(np.float32)


    X_num = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test} if X_num_train is not None else None
    X_cat = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test} if X_cat_train is not None else None
    y = {'train': y_train, 'val': y_val, 'test': y_test}

    # 2.transform data
    y_plus1 = T_dict.get('y_plus1', False)
    X_num, X_cat, y, num_transformer, cat_transformer, y_transformer, y_info = transform_dataset(raw_config, X_num, X_cat, y,
                                                                                                 T_dict["normalization"],
                                                                                                 T_dict["cat_encode_policy"],
                                                                                                 T_dict["y_policy"],
                                                                                                 y_plus1)
    # 3.make dataset
    info = util.load_json(os.path.join(real_data_path, 'info.json'))
    dataset = Dataset(X_num, X_cat, y, info=info, y_info=y_info, task_type=info.get('task_type'), n_classes=info.get('n_classes'))
    dataset.num_transformer = num_transformer
    dataset.cat_transformer = cat_transformer
    dataset.y_transformer = y_transformer
    if raw_config.get('X_num_columns_real', False):
        X = concat_features(dataset, raw_config['X_num_columns_real'], raw_config['X_cat_columns_real'])
    else:
        X = concat_features(dataset, raw_config['X_num_columns'], raw_config['X_cat_columns'])

    print(f'Train size: {X["train"].shape}, Val size: {X["val"].shape}, Test size: {X["test"].shape}')
    # print(T_dict)

    return dataset, X


# def concat_y_to_X(X, y):
#     if X is None:
#         return y.reshape(-1, 1)
#     return np.concatenate([y.reshape(-1, 1), X], axis=1)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(len(y), -1)
    return np.concatenate([y.reshape(len(y), -1), X], axis=1)

def concat_to_pd(raw_config, X_num, X_cat, y):
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    X_cat = X_cat if X_cat is not None else np.empty((X_num.shape[0], 0))
    df = pd.DataFrame(np.concatenate((X_num, X_cat, y), axis=1), columns=X_num_columns + X_cat_columns + y_column)

    return df


def make_dataset_for_uncondition(data_path, raw_config: dict):
    # 1. read data
    X_num, X_cat, y = read_data(data_path, raw_config)

    if raw_config.get('X_num_columns_real', False):
        if X_cat is None:
            X_cat = {'train': None, 'val': None, 'test': None}
        train = concat_to_pd(raw_config, X_num['train'], X_cat['train'], y['train'])
        X_num['train'] = train[raw_config['X_num_columns_real']].values.astype(np.float32)
        X_cat['train'] = train[raw_config['X_cat_columns_real']].values
        y['train'] = train[raw_config['y_column_real']].values.astype(np.float32)
        test = concat_to_pd(raw_config, X_num['test'], X_cat['test'], y['test'])
        X_num['test'] = test[raw_config['X_num_columns_real']].values.astype(np.float32)
        X_cat['test'] = test[raw_config['X_cat_columns_real']].values
        y['test'] = test[raw_config['y_column_real']].values.astype(np.float32)
        val = concat_to_pd(raw_config, X_num['val'], X_cat['val'], y['val'])
        X_num['val'] = val[raw_config['X_num_columns_real']].values.astype(np.float32)
        X_cat['val'] = val[raw_config['X_cat_columns_real']].values
        y['val'] = val[raw_config['y_column_real']].values.astype(np.float32)

    # 2.transform data
    X_num, X_cat, y, num_transformer, cat_transformer, y_transformer, y_info = transform_dataset(raw_config, X_num, X_cat, y,
                                                                                                 raw_config['Transform']['normalization'],
                                                                                                 raw_config['Transform']['cat_encode_policy'],
                                                                                                 raw_config['Transform']['y_policy'],
                                                                                                 y_plus1=False)
    # classification
    if raw_config['task_type'] == 'binclass' or raw_config['task_type'] == 'multiclass':
        if X_cat:
            for split in ['train', 'val', 'test']:
                X_cat[split] = concat_y_to_X(X_cat[split], y[split])
        else:
            X_cat={}
            for split in ['train', 'val', 'test']:
                X_cat[split] = concat_y_to_X(None, y[split])

    # regression
    else:
        if X_num:
            for split in ['train', 'val', 'test']:
                X_num[split] = concat_y_to_X(X_num[split], y[split])
        else:
            X_num = {}
            for split in ['train', 'val', 'test']:
                X_num[split] = concat_y_to_X(None, y[split])

    # 3. make dataset
    info = util.load_json(os.path.join(data_path, 'info.json'))
    dataset = Dataset(X_num, X_cat, y, info=info, y_info=y_info, task_type=info.get('task_type'), n_classes=info.get('n_classes'))
    dataset.num_transformer = num_transformer
    dataset.cat_transformer = cat_transformer
    dataset.y_transformer = y_transformer

    return dataset

@torch.no_grad()
def split_num_cat_target(syn_data, raw_config, num_transformer, cat_transformer, y_transformer):
    task_type = raw_config['task_type']

    n_num_feat = len(raw_config['X_num_columns'])

    if raw_config['Transform']['cat_encode_policy'] == 'Ordinal':
        n_cat_feat = len(raw_config['X_cat_columns'])
    elif raw_config['Transform']['cat_encode_policy'] == 'one-hot':
        n_cat_feat = sum(raw_config['num_categorical'])
    else:
        raise ValueError("wrong cat encoder name!")

    if raw_config['Transform']['y_policy'] == 'one-hot':
        n_y_feat = raw_config['num_classes']
    else:
        n_y_feat = len(raw_config['y_column'])

    if task_type == 'regression':
        n_num_feat += n_y_feat
    else:
        n_cat_feat += n_y_feat

    syn_num = syn_data[:, :n_num_feat].cpu().numpy()
    syn_cat = syn_data[:, n_num_feat:].cpu().numpy()

    if task_type == 'regression':
        syn_target = syn_num[:, :n_y_feat]
        syn_num = syn_num[:, n_y_feat:]
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :n_y_feat]
        syn_cat = syn_cat[:, n_y_feat:]

    syn_num = num_transformer.inverse_transform(syn_num).astype(np.float32) if num_transformer else syn_num.astype(np.float32)
    syn_cat = cat_transformer.inverse_transform(syn_cat) if cat_transformer else syn_cat
    syn_target = y_transformer.inverse_transform(syn_target) if y_transformer else syn_target

    return syn_num, syn_cat, syn_target


def concat_features(D: Dataset, X_num_columns, X_cat_columns) -> dict[str, pd.DataFrame]:
    """
    将 cat 和 num 数据合并起来
    :param D:
    :return:
    """
    if D.X_num is None:
        assert D.X_cat is not None
        X = {k: pd.DataFrame(v, columns=X_cat_columns) for k, v in D.X_cat.items()}
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {k: pd.DataFrame(v, columns=X_num_columns) for k, v in D.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=X_num_columns),
                    pd.DataFrame(D.X_cat[part], columns=X_cat_columns),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t is None or t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate batches 计算一共有多少个batch
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        # batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        batch = tuple(t[self.i:self.i+self.batch_size] if t is not None else None for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def prepare_dataloader_res(D: Dataset, split: str, batch_size: int):
    X_num = torch.from_numpy(D.X_num[split])
    X_cat = torch.from_numpy(D.X_cat[split]) if D.X_cat is not None else None
    y = torch.from_numpy(D.y[split])

    dataloader = FastTensorDataLoader(X_num, X_cat, y, batch_size=batch_size, shuffle=False)

    return dataloader

def prepare_dataloader_respectively(D: Dataset, all_pooler_outputs, split: str, batch_size: int):
    # if split != 'all':
    X_num = torch.from_numpy(D.X_num[split])
    if all_pooler_outputs is not None:
        X_num = X_num.type_as(all_pooler_outputs)
    X_cat = torch.from_numpy(D.X_cat[split]) if D.X_cat else None
    y = torch.from_numpy(D.y[split])
    # else:
    #     X_num = []
    #     X_cat = []
    #     y = []
    #     for s in ['train', 'val', 'test']:
    #         X_num.append(torch.from_numpy(D.X_num[s]).type_as(all_pooler_outputs))
    #         X_cat.append(torch.from_numpy(D.X_cat[s]))
    #         y.append(torch.from_numpy(D.y[s]))
    #     X_num = torch.cat(X_num)
    #     X_cat = torch.cat(X_cat)
    #     y = torch.cat(y)

    print(X_num.shape)
    dataloader = FastTensorDataLoader(X_num, X_cat, y, all_pooler_outputs, batch_size=batch_size, shuffle=False)

    return dataloader


def prepare_fast_dataloader(D: Dataset, split: str, batch_size: int):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()

    y = torch.from_numpy(D.y[split])

    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=False)

    return dataloader


def prepare_dataloader_withoutY(D: Dataset, split: str, batch_size: int):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    dataloader = FastTensorDataLoader(X, batch_size=batch_size, shuffle=(split == 'train'))
    while True:
        yield from dataloader


if __name__ == "__main__":
    # 测试.josn文件
    '''
    info = util.load_json("E:\study\自学\表格数据生成\Tab-ClassifierFree\Dataset\CICIDS2017\info.json")
    print(info.get('task_type'))
    print(type(info.get('task_type')))
    print(info.get('n_classes'))
    print(type(info.get('n_classes')))
    '''

    # 测试.toml文件
    """
    with open("E:\study\自学\表格数据生成\Tab-ClassifierFree\exp\CICIDS2017\config.toml", 'rb') as f:
        raw_config = tomli.load(f)
    print(raw_config)
    print(type(raw_config))
    print(raw_config['Transform'])
    print(raw_config['Transform']['normalization'])
    print(type(raw_config['Transform']))
    print(type(raw_config['Transform']['normalization']))
    """

    # 测试dataset
    with open("E:\study\自学\表格数据生成\Tab-ClassifierFree\exp\CICIDS2017\config.toml", 'rb') as f:
        raw_config = tomli.load(f)
    dataset = make_dataset("E:\study\自学\表格数据生成\Tab-ClassifierFree\Dataset\CICIDS2017", raw_config)
    print(dataset.X_num)
    print(dataset.y)


    # 测试dataloader
    train_loader = prepare_fast_dataloader(dataset, split='train', batch_size=1024)
    pbar = tqdm(train_loader, file=sys.stdout)
    step = 0
    for x, y in pbar:
        pbar.set_description('epoch 1, batch ' + str(step+1) + '/' + str(len(train_loader)))
        if step%100 == 0:
            tqdm.write(str(x.shape))
            tqdm.write(str(x))
            tqdm.write(str(y.shape))
            tqdm.write(str(y))
        step += 1


    # import sys
    # dic = ['a', 'b', 'c', 'd', 'e']
    # pbar = tqdm(dic, file=sys.stdout)
    # for i in pbar:
    #     tqdm.write(i)
    #     pbar.set_description('Processing ' + i)
    #     time.sleep(0.2)
