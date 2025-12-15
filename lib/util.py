import json
import tomli
import tomli_w
import enum
import os
import torch
import numpy as np
import pandas as pd
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from pathlib import Path
import matplotlib.pyplot as plt

RawConfig = Dict[str, Any]
_CONFIG_NONE = '__none__'
ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]
Normalization = Literal['standard', 'quantile', 'minmax']
CatEncoding = Literal['one-hot', 'Ordinal']
YEncoding = Literal['one-hot', 'Ordinal']
TaskType = Literal['binclass', 'multiclass', 'regression']


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')

def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


# load config
# 下面三个函数都是为了实现load config，我目前还没有看懂
def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    assert config == load_config(path)


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config

def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x
    return do(data)


def draw_loss(Loss_list, epoch, raw_config):
    plt.cla()
    x1 = range(1, epoch + 1)
    # print(x1)
    y1 = Loss_list
    # print(y1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1)
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    if raw_config['ddpm']['use_guide']:
        save_file = os.path.join(raw_config['parent_dir'], "Train_loss.png")
    else:
        save_file = os.path.join(raw_config['parent_dir'], "Train_loss_null.png")
    plt.savefig(save_file)
    print(f"loss pitcure has saved at {save_file}")
    plt.show()


def concat_to_pd(raw_config, X_num, X_cat, y):
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    X_cat = X_cat if X_cat is not None else np.empty((X_num.shape[0], 0))
    df = pd.DataFrame(np.concatenate((X_num, X_cat, y), axis=1), columns=X_num_columns + X_cat_columns + y_column)

    return df

def save_synthesis_data(raw_config, merged_df, x_num_synthesis, x_cat_synthesis, y, w):
    save_dir = raw_config['parent_dir']

    # if raw_config.get('X_num_columns_real', False):
    #     train = concat_to_pd(raw_config, x_num_synthesis, x_cat_synthesis, y)
    #     x_num_synthesis = train[raw_config['X_num_columns_real']].values.astype(np.float32)
    #     x_cat_synthesis = train[raw_config['X_cat_columns_real']].values
    #     y = train[raw_config['y_column_real']].values.astype(np.float32)

    if not os.path.exists(f"{save_dir}/synthesis_{w}"):
        os.makedirs(f"{save_dir}/synthesis_{w}")
    merged_df.to_csv(f"{save_dir}/synthesis_{w}/synthesis_{w}.csv", index=False)

    if raw_config['num_numerical_features'] != 0:
        np.save(f"{save_dir}/synthesis_{w}/X_num_synthesis", x_num_synthesis)
    if raw_config['num_categorical_features'] != 0:
        np.save(f"{save_dir}/synthesis_{w}/X_cat_synthesis", x_cat_synthesis)
    np.save(f"{save_dir}/synthesis_{w}/Y_synthesis", y)
    print(f'The generated data has been saved at {save_dir}/synthesis_{w}')