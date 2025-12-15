import torch
import numpy as np
import pandas as pd
import os
import json
import time
from lib.make_dataset import make_dataset_for_uncondition, prepare_dataloader_withoutY, split_num_cat_target
from baselines.TabDDPM.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from baselines.TabDDPM.models.modules import MLPDiffusion


def get_model(model_name, model_params):
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def sample(
    raw_config,
    model_save_path,
    sample_save_path,
    real_data_path,
    batch_size=2000,
    num_samples=0,
    task_type='binclass',
    model_type='mlp',
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type='mse',
    scheduler='cosine',
    Transform=None,
    device=torch.device('cuda'),
    change_val=False,
    ddim=False,
    steps=1000,
):
    real_data_path = os.path.normpath(real_data_path)
    dataset = make_dataset_for_uncondition(real_data_path, raw_config)

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or Transform['cat_encode_policy'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(model_type, model_params)
   
    model_path = f'{model_save_path}/model.pt'
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()
    if not ddim:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False)
    else:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True, steps=steps)

    print('x_gen.Shape', x_gen.shape)

    syn_data = x_gen
    # num_inverse = dataset.num_transformer.inverse_transform
    # cat_inverse = dataset.cat_transformer.inverse_transform


    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, raw_config, dataset.num_transformer, dataset.cat_transformer, dataset.y_transformer)
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    x_df = pd.DataFrame(np.concatenate((syn_num, syn_cat), axis=1), columns=X_num_columns + X_cat_columns)
    y_df = pd.DataFrame(syn_target, columns=y_column)
    merged_df = pd.concat([x_df, y_df], axis=1)
    end_time = time.time()

    print('Sampling time:', end_time - start_time)

    # 保存生成的数据
    save_dir = sample_save_path
    if not os.path.exists(f"{save_dir}/synthesis_null"):
        os.makedirs(f"{save_dir}/synthesis_null")
    merged_df.to_csv(f"{save_dir}/synthesis_null/synthesis_null.csv", index=False)
    if raw_config['num_numerical_features'] != 0:
        np.save(f"{save_dir}/synthesis_null/X_num_synthesis", syn_num)
    if raw_config['num_categorical_features'] != 0:
        np.save(f"{save_dir}/synthesis_null/X_cat_synthesis", syn_cat)
    y_real = np.load(os.path.join(real_data_path, f'Y_train.npy'), allow_pickle=True)
    print("type(y_real) = ", y_real.dtype)
    if y_real.dtype == bool:
        np.save(f"{save_dir}/synthesis_null/Y_synthesis", syn_target)
    else:
        np.save(f"{save_dir}/synthesis_null/Y_synthesis", syn_target.astype(y_real.dtype))
    print(f'The generated data has been saved at {save_dir}/synthesis_null')
