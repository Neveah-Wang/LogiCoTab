"""
本文件用于：

将隐空间的数据
test_z、val_z
latent_z_after_reparameterize_test
latent_z_after_reparameterize_val
存储下来
"""


import os
import torch
import argparse
import numpy as np
import lib
from lib.make_dataset import make_dataset
from TabClassifierfree.VAE import Model_VAE, Encoder_model
from lib.bert_util import make_dataset_and_encode, get_bert_model


def main(raw_config):
    # ----------- Step1: 准备数据------------
    real_data_path = raw_config['real_data_path']
    parent_dir = raw_config['parent_dir']
    save_dir = raw_config['parent_dir']
    device = torch.device(raw_config['device'])

    dataset = make_dataset(real_data_path, raw_config)

    all_pooler_outputs_val = None
    all_pooler_outputs_test = None
    if raw_config['num_categorical_features']:
        berttokenizer, bertmodel = get_bert_model(raw_config)
        # all_pooler_outputs_val = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False,
        #                                                  data_path=real_data_path, split='val')
        all_pooler_outputs_test = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False,
                                                         data_path=real_data_path, split='test')

    model = Model_VAE(
        num_layers=raw_config['VAE']['num_layers'],
        d_numerical=raw_config['num_numerical_features'],
        categories=raw_config['num_categorical'],
        d_token=raw_config['VAE']['d_token'],
        n_head=raw_config['VAE']['n_head'],
        factor=raw_config['VAE']['factor'],
        bias=True,
        bert_name=raw_config['model_params']['bert']
    ).to(device)

    pre_encoder = Encoder_model(
        num_layers=raw_config['VAE']['num_layers'],
        d_numerical=raw_config['num_numerical_features'],
        categories=raw_config['num_categorical'],
        d_token=raw_config['VAE']['d_token'],
        n_head=raw_config['VAE']['n_head'],
        factor=raw_config['VAE']['factor'],
        bert_name=raw_config['model_params']['bert']
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(parent_dir, 'vae_model.pth'), map_location=device))
    pre_encoder.load_state_dict(torch.load(os.path.join(parent_dir, 'vae_encoder_model.pth'), map_location=device))
    """
    X_val_num = torch.from_numpy(dataset.X_num['val']).to(device)
    X_val_cat = torch.from_numpy(dataset.X_cat['val']).to(device) if dataset.X_cat else None

    _, _, _, latent_data = model.VAE(X_val_num, X_val_cat, all_pooler_outputs_val)
    latent_data = latent_data.detach().cpu().numpy()
    latent_dir = os.path.join(save_dir, 'latent_data')
    np.save(os.path.join(latent_dir, 'latent_z_after_reparameterize_val.npy'), latent_data)
    
    val_z = pre_encoder(X_val_num, X_val_cat, all_pooler_outputs_val).detach().cpu().numpy()
    np.save(os.path.join(latent_dir, 'val_z.npy'), val_z)
    """

    X_test_num = torch.from_numpy(dataset.X_num['test']).to(device)
    X_test_cat = torch.from_numpy(dataset.X_cat['test']).to(device) if dataset.X_cat else None

    _, _, _, latent_data = model.VAE(X_test_num, X_test_cat, all_pooler_outputs_test)
    latent_data = latent_data.detach().cpu().numpy()
    latent_dir = os.path.join(save_dir, 'latent_data')
    np.save(os.path.join(latent_dir, 'latent_z_after_reparameterize_test.npy'), latent_data)

    test_z = pre_encoder(X_test_num, X_test_cat, all_pooler_outputs_test).detach().cpu().numpy()
    np.save(os.path.join(latent_dir, 'test_z.npy'), test_z)


if __name__ == '__main__':
    raw_config_list = []
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/buddy\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\shopper\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\churn\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/adult\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/magic\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/bean\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/yeast_me2\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/winequality\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/pageblocks\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/mammography\CoTable\config.toml"))

    for raw_config in raw_config_list:
        print(raw_config['dataname'])
        main(raw_config)