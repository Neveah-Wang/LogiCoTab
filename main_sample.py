import numpy as np
from tqdm import tqdm
import sys
import torch
import os
import pandas as pd
import argparse

import lib
from lib import util, latent_util
from lib.make_dataset import make_dataset
from lib.data_preprocess import inverse_transformer_respectively
from TabClassifierfree.VAE import Decoder_model
from lib import save_synthesis_data

def sapmle(dataset, train_z, pre_decoder, raw_config):
    device = torch.device(raw_config['sample']['device'])
    save_dir = raw_config['parent_dir']
    epoch = raw_config['ddpm_train']['epoch']
    plus1 = raw_config['Transform']['y_plus1']
    file = f"model_{epoch}.pth" if raw_config['ddpm']['use_guide'] else f"model_{epoch}_null.pth"
    ddpm = torch.load(os.path.join(save_dir, file), map_location=device)
    ddpm.eval()
    pre_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'vae_decoder_model.pth'), map_location=device))
    pre_decoder.eval()

    w = raw_config['ddpm']['guide_w'] if raw_config['ddpm']['use_guide'] else 'null'
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    print("Data is sampling...")

    with torch.no_grad():
        if raw_config['ddpm']['use_guide']:
            x_synthesis, x_synthesis_store, y = ddpm.sample_balance_bert_classifierfree(raw_config, guide_w=w)
        else:
            # x_synthesis, x_synthesis_store, y = ddpm.sample_bert(dataset, raw_config)
            x_synthesis, x_synthesis_store, y = ddpm.sample_bert_batch(dataset, raw_config)

        x_synthesis = x_synthesis * 2 + train_z.mean(0).to(device)
        x_synthesis = x_synthesis.reshape(x_synthesis.shape[0], -1, raw_config['VAE']['d_token'])
        x_hat_num, x_hat_cat = pre_decoder(torch.tensor(x_synthesis))
        syn_cat = []
        for pred in x_hat_cat:
            syn_cat.append(pred.argmax(dim=-1))

        syn_num = x_hat_num
        syn_cat = torch.stack(syn_cat).t() if syn_cat != [] else None

        x_num_synthesis, x_cat_synthesis, y = inverse_transformer_respectively(syn_num, syn_cat, y[:, None], dataset, plus1)
        x_cat_synthesis = x_cat_synthesis if x_cat_synthesis is not None else np.empty((x_num_synthesis.shape[0], 0))
        x_df = pd.DataFrame(np.concatenate((x_num_synthesis, x_cat_synthesis), axis=1), columns=X_num_columns + X_cat_columns)
        y_df = pd.DataFrame(y, columns=y_column)
        merged_df = pd.concat([x_df, y_df], axis=1)

        # 保存生成的数据
        save_synthesis_data(raw_config, merged_df, x_num_synthesis, x_cat_synthesis, y, w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--sample', action='store_true', default=False)

    args = parser.parse_args()
    raw_config = lib.util.load_config(args.config)
    """
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/bean\CoTable\config.toml")
    """
    device = torch.device(raw_config['sample']['device'])
    real_data_path = raw_config['real_data_path']

    dataset = make_dataset(real_data_path, raw_config)
    train_z, y = latent_util.get_latent_train(raw_config)

    pre_decoder = Decoder_model(num_layers=raw_config['VAE']['num_layers'],
                                d_token=raw_config['VAE']['d_token'],
                                n_head=raw_config['VAE']['n_head'],
                                factor=raw_config['VAE']['factor'],
                                d_numerical=raw_config['num_numerical_features'],
                                categories=raw_config['num_categorical']).to(device)

    sapmle(dataset, train_z, pre_decoder, raw_config)

"""
python main_sample.py --config exp/adult/CoTable/config.toml
python main_sample.py --config exp/shopper/CoTable/config.toml
python main_sample.py --config exp/covertype/CoTable/config.toml
python main_sample.py --config exp/buddy/CoTable/config.toml
python main_sample.py --config exp/obesity/CoTable/config.toml
python main_sample.py --config exp/magic/CoTable/config.toml
python main_sample.py --config exp/churn/CoTable/config.toml
python main_sample.py --config exp/bean/CoTable/config.toml
python main_sample.py --config exp/page/CoTable/config.toml
python main_sample.py --config exp/abalone/CoTable/config.toml
python main_sample.py --config exp/bike/CoTable/config.toml
python main_sample.py --config exp/insurance/CoTable/config.toml
python main_sample.py --config exp/productivity/CoTable/config.toml
"""