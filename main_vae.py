import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertModel, BertTokenizer
import warnings
import os
import sys
from tqdm import tqdm
import json
import time

import lib
from lib import util
from lib.make_dataset import make_dataset, prepare_dataloader_respectively
from lib.data_preprocess import inverse_transformer
from TabClassifierfree.VAE import Model_VAE, Encoder_model, Decoder_model
from lib.bert_util import make_dataset_and_encode

warnings.filterwarnings('ignore')

"""
def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0
    idx = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    ce_loss = torch.tensor(ce_loss/(idx + 1))
    acc = torch.tensor(acc/total_num if total_num > 0 else 0)
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

    return mse_loss, ce_loss, loss_kld, acc
"""
def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0
    if X_cat is not None:
        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim=-1)
            acc += (x_hat == X_cat[:, idx]).float().sum()
            total_num += x_hat.shape[0]

        ce_loss /= (idx + 1)
        acc /= total_num
        # loss = mse_loss + ce_loss
    else:
        ce_loss = torch.tensor(ce_loss)
        acc = torch.tensor(acc)

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

    return mse_loss, ce_loss, loss_kld, acc


def main(train_loader, all_pooler_outputs_val, raw_config, dataset):

    max_beta = 1e-2
    min_beta = 1e-5
    lambd = 0.7

    device = torch.device(raw_config['device'])
    save_dir = raw_config['parent_dir']

    # Model_VAE 的结构 = Encoder_model + Decoder_model
    model = Model_VAE(num_layers=raw_config['VAE']['num_layers'],
                      d_numerical=raw_config['num_numerical_features'],
                      categories=raw_config['num_categorical'],
                      d_token=raw_config['VAE']['d_token'],
                      n_head=raw_config['VAE']['n_head'],
                      factor=raw_config['VAE']['factor'],
                      bias=True,
                      bert_name=raw_config['model_params']['bert']).to(device)

    pre_encoder = Encoder_model(num_layers=raw_config['VAE']['num_layers'],
                                d_numerical=raw_config['num_numerical_features'],
                                categories=raw_config['num_categorical'],
                                d_token=raw_config['VAE']['d_token'],
                                n_head=raw_config['VAE']['n_head'],
                                factor=raw_config['VAE']['factor'],
                                bert_name=raw_config['model_params']['bert']).to(device)

    pre_decoder = Decoder_model(num_layers=raw_config['VAE']['num_layers'],
                                d_numerical=raw_config['num_numerical_features'],
                                categories=raw_config['num_categorical'],
                                d_token=raw_config['VAE']['d_token'],
                                n_head=raw_config['VAE']['n_head'],
                                factor=raw_config['VAE']['factor'],).to(device)

    # model.load_state_dict(torch.load(os.path.join(save_dir, 'vae_model.pth'), map_location=device))
    # pre_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'vae_encoder_model.pth'), map_location=device))
    # pre_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'vae_decoder_model.pth'), map_location=device))

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=raw_config['VAE']['lr'], weight_decay=raw_config['VAE']['wd'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=raw_config['VAE']['scheduler_fac'], patience=raw_config['VAE']['scheduler_patience'], verbose=False)

    num_epochs = raw_config['VAE']['epoch']
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta

    # 保存loss值
    df = pd.DataFrame(columns=['epoch', 'num_loss', 'cat_loss', 'kl_loss', 'val_num_loss', 'val_num_loss', 'train_acc', 'val_acc', 'lr'])
    df.to_csv(os.path.join(raw_config['parent_dir'], "vae_loss.csv"), index=False)

    print("The vae model is being trained")
    start_time = time.time()
    pbar = tqdm(range(num_epochs), file=sys.stdout)
    for ep in pbar:
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0
        curr_count = 0

        for step, [batch_num, batch_cat, batch_y, batch_cls_heads] in enumerate(train_loader):
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device) if batch_cat is not None else None

            model.train()
            optimizer.zero_grad()   # 梯度清零
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat, batch_cls_heads)
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新梯度

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count

        '''Evaluation'''
        model.eval()
        with torch.no_grad():
            X_val_num = torch.from_numpy(dataset.X_num['val']).to(device)
            X_val_cat = torch.from_numpy(dataset.X_cat['val']).to(device) if dataset.X_cat else None
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_val_num, X_val_cat, all_pooler_outputs_val)
            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_val_num, X_val_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() * 1 + val_ce_loss.item()

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                # print(f"Learning rate updated: {current_lr}")

            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), os.path.join(save_dir, 'vae_model.pth'))
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd

        pbar.set_description(f"beta = {beta:.6f}, "
                             f"Train MSE: {num_loss:.6f}, Train CE:{cat_loss:.6f}, Train KL:{kl_loss:.6f}, "
                             f"Val MSE:{val_mse_loss.item():.6f}, Val CE:{val_ce_loss.item():.6f}, "
                             f"Train ACC:{train_acc.item():6f}, Val ACC:{val_acc.item():6f}, "
                             f"lr: {optimizer.param_groups[0]['lr']:.8f}")

        data = pd.DataFrame([[ep, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item(), optimizer.param_groups[0]['lr']]])
        data.to_csv(os.path.join(raw_config['parent_dir'], "vae_loss.csv"), mode='a', header=False, index=False)

        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'
        #       .format(ep, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item()))


    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time) / 60))

    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), os.path.join(save_dir, 'vae_encoder_model.pth'))
        torch.save(pre_decoder.state_dict(), os.path.join(save_dir, 'vae_decoder_model.pth'))

        X_train_num = torch.from_numpy(dataset.X_num['train']).to(device)
        X_train_cat = torch.from_numpy(dataset.X_cat['train']).to(device) if dataset.X_cat else None

        print('Successfully load and save the model!')

        # 用训练好的VAE模型，将训练数据编码进隐空间。
        train_z = pre_encoder(X_train_num, X_train_cat, all_pooler_outputs_train).detach().cpu().numpy()
        # 将编码进隐空间的训练数据保存才来，用于Diffusion模型的训练
        if not os.path.exists(f"{save_dir}/latent_data"):
            os.makedirs(f"{save_dir}/latent_data")
        np.save(f'{save_dir}/latent_data/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', metavar='FILE')
    #
    # args = parser.parse_args()
    # raw_config = lib.util.load_config(args.config)

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/adult\CoTable\config.toml")
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/shopper\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/covertype\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/buddy\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/obesity\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/magic\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/churn\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/bean\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/page\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/abalone\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/bike\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/insurance\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/productivity\CoTable\config.toml")

    real_data_path = raw_config['real_data_path']
    device = torch.device(raw_config['device'])

    dataset = make_dataset(real_data_path, raw_config)

    """ 准备 sentences """
    all_pooler_outputs_train = None
    all_pooler_outputs_val = None
    # all_pooler_outputs_all = None

    if raw_config['num_categorical_features']:

        if raw_config['model_params']['bert'] == 'bert-base-uncased':
            berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 768
            bertmodel = BertModel.from_pretrained('bert-base-uncased').to(device)

        elif raw_config['model_params']['bert'] == 'huawei-noah/TinyBERT_General_4L_312D':
            berttokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')  # 312
            bertmodel = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D').to(device)

        elif raw_config['model_params']['bert'] == 'prajjwal1/bert-tiny':
            berttokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')  # 128
            bertmodel = BertModel.from_pretrained('prajjwal1/bert-tiny').to(device)

        else:
            raise ValueError("wrong bert name!")

        all_pooler_outputs_train = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False, split='train')
        all_pooler_outputs_val = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False, split='val')
        # all_pooler_outputs_all = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=False, split='all')
        torch.save(all_pooler_outputs_train, f"{raw_config['parent_dir']}/all_pooler_outputs_train.pth")
        torch.save(all_pooler_outputs_val, f"{raw_config['parent_dir']}/all_pooler_outputs_val.pth")
        # torch.save(all_pooler_outputs_all, f"{raw_config['parent_dir']}/all_pooler_outputs_all.pth")
        print(all_pooler_outputs_train.shape)
        print(all_pooler_outputs_val.shape)
        # print(all_pooler_outputs_all.shape)
        """
        all_pooler_outputs_train = torch.load(f"{raw_config['parent_dir']}/all_pooler_outputs_train.pth")
        all_pooler_outputs_val = torch.load(f"{raw_config['parent_dir']}/all_pooler_outputs_val.pth")
        # all_pooler_outputs_all = torch.load(f"{raw_config['parent_dir']}/all_pooler_outputs_all.pth")
        """
    train_loader = prepare_dataloader_respectively(dataset, all_pooler_outputs_train, split='train', batch_size=raw_config['VAE']['batch_size'])

    main(train_loader, all_pooler_outputs_val, raw_config, dataset)

"""
python main_vae.py --config exp/adult_ddpm_mlp/config.toml
"""