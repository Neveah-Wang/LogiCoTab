# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import os
import sys
from tqdm import tqdm
import json
import time

import lib
from lib.make_dataset import make_dataset_for_uncondition, prepare_dataloader_res
from baselines.TabSyn.models.vae import Model_VAE, Encoder_model, Decoder_model

warnings.filterwarnings('ignore')


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    # ce_loss = torch.tensor(ce_loss/(idx + 1))
    # acc = torch.tensor(acc/total_num if total_num > 0 else 0)
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

    return mse_loss, ce_loss, loss_kld, acc


def main(raw_config):

    max_beta = 1e-2
    min_beta = 1e-5
    lambd = 0.7

    save_dir = raw_config['parent_dir']
    real_data_path = raw_config['real_data_path']
    device = torch.device(raw_config['device'])
    dataset = make_dataset_for_uncondition(real_data_path, raw_config)
    train_loader = prepare_dataloader_res(dataset, split='train', batch_size=raw_config['VAE']['batch_size'])

    # Model_VAE 的结构 = Encoder_model + Decoder_model
    model = Model_VAE(num_layers=raw_config['VAE']['num_layers'],
                      d_numerical=dataset.n_num_features(),
                      categories=dataset.get_category_sizes('train'),
                      d_token=raw_config['VAE']['d_token'],
                      n_head=raw_config['VAE']['n_head'],
                      factor=raw_config['VAE']['factor'],
                      bias=True).to(device)

    pre_encoder = Encoder_model(num_layers=raw_config['VAE']['num_layers'],
                                d_numerical=dataset.n_num_features(),
                                categories=dataset.get_category_sizes('train'),
                                d_token=raw_config['VAE']['d_token'],
                                n_head=raw_config['VAE']['n_head'],
                                factor=raw_config['VAE']['factor'],).to(device)

    pre_decoder = Decoder_model(num_layers=raw_config['VAE']['num_layers'],
                                d_numerical=dataset.n_num_features(),
                                categories=dataset.get_category_sizes('train'),
                                d_token=raw_config['VAE']['d_token'],
                                n_head=raw_config['VAE']['n_head'],
                                factor=raw_config['VAE']['factor'],).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=raw_config['VAE']['lr'], weight_decay=raw_config['VAE']['wd'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    num_epochs = 4000
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader), file=sys.stdout, ncols=100)
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for batch_num, batch_cat, batch_y in pbar:
            model.train()
            optimizer.zero_grad()   # 梯度清零

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device) if batch_cat is not None else None

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
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

        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            X_test_num = torch.from_numpy(dataset.X_num['test']).to(device)
            X_test_cat = torch.from_numpy(dataset.X_cat['test']).to(device) if dataset.X_cat is not None else None
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)
            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

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

        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'
              .format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(),val_acc.item()))


    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time) / 60))

    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), os.path.join(save_dir, 'vae_encoder_model.pth'))
        torch.save(pre_decoder.state_dict(), os.path.join(save_dir, 'vae_decoder_model.pth'))

        X_train_num = torch.from_numpy(dataset.X_num['train']).to(device)
        X_train_cat = torch.from_numpy(dataset.X_cat['train']).to(device) if dataset.X_cat is not None else None

        print('Successfully load and save the model!')

        # 用训练好的VAE模型，将训练数据编码进隐空间。
        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
        # 将编码进隐空间的训练数据保存才来，用于Diffusion模型的训练
        if not os.path.exists(f"{save_dir}/latent_data"):
            os.makedirs(f"{save_dir}/latent_data")
        np.save(f'{save_dir}/latent_data/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')

    args = parser.parse_args()
    raw_config = lib.util.load_config(args.config)

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/magic\TabSyn\config.toml")


    main(raw_config)

"""
python baselines/TabSyn/train_vae.py --config exp/adult/TabSyn/config.toml
python baselines/TabSyn/train_vae.py --config exp/shopper/TabSyn/config.toml
python baselines/TabSyn/train_vae.py --config exp/buddy/TabSyn/config.toml
"""