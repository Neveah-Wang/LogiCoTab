import numpy as np
from tqdm import tqdm
import sys
import torch
import os
import pandas as pd
import argparse
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import lib
from lib import util, latent_util
from lib.make_dataset import FastTensorDataLoader
from lib.bert_util import make_dataset_and_encode
from lib.data_preprocess import inverse_transformer
from TabClassifierfree.MLP_noise_prediction import MLPDiffusion
from TabClassifierfree.Transformer_noise_prediction import Transformer
from TabClassifierfree.DDPM import DDPM
from lib.util import draw_loss
# np.set_printoptions(threshold = np.inf)



def train(train_loader, ddpm, raw_config, drawloss=True):
    print("The diffusion model is being trained")

    # 保存loss值
    df = pd.DataFrame(columns=['epoch', 'Loss', 'lr'])
    df.to_csv(os.path.join(raw_config['parent_dir'], "ddpm_loss.csv"), index=False)

    ddpm.train()
    device = torch.device(raw_config['device'])
    save_dir = raw_config['parent_dir']
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=raw_config['ddpm_train']['lr'], weight_decay=raw_config['ddpm_train']['weight_decay'])
    # optimizer = torch.optim.AdamW(ddpm.parameters(), lr=5e-05, weight_decay=raw_config['ddpm_train']['weight_decay'])
    # 学习率调度器，可以动态调整学习率
    scheduler = StepLR(optimizer, step_size=raw_config['ddpm_train']['lr_scheduler']['step_size'], gamma=raw_config['ddpm_train']['lr_scheduler']['gamma'])
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.985, patience=10, verbose=False)   # 下次试试这个

    Loss_epoch = []
    epoch = raw_config['ddpm_train']['epoch']

    pbar = tqdm(range(epoch), file=sys.stdout, ncols=130)
    for ep in pbar:
        batch = 0
        Loss_batch = []
        for step, [x, y, cls_head] in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            batch += 1
            optimizer.zero_grad()      # 梯度清零
            loss = ddpm(x, y, cls_head)
            loss.backward()            # 计算梯度
            loss_ema = loss.item()
            Loss_batch.append(loss_ema)
            optimizer.step()           # 更新梯度


        # if ep % (epoch/50) == 0 or ep == epoch-1:
        Loss_epoch.append(sum(Loss_batch)/len(train_loader))

        if optimizer.param_groups[0]['lr'] >= raw_config['ddpm_train']['lr_min']:
            scheduler.step()
            # scheduler.step(Loss_epoch[ep])

        data = pd.DataFrame([[ep, Loss_epoch[ep], optimizer.param_groups[0]['lr']]])
        data.to_csv(os.path.join(raw_config['parent_dir'], "ddpm_loss.csv"), mode='a', header=False, index=False)


        if ep in [100000, 150000]:
            file = f"model_{ep}.pth" if raw_config['ddpm']['use_guide'] else f"model_{ep}_null.pth"
            torch.save(ddpm, os.path.join(save_dir, file))
            print('saved model at ' + save_dir + f"/model_{ep}.pth")

        pbar.set_description(f"batch {batch}/{len(train_loader)}, loss: {Loss_epoch[ep]:.7f}/{Loss_epoch[0]:.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}")
        # epoch 1999/2000, batch 491/491, loss: 0.0253/0.8326, lr: 0.0002: 100%|████████| 2000/2000 [1:24:50<00:00,


    # 保存训练好的模型
    file = f"model_{epoch}.pth" if raw_config['ddpm']['use_guide'] else f"model_{epoch}_null.pth"
    torch.save(ddpm, os.path.join(save_dir, file))
    print('saved model at ' + save_dir + file)

    # 绘制loss曲线
    if drawloss == True:
        draw_loss(Loss_epoch, epoch, raw_config)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', metavar='FILE')
    # parser.add_argument('--train', action='store_true', default=False)
    # parser.add_argument('--sample', action='store_true', default=False)
    # args = parser.parse_args()

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/adult\CoTable\config.toml")
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/shopper\CoTable\config.toml")
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
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/churn\CoTable\config.toml")
    device = torch.device(raw_config['device'])

    """准备 train_z 和 y"""
    train_z, y = latent_util.get_latent_train(raw_config)
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2

    """ 准备 sentences """

    if raw_config['model_params']['bert'] == 'bert-base-uncased':
        berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')   # 768
        bertmodel = BertModel.from_pretrained('bert-base-uncased').to(device)

    elif raw_config['model_params']['bert'] == 'huawei-noah/TinyBERT_General_4L_312D':
        berttokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')  # 312
        bertmodel = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D').to(device)

    elif raw_config['model_params']['bert'] == 'prajjwal1/bert-tiny':
        berttokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')  # 128
        bertmodel = BertModel.from_pretrained('prajjwal1/bert-tiny').to(device)

    else:
        raise ValueError("wrong bert name!")

    all_pooler_outputs = make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label=True, split='train')
    torch.save(all_pooler_outputs, f"{raw_config['parent_dir']}/all_pooler_outputs.pth")
    """
    all_pooler_outputs = torch.load(f"{raw_config['parent_dir']}/all_pooler_outputs.pth")
    """
    print(f"all_pooler_outputs.shape = {all_pooler_outputs.shape}")

    train_loader = FastTensorDataLoader(train_z, y, all_pooler_outputs, batch_size=raw_config['ddpm_train']['batch_size'], shuffle=False)

    """ 实例化去噪网络 """
    raw_config['model_params']['d_in'] = train_z.shape[1]
    if raw_config['model_type'] == 'mlp':
        noise_prediction_model = MLPDiffusion(raw_config).to(device)
    elif raw_config['model_type'] == 'transformer':
        noise_prediction_model = Transformer(raw_config, device).to(device)
    else:
        raise ValueError("wrong model_type!")


    ddpm = DDPM(noise_prediction_model, raw_config).to(device)
    # ddpm = torch.load(os.path.join(raw_config['parent_dir'], 'model_5000_null.pth'), map_location=device)

    train(train_loader, ddpm, raw_config)

    util.dump_config(raw_config, os.path.join(raw_config['parent_dir'], 'config.toml'))

    """
    python main_ddpm.py 
    """