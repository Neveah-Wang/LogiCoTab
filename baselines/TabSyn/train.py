# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm
import lib
from lib.make_dataset import FastTensorDataLoader
from baselines.TabSyn.models.model import MLPDiffusion, Model
from baselines.TabSyn.models.latent_util import get_input_train

warnings.filterwarnings('ignore')


def main(raw_config):
    device = torch.device(raw_config['device'])
    save_dir = raw_config['parent_dir']
    batch_size = raw_config['model_params']['batch_size']
    num_epochs = raw_config['model_params']['num_epochs']

    train_z = get_input_train(raw_config)
    in_dim = train_z.shape[1]
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    train_data = train_z

    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4,)
    train_loader = FastTensorDataLoader(train_data, batch_size=batch_size, shuffle=False)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader), ncols=70)
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for (batch,) in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
            loss = loss.mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), f'{save_dir}/model.pth')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        # if epoch % 1000 == 0:
        #     torch.save(model.state_dict(), f'{save_dir}/model_{epoch}.pth')

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()

    raw_config = lib.util.load_config(args.config)
    main(raw_config)

"""
python baselines/TabSyn/train.py --config exp/adult/TabSyn/config.toml
python baselines/TabSyn/train.py --config exp/shopper/TabSyn/config.toml
python baselines/TabSyn/train.py --config exp/buddy/TabSyn/config.toml
"""