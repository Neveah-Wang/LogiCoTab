import sys
sys.path.append(r'D:\Study\自学\表格数据生成\v11')

import tomli
import shutil
import os
import argparse
from baselines.TabDDPM.train import train
from baselines.TabDDPM.sample import sample
import zero
import lib
import torch
import numpy as np

# torch.set_printoptions(threshold=np.inf)

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)
    args = parser.parse_args()
    """
    class Args():
        def __init__(self):
            self.config = "D:\Study\自学\表格数据生成/v11\exp/covertype\TabDDPM\config.toml"
            self.train = True
            self.sample = True
            self.change_val = False
    args = Args()
    """

    raw_config = lib.util.load_config(args.config)
    if torch.cuda.is_available():
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # timer = zero.Timer()
    # timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            raw_config=raw_config,
            model_save_path=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val
        )
    if args.sample:
        sample(
            raw_config,
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            **raw_config['diffusion_params'],
            model_save_path=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            sample_save_path=raw_config['parent_dir'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            Transform=raw_config['Transform'],
            device=device,
            change_val=args.change_val
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))

    # print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()

"""
python baselines/TabDDPM/main.py --config exp/adult/TabDDPM/config.toml --train
python baselines/TabDDPM/main.py --config exp/adult/TabDDPM/config.toml --sample
python baselines/TabDDPM/main.py --config exp/shopper/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/covertype/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/buddy/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/obesity/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/magic/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/churn/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/bean/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/page/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/abalone/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/bike/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/insurance/TabDDPM/config.toml --train --sample
python baselines/TabDDPM/main.py --config exp/productivity/TabDDPM/config.toml --train --sample
"""