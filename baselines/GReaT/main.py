# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')

import argparse
import pandas as pd

import lib
from baselines.GReaT.models.great import GReaT
from baselines.GReaT.sample import sample



def train(raw_config):
    save_dir = raw_config['parent_dir']
    real_data_path = raw_config['real_data_path']
    batch_size = raw_config['model_params']['batch_size']
    dataset_path = f'{real_data_path}/train.csv'

    train_df = pd.read_csv(dataset_path)

    great = GReaT(
        "distilgpt2",
        epochs=100,
        save_steps=2000,
        logging_steps=50,
        experiment_dir=save_dir,
        batch_size=batch_size,
        # lr_scheduler_type="constant",        # Specify the learning rate scheduler
        # learning_rate=5e-5                   # Set the inital learning rate
    )
    
    trainer = great.fit(train_df)
    great.save(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()

    # class Args():
    #     def __init__(self):
    #         self.config = "D:\Study\自学\表格数据生成/v11\exp/shopper\GReaT\config.toml"
    #         self.train = False
    #         self.sample = True
    # args = Args()

    raw_config = lib.util.load_config(args.config)

    if args.train:
        train(raw_config)
    if args.sample:
        sample(raw_config)


"""
python baselines/GReaT/main.py --config exp/adult/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/shopper/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/covertype/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/buddy/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/obesity/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/magic/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/churn/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/bean/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/abalone/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/page/GReaT/config.toml --train --sample
python baselines/GReaT/main.py --config exp/bike/GReaT/config.toml --train --sample
"""
