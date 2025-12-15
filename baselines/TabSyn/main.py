# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')


import argparse
import lib
from baselines.TabSyn.train_vae import main as train_vae
from baselines.TabSyn.train import main as train_dm
from baselines.TabSyn.sample import main as sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()
    '''
    class Args():
        def __init__(self):
            self.config = 'D:\Study\自学\表格数据生成/v11\exp/magic\TabSyn\config.toml'
            self.train = True
            self.sample = True
    args = Args()
    '''
    raw_config = lib.util.load_config(args.config)

    if args.train:
        train_vae(raw_config)
        train_dm(raw_config)
    if args.sample:
        sample(raw_config)

"""
python baselines/TabSyn/main.py --config exp/adult/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/shopper/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/covertype/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/buddy/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/obesity/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/magic/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/churn/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/bean/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/page/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/abalone/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/bike/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/insurance/TabSyn/config.toml --train --sample
python baselines/TabSyn/main.py --config exp/productivity/TabSyn/config.toml --train --sample
"""
