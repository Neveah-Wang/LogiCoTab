# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')


import argparse

import lib
from baselines.CoDi.train import main as train
from baselines.CoDi.sample import main as sample



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()
    """
    class Args():
        def __init__(self):
            self.config = "D:\Study\自学\表格数据生成/v11\exp/churn\CoDi\config.toml"
            self.train = True
            self.sample = True
    args = Args()
    """
    raw_config = lib.util.load_config(args.config)

    if args.train:
        train(raw_config)
    if args.sample:
        sample(raw_config)

"""
python baselines/CoDi/main.py --config exp/adult/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/shopper/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/covertype/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/buddy/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/obesity/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/magic/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/churn/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/page/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/bean/CoDi/config.toml --train --sample
python baselines/CoDi/main.py --config exp/insurance/CoDi/config.toml --train --sample
"""
