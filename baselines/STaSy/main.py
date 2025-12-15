# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')


import argparse

import lib
from baselines.STaSy.train import train
from baselines.STaSy.sample import sample



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()

    raw_config = lib.util.load_config(args.config)

    if args.train:
        train(raw_config)
    if args.sample:
        sample(raw_config)

"""
python baselines/STaSy/main.py --config exp/adult/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/shopper/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/buddy/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/obesity/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/obesity/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/churn/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/magic/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/bean/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/page/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/abalone/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/bike/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/insurance/STaSy/config.toml --train --sample
python baselines/STaSy/main.py --config exp/productivity/STaSy/config.toml --train --sample
"""
