import argparse

from lib import util
from evaluate.mle_simple import eval_seeds_simple
from evaluate.mle_catboost import eval_seeds_catboost



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--sampling_method', type=str, default="CoTable", choices=["CoTable", "TabDDPM", "TabSyn", "CoDi", "STaSy", "SMOTE", "ctabgan", "ctabgan-plus", "tvae"], help='The model used to synthesize the data, such as ddpm|smote|ctabgan|ctabgan-plus|tvae')
    parser.add_argument('--eval_type', type=str, default='synthetic', choices=["synthetic", "real", "merged"], help="The type of data to be evaluated, such as synthetic|real")
    parser.add_argument('--model_type', type=str, default='catboost', help='The model used to evaluate the synthetic data, such as catboost')
    parser.add_argument('--n_datasets', type=int, default=1)
    parser.add_argument('--n_sample', type=int, default=50)
    parser.add_argument('--no_dump', action='store_false', default=True)

    args = parser.parse_args()
    """
    class Args():
        def __init__(self):
            self.config = "exp/adult\CoTable\config.toml"
            self.n_seeds = 2
            self.sampling_method = "CoTable"
            self.eval_type = "merged"
            self.model_type = "catboost"
            self.n_datasets = 2
            self.n_sample = 50 * 8
            self.no_dump = True
    args = Args()
    """
    raw_config = util.load_config(args.config)
    use_guide = raw_config['ddpm'].get('use_guide', False) if args.sampling_method == 'CoTable' else False
    w = raw_config['ddpm']['guide_w'] if use_guide else 'null'

    if args.model_type == 'simple':
        eval_seeds_simple(
            raw_config,
            n_seeds=args.n_seeds,
            sampling_method=args.sampling_method,
            eval_type=args.eval_type,
            guide_w=w,
            model_type=args.model_type,
            n_datasets=args.n_datasets,
            dump=args.no_dump
        )

    elif args.model_type == 'catboost':
        eval_seeds_catboost(
            raw_config,
            n_seeds=args.n_seeds,
            sampling_method=args.sampling_method,
            eval_type=args.eval_type,
            guide_w=w,
            model_type=args.model_type,
            n_datasets=args.n_datasets,
            n_sample=args.n_sample,   # 只有指定少数类生成数量是，该参数才有效
            dump=args.no_dump
        )

'''
simple:
python main_eval.py --config exp/adult_ddpm_mlp/config.toml --n_seeds 8 --sampling_method ddpm --eval_type synthetic --model_type simple --n_datasets 5
python main_eval.py --config exp/adult_ddpm_transformer/config.toml --n_seeds 8 --sampling_method ddpm --eval_type synthetic --model_type simple --n_datasets 5
python main_eval.py --config exp/adult_ddpm_mlp/config.toml --n_seeds 3 --sampling_method ddpm --eval_type real --model_type simple --n_datasets 1
python main_eval.py --config exp/CICIDS2017/config.toml --n_seeds 3 --sampling_method ddpm --eval_type synthetic --model_type simple --n_datasets 3
python main_eval.py --config exp/default/config.toml --n_seeds 3 --sampling_method ddpm --eval_type synthetic --model_type simple --n_datasets 3
python main_eval.py --config exp/default/config.toml --n_seeds 3 --sampling_method ddpm --eval_type real --model_type simple --n_datasets 1

catboost
adult:
python main_eval.py --config exp/adult/TabDDPM/config.toml --n_seeds 5 --sampling_method TabDDPM --eval_type synthetic --model_type catboost --n_datasets 3
python main_eval.py --config exp/adult/TabSyn/config.toml --n_seeds 5 --sampling_method TabSyn --eval_type synthetic --model_type catboost --n_datasets 3
python main_eval.py --config exp/adult/CoDi/config.toml --n_seeds 5 --sampling_method CoDi --eval_type synthetic --model_type catboost --n_datasets 3
python main_eval.py --config exp/adult/CoTable/config.toml --n_seeds 5 --sampling_method CoTable --eval_type synthetic --model_type catboost --n_datasets 3
python main_eval.py --config exp/adult/CoTable/config.toml --n_seeds 3 --sampling_method CoTable --eval_type merged --model_type catboost --n_datasets 5 --n_sample 

shopper
python main_eval.py --config exp/shopper/CoTable/config.toml --n_seeds 5 --sampling_method CoTable --eval_type synthetic --model_type catboost --n_datasets 3
python main_eval.py --config exp/shopper/TabSyn/config.toml --n_seeds 5 --sampling_method TabSyn --eval_type synthetic --model_type catboost --n_datasets 3
python main_eval.py --config exp/shopper/TabDDPM/config.toml --n_seeds 5 --sampling_method TabDDPM --eval_type synthetic --model_type catboost --n_datasets 3
python main_eval.py --config exp/shopper/CoDi/config.toml --n_seeds 5 --sampling_method CoDi --eval_type synthetic --model_type catboost --n_datasets 3


churn
subprocess.run(['python', 'main_eval.py', '--config', 'exp/churn/CoTable/config.toml', '--n_seeds', '3', '--sampling_method', 'CoTable', '--eval_type', 'merged', '--model_type', 'catboost', '--n_datasets', '5', '--n_sample', f'{50*i}'])
'''
