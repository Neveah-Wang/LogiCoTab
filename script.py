import subprocess


# for data in ['bean', 'buddy', 'magic', 'churn']:
#     print(data)
#     subprocess.run(['python', 'baselines/CTGAN_TVAE/main_tvae.py', '--config', f'exp/{data}/TVAE/config.toml', '--train', '--sample'], check=True)


# for data in ['buddy']:
#     print(data)
    # subprocess.run(['python', 'baselines/CTGAN_TVAE/main_ctgan.py', '--config', f'exp/{data}/CTGAN/config.toml', '--sample'], check=True)
    # subprocess.run(['python', 'baselines/CTGAN_TVAE/main_tvae.py', '--config', f'exp/{data}/TVAE/config.toml', '--sample'], check=True)

# for method in ['TabDDPM', 'TabSyn', 'STaSy', 'CoDi']:
#     print(method)
#     subprocess.run(['python', f'baselines/{method}/main.py', '--config', f'exp/insurance/{method}/config.toml', '--train', '--sample'], check=True)
#     print("")

"""
python baselines/STaSy/main.py --config exp/insurance/STaSy/config.toml --train --sample
python baselines/CoDi/main.py --config exp/insurance/CoDi/config.toml --train --sample
"""

# subprocess.run(['python', f'baselines/GReaT/main.py', '--config', f'exp/insurance/GReaT/config.toml', '--train', '--sample'], check=True)

# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_ctgan.py', '--config', f'exp/bean/CTGAN/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/SMOTE/main.py', '--config', f'exp/bean/SMOTE/config.toml', '--sample'], check=True)
#
# subprocess.run(['python', f'baselines/SMOTE/main.py', '--config', f'exp/magic/SMOTE/config.toml', '--sample'], check=True)
#
# subprocess.run(['python', f'baselines/SMOTE/main.py', '--config', f'exp/churn/SMOTE/config.toml',  '--sample'], check=True)
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_ctgan.py', '--config', f'exp/churn/CTGAN/config.toml', '--train', '--sample'], check=True)
#
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_ctgan.py', '--config', f'exp/abalone/CTGAN/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_tvae.py', '--config', f'exp/abalone/TVAE/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/SMOTE/main.py', '--config', f'exp/abalone/SMOTE/config.toml',  '--sample'], check=True)


# subprocess.run(['python', f'baselines/SMOTE/main.py', '--config', f'exp/bike/SMOTE/config.toml',  '--sample'], check=True)
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_tvae.py', '--config', f'exp/bike/TVAE/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_ctgan.py', '--config', f'exp/bike/CTGAN/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/CoDi/main.py', '--config', f'exp/bike/CoDi/config.toml', '--train', '--sample'], check=True)
#
# subprocess.run(['python', f'baselines/CoDi/main.py', '--config', f'exp/abalone/CoDi/config.toml', '--train', '--sample'], check=True)
#
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_tvae.py', '--config', f'exp/insurance/TVAE/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_ctgan.py', '--config', f'exp/insurance/CTGAN/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/SMOTE/main.py', '--config', f'exp/insurance/SMOTE/config.toml',  '--sample'], check=True)

# subprocess.run(['python', 'baselines/SMOTE/main.py', '--config', 'exp/shopper/SMOTE/config.toml',  '--sample'], check=True)
# subprocess.run(['python', 'baselines/TabSyn/main.py', '--config',  'exp/shopper/TabSyn/config.toml',  '--train', '--sample'], check=True)
# subprocess.run(['python', 'baselines/TabDDPM/main.py', '--config', 'exp/shopper/TabDDPM/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', 'baselines/STaSy/main.py', '--config', 'exp/shopper/STaSy/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', 'baselines/CoDi/main.py', '--config', 'exp/shopper/CoDi/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_ctgan.py', '--config', f'exp/shopper/CTGAN/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/CTGAN_TVAE/main_tvae.py', '--config', f'exp/shopper/TVAE/config.toml', '--train', '--sample'], check=True)
# subprocess.run(['python', f'baselines/GReaT/main.py', '--config', f'exp/shopper/GReaT/config.toml', '--train', '--sample'], check=True)

i = 2
while 100 * i <= 5000:
    n_sample = 100 * i
    subprocess.run(['python', 'main_eval.py', '--config', 'exp/churn/CoTable/config.toml', '--n_seeds', '3', '--sampling_method', 'CoTable', '--eval_type', 'merged', '--model_type', 'catboost', '--n_datasets', '5', '--n_sample', f'{n_sample}'])
    i += 1


i = 22
n_sample = 200 * i
while n_sample <= 10000:
    subprocess.run(['python', 'main_eval.py', '--config', 'exp/adult/CoTable/config.toml', '--n_seeds', '3', '--sampling_method', 'CoTable', '--eval_type', 'merged', '--model_type', 'catboost', '--n_datasets', '5', '--n_sample', f'{n_sample}'])
    i += 1