import torch
import pandas as pd

import os

import argparse
import json
from lib import save_synthesis_data
from baselines.GReaT.models.great import GReaT
from baselines.GReaT.models.great_utils import _array_to_dataframe


def sample(raw_config):
    real_data_path = raw_config['real_data_path']
    dataname = raw_config['dataname']
    save_dir = raw_config['parent_dir']
    # device = torch.device(raw_config['device'])
 
    dataset_path = f'{real_data_path}/train.csv'
    info_path = f'{real_data_path}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    train_df = pd.read_csv(dataset_path)

    great = GReaT("distilgpt2",
                  epochs=200,
                  save_steps=2000,
                  logging_steps=50,
                  experiment_dir=save_dir,
                  batch_size=24,
                  # lr_scheduler_type="constant",        # Specify the learning rate scheduler
                  # learning_rate=5e-5                   # Set the inital learning rate
                  )
    
    model_save_path = f'{save_dir}/model.pt'
    great.model.load_state_dict(torch.load(model_save_path))

    great.load_finetuned_model(f"{save_dir}/model.pt")

    df = _array_to_dataframe(train_df, columns=None)
    great._update_column_information(df)
    great._update_conditional_information(df, conditional_col=None)

    
    n_samples = info['train_size']

    samples = great.sample(n_samples, k=100, device=raw_config['device'])


    # if not os.path.exists(f"{save_dir}/synthesis_null"):
    #     os.makedirs(f"{save_dir}/synthesis_null")
    # samples.to_csv(f"{save_dir}/synthesis_null/synthesis_null.csv", index=False)

    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']

    for c in X_cat_columns+y_column:
        if train_df.dtypes[c] != bool:
            samples[c] = samples[c].astype(train_df.dtypes[c])
    print(samples.head())


    X_num_synthesis = samples[X_num_columns].values
    X_cat_synthesis = samples[X_cat_columns].values
    y_synthesis = samples[y_column].values

    save_synthesis_data(raw_config, samples, X_num_synthesis, X_cat_synthesis, y_synthesis, w='null')



    print('Saving sampled data to {}'.format(save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GReaT')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')
    args = parser.parse_args()