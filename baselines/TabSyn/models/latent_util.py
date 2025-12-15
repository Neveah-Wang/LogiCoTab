import os
import torch
import json
import numpy as np

from lib.make_dataset import make_dataset
from baselines.TabSyn.models.vae import Decoder_model
from lib.make_dataset import make_dataset_for_uncondition


def get_latent_train(raw_config):
    parent_dir = raw_config['parent_dir']
    train_z = torch.tensor(np.load(os.path.join(parent_dir, 'latent_data/train_z.npy'))).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    print("train_z.shape = {}".format(train_z.shape))

    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y = torch.from_numpy(dataset.y['train'])

    return train_z, y


def get_input_train(raw_config):
    parent_dir = raw_config['parent_dir']
    train_z = torch.tensor(np.load(os.path.join(parent_dir, 'latent_data/train_z.npy'))).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)

    return train_z


def get_input_generate(raw_config):
    real_data_path = raw_config['real_data_path']
    parent_dir = raw_config['parent_dir']
    device = torch.device(raw_config['sample']['device'])

    with open(f'{real_data_path}/info.json', 'r') as f:
        info = json.load(f)

    dataset = make_dataset_for_uncondition(real_data_path, raw_config)

    embedding_save_path = f'{parent_dir}/latent_data/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)

    pre_decoder = Decoder_model(num_layers=raw_config['VAE']['num_layers'],
                                d_numerical=dataset.n_num_features(),
                                categories=dataset.get_category_sizes('train'),
                                d_token=raw_config['VAE']['d_token'],
                                n_head=raw_config['VAE']['n_head'],
                                factor=raw_config['VAE']['factor'], ).to(device)
    pre_decoder.load_state_dict(torch.load(os.path.join(parent_dir, 'vae_decoder_model.pth'), map_location=device))

    info['pre_decoder'] = pre_decoder
    info['token_dim'] = token_dim

    return (train_z,
            info,
            dataset.num_transformer.inverse_transform if dataset.num_transformer else None,
            dataset.cat_transformer.inverse_transform if dataset.cat_transformer else None,
            dataset.y_transformer.inverse_transform if dataset.y_transformer else None
            )

'''
@torch.no_grad()
def split_num_cat_target(syn_data, raw_config, info, num_inverse, cat_inverse):
    device = torch.device(raw_config['sample']['device'])
    task_type = info['task_type']
    n_num_feat = len(raw_config['X_num_columns'])
    n_cat_feat = len(raw_config['X_cat_columns'])
    n_y_feat = len(raw_config['y_column'])

    if task_type == 'regression':
        n_num_feat += n_y_feat
    else:
        n_cat_feat += n_y_feat

    pre_decoder = info['pre_decoder'].to(device)
    token_dim = info['token_dim']

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim).to(device)
    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    if task_type == 'regression':
        syn_target = syn_num[:, :n_y_feat]
        syn_num = syn_num[:, n_y_feat:]
    else:
        syn_target = syn_cat[:, :n_y_feat]
        syn_cat = syn_cat[:, n_y_feat:]

    return syn_num, syn_cat, syn_target
'''

def split_num_cat_target(syn_data, raw_config, info, num_inverse, cat_inverse, y_inverse):
    device = torch.device(raw_config['sample']['device'])
    task_type = info['task_type']
    n_num_feat = len(raw_config['X_num_columns'])
    n_cat_feat = len(raw_config['X_cat_columns'])
    if raw_config['Transform']['y_policy'] == 'one-hot':
        n_y_feat = raw_config['num_classes']
    else:
        n_y_feat = len(raw_config['y_column'])

    if task_type == 'regression':
        n_num_feat += n_y_feat
    else:
        n_cat_feat += n_y_feat

    pre_decoder = info['pre_decoder'].to(device)
    token_dim = info['token_dim']

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim).to(device)
    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    syn_num = x_hat_num.detach().cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    if task_type == 'regression':
        syn_target = syn_num[:, :n_y_feat]
        syn_num = syn_num[:, n_y_feat:]
    else:
        syn_target = syn_cat[:, :n_y_feat]
        syn_cat = syn_cat[:, n_y_feat:]

    syn_num = num_inverse(syn_num) if num_inverse else syn_cat
    syn_cat = cat_inverse(syn_cat) if cat_inverse else syn_cat
    syn_target = y_inverse(syn_target) if y_inverse else syn_target

    return syn_num, syn_cat, syn_target
