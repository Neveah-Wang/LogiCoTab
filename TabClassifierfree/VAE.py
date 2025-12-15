import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import typing as ty
import math


class Tokenizer(nn.Module):
    """
    跟 Revisiting Deep Learning Models for Tabular Data 的 Tokenizer 是一模一样的。
    """

    def __init__(self, d_numerical, categories, d_token, bias, bert_name):
        super().__init__()

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))

        if bert_name == 'bert-base-uncased':
            self.linear = nn.Linear(768, d_token)
        elif bert_name == 'huawei-noah/TinyBERT_General_4L_312D':
            self.linear = nn.Linear(312, d_token)
        elif bert_name == 'prajjwal1/bert-tiny':
            self.linear = nn.Linear(128, d_token)
        else:
            raise ValueError("wrong bert name!")


    def forward(self, x_num, x_cat, all_pooler_outputs):
        temp = 1+x_num.shape[1]+x_cat.shape[1] if x_cat is not None else 1+x_num.shape[1]

        # 这个 x_some 只是起到了一个获取维度的作用
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None

        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]

        if all_pooler_outputs is not None:
            x_cat_bert = self.linear(all_pooler_outputs)
            x = torch.cat([x, x_cat_bert], dim=1)

        assert x.shape[1] == temp
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization='kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):
  
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a/b, dim=-1)

        
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x
        
class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        activation='relu',
        prenormalization=True,
        initialization='kaiming',
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(d_token, n_heads, attention_dropout, initialization),
                    'linear0': nn.Linear(d_token, d_hidden),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
   
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)


    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x


class VAE(nn.Module):
    def __init__(self, d_numerical, categories, num_layers, hid_dim, n_head=1, factor=4, bias=True, bert_name="prajjwal1/bert-tiny"):
        super(VAE, self).__init__()
 
        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head
 
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias, bert_name=bert_name)

        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def get_embedding(self, x):
        return self.encoder_mu(x, x).detach() 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat, cls_heads):
        x = self.Tokenizer(x_num, x_cat, cls_heads)

        mu_z = self.encoder_mu(x)
        std_z = self.encoder_logvar(x)

        z = self.reparameterize(mu_z, std_z)

        h = self.decoder(z[:, 1:])
        
        return h, mu_z, std_z

class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class Model_VAE(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head=1, factor=4,  bias=True, bert_name="prajjwal1/bert-tiny"):
        super(Model_VAE, self).__init__()

        self.VAE = VAE(d_numerical, categories, num_layers, d_token, n_head=n_head, factor=factor, bias=bias, bert_name=bert_name)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def get_embedding(self, x_num, x_cat, cls_heads):
        x = self.Tokenizer(x_num, x_cat, cls_heads)
        return self.VAE.get_embedding(x)

    def forward(self, x_num, x_cat, cls_heads):

        h, mu_z, std_z = self.VAE(x_num, x_cat, cls_heads)

        # recon_x_num, recon_x_cat = self.Reconstructor(h[:, 1:])
        recon_x_num, recon_x_cat = self.Reconstructor(h)

        return recon_x_num, recon_x_cat, mu_z, std_z


class Encoder_model(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True, bert_name="prajjwal1/bert-tiny"):
        super(Encoder_model, self).__init__()
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias, bert_name=bert_name)
        self.VAE_Encoder = Transformer(num_layers, d_token, n_head, d_token, factor)

    def load_weights(self, Pretrained_VAE):
        self.Tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(Pretrained_VAE.VAE.encoder_mu.state_dict())   # 这里只是load了 encoder_mu 的参数，为什么？

    def forward(self, x_num, x_cat, cls_heads):
        x = self.Tokenizer(x_num, x_cat, cls_heads)
        z = self.VAE_Encoder(x)         # 为什么 VAE_Encoder(x) 的输出是z，VAE_Encoder(x)仅仅是 encoder_mu，logvar不需要生成吗？？

        return z

class Decoder_model(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True):
        super(Decoder_model, self).__init__()
        self.VAE_Decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def load_weights(self, Pretrained_VAE):
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(Pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z):

        h = self.VAE_Decoder(z)
        x_hat_num, x_hat_cat = self.Detokenizer(h)

        return x_hat_num, x_hat_cat