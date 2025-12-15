import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchsummary import summary
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from lib import util

ModuleType = Union[str, Callable[..., nn.Module]]


class MLPDiffusion(nn.Module):
    def __init__(self, raw_config, dim_t=128):
        super().__init__()
        self.device = torch.device(raw_config['device'])
        self.d_in = raw_config['model_params']['d_in']
        self.num_classes = raw_config['model_params']['num_classes']
        self.rtdl_params = raw_config['model_params']['rtdl_params']
        self.dim_t = dim_t
        self.use_guide = raw_config['ddpm']['use_guide']

        # d0 = rtdl_params['d_layers'][0]

        self.rtdl_params['d_in'] = self.dim_t
        self.rtdl_params['d_out'] = self.d_in

        self.mlp = MLP.make_baseline(**self.rtdl_params)

        # if self.num_classes > 0:
        #     self.label_emb = nn.Embedding(self.num_classes+1, dim_t)
        # elif self.num_classes == 0:
        #     self.label_emb = nn.Linear(1, dim_t)

        if self.num_classes > 0:
            if not self.use_guide:
                self.label_emb = nn.Embedding(self.num_classes, dim_t)
            else:
                self.label_emb = nn.Embedding(self.num_classes+1, dim_t)
        elif self.num_classes == 0:
            self.label_emb = nn.Linear(1, dim_t)

        self.proj = nn.Linear(self.d_in, self.dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x: torch.tensor, y: torch.tensor, timesteps, context_mask: torch.int):
        """

        :param x: noise data
        :param y: label
        :param timesteps:
        :param context_mask:
        :return:
        """
        x = x.to(self.device)
        y = y.to(self.device)
        timesteps = timesteps.to(self.device)
        context_mask = context_mask.to(self.device)

        if self.use_guide:
            # 一部分数据使用无条件，即将其label置为0
            y = y * context_mask
            # 将0对应的嵌入权重置为全零向量
            empty_char_index = 0
            self.label_emb.weight.data[empty_char_index] = torch.zeros(self.dim_t)

        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))

        if self.num_classes > 0:
            y = y.squeeze()
        else:
            y = y.resize(y.size(0), 1).float()

        emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(self, *, d_in: int, d_out: int, bias: bool, activation: ModuleType, dropout: float,) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(self, *, d_in: int, d_layers: List[int], dropouts: Union[float, List[float]], activation: Union[str, Callable[[], nn.Module]], d_out: int,) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(cls: Type['MLP'], d_in: int, d_layers: List[int], dropout: float, d_out: int,) -> 'MLP':
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two layers, then all of them except for the first and the last ones must have the same dimension.
                Valid examples: :code:`[]`, :code:`[8]`, :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`.
                Invalid example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(d_in=d_in, d_layers=d_layers, dropouts=dropout, activation='ReLU', d_out=d_out,)


    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)



if __name__ == '__main__' :
    # test
    raw_config = util.load_config('E:\study\自学\表格数据生成\Tab-ClassifierFree\exp\CICIDS2017\config.toml')
    MLPnet =MLPDiffusion(raw_config)

    # 两种打印方式
    summary(MLPnet, input_size=[(20, 78), 1, 20, 20])
    print(MLPnet)