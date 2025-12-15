import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import module
from lib.bert_util import sum_cls_head_mask


class Transformer(nn.Module):
    def __init__(self, raw_config, device) -> None:
        super().__init__()
        d_in = raw_config['model_params']['d_in']
        dim_t = raw_config['model_params']['dim_t']
        ff_dropout = raw_config['model_params']['ff_dropout']
        n_head = raw_config['model_params']['attention']['n_head']
        num_classes = raw_config['num_classes']

        self.device = torch.device(device)
        self.dim_t = dim_t
        self.depth = raw_config['model_params']['depth']
        self.use_guide = raw_config['ddpm']['use_guide']

        if num_classes > 0:
            if not self.use_guide:
                self.label_emb = nn.Embedding(num_classes, dim_t)
            else:
                self.label_emb = nn.Embedding(num_classes+1, dim_t)
        elif self.num_classes == 0:
            self.label_emb = nn.Linear(1, dim_t)

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        self.proj = nn.Linear(d_in, dim_t)
        self.norm = nn.LayerNorm(dim_t)
        self.attention1 = MultiHeadSelfAttention(dim_t, n_head)
        self.attention2 = MultiHeadSelfAttention(dim_t, n_head)

        """ 实例化Bert """
        if raw_config['model_params']['bert'] == 'bert-base-uncased':
            self.linear = nn.Linear(768, dim_t)
        elif raw_config['model_params']['bert'] == 'huawei-noah/TinyBERT_General_4L_312D':
            self.linear = nn.Linear(312, dim_t)
        elif raw_config['model_params']['bert'] == 'prajjwal1/bert-tiny':
            if dim_t == 128:
                self.linear = None
            else:
                self.linear = nn.Linear(128, dim_t)
        else:
            raise ValueError("wrong bert name!")

        layers = []
        layers.append(nn.LayerNorm(dim_t))
        input_dim = dim_t
        for output_dim in raw_config['model_params']['feedforward_layer']:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.SiLU())  # 激活函数
            input_dim = output_dim
        layers.append(nn.Dropout(ff_dropout))
        layers.append(nn.Linear(input_dim, dim_t))
        self.feedforward = nn.Sequential(*layers)

        self.out = nn.Linear(dim_t, d_in)


    def forward(self,
                x: torch.tensor,
                y: torch.tensor,
                cls_head: torch.Tensor,
                timesteps,
                context_mask: torch.int,
                if_mask: bool,
                device
                ):
        x = x.to(device)
        y = y.squeeze().to(device)
        cls_head = cls_head.to(device)
        timesteps = timesteps.to(device)
        context_mask = context_mask.squeeze().to(device)

        if self.use_guide:
            # 一部分数据使用无条件，即将其label置为0
            y = y * context_mask
            # 将0对应的嵌入权重置为全零向量
            empty_char_index = 0
            self.label_emb.weight.data[empty_char_index] = torch.zeros(self.dim_t)

        x = self.proj(x)
        time_emb = self.time_embed(module.timestep_embedding(timesteps, self.dim_t)).to(device)
        label_emb = F.silu(self.label_emb(y)).to(device)
        x += (time_emb + label_emb)

        cls_sum, cls_label = sum_cls_head_mask(cls_head, device, if_mask)

        if self.linear is not None:
            cls_sum = self.linear(cls_sum)
            cls_label = self.linear(cls_label)

        # for _ in range(self.depth):
        #     x_residual = x
        #     x = self.norm(x)
        #     x = self.attention(cls_head, x, x)
        #     x = x_residual + x
        #     x = self.feedforward(x) + x

        x_residual = x
        x = self.norm(x)
        x = self.attention1(cls_sum, x, x)
        x = x_residual + x
        x = self.feedforward(x) + x

        x_residual = x
        x = self.norm(x)
        x = self.attention2(cls_label, x, x)
        x = x_residual + x
        x = self.feedforward(x) + x

        return self.out(x)



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_t, n_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_head = n_head
        self.query_layers = nn.ModuleList([nn.Linear(dim_t, dim_t, bias=False) for _ in range(self.n_head)])
        self.key_layers = nn.ModuleList([nn.Linear(dim_t, dim_t, bias=False) for _ in range(self.n_head)])
        self.value_layers = nn.ModuleList([nn.Linear(dim_t, dim_t, bias=False) for _ in range(self.n_head)])
        self.fc_out = nn.Linear(dim_t * self.n_head, dim_t, bias=False)  # 输出线性层

    def forward(self, x_q, x_k, x_v):
        # x_q == x_k == x_v
        seq_length = x_k.shape[1]

        all_heads_output = []
        for i in range(self.n_head):
            # 生成q, k, v
            queries = self.query_layers[i](x_q).unsqueeze(-1)  # 生成 q_i
            keys = self.key_layers[i](x_k).unsqueeze(-1)       # 生成 k_i
            values = self.value_layers[i](x_v).unsqueeze(-1)   # 生成 v_i

            # 计算attention scores
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
            attention_scores = attention_scores / (seq_length ** 0.5)

            # 计算softmax权重
            attention_weights = F.softmax(attention_scores, dim=-1)

            # 计算加权和
            head_output = torch.matmul(attention_weights, values)
            head_output = head_output.squeeze(-1)  # 移除最后一维
            all_heads_output.append(head_output)

        # 拼接所有头的输出
        concatenated = torch.cat(all_heads_output, dim=-1)

        # 输出线性层
        output = self.fc_out(concatenated)

        return output


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

        # 定义生成q, k, v的线性层
        self.query = nn.Linear(4, 4, bias=False)
        self.key = nn.Linear(4, 4, bias=False)
        self.value = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        seq_length = x.shape[1]

        # 生成q, k, v
        queries = self.query(x).unsqueeze(-1)  # 生成 q_i
        keys = self.key(x).unsqueeze(-1)       # 生成 k_i
        values = self.value(x).unsqueeze(-1)   # 生成 v_i

        # 计算attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (seq_length ** 0.5)

        # 计算softmax权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权和
        attention_output = torch.matmul(attention_weights, values)
        attention_output = attention_output.squeeze(-1)  # 移除最后一维

        return attention_output



if __name__ == '__main__':
    # 示例输入数据
    x = torch.tensor([[1., 1., 1., 1.], [2., 2., 2., 2.]])

    # 初始化模型
    multi_head_self_attention_scalar = MultiHeadSelfAttention(dim_t=4, n_head=2)

    # 获取输出
    # output = multi_self_attention_scalar(x)
    output = multi_head_self_attention_scalar(x, x, x)
    print(output)
