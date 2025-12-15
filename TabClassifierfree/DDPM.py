import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from lib.bert_util import default_sentences
from lib.make_dataset import FastTensorDataLoader


class DDPM(nn.Module):
    def __init__(self, noise_prediction_model, raw_config):
        super(DDPM, self).__init__()
        self.device = torch.device(raw_config['device'])
        self.noise_prediction_model = noise_prediction_model
        self.n_T = raw_config['ddpm']['num_Timesteps']
        self.schedule_name = raw_config['ddpm']['schedule_name']
        self.use_guide = raw_config['ddpm']['use_guide']
        self.drop_prob_of_context_mask = raw_config['ddpm']['drop_prob_of_context_mask']
        self.loss_mse = nn.MSELoss()

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(self.n_T, self.schedule_name).items():
            self.register_buffer(k, v)



    def forward(self, x: torch.Tensor, y: torch.Tensor, cls_head: torch.Tensor):
        """
        this method is used in training, so samples t and noise randomly
        """
        x = x.to(self.device)    # x.shape = torch.Size([512, 78])
        y = y.to(self.device)    # y.shape = torch.Size([512, 1])

        '''t ~ Uniform(0, n_T)'''   # 这里为什么是给每个图片的加噪时间步是不一样的呢？？？？ 256张图片对应256种随机的t
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        # _ts = tensor([102, 234,  52, 135, 250, 236, 239, 246,  18, 156, ... , 55,  42, 398,   9])
        # _ts.shape = torch.Size([256])

        '''eps ~ N(0, 1)'''
        noise = torch.randn_like(x).to(self.device)    # noise.shape = torch.Size([512, 78])
        # print(f"noise.shape = {noise.shape}")

        '''前向加噪  This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps'''
        x_t = ((self.sqrtab[_ts-1, None]).to(self.device) * x + (self.sqrtmab[_ts-1, None]).to(self.device) * noise)   # x_t.shape = torch.Size([512, 78])
        # print('正向加噪后的噪声：')
        # print(pd.DataFrame(x_t.detach().numpy()))
        # print(f"x_t.shape = {x_t.shape}")

        '''context_mask'''
        if self.use_guide:
            # dropout context with some probability。取1的概率为self.drop_prob=0.9，取0的概率为0.1
            context_mask = torch.bernoulli(torch.zeros_like(y)+self.drop_prob_of_context_mask)   # context_mask.shape = torch.Size([512, 1])
            context_mask = context_mask.type(torch.int).to(self.device)
            # print(f"context_mask.shape = {context_mask.shape}")
            # print(context_mask)
            # 下面展示了 drop_prob=0.9 时的 context_mask:
            # tensor([[1], [1], [1], [1], ... ,[0], [0], [1], [1], [1]], dtype=torch.int32)
        else:
            context_mask = torch.ones_like(y).type(torch.int).to(self.device)

        '''预测噪声'''
        predicted_noise = self.noise_prediction_model(x_t, y, cls_head, _ts / self.n_T, context_mask, if_mask=True, device=self.device)   # predicted_noise.shape = torch.Size([512, 78])
        # print(f"predicted_noise.shape = {predicted_noise.shape}")

        return self.loss_mse(noise, predicted_noise)


    def sample(self, dataset, raw_config):
        """
        * 使用训练集的 y 作为条件c，生成跟训练集一样多的数据。
        * 不使用 ClassifierFree Guidance。
        * 不使用 bert 引入语言上的引导。
        """
        device = raw_config['sample']['device']
        size = (raw_config['model_params']['d_in'],)
        n_sample = len(dataset.y['train'])

        x_i = torch.randn(n_sample, *size, device=device)
        c_i = torch.tensor(dataset.y['train']).to(device)
        c_i = torch.reshape(c_i, (-1,))
        context_mask = torch.ones_like(c_i).type(torch.int).to(device)

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print("\r", end='')
            print(f'sampling timestep {i}', flush=True, end=" ")
            t_is = torch.tensor(i / self.n_T).to(device)   # 是个数字
            t_is = t_is.repeat(n_sample)                   # torch.Size([150])

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # 预测噪声
            eps = self.noise_prediction_model(x_i, c_i, t_is, context_mask, if_mask=False, device=device).to(device)

            # 去噪 x_t -> x_{t-1}
            mean = (self.oneover_sqrta[i - 1].to(device) * (x_i.to(device) - eps.to(device) * self.mab_over_sqrtmab[i - 1].to(device)))
            sigma_sqrt = torch.sqrt(self.posterior_variance)[i - 1].to(device)
            x_i = (mean + sigma_sqrt * z)

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store, c_i



    def sample_balance_classifierfree(self, raw_config, guide_w):
        """
        * 使用类平衡的 label 作为条件 c 对生成过程进行引导，每个生成的数量由 raw_config['sample']['n_sample_of_each_class'] 决定
        * 使用 ClassifierFree Guidance，需要设置 guide_w 。
        * 不使用 bert 引入语言上的引导。
        """

        device = raw_config['sample']['device']
        size = (raw_config['model_params']['d_in'],)
        num_classes = raw_config['num_classes']
        n_sample_of_each_class = raw_config['sample']['n_sample_of_each_class']
        n_sample = n_sample_of_each_class * num_classes

        x_i = torch.randn(n_sample, *size, device=device)      # x_T ~ N(0, 1), sample initial noise, x_i.shape = torch.Size([150, 78])
        # print("采样的输入：随机噪声")
        # print(pd.DataFrame(x_i[0:, 0:]))

        c_i = torch.zeros((num_classes, n_sample_of_each_class), dtype=torch.int).to(device)
        for i in range(num_classes):
            c_i[i, :] = i+1
        c_i = torch.reshape(c_i, (-1,))
        # print(f"c_i = {c_i}")

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).type(torch.int).to(device)  # context_mask.shape = torch.Size([150]),  context_mask = tensor([0, 0,..., 0, 0], dtype=torch.int32)

        # double the batch
        c_i = c_i.repeat(2)                                # c_i.shape = torch.Size([300])
        context_mask = context_mask.repeat(2)              # context_mask.shape = torch.Size([300])
        context_mask[:n_sample] = 1                        # makes second half of batch context free, 前半部分是1，后半部分是0
        # print(f"c_i = {c_i}")
        # print(f"context_mask.shape = {context_mask.shape}")

        # df = pd.DataFrame(columns=['t', '1', '2', '3', '4', '5', '6','x_t_0','x_t_1','x_t_2','x_t_3','eps1','eps2','eps_0','eps_1', 'eps_2','eps_3','mean_0','mean_1','mean_2','mean_3','sigma'])
        x_i_store = []                                     # keep track of generated steps in case want to plot something
        for i in range(self.n_T, 0, -1):
            print("\r", end='')
            print(f'sampling timestep {i}', flush=True, end=" ")
            t_is = torch.tensor(i / self.n_T).to(device)   # 是个数字
            t_is = t_is.repeat(n_sample)                   # torch.Size([150])

            # double batch
            x_i = x_i.repeat(2, 1)                   # torch.Size([300, 78])
            t_is = t_is.repeat(2)                          # torch.Size([300])

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # 预测噪声
            """
            split predictions and compute weighting
            噪声预测网络一共输入300张图片，其中150张随机生成的噪声图片，另外150张是复制而来的。
            context_mask中一共有300个数字，前150个是1，后150个0。对应前150个图片是有条件，后150个无条件。
            """
            eps = self.noise_prediction_model(x_i, c_i, t_is, context_mask, if_mask=False, device=device).to(device)   # eps.shape = torch.Size([300, 78])
            # print(f"eps.shape = {eps.shape}")
            # print(f"eps = {eps}")
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            # 去噪 x_t -> x_{t-1}
            x_i = x_i[:n_sample]
            a = x_i
            mean = self.oneover_sqrta[i-1].to(device) * (x_i.to(device) - eps.to(device) * self.mab_over_sqrtmab[i-1].to(device))
            sigma_sqrt = torch.sqrt(self.posterior_variance)[i-1].to(device)
            x_i = (mean + sigma_sqrt * z)

            # 存储去噪中间值
            """
            new = {'t':i,
                   '1':1.0-(self.alpha_t[i-1]).detach().numpy(),
                   '2':self.alpha_t[i-1].detach().numpy(),
                   '3':self.alphabar_t[i-1].detach().numpy(),
                   '4':self.sqrtmab[i-1].detach().numpy(),
                   '5':self.oneover_sqrta[i-1].detach().numpy(),
                   '6':self.mab_over_sqrtmab[i-1].detach().numpy(),
                   'x_t_0':a[0][0].detach().numpy(),
                   'x_t_1':a[0][1].detach().numpy(),
                   'x_t_2':a[0][2].detach().numpy(),
                   'x_t_3':a[0][3].detach().numpy(),
                   'eps1':eps1[0][0].detach().numpy(),
                   'eps2':eps2[0][0].detach().numpy(),
                   'eps_0':eps[0][0].detach().numpy(),
                   'eps_1':eps[0][1].detach().numpy(),
                   'eps_2':eps[0][2].detach().numpy(),
                   'eps_3':eps[0][3].detach().numpy(),
                   'mean_0':mean[0][0].detach().numpy(),
                   'mean_1':mean[0][1].detach().numpy(),
                   'mean_2':mean[0][2].detach().numpy(),
                   'mean_3':mean[0][3].detach().numpy(),
                   'sigma':sigma_sqrt.detach().numpy()
            }
            df = pd.concat([df, pd.DataFrame(new, index=[0])])
            """

            # 存储中间时刻的图像
            """
            if i % 200 == 0 or i == self.n_T or (i < 100 and i%20==0):
                x_i_store.append(x_i.detach().cpu().numpy())
            """

        x_i_store = np.array(x_i_store)
        # print(df)  # 打印去噪中间值
        return x_i, x_i_store, c_i[:n_sample]


    def sample_balance_bert_classifierfree(self, raw_config, guide_w):
        """
        * 使用类平衡的 label 作为条件 c 对生成过程进行引导。每个生成的数量由 raw_config['sample']['n_sample_of_each_class'] 决定
        * 使用 ClassifierFree Guidance。需要设置 guide_w 。
        * 使用 bert 引入语言上的引导。
        """
        device = raw_config['sample']['device']
        size = (raw_config['model_params']['d_in'],)
        num_classes = raw_config['num_classes']
        n_sample_of_each_class = raw_config['sample']['n_sample_of_each_class']
        n_sample = n_sample_of_each_class * num_classes

        x_i = torch.randn(n_sample, *size, device=device)      # x_T ~ N(0, 1), sample initial noise, x_i.shape = torch.Size([150, 78])
        c_i = torch.zeros((num_classes, n_sample_of_each_class), dtype=torch.int).to(device)
        for i in range(num_classes):
            c_i[i, :] = i+1
        c_i = torch.reshape(c_i, (-1,))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).type(torch.int).to(device)  # context_mask.shape = torch.Size([150]),  context_mask = tensor([0, 0,..., 0, 0], dtype=torch.int32)

        # double the batch
        c_i = c_i.repeat(2)                                # c_i.shape = torch.Size([300])
        context_mask = context_mask.repeat(2)              # context_mask.shape = torch.Size([300])
        context_mask[:n_sample] = 1                        # makes second half of batch context free, 前半部分是1，后半部分是0

        cls_head = default_sentences(c_i, raw_config)
        cls_head = cls_head.repeat(2, 1, 1).to(device)

        x_i_store = []                                     # keep track of generated steps in case want to plot something
        for i in range(self.n_T, 0, -1):
            print("\r", end='')
            print(f'sampling timestep {i}', flush=True, end=" ")
            t_is = torch.tensor(i / self.n_T).to(device)   # 是个数字
            t_is = t_is.repeat(n_sample)                   # torch.Size([150])

            # double batch
            x_i = x_i.repeat(2, 1)                   # torch.Size([300, 78])
            t_is = t_is.repeat(2)                          # torch.Size([300])

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # 预测噪声
            """
            split predictions and compute weighting
            噪声预测网络一共输入300张图片，其中150张随机生成的噪声图片，另外150张是复制而来的。
            context_mask中一共有300个数字，前150个是1，后150个0。对应前150个图片是有条件，后150个无条件。
            """
            eps = self.noise_prediction_model(x_i, c_i, cls_head, t_is, context_mask, if_mask=False, device=device).to(device)   # eps.shape = torch.Size([300, 78])
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            # 去噪 x_t -> x_{t-1}
            x_i = x_i[:n_sample]
            a = x_i
            mean = self.oneover_sqrta[i-1].to(device) * (x_i.to(device) - eps.to(device) * self.mab_over_sqrtmab[i-1].to(device))
            sigma_sqrt = torch.sqrt(self.posterior_variance)[i-1].to(device)
            x_i = (mean + sigma_sqrt * z)


        x_i_store = np.array(x_i_store)
        # print(df)  # 打印去噪中间值
        return x_i, x_i_store, c_i[:n_sample]


    def sample_balance_bert(self, raw_config):
        """
        * 使用类平衡的 label 作为条件 c 对生成过程进行引导。每个生成的数量由 raw_config['sample']['n_sample_of_each_class'] 决定
        * 不使用 ClassifierFree Guidance。。
        * 使用 bert 引入语言上的引导。
        """
        device = raw_config['sample']['device']
        size = (raw_config['model_params']['d_in'],)
        num_classes = raw_config['num_classes']
        n_sample_of_each_class = raw_config['sample']['n_sample_of_each_class']
        n_sample = n_sample_of_each_class * num_classes

        x_i = torch.randn(n_sample, *size, device=device)  # x_T ~ N(0, 1), sample initial noise, x_i.shape = torch.Size([150, 78])
        c_i = torch.zeros((num_classes, n_sample_of_each_class), dtype=torch.int).to(device)
        for i in range(num_classes):
            c_i[i, :] = i
        c_i = torch.reshape(c_i, (-1,))

        # don't drop context at test time
        context_mask = torch.ones_like(c_i).type(torch.int).to(device)  # context_mask.shape = torch.Size([150]),  context_mask = tensor([0, 0,..., 0, 0], dtype=torch.int32)

        cls_head = default_sentences(c_i, raw_config)
        print("cls_head.shape: ", cls_head.shape)

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print("\r", end='')
            print(f'sampling timestep {i}', flush=True, end=" ")
            t_is = torch.tensor(i / self.n_T).to(device)   # 是个数字
            t_is = t_is.repeat(n_sample)                   # torch.Size([150])

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # 预测噪声
            eps = self.noise_prediction_model(x_i, c_i, cls_head, t_is, context_mask, if_mask=False, device=device).to(device)

            # 去噪 x_t -> x_{t-1}
            mean = (self.oneover_sqrta[i - 1].to(device) * (x_i.to(device) - eps.to(device) * self.mab_over_sqrtmab[i - 1].to(device)))
            sigma_sqrt = torch.sqrt(self.posterior_variance)[i - 1].to(device)
            x_i = (mean + sigma_sqrt * z)

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store, c_i


    def sample_bert(self, dataset, raw_config):
        """
        * 使用训练集的 y 作为条件c，生成跟训练集一样多的数据。
        * 不使用 ClassifierFree Guidance。
        * 使用 bert 引入语言上的引导。
        """
        device = raw_config['sample']['device']
        size = (raw_config['model_params']['d_in'],)
        n_sample = len(dataset.y['train'])

        x_i = torch.randn(n_sample, *size, device=device)
        c_i = torch.tensor(dataset.y['train']).to(device)
        c_i = torch.reshape(c_i, (-1,))
        context_mask = torch.ones_like(c_i).type(torch.int).to(device)

        cls_head = default_sentences(c_i, dataset, raw_config)
        print("cls_head.shape: ", cls_head.shape)

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print("\r", end='')
            print(f'sampling timestep {i}', flush=True, end=" ")
            t_is = torch.tensor(i / self.n_T).to(device)  # 是个数字
            t_is = t_is.repeat(n_sample)  # torch.Size([150])

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # 预测噪声
            eps = self.noise_prediction_model(x_i, c_i, cls_head, t_is, context_mask, if_mask=False, device=device).to(device)

            # 去噪 x_t -> x_{t-1}
            mean = (self.oneover_sqrta[i - 1].to(device) * (x_i.to(device) - eps.to(device) * self.mab_over_sqrtmab[i - 1].to(device)))
            sigma_sqrt = torch.sqrt(self.posterior_variance)[i - 1].to(device)
            x_i = (mean + sigma_sqrt * z)

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store, c_i


    def sample_bert_batch(self, dataset, raw_config):
        """
        * 使用训练集的 y 作为条件c，生成跟训练集一样多的数据。
        * 不使用 ClassifierFree Guidance。
        * 使用 bert 引入语言上的引导。
        """
        device = raw_config['sample']['device']
        batch_size = raw_config['sample']['batch_size']
        size = (raw_config['model_params']['d_in'],)

        c_i = torch.tensor(dataset.y['train']).to(device)
        c_i = torch.reshape(c_i, (-1,))

        cls_head_all = default_sentences(c_i, dataset, raw_config)
        print("cls_head.shape: ", cls_head_all.shape)

        x_sample = []
        x_i_store = []
        c_loader = FastTensorDataLoader(c_i, cls_head_all, batch_size=batch_size)

        for step, [c, cls_head] in enumerate(c_loader):
            x_i = torch.randn(len(c), *size, device=device)
            context_mask = torch.ones_like(c).type(torch.int).to(device)

            for i in range(self.n_T, 0, -1):
                # 不打印了，试一下会不会变快，好像不会变快
                print("\r", end='')
                print(f'sampling timestep {i}', flush=True, end=" ")
                t_is = torch.tensor(i / self.n_T).to(device)  # 是个数字
                t_is = t_is.repeat(len(c))  # torch.Size([150])

                z = torch.randn(len(c), *size).to(device) if i > 1 else 0

                # 预测噪声
                eps = self.noise_prediction_model(x_i, c, cls_head, t_is, context_mask, if_mask=False, device=device).to(device)

                # 去噪 x_t -> x_{t-1}
                mean = (self.oneover_sqrta[i - 1].to(device) * (x_i.to(device) - eps.to(device) * self.mab_over_sqrtmab[i - 1].to(device)))
                sigma_sqrt = torch.sqrt(self.posterior_variance)[i - 1].to(device)
                x_i = (mean + sigma_sqrt * z)

            x_sample.append(x_i)

        x_sample = torch.cat(x_sample, dim=0)
        x_i_store = np.array(x_i_store)
        return x_sample, x_i_store, c_i


def ddpm_schedules(T, schedule_name, beta1=1e-4, beta2=0.02, s=0.008):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    if schedule_name == 'linear_beta_schedule':
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
        beta_t = torch.linspace(beta1, beta2, T)

    elif schedule_name == 'quadratic_beta_schedule':
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
        beta_t = torch.linspace(beta1 ** 0.5, beta2 ** 0.5, T) ** 2

    elif schedule_name == 'sigmoid_beta_schedule':
        betas = torch.linspace(-6, 6, T)
        beta_t = torch.sigmoid(betas) * (beta2 - beta1) + beta1

    elif schedule_name == 'cosine_beta_schedule':
        # cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        beta_t = torch.clip(betas, 0.0001, 0.99)

    else:
        raise ValueError('Schedule_name is worry')


    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1. - beta_t
    # log_alpha_t = torch.log(alpha_t)
    # alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    alphabar_t = torch.cumprod(alpha_t, dim=0)
    alphabar_prev_t = torch.cat((torch.tensor([1.]), alphabar_t[:-1]))

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1. / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1. - alphabar_t)
    mab_over_sqrtmab_inv = (1. - alpha_t) / sqrtmab

    posterior_variance = beta_t * (1.0 - alphabar_prev_t) / (1.0 - alphabar_t)

    return {
        "alpha_t": alpha_t,              # \alpha_t                包含了所有t时刻的值 tensor([\alpha_0, ... ,\alpha_t, ... , \alpha_T])
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}       包含了所有t时刻的值
        "sqrt_beta_t": sqrt_beta_t,      # \sqrt{\beta_t}          包含了所有t时刻的值
        "alphabar_t": alphabar_t,        # \bar{\alpha_t}          包含了所有t时刻的值 tensor([\bar{\alpha_0}, \bar{\alpha_2}, ... , \bar{\alpha_t}, ... , \bar{\alpha_T}])
        "sqrtab": sqrtab,                # \sqrt{\bar{\alpha_t}}   包含了所有t时刻的值
        "sqrtmab": sqrtmab,              # \sqrt{1-\bar{\alpha_t}} 包含了所有t时刻的值
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}   包含了所有t时刻的值
        "posterior_variance": posterior_variance
    }



if __name__ == '__main__' :
    # test
    dic = ddpm_schedules(1e-4, 0.02, 10, 'linear_beta_schedule', s=0.008)
    print(dic)
    _ts = torch.randint(1, 10 + 1, (25,))
    print(_ts)
    print(dic["sqrtab"][_ts-1, None, None, None])