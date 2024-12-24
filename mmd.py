import torch
import torch.nn as nn
import numpy as np

class MMD_loss(nn.Module):
    def __init__(self, class_num, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def compute_pairwise_distance(self, x):
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)
        dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
        dist = torch.clamp(dist, min=0.0)
        return dist

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        total = torch.cat([source, target], dim=0)
        L2_distance = self.compute_pairwise_distance(total)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            n_samples = total.size(0)
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw_temp) for bw_temp in bandwidth_list]
        return sum(kernel_val)  # (batch_size, batch_size)

    def get_loss(self, source, target, s_label=None, t_label=None):
        batch_size = source.size(0)
        if s_label is not None and t_label is not None:
            weight_ss, weight_tt, weight_st = self.cal_weight(s_label, t_label, class_num=self.class_num)
        else:
            weight_ss = torch.ones(batch_size, batch_size).to(source.device) / (batch_size * batch_size)
            weight_tt = torch.ones(batch_size, batch_size).to(source.device) / (batch_size * batch_size)
            weight_st = torch.ones(batch_size, batch_size).to(source.device) / (batch_size * batch_size)

        kernels = self.gaussian_kernel(source, target,
                                       kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.sum(weight_ss * kernels[:batch_size, :batch_size]) + \
               torch.sum(weight_tt * kernels[batch_size:, batch_size:]) - \
               2 * torch.sum(weight_st * kernels[:batch_size, batch_size:])
        return loss

    def cal_weight(self, s_label, t_label, class_num):
        s_onehot = torch.nn.functional.one_hot(s_label, num_classes=class_num).float()
        t_onehot = torch.nn.functional.one_hot(t_label, num_classes=class_num).float()

        s_sum = torch.sum(s_onehot, dim=0, keepdim=True)
        s_sum[s_sum == 0] = 1
        s_vec_label = s_onehot / s_sum

        t_sum = torch.sum(t_onehot, dim=0, keepdim=True)
        t_sum[t_sum == 0] = 1
        t_vec_label = t_onehot / t_sum

        weight_ss = torch.mm(s_vec_label, s_vec_label.t())
        weight_tt = torch.mm(t_vec_label, t_vec_label.t())
        weight_st = torch.mm(s_vec_label, t_vec_label.t())

        return weight_ss, weight_tt, weight_st
