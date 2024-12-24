# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from GCN import GCN
from mmd import MMD_loss
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class PSFAN(nn.Module):
    def __init__(self, num_classes, num_features):
        super(PSFAN, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = 128  # 根据模型最后一层卷积的输出维度

        # 特征提取器
        self.feature_extractor = GCN(num_classes, num_features)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # 域判别器
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

        # MMD 损失
        self.mmd_loss = MMD_loss(class_num=num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def extract_features(self, x, adjs):
        x = x.to(self.device)
        for i, (edge_index, _, size) in enumerate(adjs):
            edge_index = edge_index.to(self.device)
            x_target = x[:size[1]]
            x = self.feature_extractor.convs[i]((x, x_target), edge_index)
            x = self.feature_extractor.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        return x

    def predict(self, data):
        x, adjs = data  # 解包输入数据
        x = self.extract_features(x, adjs)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # 返回 logits

    def forward(self, source_input, target_input, s_label=None, t_label=None, alpha=0.0):
        # 解包源域输入
        x_s, adjs_s = source_input
        # 解包目标域输入
        x_t, adjs_t = target_input

        # 初始化特征变量
        source_feature_flat = None
        target_feature_flat = None

        # 处理源域数据
        if x_s is not None and adjs_s is not None:
            source_feature = self.extract_features(x_s, adjs_s)
            source_feature_flat = source_feature.view(source_feature.size(0), -1)
            s_pred = self.classifier(source_feature_flat)
            reverse_source_feature = ReverseLayerF.apply(source_feature_flat, alpha)
            source_domain_pred = self.domain_classifier(reverse_source_feature)
        else:
            s_pred = None
            source_domain_pred = None

        # 处理目标域数据
        if x_t is not None and adjs_t is not None:
            target_feature = self.extract_features(x_t, adjs_t)
            target_feature_flat = target_feature.view(target_feature.size(0), -1)
            t_pred = self.classifier(target_feature_flat)
            reverse_target_feature = ReverseLayerF.apply(target_feature_flat, alpha)
            target_domain_pred = self.domain_classifier(reverse_target_feature)
        else:
            t_pred = None
            target_domain_pred = None

        # 计算MMD损失
        if source_feature_flat is not None and target_feature_flat is not None:
            if s_label is not None and t_label is not None:
                loss_mmd = self.mmd_loss.get_loss(source_feature_flat, target_feature_flat, s_label, t_label).to(self.device)
            else:
                loss_mmd = self.mmd_loss.get_loss(source_feature_flat, target_feature_flat).to(self.device)
        else:
            loss_mmd = torch.tensor(0.0, device=self.device)

        return s_pred, t_pred, source_domain_pred, target_domain_pred, loss_mmd


