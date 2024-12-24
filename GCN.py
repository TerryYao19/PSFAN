import torch
from torch_geometric.nn import GraphConv
from torch.nn import BatchNorm1d
import torch.nn.functional as F
class GCN(torch.nn.Module):
    def __init__(self, num_classes, num_features):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GraphConv(num_features, 128))
        self.bns.append(BatchNorm1d(128))

        self.convs.append(GraphConv(128, 128))
        self.bns.append(BatchNorm1d(128))

        self.convs.append(GraphConv(128, 128))
        self.bns.append(BatchNorm1d(128))

        # 可以根据需要添加更多层

        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

