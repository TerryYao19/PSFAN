import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, sample_ratio=1.0, time_threshold=100):
        self.sample_ratio = sample_ratio
        self.time_threshold = time_threshold
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['2_target_30%_cleaned.csv']

    @property
    def processed_file_names(self):
        return ['2_target_30%_99%.pt']

    def process(self):
        data = construct_single_graph_from_csv(
            os.path.join(self.root, self.raw_file_names[0]),
            sample_ratio=self.sample_ratio,
            time_threshold=self.time_threshold)
        torch.save(self.collate([data]), self.processed_paths[0])

def construct_single_graph_from_csv(file_path, sample_ratio=1.0, time_threshold=100):
    df = pd.read_csv(file_path)

    # 编码标签
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df.iloc[:, -1])

    # 分层抽样
    df_sampled = df.groupby('Label', group_keys=False).apply(
        lambda x: x.sample(frac=sample_ratio, random_state=42)).reset_index(drop=True)

    # 排序数据
    df_sampled = df_sampled.sort_values('frame.time_relative').reset_index(drop=True)

    # 重置索引
    df_sampled = df_sampled.reset_index(drop=True)

    # 标准化特征
    features = df_sampled.select_dtypes(include=[np.number]).drop(columns=['frame.time_delta','frame.time_relative', 'Label'])  # 排除时间戳和标签列
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    num_nodes = df_sampled.shape[0]
    x = torch.tensor(features_scaled, dtype=torch.float)
    y = torch.tensor(df_sampled['Label'].values, dtype=torch.long)

    timestamps = df_sampled['frame.time_relative'].values

    edge_index = []
    # 构建边
    print("正在构建边...")
    for i in tqdm(range(num_nodes)):
        t_i = timestamps[i]
        for j in range(i + 1, num_nodes):
            delta_t = timestamps[j] - t_i
            if delta_t > time_threshold:
                break  # 超过阈值，停止遍历
            else:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 无向图，添加双向边

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)

    # 打印节点和边的数量
    num_edges = edge_index.size(1)
    print(f"图中共有 {num_nodes} 个节点，{num_edges} 条边。")

    return data

if __name__ == "__main__":
    sample_ratio = 1.0  # 选择数据的比例
    time_threshold =  0.06200307999999981  # 时间阈值，根据您的数据进行调整
    dataset = CustomGraphDataset(root='Dataset', sample_ratio=sample_ratio, time_threshold=time_threshold)
    data = dataset[0]  # 获取图数据

    # 保存数据
    torch.save(data, '2_target_30%_99%.pt')  # 保存为 data.pt，供训练和评估使用
