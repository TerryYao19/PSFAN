import pandas as pd
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from tqdm import tqdm

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, sample_ratio=1.0):
        self.sample_ratio = sample_ratio
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['http3_1.csv']

    @property
    def processed_file_names(self):
        return ['http3_1.pt']

    def download(self):
        # Download to `self.raw_dir`
        pass

    def process(self):
        data = construct_single_graph_from_csv(os.path.join(self.root, self.raw_file_names[0]), sample_ratio=self.sample_ratio)
        torch.save(self.collate([data]), self.processed_paths[0])

def construct_single_graph_from_csv(file_path, threshold=0.000001, sample_ratio=1.0):
    df = pd.read_csv(file_path)

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df.iloc[:, -1])

    # 分层抽样
    df_sampled = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=sample_ratio, random_state=42)).reset_index(drop=True)

    # Select only numeric features for standardization
    features = df_sampled.iloc[:, 2:-1].select_dtypes(include=[float, int])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Prepare graph
    num_rows = df_sampled.shape[0]
    G = nx.Graph()

    # Add nodes with features
    for i in range(num_rows):
        G.add_node(i, x=torch.tensor(features[i], dtype=torch.float))

    # Add edges based on time difference using vectorized approach
    times = df_sampled.iloc[:, 1].values
    edges = []

    # Initialize tqdm progress bar
    pbar = tqdm(total=num_rows * (num_rows - 1) // 2, desc='Building edges', unit='edges')

    for i in range(num_rows):
        time_diff = np.abs(times[i] - times[i+1:])
        valid_edges = np.where(time_diff < threshold)[0]
        for j in valid_edges:
            j = j + i + 1
            weight = 1.0 / (time_diff[j - i - 1] + 1e-5)
            edges.append((i, j, weight))
            pbar.update(1)

    pbar.close()

    # Add edges to the graph
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)

    # Convert to PyTorch Geometric Data
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float).view(-1, 1)
    x = torch.stack([G.nodes[i]['x'] for i in range(num_rows)])
    y = torch.tensor(df_sampled['label'].values, dtype=torch.long)

    # Create masks
    num_nodes = num_rows
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    # Ensure each label is included in the training set
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    unique_labels = df_sampled['label'].unique()
    for label in unique_labels:
        label_indices = indices[df_sampled['label'].values[indices] == label]
        train_size = int(0.5 * len(label_indices))
        train_mask[label_indices[:train_size]] = True
        test_mask[label_indices[train_size:]] = True

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                train_mask=train_mask, test_mask=test_mask)

    return data

if __name__ == "__main__":
    sample_ratio = 1.0  # 选择数据的比例
    dataset = CustomGraphDataset(root='Dataset', sample_ratio=sample_ratio)

    #data = torch.load('Dataset/processed/http3_1.pt')  # 修改为你的文件路径


