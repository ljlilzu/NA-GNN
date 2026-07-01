import os
import numpy as np
import torch
import dgl
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Actor
import time
import scipy.io
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def preprocess_data(dataset, train_ratio=0.5, val_ratio=0.25, param=None):
    """
    A data preprocessing function that loads data based on the dataset type and returns a DGL graph, features, labels, and training/validation/test masks.
    param dataset: Dataset name
    param train_ratio: Training set ratio
    param val_ratio: Validation set ratio
    param: Dictionary of other parameters
    return: DGL graph, features, labels, training mask, validation mask, test mask
     """

    if dataset in ['cora', 'citeseer', 'pubmed']:
        data_folder = '/root/autodl-tmp/data'
        dataset_data = Planetoid(root=data_folder, name=dataset)
        data = dataset_data[0]


        features = data.x
        labels = data.y
        la=labels
        edges = (data.edge_index[0], data.edge_index[1])

        g = create_dgl_graph_from_edges(edges, num_nodes=data.num_nodes)

    elif dataset in ['cornell', 'texas', 'wisconsin','chameleon']:
        data = torch.load(f'/root/autodl-tmp/data/{dataset}.pt')

        features = data['x'].numpy()
        labels = data['y'].numpy()
        la=labels
        edges = (data['edge_index'][0].numpy(), data['edge_index'][1].numpy())

        g = create_dgl_graph_from_edges(edges, num_nodes=labels.shape[0])
        features = torch.FloatTensor(normalize_features(features))
        labels = torch.LongTensor(labels)

    elif dataset in ['chameleon']:
        file_path = os.path.join('/root/autodl-tmp/data/', f"{dataset}_filtered.npz")
        data = np.load(file_path)
        features = data['node_features']
        labels = data['node_labels']
        la = labels
        edges = data['edges']
        g = create_dgl_graph_from_edges(edges, num_nodes=labels.shape[0])
        features = torch.FloatTensor(normalize_features(features))
        labels = torch.LongTensor(labels)

    elif dataset == 'twitch_gamers':

        base_path = '/root/autodl-tmp/data/twitch_gamers'
        edges_path = os.path.join(base_path, 'large_twitch_edges.csv')
        features_path = os.path.join(base_path, 'large_twitch_features.csv')

        edges_df = pd.read_csv(edges_path)
        edges = (edges_df.iloc[:, 0].values, edges_df.iloc[:, 1].values)

        feat_df = pd.read_csv(features_path)
        labels = feat_df['dead_account'].values

        cols_to_exclude = ['numeric_id', 'dead_account', 'created_at', 'updated_at']
        target_cols = [c for c in feat_df.columns if c not in cols_to_exclude]
        features_raw = feat_df[target_cols]

        features_processed = pd.get_dummies(features_raw)
        features = features_processed.values.astype(np.float32)

        la = labels
        g = create_dgl_graph_from_edges(edges, num_nodes=labels.shape[0])
        features = torch.FloatTensor(normalize_features(features))
        labels = torch.LongTensor(labels)


    elif dataset == 'penn94':
        file_path = '/root/autodl-tmp/data/penn94.mat'
        data = scipy.io.loadmat(file_path)
        adj = data['A']
        info = data['local_info'].astype(np.float32)


        raw_labels = info[:, 1].astype(np.int64)
        valid_mask = (raw_labels > 0)

        labels = raw_labels[valid_mask] - 1
        labels = torch.LongTensor(labels)

        features_raw = info[valid_mask][:, feature_cols]

        enc = OneHotEncoder(sparse_output=False)
        features_onehot = enc.fit_transform(features_raw).astype(np.float32)


        features = torch.FloatTensor(normalize_features(features_onehot))

        adj_trimmed = adj[valid_mask][:, valid_mask]

        src, dst = adj_trimmed.nonzero()
        edges = (src, dst)

        g = create_dgl_graph_from_edges(edges, num_nodes=labels.shape[0])

        train_mask, val_mask, test_mask = create_train_val_test_masks(labels.shape[0], train_ratio, val_ratio)

        return g, features, labels, train_mask, val_mask, test_mask, labels.numpy()
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

    train_mask, val_mask, test_mask = create_train_val_test_masks(features.shape[0], train_ratio, val_ratio)

    return g, features, labels, train_mask, val_mask, test_mask,la
