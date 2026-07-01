import os
import warnings

import heapq
import random
import collections
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

import dgl
from dgl import DGLGraph
import networkx as nx
import torch
from torch.utils.data import DataLoader


from dataset import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def collate(samples):

    graphs, labels, gt_adjs = map(list, zip(*samples))
    batched_graphs = dgl.batch(graphs)

    return batched_graphs, torch.cat(tuple(labels), 0), gt_adjs


def evaluate_f1(logits, labels):
    y_pred = torch.where(logits > 0.0, torch.ones_like(logits), torch.zeros_like(logits))
    y_pred = y_pred.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average='micro')


def dgl_to_adj(dgl_graph):

    adjs_list = []

    for i in range(16):
        if f'factor_{i}' not in dgl_graph.edata:
            break
        
        srt, dst = dgl_graph.edges()
        esge_weights = dgl_graph.edata[f'factor_{i}'].squeeze()
        srt, dst = srt.detach().cpu().numpy(), dst.detach().cpu().numpy()
        esge_weights = esge_weights.detach().cpu().numpy()
        
        num_node = dgl_graph.number_of_nodes()
        adjs = np.zeros((num_node, num_node))

        adjs[srt, dst] = esge_weights
        adjs += np.transpose(adjs)
        adjs /= 2.0
        adjs_list.append(adjs)
    
    return adjs_list


def translate_gt_graph_to_adj(gt_graph):
    gt_adjs = []
    gt_g_list = dgl.unbatch(gt_graph)

    for gt_g in gt_g_list:
        gt_list = []
        gt_ids = []

        n_node = gt_g.number_of_nodes()
        srt, dst = gt_g.edges()
        srt, dst = srt.detach().cpu().numpy(), dst.detach().cpu().numpy()
        edge_factor = gt_g.edata['feat'].detach().cpu().numpy()
        assert srt.shape[0] == edge_factor.shape[0]

        for edge_id in set(edge_factor):
            org_g = np.zeros((n_node, n_node))
            edge_factor_edge_id = np.zeros_like(edge_factor)
            idx = np.where(edge_factor == edge_id)[0] 
            edge_factor_edge_id[idx] = 1.0
            org_g[srt, dst] = edge_factor_edge_id
            gt_list.append(org_g)
            gt_ids.append(edge_id)

        gt_adjs.append((gt_list, gt_ids))

    return gt_adjs


def compute_consistant(total_factor_map):

    scores = []

    for idx in total_factor_map.keys():
        inds = total_factor_map[idx]
        most_id = max(set(inds), key = inds.count)
        scores.append(float(inds.count(most_id)) / len(inds))

    return np.mean(scores)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
if __name__ == '__main__':

    path = '../log/run0000/best_model.pt'
    best_model = torch.load(path)
    evaluate_graph(best_model)
