import sys
from unittest.mock import MagicMock
from matplotlib.lines import Line2D
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import os
import csv
import time
import json
import argparse
import warnings
import datetime
import statistics
import nni
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.nn import GraphConv

from utils import *
from loaddata import *
from model_self import *

from CAGNN import *
from RFAGNN import *
from GCN import *
from xiangsix import *
from KNNGNN import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_class_weights(labels, num_classes):
    """
    Compute class weights to give larger weights to minority classes
    """
    class_counts = torch.bincount(labels)  # 计算每个类别中的样本数
    total_samples = labels.shape[0]
    class_weights = total_samples / (num_classes * class_counts)  # 类别权重 = 总样本数 / (类别数 * 该类别样本数)
    return class_weights

def  exponential_penalty_balance_rati(s_in,s_out,n):
    epbr= 10*(s_in-s_out)/np.exp(n*(s_out-0.3))
    return epbr

def main(param, seed=None):

    if param['save_mode'] == 0:
        set_seed(param['seed'])
    else:
        set_seed(seed)

    g, features, labels, train_mask, val_mask, test_mask,la = preprocess_data(param['dataset'], param['train_ratio'],param['val_ratio'],
                                                                           param)
    print(g)

    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print('using cpu')

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    g = g.to(device)

    within_selfs = []
    within_neis = []

    if param['model'] == 'NAGNN':
        for i in range(10):
            within_self, within_neighbors, between_self, between_neighbors = compute_average_similarity_per_label(
                g, features, labels,
                train_mask, ratio=0.1,N_min=30)


            within_selfs.append(within_self)
            within_neis.append(within_neighbors)


        within_selfs_tensor = torch.tensor(within_selfs)
        within_selfs_mean = np.mean(within_selfs_tensor.cpu().numpy())
        within_neis_tensor = torch.tensor(within_neis)
        within_neis_mean = np.mean(within_neis_tensor.cpu().numpy())
        between_self_tensor = torch.tensor(between_self)
        between_self_mean = np.mean(between_self_tensor.cpu().numpy())
        between_neis_tensor = torch.tensor(between_neighbors)
        between_neis_mean = np.mean(between_neis_tensor.cpu().numpy())

        n=param['n']

        epbr_self = exponential_penalty_balance_rati(within_selfs_mean,between_self_mean,n)

        epbr_nei = exponential_penalty_balance_rati(within_neis_mean,between_neis_mean,n)

        alpha = (epbr_nei-epbr_self)/epbr_self
        param['self_nei']=alpha

        a=param['a']

        if alpha > 1+a:
            param['self_nei'] = 1
        elif alpha < a:
            param['self_nei'] = 0
        else:
            param['self_nei'] = alpha-a

        print(f"alpa-0.5:{param['self_nei']}")


    if param['model'] == 'MLP':
        model = MLP(g, param).to(device)
    elif param['model'] == 'GCN':
        model = GCN(g, param).to(device)
    elif param['model'] == 'GAT':
        model = GAT(g, param).to(device)
    elif param['model'] == 'H2GCN':
        model = H2GCN(g, param).to(device)
    elif param['model'] == 'MixHop':
        model = MixHop(g, param).to(device)
    elif param['model'] == 'CAGNN':
        model = CAGNN(g, param).to(device)
    elif param['model'] == 'NAGNN':
        model = NAGNN(g, param).to(device)
    elif param['model'] == 'GCN_MLP':
        model = GCN_MLP(g, param).to(device)
    elif param['model'] == 'GraphSAGE':
        model = GraphSAGE(g, param).to(device)
    else:
        raise ValueError(f"There is no model named '{param['model']}'.")


    num_classes = int(labels.max().item()) + 1
    class_weights = compute_class_weights(labels[train_mask], num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    val_best = 0
    test_best = 0
    test_val = 0
    val_best_epoch = 0
    best_logits = None

    start_time = datetime.datetime.now()

    for epoch in range(param['epochs']):
        model.train()
        optimizer.zero_grad()


        logits = model(features)

        loss_cla = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights)

        train_loss = loss_cla
        train_acc = accuracy(logits[train_mask], labels[train_mask])

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask], weight=class_weights).item()
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])

        if val_acc >  val_best and epoch>20:
            val_best = val_acc
            min_loss = val_loss
            test_val = test_acc
            val_best_epoch = epoch
            test_best = test_acc
            best_logits = logits.detach().cpu()
            torch.save(model.state_dict(), os.path.join(param['save_dir'], 'best_model.pth'))
        print(
            "\033[0;30;46m Epoch: {} | Loss: loss_cla-{:.6f}, train_loss-{:.6f} | Acc: train_acc-{:.5f}, val_acc-{:.5f}, test_acc-{:.5f}, val_best_epoch-{} ({:.5f}), test_best-{:.5f} \033[0m".format(
                epoch, loss_cla.item(), train_loss.item(), train_acc, val_acc, test_acc, val_best_epoch, test_val,
                test_best
            )
        )
    end_time = datetime.datetime.now()
    runtime = end_time - start_time

    if param['save_mode'] == 0:
        nni.report_final_result(test_val)
        outFile = open('/root/autodl-tmp/PerformMetrics.csv', 'a+', newline='')
        writer = csv.writer(outFile, dialect='excel')
        results = [
            str(param["dataset"]),
            str(param["k"]),
            str(param['lr']),
            str(param['hidden_dim']),
            str(param['epochs']),
            str(test_acc),
            str(test_val),
            str(test_best),
            str(val_best_epoch),
            str(runtime.total_seconds()),

            float(param['a']),
            float(param['n']),
            float(param['tau']),
        ]
        writer.writerow(results)

        return test_acc, test_val,val_best_epoch,test_best,runtime,best_logits,la
    else:
        return test_acc, test_val, test_best,runtime

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'texas',
                        choices=['cora', 'citeseer', 'pubmed', 'texas', 'cornell', 'wisconsin', 'chameleon',
                                'twitch_gamers','penn94'])
    parser.add_argument("--dataset_num", type=int, default=-1)
    parser.add_argument("--save_mode", type=int, default=1)

    parser.add_argument("--out_dim", type=int, default=6, choices=[7, 6, 3, 5, 5, 5, 5, 5, 5])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument("--dataset_name", type=str, default='h0.00-r1')
    parser.add_argument('--train_ratio', type=float, default=0.48)
    parser.add_argument('--val_ratio', type=float, default=0.32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_graph', type=int, default=4)
    parser.add_argument("--graph_mode", type=int, default=0)
    parser.add_argument("--model_mode", type=int, default=1)
    parser.add_argument("--num_hop", type=int, default=6)
    parser.add_argument("--model", type=str, default='GAT')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--a', type=float, default=0.5)
    parser.add_argument('--n', type=float, default=3)#惩罚系数上的
    parser.add_argument('--tau', type=float, default=1)


    args = parser.parse_args()

    if args.dataset in ['cora', 'citeseer', 'pubmed', 'texas', 'cornell', 'wisconsin', 'chameleon','twitch_gamers','penn94']:
        load_json_path = os.path.join("/root/autodl-tmp/data/Param", f"param_{args.dataset}.json")
        jsontxt = open(load_json_path, 'r').read()
        param = json.loads(jsontxt)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")


    param.update(nni.get_next_parameter())

    param['hidden_dim'] = args.hidden_dim
    param['dropout'] = args.dropout
    param['epochs'] = args.epochs
    param['k'] = args.k
    param['save_mode'] = 0
    param['seed'] = args.seed
    param['lr'] = args.lr
    param['weight_decay'] = args.weight_decay
    param['num_graph'] = args.num_graph
    param['graph_mode'] = args.graph_mode
    param['model_mode'] = args.model_mode
    param['num_hop'] = args.num_hop
    #param['beta'] = 1
    param['model'] = args.model
    param['train_ratio'] = args.train_ratio
    param['val_ratio'] = args.val_ratio
    param['a'] = args.a
    param['n'] = args.n
    param['tau']=args.tau
    if param['save_mode'] == 0:
        main(param)


