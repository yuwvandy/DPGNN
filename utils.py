import torch
import numpy as np
from sklearn.metrics import f1_score
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops, degree
import torch.nn.functional as F
import random
import torch.nn as nn
import matplotlib.pyplot as plt

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def adj2edgeindex(adj):
    edge_index = []
    head, tail = torch.where(adj == 1)
    edge_index.append(head)
    edge_index.append(tail)

    return torch.stack(edge_index)


def label_prop(edge_index, train_mask, c_train_num, y, epochs=20):
    y_onehot = F.one_hot(y).type(torch.float)
    y_onehot[~train_mask, :] = 0

    ratio = (c_train_num / sum(c_train_num)).to(y.device)

    y_onehot = y_onehot * (1 / ratio[y].view(-1, 1))

    h = y_onehot
    for epoch in range(epochs):
        h = propagate(edge_index, h)

    return h


def propagate(edge_index, x):
    """ feature propagation procedure: sparsematrix
    """

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[row]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def y_aug_trail(y, y_prop, c_train_num, val_mask):
    conf = (y_prop.max(dim=1)[0] - (y_prop.sum(dim=1) - y_prop.max(dim=1)[0]) / (
        y_prop.size(1) - 1)).view(-1, 1) / y_prop.mean(dim=1).view(-1, 1)
    # conf = (y_prop.max(dim=1)[0] - (y_prop.sum(dim=1) - y_prop.max(dim=1)[0]) / (
    #     y_prop.size(1) - 1)).view(-1, 1) / y_prop.sum(dim=1).view(-1, 1)

    tmp = conf.view(-1,)
    tmp = tmp[~torch.isnan(tmp)]

    y_aug = y_prop.max(dim=1)[1]
    acc, count = [], []
    for eta in np.arange(0, max(tmp).item() + 1, 0.1):
        idx = (conf > eta).view(-1, ) & val_mask
        acc.append(((y[idx] == y_aug[idx]).sum() / idx.sum()).item())
        count.append((idx.sum()).item())

    # plt.figure()
    # plt.plot(np.arange(0, max(tmp).item(), 0.1), acc)
    # plt.xlabel('Threshold', fontsize=15, fontweight='bold')
    # plt.ylabel('ACC', fontsize=15, fontweight='bold')
    # np.save('acc_Cora', acc)

    # plt.savefig('Cora_labelprop.pdf', dpi = 1000, bbox_inches = 'tight')
    # plt.close()

    # plt.figure()
    # plt.plot(np.arange(0, max(tmp).item(), 0.1), count)
    # plt.xlabel('Threshold', fontsize=15, fontweight='bold')
    # plt.ylabel('Number', fontsize=15, fontweight='bold')
    # np.save('num_Cora', count)

    # plt.savefig('Cora_labelnumber.pdf', dpi = 1000, bbox_inches = 'tight')
    # plt.close()

    # np.save('labelprop_acc_Pubmed', acc, allow_pickle = True)
    # np.save('labelprop_num_Pubmed', count, allow_pickle = True)


def sample(train_mask, c_train_num, y_prop, y, eta):
    conf = (y_prop.max(dim=1)[0] - (y_prop.sum(dim=1) - y_prop.max(dim=1)[0]) / (
        y_prop.size(1) - 1)).view(-1, 1) / y_prop.mean(dim=1).view(-1, 1)

    idx = (conf > eta).view(-1, )

    y_aug = y.clone().detach().cpu()
    idx, y_prop, y = idx.detach().cpu(), y_prop.detach().cpu(), y.detach().cpu()
    y_aug[idx] = y_prop.max(dim=1)[1][idx]
    y_aug[train_mask] = y[train_mask]
    new_train_mask = train_mask.clone()
    new_train_mask[idx] = True

    return y_aug, new_train_mask

def episodic_generator(data, ratio, classes, nodenum):

    idx = torch.arange(0, nodenum)
    support = []
    query = []
    for i in range(len(classes)):
        train_idx = idx[(data.y_aug == classes[i]) & data.train_mask].tolist()
        sample_train_idx = random.sample(
            train_idx, int(len(train_idx) * ratio))

        if(len(sample_train_idx) >= 2):
            support_idx = sample_train_idx[1:]
            query_idx = sample_train_idx[0]
        else:
            support_idx = sample_train_idx[:]
            query_idx = sample_train_idx[0]

        support.append(support_idx)
        query.append(query_idx)

    return support, query


def deg(edge_index, x):
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)

    deg_inv_sqrt = deg.pow(-0.5)

    return deg_inv_sqrt


def cos_sim(x1, x2):
    """ Calculate the similarity between x1 and x2
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if(len(x1.size()) == 1):  # in case that x1 and x2 are just 1-dimension tensor
        return cos(x1.view(1, -1), x2.view(1, -1))
    else:
        return cos(x1, x2)


def cos_sim_pair(representation):
    nodenum = representation.size(0)

    cos_pair = torch.zeros((nodenum, nodenum), dtype=torch.float)

    for i in range(nodenum):
        cos = cos_sim(representation[i].repeat(
            representation.size(0), 1), representation)
        cos_pair[i] = cos

    return cos_pair
