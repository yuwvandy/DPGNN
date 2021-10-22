import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch.nn import Linear
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.n_hidden)
        self.conv2 = GCNConv(args.n_hidden, args.n_hidden)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class prototype(torch.nn.Module):
    def __init__(self):
        super(prototype, self).__init__()
        

    def forward(self, x):
        return torch.mean(x, dim = 0)


class dist_embed(torch.nn.Module):
    def __init__(self, args):
        super(dist_embed, self).__init__()
        self.lin = Linear(args.n_hidden*args.num_classes, args.num_classes)

    def forward(self, query, proto, classes):
        d1 = query.size(0)
        d2 = proto.size(0)

        query = torch.repeat_interleave(query, d2, dim = 0)
        proto = torch.tile(proto, (d1, 1))

        dist = self.lin((query - proto).view(d1, -1))

        return dist



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


class Prop(MessagePassing):
    def __init__(self, num_classes, K=10, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)


        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)

        pps = torch.stack(preds)
        out = torch.sum(pps, dim = 0)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
