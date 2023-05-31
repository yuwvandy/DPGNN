from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid, Amazon, CitationFull
import os.path as osp
import torch_geometric.transforms as T
import torch
from utils import *
import numpy as np
from typing import Optional, Callable
import copy


class Twitch(InMemoryDataset):
    r"""The Twitch Gamer networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent gamers on Twitch and edges are followerships between them.
    Node features represent embeddings of games played by the Twitch users.
    The task is to predict whether a user streams mature content.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"DE"`, :obj:`"EN"`,
            :obj:`"ES"`, :obj:`"FR"`, :obj:`"PT"`, :obj:`"RU"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    url = 'https://graphmining.ai/datasets/ptg/twitch'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        assert self.name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)

        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).to(index.device)
    mask[index] = 1

    return mask


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name, transform=T.NormalizeFeatures())

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_twitch_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Twitch(path, name, transform=T.NormalizeFeatures())

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_WebKB_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WebKB(path, name, transform=T.NormalizeFeatures())

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Amazon(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_citationalfull_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = CitationFull(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def load_data(dataname, imb_ratio):
    if(dataname in ['Cora', 'Citeseer', 'Pubmed']):
        dataset = get_planetoid_dataset(
            dataname, transform=T.NormalizeFeatures())
        class_sample_num = 2
        classes = torch.unique(dataset.data.y)
        if(dataname == 'Cora'):
            ratio = [1] * 5 + [imb_ratio] * 2
        if(dataname == 'Citeseer'):
            ratio = [1] * 4 + [imb_ratio] * 2
        if(dataname == 'Pubmed'):
            ratio = [1] * 2 + [imb_ratio]

        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features

    if(dataname in ['Cora_ML', 'DBLP']):
        dataset = get_citationalfull_dataset(
            dataname, transform=T.NormalizeFeatures())
        class_sample_num = 2
        if(dataname == 'Cora_ML'):
            ratio = [1] * 5 + [imb_ratio] * 2
        if(dataname == 'DBLP'):
            ratio = [1] * 3 + [imb_ratio]

        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features

    if(dataname in ['computers', 'photo']):
        dataset = get_amazon_dataset(dataname, transform=T.NormalizeFeatures())
        if(dataname == 'computers'):
            class_sample_num = 50
        if(dataname == 'photo'):
            class_sample_num = 30
        classes = torch.unique(dataset.data.y)
        ratio = [((dataset.data.y == classes[i].item()).sum() /
                  dataset.data.y.size(0)).item() for i in range(len(classes))]

        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features

    if(dataname in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']):
        dataset = get_twitch_dataset(
            dataname, transform=T.NormalizeFeatures())
        class_sample_num = 2
        classes = torch.unique(dataset.data.y)
        if(dataname == 'PT'):
            ratio = [1] * 1 + [imb_ratio] * 1

        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features

    c_train_num = [int(ratio[i] * class_sample_num) for i in range(len(ratio))]

    print('homophily', (dataset.data.y[dataset.data.edge_index[0]] == dataset.data.y[dataset.data.edge_index[1]]).sum()/len(dataset.data.edge_index[0]))


    return data, class_sample_num, num_classes, num_features, torch.tensor(c_train_num), torch.arange(0, num_classes)


def shuffle(data, num_classes, num_training_nodes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index, rest_index = [], []
    for i in range(num_classes):
        train_index.append(indices[i][:num_training_nodes[i]])
        rest_index.append(indices[i][num_training_nodes[i]:])

    train_index = torch.cat(train_index, dim=0)
    rest_index = torch.cat(rest_index, dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data

