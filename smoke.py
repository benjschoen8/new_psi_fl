"""Small real training workload for CPU integration checks, not research data."""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from client import Client


class TinyGenerator(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.embedding = nn.Embedding(count, 2)
        self.linear = nn.Linear(6, 12)

    def forward(self, noise, labels):
        return self.linear(torch.cat((noise, self.embedding(labels)), dim=1)).tanh().reshape(-1, 3, 2, 2)


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2, 2)
        self.linear = nn.Linear(14, 1)

    def forward(self, images, labels):
        return self.linear(torch.cat((images.flatten(1), self.embedding(labels)), dim=1))


class TinyClassifier(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.linear = nn.Linear(12, count)

    def forward(self, images):
        features = images.flatten(1)
        return features, self.linear(features)


def build_smoke_clients(config, device='cpu'):
    clients, spaces, tests, metadata = [], {}, [], {}
    for cid in range(2):
        # Equal data gives PACFL a well-defined compatible cluster.
        images = torch.linspace(-1, 1, 96).reshape(8, 3, 2, 2)
        labels = torch.tensor([0, 1] * 4)
        train = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=True)
        test = DataLoader(TensorDataset(images.flip(0), labels.flip(0)), batch_size=4)
        clients.append(Client(cid, TinyClassifier(2), TinyGenerator(2), TinyDiscriminator(),
                              train, 2, config, device))
        spaces[cid] = ('zero', 'one')
        tests.append(('synthetic', cid, test))
        metadata[cid] = dict(dataset='synthetic', labels=spaces[cid], architecture='tiny',
                             train_indices=list(range(8)), test_indices=list(range(8)))
    return clients, spaces, tests, metadata, TinyGenerator, TinyClassifier

