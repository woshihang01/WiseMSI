import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):

    def __init__(self, attention_mechanism='dot',  L=1024, D=256, dropout=False, n_tasks=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_mechanism = attention_mechanism
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)

        if self.attention_mechanism == 'dot':
            A = a.mul(b)

        elif self.attention_mechanism == 'cosine':
            A = torch.unsqueeze(torch.cosine_similarity(a, b), 1)  # N
        else:
            raise Exception('model is error')
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""


class TOAD_fc_mtl_concat(nn.Module):
    def __init__(self, model_type='toad', size_arg="big", dropout=False, n_classes=2):
        super(TOAD_fc_mtl_concat, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))

        attention_net = Attn_Net_Gated(attention_mechanism='dot' if model_type=='toad' else 'cosine', L=size[1], D=size[2], dropout=dropout, n_tasks=1)
        self.classifier = nn.Linear(size[1], n_classes)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, h, return_features=False, attention_only=False):

        A, h = self.attention_net(h)  # h: N * n_dim (N个patch，K个特征维度)

        A = torch.transpose(A, 1, 0)  # A: N * 1    ->     1 * N
        if attention_only:
            return A[0]
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        # M = torch.cat([M, sex.repeat(M.size(0),1)], dim=1)

        logits = self.classifier(M[0].unsqueeze(0))
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A})

        return results_dict
