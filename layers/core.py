import torch
import torch.nn as nn
from deepctr.layers.activation import activation_layer
from torch.nn import functional as F


class PredictionLayer(nn.Module):
    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))
            self.scale = nn.Parameter(torch.ones((1,)))

    def forward(self, X):
        if self.use_bias:
            output = X * self.scale + self.bias
        else:
            output = X
        if self.task == "binary":
            output = torch.sigmoid(X)
        return output


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0.0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        if inputs_dim > 0:
            hidden_units = [inputs_dim] + list(hidden_units)
        else:
            hidden_units = list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0,
                 dice_dim=3, l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavier):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavier_len = user_behavier.size(1)

        queries = query.expand(-1, user_behavier_len, -1)

        attention_input = torch.cat([queries, user_behavier, queries - user_behavier, queries * user_behavier],
                                    dim=-1)  # [B, T, 4*E]
        attention_out = self.dnn(attention_input)

        attention_score = self.dense(attention_out)  # [B, T, 1]

        return attention_score


class SparseEncoding(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0.0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu', output_dim=4, norm_weight=0.0, beta=0.1,
                 low=-0.1, high=1.1):
        super(SparseEncoding, self).__init__()
        last_hidden_dim = hidden_units[-1]
        self.seed = seed
        self.norm = nn.BatchNorm1d(output_dim, affine=False, momentum=0.5)
        self.norm_weight = norm_weight
        self.shared = DNN(inputs_dim=inputs_dim,
                          hidden_units=hidden_units,
                          activation=activation,
                          l2_reg=l2_reg,
                          dropout_rate=dropout_rate,
                          dice_dim=dice_dim,
                          use_bn=use_bn)
        # self.reg_tower = DNN(inputs_dim=last_hidden_dim,
        #                      hidden_units=(output_dim,),
        #                      activation="linear",
        #                      l2_reg=l2_reg,
        #                      dropout_rate=0.0,
        #                      dice_dim=dice_dim,
        #                      use_bn=use_bn)
        # self.embed_tower = DNN(inputs_dim=last_hidden_dim,
        #                        hidden_units=(output_dim,),
        #                        activation="linear",
        #                        l2_reg=l2_reg,
        #                        dropout_rate=0.0,
        #                        dice_dim=dice_dim,
        #                        use_bn=use_bn)
        self.embed_tower = nn.Sequential(
            nn.ReLU(),
            nn.Linear(last_hidden_dim, output_dim),
        )
        self.reg_tower = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(last_hidden_dim, output_dim),
        )
        self.beta = beta
        self.high = high
        self.low = low
        self.to(device)
        self.gate = True

    def forward(self, inputs):
        import numpy as np
        import random
        shared = self.shared(inputs)
        embedding = self.embed_tower(shared)
        embedding = nn.functional.normalize(embedding)
        if self.gate:
            return embedding
        alpha0 = self.reg_tower(shared)
        alpha0 = self.norm(alpha0)
        alpha = alpha0 + self.norm_weight
        weight = self.sample_attention(alpha)
        if random.random() < 0.01:
            print(f'{self.norm_weight} alpha0 {np.mean(alpha0.cpu().detach().numpy())} alpha {np.mean(alpha.cpu().detach().numpy())} weight {np.mean(weight.cpu().detach().numpy())}')
        embedding = embedding * weight
        return embedding

    def sample_attention(self, weights):
        if self.training:
            eps = torch.rand_like(weights)
            s = torch.sigmoid((torch.log(eps) - torch.log(1.0 - eps) + weights) / self.beta)
        else:
            s = torch.sigmoid(weights / 0.001)
        s = s * (self.high - self.low) + self.low
        return F.hardtanh(s, min_val=0, max_val=1)
