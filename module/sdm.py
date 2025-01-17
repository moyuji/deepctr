import pytorch_lightning as pl
from deepctr.preprocessing.inputs import combined_dnn_input, compute_input_dim
from deepctr.layers.core import SparseEncoding
from deepctr.preprocessing.utils import dot_similarity
import torch
import torch.nn as nn
from torch.nn import functional as F

class SDM(pl.LightningModule):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None, output_dim=1024,
                 item_norm_weight_start=0.0, user_norm_weight_start=0.0, item_norm_weight_end=0.0, user_norm_weight_end=0.0,
                 norm_weight_warmup=1, gating_warmup=5, beta=0.1, momentum=0.1):
        super(SDM, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        if len(user_dnn_feature_columns) > 0:
            self.user_dnn = SparseEncoding(compute_input_dim(user_dnn_feature_columns), dnn_hidden_units, beta=beta,
                                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                           use_bn=dnn_use_bn, init_std=init_std, device=device, output_dim=output_dim,
                                           norm_weight=user_norm_weight_start, momentum=momentum)
            self.user_dnn_embedding = None

        if len(item_dnn_feature_columns) > 0:
            self.item_dnn = SparseEncoding(compute_input_dim(item_dnn_feature_columns), dnn_hidden_units, beta=beta,
                                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                           use_bn=dnn_use_bn, init_std=init_std, device=device, output_dim=output_dim,
                                           norm_weight=item_norm_weight_start, momentum=momentum)
            self.item_dnn_embedding = None
        self.item_norm_weight_inc = 1.0 * (item_norm_weight_end - item_norm_weight_start) / norm_weight_warmup
        self.user_norm_weight_inc = 1.0 * (user_norm_weight_end - user_norm_weight_start) / norm_weight_warmup
        self.norm_weight_warmup = norm_weight_warmup
        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus
        self.gating_warmup = gating_warmup

    def forward(self, inputs):
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            self.user_dnn_embedding = self.user_dnn(user_dnn_input)

        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            self.item_dnn_embedding = self.item_dnn(item_dnn_input)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            score = dot_similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
            output = self.out(score)
            return output

        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding

        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def loss(self, batch, export=False):
        x_train, y_train = batch
        x = x_train.float()
        y = y_train.float()
        y_pred = self(x)
        losses = {}
        loss = F.binary_cross_entropy(y_pred, y)
        losses['loss'] = loss
        return losses

    def validation_step(self, batch, batch_idx):
        losses = self.loss(batch, export=True)
        self.logall('val', losses)
        return losses
        
    def training_step(self, batch, batch_idx):
        losses = self.loss(batch)
        self.logall('train', losses)
        return losses

    def logall(self, prefix, losses):
        for k, v in losses.items():
            self.log(f'{prefix}_{k}', v)    

    def validation_epoch_end(self):
        epoch = self.current_epoch
        if epoch < self.norm_weight_warmup:
            self.item_dnn.norm_weight += self.item_norm_weight_inc
            self.user_dnn.norm_weight += self.user_norm_weight_inc
            # print(self.item_dnn.norm_weight, self.user_dnn.norm_weight)
        if epoch == self.gating_warmup:
            self.item_dnn.gate = False
            self.user_dnn.gate = False
