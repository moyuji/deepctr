from deepctr.model.base_tower import BaseTower
from deepctr.preprocessing.inputs import combined_dnn_input, compute_input_dim
from deepctr.layers.core import SparseEncoding
from deepctr.preprocessing.utils import dot_similarity
import numpy as np

class SDM(BaseTower):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None, output_dim=1024,
                 item_norm_weight_start=0.0, user_norm_weight_start=0.0, item_norm_weight_end=0.0, user_norm_weight_end=0.0,
                 norm_weight_warmup=1, gating_warmup=5, beta=0.1, momentum=0.1, logger=None, nd_sample=False):
        super(SDM, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus, logger=logger)
        if len(user_dnn_feature_columns) > 0:
            self.user_dnn = SparseEncoding(compute_input_dim(user_dnn_feature_columns), dnn_hidden_units, beta=beta,
                                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                           use_bn=dnn_use_bn, init_std=init_std, device=device, output_dim=output_dim,
                                           norm_weight=user_norm_weight_start, momentum=momentum, nd_sample=nd_sample)
            self.user_dnn_embedding = None

        if len(item_dnn_feature_columns) > 0:
            self.item_dnn = SparseEncoding(compute_input_dim(item_dnn_feature_columns), dnn_hidden_units, beta=beta,
                                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                           use_bn=dnn_use_bn, init_std=init_std, device=device, output_dim=output_dim,
                                           norm_weight=item_norm_weight_start, momentum=momentum, nd_sample=nd_sample)
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

    def val_epoch_end(self, epoch):
        if epoch < self.norm_weight_warmup:
            self.item_dnn.norm_weight += self.item_norm_weight_inc
            self.user_dnn.norm_weight += self.user_norm_weight_inc
            # print(self.item_dnn.norm_weight, self.user_dnn.norm_weight)
        if epoch == self.gating_warmup:
            self.item_dnn.gate = False
            self.user_dnn.gate = False

        item_embedding=np.abs(self.item_dnn_embedding.cpu().data.numpy())
        user_embedding=np.abs(self.user_dnn_embedding.cpu().data.numpy())
        self.log('val/ir', np.mean(item_embedding / (item_embedding+0.00000001)))
        self.log('val/ur', np.mean(user_embedding / (user_embedding+0.00000001)))
        v = np.matmul(user_embedding, item_embedding.T)
        self.log('val/cb', np.mean(v / (v+0.00000001)))

    def train_epoch_end(self, epoch):
        item_embedding=np.abs(self.item_dnn_embedding.cpu().data.numpy())
        user_embedding=np.abs(self.user_dnn_embedding.cpu().data.numpy())
        self.log('train/ir', np.mean(item_embedding / (item_embedding+0.00000001)))
        self.log('train/ur', np.mean(user_embedding / (user_embedding+0.00000001)))
        v = np.matmul(user_embedding, item_embedding.T)
        self.log('train/cb', np.mean(v / (v+0.00000001)))