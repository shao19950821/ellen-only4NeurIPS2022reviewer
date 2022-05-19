import os
import torch
import logging
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from config.configs import Ellen_Config, General_Config
from utils.function_utils import generate_pair_index, create_embedding_matrix
from .baseModel import BaseModel
from utils.function_utils import slice_arrays
from sklearn.metrics import *
from layer.interactionLayer import MixedOp, Interaction_Types
from model.architect import Architect


class NasModel(BaseModel):
    def __init__(self, feature_columns, feature_index, interaction_fc_output_dim=1, embedding_size=20,
                 seed=1024, device='cpu'):
        super(NasModel, self).__init__()
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.embedding_size = embedding_size
        self.device = device
        if device == 'cpu':
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        self.register_buffer('pair_indexes',
                             torch.tensor(generate_pair_index(len(self.feature_columns), 2)))
        self.interaction_pair_number = len(self.pair_indexes[0])
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std=0.001, sparse=False)
        self.structure_param = self.create_structure_param(length=self.interaction_pair_number,
                                                           init_mean=0.5,
                                                           init_radius=0.001)
        self.mixed_operation = MixedOp(input_dim=embedding_size,
                                       output_dim=interaction_fc_output_dim, device=device)

    def forward(self, input):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            input[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        embedding_matrix = torch.cat(embedding_list, dim=1)
        feat_i, feat_j = self.pair_indexes
        embedding_matrix_i = torch.index_select(embedding_matrix, 1, feat_i)
        embedding_matrix_j = torch.index_select(embedding_matrix, 1, feat_j)
        weights = F.softmax(self.structure_param, dim=-1)
        batch_product_matrix = self.mixed_operation(embedding_matrix_i, embedding_matrix_j, weights)
        batch_product_vector = torch.flatten(batch_product_matrix, start_dim=-2, end_dim=-1)
        out = torch.sum(batch_product_vector, dim=1)
        return torch.sigmoid(out)

    def create_structure_param(self, length, init_mean, init_radius):
        num_ops = len(Interaction_Types)
        structure_param = nn.Parameter(
            torch.empty((length, num_ops)).uniform_(
                init_mean - init_radius,
                init_mean + init_radius))
        structure_param.requires_grad = True
        return structure_param

    def new(self):
        model_new = NasModel(feature_columns=self.feature_columns, feature_index=self.feature_index,
                             embedding_size=self.embedding_size, device=self.device)
        for x, y in zip(model_new.structure_param, self.structure_param):
            x.data.copy_(y.data)
        model_new.before_train()
        return model_new.to(self.device)

    def get_loss(self, input, target):
        y_pred = self(input).squeeze()
        return self.loss_func(y_pred, target.squeeze(), reduction='sum')

    def before_train(self):
        self.metrics_names = ["loss"]
        all_parameters = self.parameters()
        structure_param = set([self.structure_param])
        self.net_param = [i for i in all_parameters if i not in structure_param]
        self.structure_optim = self.get_structure_optim(structure_param)
        self.net_optim = self.get_net_optim(self.net_param)
        self.loss_func = F.binary_cross_entropy
        self.metrics = self.get_metrics(["binary_crossentropy", "auc"])

    def get_structure_optim(self, structure_params):
        optimizer = torch.optim.Adam(structure_params,
                                     lr=Ellen_Config['nas']['structure_optim_lr'], betas=(0.5, 0.999),
                                     weight_decay=Ellen_Config['nas']['structure_optim_weight_decay'])
        return optimizer

    def get_net_optim(self, net_params):
        optimizer = torch.optim.Adam(net_params,
                                     lr=Ellen_Config['nas']['structure_optim_lr'], betas=(0.5, 0.999),
                                     weight_decay=Ellen_Config['nas']['structure_optim_weight_decay'])
        return optimizer

    def get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def fit(self, x=None, y=None, batch_size=256

            , epochs=1, initial_epoch=0, validation_split=0.,
            shuffle=True):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        do_validation = False
        if validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        for i in range(len(val_x)):
            if len(val_x[i].shape) == 1:
                val_x[i] = np.expand_dims(val_x[i], axis=1)
        train_tensor_data = TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y))
        valid_tensor_data = TensorDataset(torch.from_numpy(np.concatenate(val_x, axis=-1)), torch.from_numpy(val_y))
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)
        valid_loader = DataLoader(
            dataset=valid_tensor_data, shuffle=shuffle, batch_size=batch_size)
        architect = Architect(self, self.structure_optim)
        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        print("nas period")
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            epoch_start_time = time.time()
            epoch_logs = {}
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for index, (x_train, y_train) in t:
                        x_valid, y_valid = next(iter(valid_loader))
                        x_valid = x_valid.to(self.device).float()
                        y_valid = y_valid.to(self.device).float()
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        architect.step(x, y, x_valid, y_valid,
                                       Ellen_Config['nas']['structure_optim_lr'], net_optim)
                        y_pred = model(x).squeeze()
                        net_optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        total_loss_epoch += loss.item()
                        loss.backward()
                        net_optim.step()
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s - loss: {1: .4f}".format(
                epoch_time, epoch_logs["loss"])

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            logging.info(eval_str)

    def after_train(self, param_save_dir):
        prob = F.softmax(self.structure_param, dim=-1)
        prob_ndarray = prob.cpu().detach().numpy()
        selected_interaction_type = np.argmax(prob_ndarray, axis=1)
        param_save_file_path = os.path.join(param_save_dir, 'interaction_type-embedding_size-' + str(
            General_Config['general']['embedding_size']) + '.pkl')
        with open(param_save_file_path, 'wb') as f:
            pkl.dump(selected_interaction_type, f)
