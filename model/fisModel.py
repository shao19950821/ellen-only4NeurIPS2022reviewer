import os
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimizer.gRDA import gRDA
from .baseModel import BaseModel
from layer.interactionLayer import InteractionLayer
from layer.linearLayer import NormalizedWeightedLinearLayer
from utils.function_utils import generate_pair_index, slice_arrays
from sklearn.metrics import *
from config.configs import Ellen_Config, General_Config


class FisModel(BaseModel):
    def __init__(self, feature_columns, feature_index, selected_interaction_type, interaction_fc_output_dim=1,
                 embedding_size=20,mutation=False,mutation_threshold=0.5,
                 activation='tanh', seed=1024, device='cpu'):
        super(FisModel, self).__init__()
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
        self.linear = NormalizedWeightedLinearLayer(feature_columns=feature_columns, feature_index=feature_index,
                                                    alpha_activation=activation, use_alpha=True,
                                                    device=device)
        self.interaction_operation = InteractionLayer(input_dim=embedding_size, feature_columns=feature_columns,
                                                      feature_index=feature_index, use_beta=True,
                                                      interaction_fc_output_dim=interaction_fc_output_dim,
                                                      selected_interaction_type=selected_interaction_type,
                                                      mutation_threshold=mutation_threshold,
                                                      device=device)
        self.mutation=mutation

    def forward(self, x, mutation=False):
        linear_out = self.linear(x)
        interation_out = self.interaction_operation(x,mutation)
        out = linear_out + interation_out
        return torch.sigmoid(out)

    def before_train(self):
        self.metrics_names = ["loss"]
        all_parameters = self.parameters()
        structure_params = set([self.linear.alpha, self.interaction_operation.beta])
        net_params = [i for i in all_parameters if i not in structure_params]
        self.structure_optim = self.get_structure_optim(structure_params)
        self.net_optim = self.get_net_optim(net_params)
        self.loss_func = F.binary_cross_entropy
        self.metrics = self.get_metrics(["binary_crossentropy", "auc"])

    def get_net_optim(self, net_params):
        optimizer = optim.Adam(net_params, lr=float(Ellen_Config['fis']['net_optim_lr']))
        return optimizer

    def get_structure_optim(self, structure_params):
        optimizer = gRDA(structure_params, lr=float(Ellen_Config['fis']['gRDA_optim_lr']),
                         c=Ellen_Config['fis']['c'], mu=Ellen_Config['fis']['mu'])
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

    def fit(self, x=None, y=None, batch_size=None, epochs=1, initial_epoch=0, validation_split=0.,
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

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("fis period")
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
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        if self.mutation and index % 100 == 0:
                            y_pred = model(x, mutation=True).squeeze()
                        else:
                            y_pred = model(x, mutation=False).squeeze()
                        net_optim.zero_grad()
                        structure_optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        total_loss_epoch += loss.item()
                        loss.backward()
                        net_optim.step()
                        structure_optim.step()
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
        state = {'alpha': self.linear.alpha,
                 'beta': self.interaction_operation.beta,
                 }
        param_save_path = os.path.join(param_save_dir,
                                       'alpha_beta-c' + str(Ellen_Config['fis']['c']) + '-mu' + str(
                                           Ellen_Config['fis']['mu']) + '-embedding_size' + str(
                                           General_Config['general']['embedding_size']) + '.pth')
        torch.save(state, param_save_path)
