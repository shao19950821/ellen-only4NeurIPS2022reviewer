import os
import time
import torch
import logging
import pickle as pkl
import numpy as np
from utils.function_utils import get_feature_names
from config.configs import Ellen_Config, General_Config
from utils.function_utils import get_param_sum
from model.preModel import PreModel
from sklearn.metrics import log_loss, roc_auc_score


def predict_period_train_and_test(feature_columns, feature_index, data_train, data_test, param_save_dir, label_name,
                                  device='cpu', use_nas=True, use_fis=True, use_mlp=True):
    logging.info('pre period param:')
    logging.info(Ellen_Config['pre'])
    feature_names = get_feature_names(feature_columns)
    if use_nas:
        selected_interaction_type = pkl.load(open(os.path.join(param_save_dir, 'interaction_type-embedding_size-' + str(
            General_Config['general']['embedding_size']) + '.pkl'), 'rb'))
    else:
        feature_num = len(feature_columns)
        interaction_pair_num = int((feature_num * (feature_num - 1)) / 2)
        selected_interaction_type = np.random.randint(low=0, high=4, size=interaction_pair_num)
        logging.info("generate interaction type randomly")
    logging.info(selected_interaction_type)
    if use_fis:
        checkpoint = torch.load(
            os.path.join(param_save_dir,
                         'alpha_beta-c' + str(Ellen_Config['fis']['c']) + '-mu' + str(
                             Ellen_Config['fis']['mu']) + '-embedding_size' + str(
                             General_Config['general']['embedding_size']) + '.pth'))
        alpha = checkpoint['alpha']
        beta = checkpoint['beta']
        logging.info(alpha)
        logging.info(beta)
        preModel = PreModel(feature_columns=feature_columns, feature_index=feature_index,
                            selected_interaction_type=selected_interaction_type,
                            interaction_fc_output_dim=Ellen_Config['pre']['interaction_fc_output_dim'],
                            alpha=alpha, beta=beta,
                            embedding_size=General_Config['general']['embedding_size'],
                            device=device, use_mlp=use_mlp)
    else:
        preModel = PreModel(feature_columns=feature_columns, feature_index=feature_index,
                            selected_interaction_type=selected_interaction_type,
                            interaction_fc_output_dim=Ellen_Config['pre']['interaction_fc_output_dim'],
                            embedding_size=General_Config['general']['embedding_size'],
                            device=device, use_mlp=use_mlp)
    train_model_input = {name: data_train[name] for name in feature_names}
    test_model_input = {name: data_test[name] for name in feature_names}
    logging.info("predict period start")
    preModel.to(device)
    preModel.before_train()
    start_time = time.time()
    get_param_sum(model=preModel)
    preModel.fit(train_model_input, data_train[label_name].values, batch_size=General_Config['general']['batch_size'],
                 epochs=General_Config['general']['epochs'],
                 validation_split=General_Config['general']['validation_split'])
    predict_result = preModel.predict(test_model_input, 256)
    logging.info("test LogLoss:{}".format(round(log_loss(data_test[label_name].values, predict_result), 4)))
    logging.info("test AUC:{}".format(round(roc_auc_score(data_test[label_name].values, predict_result), 4)))
    end_time = time.time()
    cost_time = (int)(end_time - start_time)
    logging.info("predict period end")
    logging.info('predict period cost:' + str(cost_time))