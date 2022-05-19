import os
import time
import logging
import pickle as pkl
from utils.function_utils import get_feature_names
from config.configs import Ellen_Config, General_Config
from utils.function_utils import get_param_sum
from model.fisModel import FisModel


def feature_interaction_search(feature_columns, feature_index, data_train, param_save_dir, label_name,mutation=False,
                               device='cpu', use_fis=True, retrain=False):
    logging.info('fis period param:')
    logging.info(Ellen_Config['fis'])
    if use_fis:
        if retrain:
            feature_names = get_feature_names(feature_columns)
            selected_interaction_type = pkl.load(open(os.path.join(param_save_dir,
                                                                   'interaction_type-embedding_size-' + str(
                                                                       General_Config['general'][
                                                                           'embedding_size']) + '.pkl'), 'rb'))
            train_model_input = {name: data_train[name] for name in feature_names}
            logging.info('feature interaction search period start')
            fisModel = FisModel(feature_columns=feature_columns, feature_index=feature_index,
                                selected_interaction_type=selected_interaction_type,
                                mutation=mutation,mutation_threshold=Ellen_Config['fis']['mutation_threshold'],
                                interaction_fc_output_dim=Ellen_Config['fis']['interaction_fc_output_dim'],
                                embedding_size=General_Config['general']['embedding_size'],
                                device=device)
            fisModel.to(device)
            fisModel.before_train()
            start_time = time.time()
            get_param_sum(model=fisModel)
            fisModel.fit(train_model_input, data_train[label_name].values,
                         batch_size=General_Config['general']['batch_size'],
                         epochs=General_Config['general']['epochs'],
                         validation_split=Ellen_Config['fis']['validation_split'])
            fisModel.after_train(param_save_dir=param_save_dir)
            end_time = time.time()
            cost_time = (int)(end_time - start_time)
            logging.info('feature interaction search period end')
            logging.info('feature interaction search period cost:' + str(cost_time))
        else:
            logging.info('feature interaction search has done, result has generated')
    else:
        logging.info('no feature interaction search period')
