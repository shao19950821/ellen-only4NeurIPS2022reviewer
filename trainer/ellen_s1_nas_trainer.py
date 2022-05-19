import time
import logging
from utils.function_utils import get_feature_names
from config.configs import Ellen_Config, General_Config
from utils.function_utils import get_param_sum
from model.nasModel import NasModel


def model_search(feature_columns, feature_index, data_train, param_save_dir, label_name, device='cpu',
                 use_nas=True, retrain=False):
    logging.info('nas period param:')
    logging.info(Ellen_Config['nas'])
    if use_nas:
        if retrain:
            logging.info('model search period start')
            feature_names = get_feature_names(feature_columns)
            train_model_input = {name: data_train[name] for name in feature_names}
            print("epoch num:{}".format(General_Config['general']['epochs']))
            nasModel = NasModel(feature_columns=feature_columns, feature_index=feature_index,
                                interaction_fc_output_dim=Ellen_Config['nas']['interaction_fc_output_dim'],
                                embedding_size=General_Config['general']['embedding_size'],
                                device=device)
            nasModel.to(device)
            nasModel.before_train()
            start_time = time.time()
            get_param_sum(model=nasModel)
            nasModel.fit(train_model_input, data_train[label_name].values,
                         batch_size=General_Config['general']['batch_size'],
                         epochs=General_Config['general']['epochs'],
                         validation_split=Ellen_Config['nas']['validation_split'])
            nasModel.after_train(param_save_dir=param_save_dir)
            end_time = time.time()
            cost_time = (int)(end_time - start_time)
            logging.info('model search period end')
            logging.info('model search period cost:' + str(cost_time))
        else:
            logging.info('model search has done, result has generated')
    else:
        logging.info('no model search period')
        logging.info('generate feature interaction type randomly')
