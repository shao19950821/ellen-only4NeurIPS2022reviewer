import logging
import time
import torch
import pickle as pkl
import pandas as pd
from data.avazuPreprocess import AvazuPreprocess
from data.featureDefiniton import SparseFeat
from config.configs import General_Config
from utils.function_utils import build_input_features, log
from trainer.ellen_s1_nas_trainer import model_search
from trainer.ellen_s2_fis_trainer import feature_interaction_search
from trainer.ellen_s3_pre_trainer import predict_period_train_and_test

feat_names = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
              'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15',
              'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
param_save_dir = '../param/avazu'

def train(params):
    log(dataset=params.dataset, model=params.model)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    avazuData = AvazuPreprocess()
    train_data_file_path = pkl.load(open(avazuData.train_path, 'rb'))
    test_data_file_path = pkl.load(open(avazuData.test_path, 'rb'))
    feat_size = pkl.load(open(avazuData.feature_size_file_path, 'rb'))
    fixlen_feature_columns = [SparseFeat(feat_name, feat_size[feat_name], General_Config['general']['embedding_size'])
                              for feat_name in feat_names]
    feature_index = build_input_features(fixlen_feature_columns)
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        logging.info('cuda ready...')
        device = 'cuda:' + str(params.gpu)
    logging.info(params)
    logging.info(General_Config)
    if (General_Config['general']['data'] == -1):
        data_train = pd.read_csv(train_data_file_path)
        data_test = pd.read_csv(test_data_file_path)
    else:
        data_train = pd.read_csv(train_data_file_path, nrows=General_Config['general']['data'])
        data_test = pd.read_csv(test_data_file_path, nrows=General_Config['general']['data'])


    retrain_nas = bool(params.retrain_nas)
    retrain_fis = bool(params.retrain_fis)
    use_nas = bool(params.nas)
    model_search(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                 data_train=data_train, param_save_dir=param_save_dir,
                 label_name='click',
                 device=device, use_nas=use_nas, retrain=retrain_nas)
    use_fis = bool(params.fis)
    mutation = bool(params.mutation)
    feature_interaction_search(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                               data_train=data_train, param_save_dir=param_save_dir,mutation=mutation,
                               label_name='click',
                               device=device, use_fis=use_fis, retrain=retrain_fis)
    use_mlp = bool(params.mlp)
    predict_period_train_and_test(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                                  data_train=data_train, data_test=data_test,
                                  label_name='click', param_save_dir=param_save_dir,
                                  device=device, use_nas=use_nas, use_fis=use_fis, use_mlp=use_mlp)