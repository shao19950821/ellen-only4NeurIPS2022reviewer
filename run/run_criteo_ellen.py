import time
import logging
import torch
import pickle as pkl
import pandas as pd
from data.criteoPreprocess import CriteoProcessor
from data.featureDefiniton import SparseFeat, DenseBucketFeat
from config.configs import General_Config
from utils.function_utils import build_input_features, log
from trainer.ellen_s1_nas_trainer import model_search
from trainer.ellen_s2_fis_trainer import feature_interaction_search
from trainer.ellen_s3_pre_trainer import predict_period_train_and_test

param_save_dir = '../param/criteo'


def train(params):
    log(dataset=params.dataset, model=params.model)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    dataProcessor = CriteoProcessor()
    train_data_file_path = pkl.load(open(dataProcessor.processed_full_train_data_file_path, 'rb'))
    test_data_file_path = pkl.load(open(dataProcessor.processed_full_test_data_file_path, 'rb'))
    feat_size = pkl.load(open(dataProcessor.feature_size_file_path, 'rb'))
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    fixlen_feature_columns = [SparseFeat(feat, feat_size[feat], General_Config['general']['embedding_size']) for feat
                              in sparse_features] \
                             + [DenseBucketFeat(feat, feat_size[feat], General_Config['general']['embedding_size'])
                                for feat in dense_features]
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:' + str(params.gpu)
    feature_index = build_input_features(fixlen_feature_columns)
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
                 label_name='label', device=device, use_nas=use_nas, retrain=retrain_nas)
    use_fis = bool(params.fis)
    mutation = bool(params.mutation)
    feature_interaction_search(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                               data_train=data_train, param_save_dir=param_save_dir,mutation=mutation,
                               label_name='label', device=device, use_fis=use_fis, retrain=retrain_fis)
    use_mlp = bool(params.mlp)
    predict_period_train_and_test(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                                  data_train=data_train, data_test=data_test,
                                  label_name='label', param_save_dir=param_save_dir,
                                  device=device, use_nas=use_nas, use_fis=use_fis, use_mlp=use_mlp)
