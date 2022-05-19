import time
import logging
import torch
import pickle as pkl
import pandas as pd
from data.fintechPreprocess import FintechPreprocess
from data.featureDefiniton import SparseFeat, DenseBucketFeat
from config.configs import General_Config
from utils.function_utils import build_input_features, log
from trainer.ellen_s1_nas_trainer import model_search
from trainer.ellen_s2_fis_trainer import feature_interaction_search
from trainer.ellen_s3_pre_trainer import predict_period_train_and_test

param_save_dir = '../param/fintech'
dense_features = ['h2q7','h2q6','g1k3','c1z1','h1l3','e1g5','g1i1','d1j9','g1s1','b1u7','h1i3','d1m3','h1k7','d1g2','d1m6',
                 'd1d5','d1d4','a1s7','d1e1']
sparse_features = ['e1g6','h1w5','a1y2','e1g4','h2h4','a1t3','a1t5']


def train(params):
    log(dataset=params.dataset, model=params.model)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    dataProcessor = FintechPreprocess()
    train_data_file_path = pkl.load(open(dataProcessor.processed_full_train_data_file_path, 'rb+'))
    test_data_file_path = pkl.load(open(dataProcessor.processed_full_test_data_file_path, 'rb+'))
    feat_size = pkl.load(open(dataProcessor.feature_size_file_path, 'rb'))
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
                 label_name='y4_if_deal', device=device, use_nas=use_nas, retrain=retrain_nas)
    use_fis = bool(params.fis)
    mutation = bool(params.mutation)
    feature_interaction_search(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                               data_train=data_train, param_save_dir=param_save_dir,mutation=mutation,
                               label_name='y4_if_deal', device=device, use_fis=use_fis, retrain=retrain_fis)
    use_mlp = bool(params.mlp)
    predict_period_train_and_test(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                                  data_train=data_train, data_test=data_test,
                                  label_name='y4_if_deal', param_save_dir=param_save_dir,
                                  device=device, use_nas=use_nas, use_fis=use_fis, use_mlp=use_mlp)
