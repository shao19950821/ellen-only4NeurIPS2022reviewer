import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Ellen example')
    parser.add_argument('--model', type=str, default='Ellen', help='use model',
                        choices=['Ellen'])
    parser.add_argument('--dataset', type=str, default='criteo', help='use dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--nas', type=int, default=1, help='nas period: 1 use 0 not used')
    parser.add_argument('--fis', type=int, default=1, help='fis period: 1 use 0 not used')
    parser.add_argument('--mlp', type=int, default=1, help='use mlp predict: 1 use 0 not used')
    parser.add_argument('--retrain_nas', type=int, default=0, help='retrain nas period: 1 retrain 0 not retrain')
    parser.add_argument('--retrain_fis', type=int, default=0, help='retrain fis period: 1 retrain 0 not retrain')
    parser.add_argument('--mutation', type=int, default=1, help='use mutation: 1 use 0 not used')
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'criteo':
        from run.run_criteo_ellen import train
    elif dataset == 'avazu':
        from run.run_avazu_ellen import train
    elif dataset == 'huawei':
        from run.run_huawei_ellen import train
    elif dataset == 'fintech':
        from run.run_fintech_ellen import train
    train(params=args)
