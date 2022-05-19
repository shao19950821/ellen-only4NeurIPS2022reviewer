General_Config = {
    'general': {
        'batch_size': 2000,
        'data': 10000,
        'epochs': 1,
        'validation_split': 0.11,
        'net_optim_lr': 1e-3,
        'embedding_size': 15,
    },
}
Ellen_Config = {
    'nas': {
        'momentum': 0.9,
        'net_optim_lr': 0.025,
        'net_optim_weight_decay': 3e-4,
        'structure_optim_lr': 3e-4,
        'structure_optim_weight_decay': 1e-3,
        'interaction_fc_output_dim': 1,
        'validation_split': 0.5,
        'mutation_threshold': 0.005
    },
    'fis': {
        'c': 0.5,
        'mu': 0.8,
        'gRDA_optim_lr': 1e-3,
        'net_optim_lr': 1e-3,
        'interaction_fc_output_dim': 1,
        'validation_split': 0.11,
        'mutation_threshold': 0.5,
    },
    'pre': {
        'interaction_fc_output_dim': 15,
        'dnn_hidden_units': [400, 400],
    }
}