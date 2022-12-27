"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'load_dataset': bool,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'time_steps': int,
        'tt_split': float,
        'dataset_creation': str,
        'exp_type': str,
        'paths': {
            'train': str,
            'validation': str,
            'logs': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'batch_size': int,
        'loss': str,
        'save_model': bool,
        'exp_type': str,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
    'model': {
        'dropout': float,
        'num_hid_mlp': int,
        'num_hid_cnn': int,
        'exp_type': str,
        'pooling': int,
        'res': int,
        'conv_params': {
            'kernel_size': int,
            'stride': int,
            'padding': int,
        },
    },
}
