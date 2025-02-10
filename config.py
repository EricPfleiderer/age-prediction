import torch.optim

config = {
    'settings': {
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
    },

    'data_space': {
        'log_transform': False,
    },

    'training_space': {
        'epochs': 250,
        'batch_size': 256,
        'optimizer': torch.optim.Adam,
        'optimizer_space': {
            'lr': 0.0001,
        },
    }
}


def confirm_config(config):
    assert config['settings']['train_size'] + config['settings']['val_size'] + config['settings']['train_size'] + \
           config['settings']['test_size'] == 1