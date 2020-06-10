import torch
import torch.nn as nn
import torch.nn.functional as F


def load_criterion(cfg):
    def get_criterion(name):
        if name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError()

    crittype_train = cfg['training']['train_criterion']
    crittype_test = cfg['training']['test_criterion']

    train_criterion = get_criterion(crittype_train)
    test_criterion = get_criterion(crittype_test)

    return train_criterion, test_criterion