import torch
import torch.nn as nn
from lib.unet1d import UNet1D


def load_model(cfg, load_model_only=False):
    modeltype = cfg['training']['model']
    in_channels = len(cfg['data']['input_channels'])
    out_channels = cfg['data']['n_output_channels']
    dp = cfg['training']['data_parallel']

    if modeltype == 'unet1d':
        model = UNet1D(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model type {modeltype}")

    if dp:
        model = nn.DataParallel(model).to(cfg['training']['device'])
        m = model.module
    else:
        m = model

    if load_model_only:
        return model

    modelpath = cfg['resume'].get('path', None)
    config_epoch = cfg['resume'].get('epoch', None)

    if modelpath:
        state = torch.load(modelpath)
        m.load_state_dict(state['state_dict'])
        starting_epoch = state['epoch']
        if config_epoch:
            print(f"WARNING: Loaded model trained for {starting_epoch - 1} epochs but config explicitly overrides to {config_epoch}")
            starting_epoch = config_epoch
    else:
        starting_epoch = 1
        state = {}

    return model, starting_epoch, state