import math
import itertools as it

import torch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

def load_optimizer(model, cfg, state, steps_per_epoch=None):
    resuming = cfg['resume'].get('path', False) is not False
    resetting_epoch = cfg['resume'].get('epoch', 0) == 1 and resuming
    resetting_optimizer = cfg['resume'].get('reset_optimizer', False) is not False

    # Create optimizer
    lr = cfg['training']['lr']
    wd = cfg['training']['weight_decay']
    opt = cfg['training']['optimizer']
    if opt == 'adam':
        optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer {opt}")

    # Load optimizer weights if in state dict
    opt_path = state.get('optimizer', None)
    if opt_path:
        if resetting_optimizer:
            print(f"Resetting optimizer state")
        else:
            optimizer.load_state_dict(opt_path)

    # Load scheduler if in state dict AND if we're not resetting the epoch or optimizer
    scheduler = state.get('scheduler', None)
    sched = cfg['training'].get('scheduler', None)
    if scheduler and not resetting_epoch and not resetting_optimizer:
        print(f"Loaded scheduler from state dict")
        return optimizer, scheduler

    # Otherwise create scheduler if needed

    elif sched:
        # If we are resuming but not resetting the epoch to 1, user should be warned we aren't continuing the scheduler

        if resuming and not resetting_epoch and not resetting_optimizer:
            print(f"WARNING: Resuming training from a checkpoint without resetting the epochs or optimzier, and yet no"
                  f"scheduler found - creating new scheduler")

        if sched == 'one_cycle':
            assert steps_per_epoch
            scheduler = OneCycleLR(optimizer,
                                   max_lr=cfg['training']['lr'],
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=cfg['training']['n_epochs'])
            print(f"Using one-cycle LR")

    else:
        scheduler = None
    return optimizer, scheduler