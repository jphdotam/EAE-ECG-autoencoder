import os
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(cfg):
    if not cfg['output']['use_tensorboard']:
        return None
    else:
        log_dir = os.path.join(cfg['output']['log_dir'], cfg['experiment_id'])
        return SummaryWriter(log_dir=log_dir)
