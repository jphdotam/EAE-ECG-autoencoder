import os
import yaml

def load_config(configpath):
    with open(configpath) as f:
        cfg = yaml.safe_load(f)

    experiment_id = os.path.splitext(os.path.basename(configpath))[0]
    cfg['experiment_id'] = experiment_id

    model_dir = cfg['output']['model_dir']
    if model_dir:
        model_dir = os.path.join(model_dir, experiment_id)
        print(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    log_dir = cfg['output']['log_dir']
    if log_dir:
        log_dir = os.path.join(log_dir, experiment_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    vis_dir = cfg['output']['vis_dir']
    if vis_dir:
        vis_dir = os.path.join(vis_dir, experiment_id)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    return cfg