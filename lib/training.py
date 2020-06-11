import os
import torch
from collections import deque


class Am:
    """Simple average meter which stores progress as a running average"""

    def __init__(self, n_for_running_average=100):  # n is in samples not batches
        self.n_for_running_average = n_for_running_average
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.running = deque(maxlen=self.n_for_running_average)
        self.running_average = -1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.running.extend([val] * n)
        self.count += n
        self.avg = self.sum / self.count
        self.running_average = sum(self.running) / len(self.running)

def cycle(train_or_test, model, dataloader, epoch, criterion, optimizer, cfg, scheduler=None, writer=None):
    log_freq = cfg['output']['log_freq']
    device = cfg['training']['device']
    meter_loss = Am()

    model = model.to(device)

    if train_or_test == 'train':
        model.train()
        training = True
    elif train_or_test == 'test':
        model.eval()
        training = False
    else:
        raise ValueError(f"train_or_test must be 'train', or 'test', not {train_or_test}")

    for i_batch, (x, y_true, _info_dict) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)
        optimizer.zero_grad()

        # Forward pass
        if training:
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
        else:
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y_true)

        # Backward pass
        if training:
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        meter_loss.update(loss, x.size(0))

        # Loss intra-epoch printing
        if (i_batch+1) % log_freq == 0:
            print(f"{train_or_test.upper(): >5} [{i_batch+1:04d}/{len(dataloader):04d}] \t\tLOSS: {meter_loss.running_average:.5f}")

            if train_or_test == 'train':
                i_iter = ((epoch - 1) * len(dataloader)) + i_batch+1
                writer.add_scalar(f"LossIter/{train_or_test}", meter_loss.running_average, i_iter + 1)

    loss = float(meter_loss.avg.detach().cpu().numpy())

    print(f"{train_or_test.upper(): >5} Complete!\t\t\tLOSS: {meter_loss.avg:.5f}")

    if writer:
        writer.add_scalar(f"LossEpoch/{train_or_test}", loss, epoch)

    return loss

def save_state(state, filename, test_metric, best_metric, cfg, last_save_path, lowest_best=True):
    save = cfg['output']['save']
    save_path = os.path.join(cfg['output']['model_dir'], cfg['experiment_id'], filename)
    if save == 'all':
        torch.save(state, save_path)
    elif (test_metric < best_metric) == lowest_best:
        print(f"{test_metric:.5f} better than {best_metric:.5f} -> SAVING")
        if save == 'best':  # Delete previous best if using best only; otherwise keep previous best
            if last_save_path:
                try:
                    os.remove(last_save_path)
                except FileNotFoundError:
                    print(f"Failed to find {last_save_path}")
        best_metric = test_metric
        torch.save(state, save_path)
        last_save_path = save_path
    else:
        print(f"{test_metric:.5g} not improved from {best_metric:.5f}")
    return best_metric, last_save_path