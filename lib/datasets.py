import os
import hashlib
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class ClaimDataset(Dataset):
    def __init__(self, cfg, train_or_test, fold, final_test=False):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.fold = fold
        self.final_test = final_test

        self.datapath = cfg['data']['dataset_path']
        self.input_lead_names = cfg['data']['input_channels']
        self.input_lead_ids = [cfg['data']['lead_names'].index(name) for name in self.input_lead_names]
        self.iqr = cfg['data']['iqr']
        self.transforms = cfg['transforms'][train_or_test]
        self.npz_files = self.load_npz_files()

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        """Will return:
        x: float tensor
        y: float tensor
        npy_path: path to loaded datafile
        seg_labels: the segmentation labels
        start_time: the time 0th datapoint corresponds to in the seg_labels"""

        npy_path = self.npz_files[idx]
        ecg_data = np.load(npy_path)['data']
        seg_labels = np.load(npy_path, allow_pickle=True)['labels'].item()  # Items ensures object -> dict
        start_time = np.load(npy_path, allow_pickle=True)['starttime']
        ecg_data = ecg_data - ecg_data[0]  # iso-electric=0

        if self.iqr:
            ecg_data = ecg_data / self.iqr
        else:
            print(f"WARNING: Not normalising ECG as no IQR supplied")

        x = ecg_data[:, self.input_lead_ids].T
        y = ecg_data.T

        if self.transforms:
            x, y = self.transform(x, y)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return x, y, {'path': npy_path, 'seg_labels': seg_labels, 'start_time': start_time}

    def transform(self, x, y):
        raise NotImplementedError("This is where augmentations will go, but not implemented")

    def get_iqr(self):
        npy_files = glob(os.path.join(self.datapath, "**/*.npz"), recursive=True)
        iqrs = np.zeros((len(npy_files), 12))
        for i_npy, npy_file in tqdm(enumerate(npy_files), total=len(npy_files)):
            ecg_data = np.load(npy_file)['data']
            ecg_data = ecg_data - ecg_data[0]
            upper_q, lower_q = np.percentile(ecg_data.T, [75, 25], axis=1)
            iqrs[i_npy] = upper_q - lower_q
        return np.mean(iqrs, axis=0)

    def load_npz_files(self):
        n_folds = self.cfg['data']['n_folds']
        excluded_folds = self.cfg['data']['excluded_folds']
        assert 1 <= self.fold <= n_folds, f"Fold should be between 1 and {n_folds}, not {self.fold}"

        def get_train_test_exclude_for_file(npypath):
            """gets a randum number between 0 and 1
            Translates this into a fold, 1 to n_folds, inclusive
            If that is our fold, it's a test case, if it's the excluded case we exclude, and otherwise it's train"""
            case_id = os.path.basename(npypath).split('_', 1)[0]
            randnum = int(hashlib.md5(str.encode(case_id)).hexdigest(), 16) / 16**32
            test_fold = int(randnum * n_folds)+1  # 4 folds -> 1,2,3 or 4

            if test_fold == self.fold:
                return 'test'
            elif test_fold in excluded_folds:
                return 'exclude'
            else:
                return 'train'

        npy_files = glob(os.path.join(self.datapath, "**/*.npz"), recursive=True)
        npy_files_filt = [f for f in npy_files if get_train_test_exclude_for_file(f) == self.train_or_test]
        print(f"{self.train_or_test.upper()}: Loaded {len(npy_files_filt)} of {len(npy_files)} ECGs")
        return npy_files_filt
