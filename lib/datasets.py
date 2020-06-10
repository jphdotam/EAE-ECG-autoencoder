import os
import random
import numpy as np
from glob import glob
from datetime import datetime
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset

try:
    from Code.export_CLAIM import LEADS
except ImportError:
    from export_CLAIM import LEADS

LEAD_DICT = {lead: i for i, lead in
             enumerate(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])}
LEAD_POSITIONS = {'I': 1, 'II': 5, 'III': 9, 'aVR': 2, 'aVL': 6, 'aVF': 10, 'V1': 3, 'V2': 7, 'V3': 11, 'V4': 4,
                  'V5': 8, 'V6': 12}


class ClaimDataset(Dataset):
    def __init__(self, path, input_lead_names, train_val_test, augmentation=False, val_proportion=0.25,
                 test_proportion=0,
                 n_fold=0, return_filename=False, return_segmentation_labels=False, print_n=False, interp_len=1024):
        self.path = path
        self.input_lead_names = input_lead_names
        self.train_val_test = train_val_test
        self.augmentation = augmentation
        self.val_proportion = val_proportion
        self.test_proportion = test_proportion
        self.n_fold = n_fold
        self.return_filename = return_filename
        self.return_segmentation_labels = return_segmentation_labels
        self.print_n = print_n
        self.interp_len = interp_len

        self.iqr, self.maxlen = self.get_dataset_iqr_and_maxlen()

        self.input_lead_ids = [LEAD_DICT[name] for name in self.input_lead_names]
        self.np_files = self.load_npy_files()

    def __len__(self):
        return len(self.np_files)

    def __getitem__(self, idx):
        npy_path = self.np_files[idx]
        ecg_data = np.load(npy_path)['data']  # Transpose for channels 1st
        seg_labels = np.load(npy_path, allow_pickle=True)['labels'].item()  # Items ensures object -> dict
        start_time = np.load(npy_path, allow_pickle=True)['starttime']
        ecg_data = ecg_data - ecg_data[0]  # iso-electric=0
        ecg_data = ecg_data / self.iqr

        ecg_data = self._pad_to_length_with_zeros(ecg_data, self.maxlen)

        x = ecg_data[:, self.input_lead_ids].T
        y = ecg_data.T

        if self.augmentation:
            x, y = self.augment_xy(x, y)

        if self.interp_len:
            x = self.interp(x, self.interp_len)
            y = self.interp(y, self.interp_len)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        if self.return_filename:
            if self.return_segmentation_labels:
                return npy_path, x, y, seg_labels, start_time
            else:
                return npy_path, x, y
        else:
            if self.return_segmentation_labels:
                return x, y, seg_labels, start_time
            else:
                return x, y

    @staticmethod
    def interp(data, interplen):
        x = np.linspace(0, 1, data.shape[-1])
        y = data
        f = interp1d(x, y, kind='cubic')
        x_new = np.linspace(0, 1, interplen)
        y_new = f(x_new)
        return y_new

    @staticmethod
    def augment_xy(x, y):
        # Stretch vertically
        stretch = random.uniform(0.5, 2)
        x = x * stretch
        y = y * stretch

        # Trim off start
        trim_start = random.randint(1, 10)  # Roll up to 10%
        x_new = np.zeros_like(x)
        y_new = np.zeros_like(y)
        x_new[:, :-trim_start] = x[:, trim_start:]
        y_new[:, :-trim_start] = y[:, trim_start:]
        x = x_new
        y = y_new
        return x, y

    @staticmethod
    def _pad_to_length_with_zeros(data, maxlen):
        new_data = np.zeros((maxlen, data.shape[-1]))
        new_data[:len(data), :] = data
        return new_data

    def get_dataset_iqr_and_maxlen(self):
        npy_files = glob(os.path.join(self.path, "**/*.npz"), recursive=True)
        iqrs = np.zeros((len(npy_files), 12))
        maxlen = 0
        for i_npy, npy_file in enumerate(npy_files):
            npy = np.load(npy_file)['data'].T
            maxlen = max(npy.shape[1], maxlen)
            upper_q, lower_q = np.percentile(npy, [75, 25], axis=1)
            iqrs[i_npy] = upper_q - lower_q
        return np.mean(iqrs, axis=0), maxlen

    def load_npy_files(self):
        npy_files = glob(os.path.join(self.path, "**/*.npz"), recursive=True)
        npy_files_filt = self._filter_npyfiles_by_train_val_test(npy_files)
        if self.print_n:
            print(f"{self.train_val_test.upper()}: Loaded {len(npy_files_filt)} of {len(npy_files)} ECGs")
        return npy_files_filt

    def _filter_npyfiles_by_train_val_test(self, npy_files):
        selected_npys = []
        if self.train_val_test not in ('train', 'val', 'test'):
            raise ValueError("train_or_test but be 'train', 'val' or 'test'")
        for npy_file in npy_files:
            # Randomise ECGs at the case level
            case_id = os.path.basename(npy_file).split('_', 1)[0]
            thresh_test = self.test_proportion
            thresh_val = self.test_proportion + self.val_proportion
            random.seed(case_id)
            rand = random.random()  # e.g. 0.78
            rand = (rand + (self.n_fold * 0.25)) % 1  # If n_fold 3, 0.78 + 0.75 % 1 = 0.53
            if self.train_val_test == 'test':
                if thresh_test > rand:
                    selected_npys.append(npy_file)
            elif self.train_val_test == 'val':
                if thresh_test <= rand < thresh_val:
                    selected_npys.append(npy_file)
            elif self.train_val_test == 'train':
                if thresh_val <= rand:
                    selected_npys.append(npy_file)
        return selected_npys


class PhysioNetDataset(Dataset):
    def __init__(self, dataset_path, input_lead_names, train_val_test, val_proportion=0.25, test_proportion=0,
                 n_fold=0, seq_len=1024, print_n=True):
        self.dataset_path = dataset_path
        self.input_lead_names = input_lead_names
        self.train_val_test = train_val_test
        self.val_proportion = val_proportion
        self.test_proportion = test_proportion
        self.n_fold = n_fold
        self.seq_len = seq_len
        self.print_n = print_n

        self.iqr, self.maxlen = self.get_dataset_iqr_and_maxlen()  # All files
        self.patients = self.load_patients()
        self.ecgs = self.load_ecgs()

    def __len__(self):
        return len(self.ecgs)

    def __getitem__(self, idx):
        npz_path = self.ecgs[idx]
        ecg_data = np.load(npz_path)['data']  # Transpose for channels 1st

        ecg_data = ecg_data - ecg_data[0]  # iso-electric=0

        if self.train_val_test == 'train':
            random.seed(datetime.now())  # Random segment for training set
        else:
            random.seed(idx)  # If test/val, get same segment every epoch
        start_snip = random.randint(0, len(ecg_data) - self.seq_len - 1)
        ecg_data = ecg_data[start_snip:start_snip + self.seq_len]
        ecg_data = ecg_data - ecg_data[0]  # Isoelectric at start

        ecg_data = ecg_data / self.iqr

        input_lead_ids = [LEADS.index(l) for l in self.input_lead_names]

        x = ecg_data[:, input_lead_ids].T
        y = ecg_data.T

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return x, y

    def get_dataset_iqr_and_maxlen(self):
        npy_files = glob(os.path.join(self.dataset_path, "**/*.npz"), recursive=True)
        iqrs = np.zeros((len(npy_files), 12))
        maxlen = 0
        for i_npy, npy_file in enumerate(npy_files):
            npy = np.load(npy_file)['data'].T
            maxlen = max(npy.shape[1], maxlen)
            upper_q, lower_q = np.percentile(npy, [75, 25], axis=1)
            iqrs[i_npy] = upper_q - lower_q
        return np.mean(iqrs, axis=0), maxlen

    def load_patients(self):
        # Only load patients with existing NPZ files
        pts_all = [os.path.basename(os.path.dirname(f)) for f in
                   glob(os.path.join(self.dataset_path, "**/*.npz"), recursive=True)]
        pts_sel = self._filter_pts_by_train_val_test(pts_all)
        if self.print_n:
            print(f"{self.train_val_test.upper()}: Loaded {len(pts_sel)} of {len(pts_all)} patients")
        return pts_sel

    def _filter_pts_by_train_val_test(self, pts):
        selected_pts = []
        if self.train_val_test not in ('train', 'val', 'test'):
            raise ValueError("train_or_test but be 'train', 'val' or 'test'")
        for pt in pts:
            # Randomise ECGs at the case level
            thresh_test = self.test_proportion
            thresh_val = self.test_proportion + self.val_proportion
            random.seed(pt)
            rand = random.random()  # e.g. 0.78
            rand = (rand + (self.n_fold * 0.25)) % 1  # If n_fold 3, 0.78 + 0.75 % 1 = 0.53
            if self.train_val_test == 'test':
                if thresh_test > rand:
                    selected_pts.append(pt)
            elif self.train_val_test == 'val':
                if thresh_test <= rand < thresh_val:
                    selected_pts.append(pt)
            elif self.train_val_test == 'train':
                if thresh_val <= rand:
                    selected_pts.append(pt)
        return selected_pts

    def load_ecgs(self):
        ecgs = []
        for patient in self.patients:
            ecgs.extend(glob(os.path.join(self.dataset_path, patient, "*.npz")))
        return ecgs


class MergedDataset(Dataset):
    def __init__(self, list_of_datasets):
        assert type(list_of_datasets) == list
        assert len(list_of_datasets) > 1
        self.datasets = list_of_datasets
        self.n_datasets = len(list_of_datasets)
        self.indices_to_dataset_and_sample_id = self.get_indices()

        print(f"Loaded MergedDataset of {len(self.datasets)} datasets of {len(self)} samples")

    def __len__(self):
        return len(self.indices_to_dataset_and_sample_id)

    def __getitem__(self, idx):
        dataset_id, sample_id = self.indices_to_dataset_and_sample_id[idx]
        return self.datasets[dataset_id][sample_id]

    def get_indices(self):
        indices = {}
        for i_dataset, dataset in enumerate(self.datasets):
            indices.update({i_sample+len(indices): (i_dataset, i_sample) for i_sample in range(len(dataset))})
        return indices
