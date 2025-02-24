from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from os.path import join as pjoin
import pickle as pkl
import numpy as np
import random
import torch
import wfdb
from wfdb import processing


def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class ECGDataset(Dataset):
    def _calc_split_params(self, mode, tot_smpl_no, tvt_split):
        smpl_start_i = 0
        split_norm = 0
        res_dct = dict()
        for m in ['train', 'valid', 'test']:
            split_norm += tvt_split[m] / sum(tvt_split.values())
            last_smpl_i = int(np.floor(tot_smpl_no * split_norm))
            if m == mode:
                return smpl_start_i, last_smpl_i - smpl_start_i
            smpl_start_i = last_smpl_i
    
    def __init__(self, mode, data_dir, file_gen, file_sigs, file_noise, tvt_split, transform=None):
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        
        self.inp_dir = data_dir
        self.file_gen = file_gen
        self.file_sigs = file_sigs
        self.file_noise = file_noise
        self.transform = transform

        # load data files
        with open(pjoin(self.inp_dir, self.file_gen), 'rb') as file:
            data_temp_ = pkl.load(file)
        sigs_temp_ = np.load(pjoin(self.inp_dir, self.file_sigs))

        # save data generation parameters
        self.tot_smpl_no = data_temp_['params']['GEN_DATA_SIZE']
        self.inp_smpl_freq = data_temp_['params']['INP_SMPL_FREQ']
        self.gen_sig_secs = data_temp_['params']['GEN_SIG_SECS']
        self.valid_inp_sigs = data_temp_['params']['VALID_INP_SIGS']
        self.gen_sig_len = self.inp_smpl_freq * self.gen_sig_secs

        # load and process relevant noise data
        noise_temp_ = wfdb.rdrecord(pjoin(self.inp_dir, self.file_noise))
        self.data_noise_, _ = processing.resample_sig(noise_temp_.p_signal[:, 1], noise_temp_.fs, self.inp_smpl_freq)
        
        # save only valid (non-zero) test signals
        self.data_sigs_ = sigs_temp_[:, :self.valid_inp_sigs]
        
        # perform tvt split
        smpl_start_i, self.smpl_no = self._calc_split_params(mode, self.tot_smpl_no, tvt_split)
        self.data_cuts_ = np.int_(data_temp_['data'][:, smpl_start_i:smpl_start_i+self.smpl_no])
        
        # transform the input tensor into required formats
        if self.transform:
            input_m = self.transform(input_m)
     
    def __len__(self):
        return self.smpl_no

    def __getitem__(self, idx):
        sig_item = self.data_sigs_[self.data_cuts_[1, idx]:self.data_cuts_[1, idx]+self.gen_sig_len, self.data_cuts_[0, idx]]
        return sig_item + self.data_noise_[self.data_cuts_[2, idx]:self.data_cuts_[2, idx]+self.gen_sig_len], sig_item


def DataSplit(data_dir, file_gen, file_sigs, file_noise, tvt_split, batch_size=128, generator=None, transform=None):
    ds_trn = ECGDataset('train', data_dir, file_gen, file_sigs, file_noise, tvt_split, transform)
    ds_val = ECGDataset('valid', data_dir, file_gen, file_sigs, file_noise, tvt_split, transform)
    ds_tst = ECGDataset('test', data_dir, file_gen, file_sigs, file_noise, tvt_split, transform)

    if generator != None:
        trn_smplr = SubsetRandomSampler(list(range(len(ds_trn))), generator=generator)
        trn_ldr = DataLoader(ds_trn, batch_size=batch_size, sampler=trn_smplr, worker_init_fn=seed_worker)
        val_ldr = DataLoader(ds_val, batch_size=batch_size, worker_init_fn=seed_worker, generator=generator)
        tst_ldr = DataLoader(ds_tst, batch_size=batch_size, worker_init_fn=seed_worker, generator=generator)
    else:
        trn_smplr = SubsetRandomSampler(list(range(len(ds_trn))))
        trn_ldr = DataLoader(ds_trn, batch_size=batch_size, sampler=trn_smplr)
        val_ldr = DataLoader(ds_val, batch_size=batch_size)
        tst_ldr = DataLoader(ds_tst, batch_size=batch_size)

    return trn_ldr, val_ldr, tst_ldr
