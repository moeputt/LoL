import re
import numpy as np
import pandas as pd
import random
import os
import csv
import pickle
import matplotlib.pyplot as plt
import math
import time
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

ALL_SOURCES = ["MCAS", "NYSEDREGENTS", "TIMSS", "ACTAAP", "NCEOGA", "MDSA", "VASoL", "AKDE&ED", "CSZ", "MEA", "LEAP", "NAEP", "MSA", "OHAT", "AIMS", "TAKS", "MEAP", "FCAT", "WASL"] # for ARC dataset

class FlattenWeights():
    def __call__(self, sd):
        return torch.cat([v.flatten() for v in sd.values()])

class MultiplyLoraFlatten():
    def __init__(self, device=None):
        self.device = device

    @torch.no_grad
    def __call__(self, sd):
        result = []
        keys = list(sd.keys())
        if self.device is not None:
            for k in keys:
                sd[k] = sd[k].to(self.device)
        for i in range(0, len(keys), 2):
            a_key = keys[i]
            b_key = keys[i+1]
            prefix_idx = a_key.find("lora_A")
            assert a_key[:prefix_idx] == b_key[:prefix_idx], "expected keys to be in order, for each layer first lora_A then lora_B"
            A, B = sd[a_key], sd[b_key]
            mat = B @ A
            result.append(mat.flatten())
        return torch.cat(result)

class PairWeights():
    def __call__(self, sd):
        uvs = []
        keys = list(sd.keys())
        for i in range(0, len(keys), 2):
            a_key = keys[i]
            b_key = keys[i+1]
            prefix_idx = a_key.find("lora_A")
            assert a_key[:prefix_idx] == b_key[:prefix_idx], "expected keys to be in order, for each layer first lora_A then lora_B"
            #mat = sd[b_key] @ sd[a_key]
            A = sd[a_key]
            B = sd[b_key]
            assert A.ndim == 2, "requires matrix weights"
            A = A.T
            uvs.append((B, A))
        return uvs

class SingularValues():
    def __call__(self, sd):
        # for efficiency, compute singular values of R1 @ R2.T
        # where B = Q1R1 and A = Q2R2
        # this is equivalent to nonzero singular values of B @ A
        svals = []
        keys = list(sd.keys())
        for i in range(0, len(keys), 2):
            a_key = keys[i]
            b_key = keys[i+1]
            prefix_idx = a_key.find("lora_A")
            assert a_key[:prefix_idx] == b_key[:prefix_idx], "expected keys to be in order, for each layer first lora_A then lora_B"
            #mat = sd[b_key] @ sd[a_key]
            A = sd[a_key]
            B = sd[b_key]
            assert A.ndim == 2, "requires matrix weights"
            Q1, R1 = torch.linalg.qr(B)
            Q2, R2 = torch.linalg.qr(A.T)
            s = torch.linalg.svdvals(R1 @ R2.T)
            svals.append(s)
        return torch.cat(svals)

class PairFast():
    def __call__(self, sd):
        us = []
        vs = []
        #uvs = []
        keys = list(sd.keys())
        for i in range(0, len(keys), 2):
            a_key = keys[i]
            b_key = keys[i+1]
            prefix_idx = a_key.find("lora_A")
            assert a_key[:prefix_idx] == b_key[:prefix_idx], "expected keys to be in order, for each layer first lora_A then lora_B"
            #mat = sd[b_key] @ sd[a_key]
            A = sd[a_key]
            B = sd[b_key]
            assert A.ndim == 2, "requires matrix weights"
            A = A.T
            us.append(B)
            vs.append(A)
        uvs = torch.stack(us + vs, 0)
        return uvs


def orthog_procrustes(A, B):
    """ A and B are n x d matrices
    returns AQ, where Q is orthog and minimizes min_Q ||AQ - B||_F
    """
    M = B.T @ A # d x d
    U, _, Vt = torch.linalg.svd(M)
    Q = U @ Vt
    return A @ Q.T

class OrthogAlignFlatten():
    def __init__(self, device=None):
        self.device = device
        self.template = None

    def __call__(self, sd):
        assert self.template is not None, "need to save template first"
        keys = list(sd.keys())
        result = []
        if self.device is not None:
            for k in keys:
                sd[k] = sd[k].to(self.device)
        for i in range(0, len(keys), 2):
            a_key = keys[i]
            b_key = keys[i+1]
            prefix_idx = a_key.find("lora_A")
            assert a_key[:prefix_idx] == b_key[:prefix_idx], "expected keys to be in order, for each layer first lora_A then lora_B"
            #mat = sd[b_key] @ sd[a_key]
            A = sd[a_key]
            B = sd[b_key]
            mat = torch.cat([B, A.T], dim=0) # m+n x r
            template_mat = torch.cat([self.template[b_key], self.template[a_key].T], dim=0)
            mat_aligned = orthog_procrustes(mat, template_mat)
            result.append(mat_aligned.flatten())
        return torch.cat(result)

    def save_template(self, sd):
        # save A and B weights
        if self.template is not None:
            raise ValueError("Already saved template")
        print("Saving template for orthogonal alignment")
        self.template = deepcopy(sd)
        if self.device is not None:
            for k in self.template:
                self.template[k] = self.template[k].to(self.device)


class LoraDataset(torch.utils.data.Dataset):
    """ built off of Graph Metanetworks and Empirical Impact of Param Symmetries code"""
    def __init__(self, root='lora_data/qwen2_arc_data/', hparams_path="hparams.csv", target='eval_loss', return_hparams=False, transform=FlattenWeights()):
        """ transform takes in a state dict and returns some other data structure
            holding weights """
        self.root = root
        self.transform = transform
        self.paths = [os.path.join(root, name) for name in os.listdir(root) if os.path.isfile(os.path.join(root, name, "adapter_model.bin"))]
        all_hparams = []
        with open(hparams_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    assert row[0] == 'learning_rate'
                    assert row[1] == 'weight_decay'
                    assert row[2] == 'num_epochs'
                    assert row[3] == 'batch_size'
                    assert row[4] == 'lora_dropout'
                    assert row[5] == 'filter_sources'
                else:
                    lr = float(row[0])
                    wd = float(row[1])
                    num_epochs = int(row[2])
                    batch_size = int(row[3])
                    lora_dropout = float(row[4])
                    filter_sources = row[5]
                    all_hparams.append([lr, wd, num_epochs, batch_size, lora_dropout, filter_sources])

        #self._prune_nans()

        self.hparams = []
        for idx in range(len(self.paths)):
            orig_idx = self._idx_to_orig(idx)
            self.hparams.append(all_hparams[orig_idx])

        self.target = target
        self.return_hparams = return_hparams
        del all_hparams

    def _prune_nans(self):
        # if path exists, load from there
        print("Keeping non-nan paths")
        nan_path = self.root + f'{self.model_type}_no_nan'
        new_paths = []
        if os.path.exists(nan_path):
            with open(nan_path, "rb") as f:
                new_paths = pickle.load(f)
        else:
            # if not, determine paths
            print("Computing non-nans")
            for idx, path in enumerate(self.paths):
                if idx % 500 == 0: print(idx)
                sd, _ = torch.load(path)
                isnan = any(v.isnan().any() for v in sd.values())
                if not isnan:
                    new_paths.append(path)
            with open(nan_path, "wb") as f:
                pickle.dump(new_paths, f)
        self.paths = new_paths
                

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sd = torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu", weights_only=True)
        if self.transform is not None:
            if hasattr(self.transform, "save_template") and self.transform.template is None:
                self.transform.save_template(sd)
            weights = self.transform(sd)
        with open(path + "/eval_loss.txt", "r") as f: eval_loss = float(f.read())
        if self.target == "arc-c-test":
            with open(path + "/ARC-Challenge-Acc.txt", "r") as f: arc_c_acc = float(f.read())
        hparams = self.hparams[idx]
        if self.target == 'eval_loss':
            y = torch.tensor([eval_loss], dtype=torch.float)
        elif self.target == 'arc-c-test':
            y = torch.tensor([arc_c_acc], dtype=torch.float)
        elif self.target == 'arc-e-test':
            raise NotImplemented
        elif self.target == 'filter_sources':
            filter_sources = hparams[-1].split("|")
            # 1.0 means source included, 0.0 means excluded
            y = torch.ones(len(ALL_SOURCES), dtype=torch.float32)
            for source in filter_sources:
                source_idx = ALL_SOURCES.index(source)
                y[source_idx] = 0.0 
        elif self.target == 'filter_sources_top':
            filter_sources = hparams[-1].split("|")
            # 1.0 means source included, 0.0 means excluded
            y = torch.ones(len(ALL_SOURCES), dtype=torch.float32)
            for source in filter_sources:
                source_idx = ALL_SOURCES.index(source)
                y[source_idx] = 0.0 
            y = y[:3] # only take label for top 3 sources
        else:
            raise ValueError('Invalid target')
        if not self.return_hparams:
            return weights, y
        else:
            return weights, y, hparams

    def _idx_to_orig(self, idx):
        orig_idx = int(re.search(r'\d+$', self.paths[idx]).group())
        return orig_idx

    @property
    def dim_weights(self):
        w = self[0][0]
        if isinstance(w, torch.Tensor):
            assert w.ndim == 1, "Expected flattened tensor"
            return w.shape[0] 
        elif isinstance(w, list):
            # when PairWeights() is used
            return w[0][0].shape[0]
        else:
            raise ValueError("Invalid weight type")
            
class RandomLoraGPTDataset(torch.utils.data.Dataset):
    """ for timing on state dicts of GPT-3 size """
    def __init__(self, size=1000, idx=0, rank=4, transform=FlattenWeights(), dtype=torch.float16):
        """ transform takes in a state dict and returns some other data structure
            holding weights """
        num_layers = [12, 24, 24, 24, 32, 32, 40, 96]
        num_params = [.125, .350, .750, 1.3, 2.7, 6.7, 13.0, 175.0]
        dims = [768, 1024, 1536, 2048, 2560, 4096, 5140, 12288]

        self.size = size
        self.dtype = dtype
        self.num_layers = num_layers[idx]
        self.num_params = num_params[idx]
        self.dim = dims[idx]
        self.rank = rank
        self.transform = transform

    def __len__(self):
        return self.size

    def make_sd(self):
        sd = OrderedDict()
        for layer in range(self.num_layers):
            a_key = f"{layer}.lora_A"
            b_key = f"{layer}.lora_B"
            A = torch.randn(self.rank, self.dim, dtype=self.dtype) / math.sqrt(self.dim)
            B = torch.randn(self.dim, self.rank, dtype=self.dtype) / math.sqrt(self.dim)
            sd[a_key] = A
            sd[b_key] = B
        return sd

    def __getitem__(self, idx):
        sd = self.make_sd()
        if self.transform is not None:
            if hasattr(self.transform, "save_template") and self.transform.template is None:
                self.transform.save_template(sd)
            start_time = time.time()
            weights = self.transform(sd)
            transform_time = time.time() - start_time

        return weights, transform_time


    @property
    def dim_weights(self):
        w = self[0][0]
        if isinstance(w, torch.Tensor):
            return w.shape[0] 
        elif isinstance(w, list):
            # when PairWeights() is used
            return w[0][0].shape[0]
        else:
            raise ValueError("Invalid weight type")


if __name__ == "__main__":
    #dataset = LoraDataset(target="filter_sources", return_hparams=False, transform=FlattenWeights())
    #dataset = LoraDataset(target="filter_sources", return_hparams=False, transform=MultiplyLoraFlatten())
    #dataset = LoraDataset(target="filter_sources", return_hparams=False, transform=PairWeights())
    #dataset = LoraDataset(target="filter_sources", return_hparams=False, transform=SingularValues())
    A = torch.randn(5, 3)
    B = A @ torch.linalg.qr(torch.randn(3,3)).Q
    A_ = orthog_procrustes(A, B)
    print("Orthog procrustes error", (A_ - B).norm().item())
    A2 = orthog_procrustes(A, A)
    print("Orthog procrustes error", (A2 - A).norm().item())
