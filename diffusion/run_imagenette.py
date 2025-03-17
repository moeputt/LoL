# !pip install pandas
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.numpy import load_file
from itertools import chain, combinations
import os
from utils.aligner import ModelAligner
import pandas as pd
import numpy as np
import argparse
import wandb
from tqdm import tqdm
from models.models import *

from utils.utils import AlignedModelDataset, get_equiv_shapes, get_standard_shapes
from utils.utils import svd, get_uvs_from_file, get_tensors_from_file
from utils.utils import train, valid, test
def load_data(mode = 'standard', to_keep = None, verbose = False):
    
    a = []
    for i, model in tqdm(enumerate(sorted(os.listdir(dir))), desc="Loading Data"):
        if 'model' not in model or ((to_keep != None) and (model not in to_keep)):
            continue
 
        
        label = torch.tensor([int(c) for c in model.split('_')[2]])
        
        path = os.path.join(dir, model, 'unet')
        if mode == 'standard':
            d = get_uvs_from_file(path, do_svd)
        elif mode =='tensor':
            d = get_tensors_from_file(path)
            
        a.append((d, label))
    return a
class model_dataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'standard'):
        if mode == 'standard':
            self.data = load_data(mode = mode, verbose = args.verbose)
        else:
            self.data = [model for model in sorted(os.listdir(dir)) if 'model' in model]
            
        self.mode = mode
    def __getitem__(self, idx):
        
        if self.mode == 'standard':
            return self.data[idx]
        else:
            
            model = self.data[idx]
            label = torch.tensor([int(c) for c in model.split('_')[2]])
            path = os.path.join(dir, model, 'unet')
            d = get_tensors_from_file(path)
            return d, label
    def __len__(self):
        return len(self.data)


import argparse
parser = argparse.ArgumentParser(description='Properties for ResNets for CIFAR10 in pytorch')
if __name__ == '__main__':

    from torchvision import datasets, transforms
    parser.add_argument('--train_directory', default="/workspace/b/s1-4models", type=str,
                        help='Path of directory for in distribution training, testing?')
    parser.add_argument('--test_directory', default="/workspace/b/s1-5models", type=str,
                        help='Path of directory for OOD testing?')
    parser.add_argument('--model_type', default=0, type=int,
                        help='0:GLNet, 1: MLP+Mul, 2: MLP, 3: MLP+SVD, 4: MLP+Align')
    parser.add_argument('--epochs', default=3, type=int,
                    help='number of epochs to train for')
    parser.add_argument('--batch_size', default=8, type=int,
                    help='batch size for optimization')
    parser.add_argument('--n_layers', default=1, type=int,
                    help='number of layers for GLNet')
    parser.add_argument('--hidden_dim', default=128, type=int,
                    help='hidden dimension of MLP')
    parser.add_argument('--lr', default=.001, type=float,
                    help='learning rate to use for optimization')
    parser.add_argument('--verbose', default=True, type=bool, 
                    help='verbose?')
    args = parser.parse_args()
    
    train_on = args.train_directory
    test_on = args.test_directory
    
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="LOL-Imagenette",
        config = args
        # track hyperparameters and run metadata
    )
    
    #---------------------------------------------------#

    do_tensor = (args.model_type == 1)
    do_svd = (args.model_type ==3)
    device = torch.device('cuda')
    dir = train_on
    
    #---------------------------------------------------#
    dataset = model_dataset(mode = 'tensor' if do_tensor else 'standard')
    train_set, valid_set, test_set =  torch.utils.data.random_split(dataset, [1534, 256, 256], generator = torch.Generator().manual_seed(42))
    

    num_input_layers = len(train_set[0][0])
    num_pred = 10
    if args.model_type == 4:
        ind = np.random.randint(len(train_set))
        x = train_set[ind][0]
        model_aligner = ModelAligner([(u.cpu(), v.cpu()) for u,v in x])
        train_set = AlignedModelDataset(model_aligner, train_set, batch_size = 10 )
        valid_set = AlignedModelDataset(model_aligner, valid_set, batch_size = 10 )
        test_set = AlignedModelDataset(model_aligner, test_set, batch_size = 10 )
    
    print(f"FINISHED LOADING TRAINING DATA FOR {dir}")
    
    if args.model_type == 0:
        point = train_set[0][0]
        ns, ms = get_equiv_shapes(point)
        model = GLInvariantMLP(
            ns, ms, num_input_layers, num_pred, n_layers = args.n_layers, hidden_dim_equiv=64, hidden_dim_inv =  args.hidden_dim).to(device)
    elif args.model_type == 1:
        model = BaselineNet(46755840, args.hidden_dim, num_pred).to(device)
    elif args.model_type in (2,4):
        point = train_set[0][0]
        ns,ms = get_standard_shapes(point)
        model = SimpleNet(ns, ms, args.hidden_dim, num_pred).to(device)
    elif args.model_type == 3:
        num_inputs = train_set[0][0].shape[-1]
        model = BaselineNet(num_inputs, args.hidden_dim, num_pred).to(device)
    else:
        assert False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = .0)
    train(model, device, train_set, optimizer, args.epochs, args.batch_size)
    
    id_loss, id_acc = test(model, device, test_set, num_pred)
    dir = test_on
    
    dataset = model_dataset(mode = 'tensor' if do_tensor else 'standard')
    
    train_set, valid_set, test_set =  torch.utils.data.random_split(dataset, [1534, 256, 256], generator = torch.Generator().manual_seed(42))
    if args.model_type == 4:
        model_aligner = ModelAligner([(u.cpu(), v.cpu()) for u,v in x])
        test_set = AlignedModelDataset(model_aligner, test_set, batch_size = 10 )
    print(f"FINISHED LOADING TRAINING DATA FOR {dir}")
    
    ood_loss, ood_acc = (test(model, device, test_set, num_pred))
    log_dict = {"test_acc" : id_acc, "test_loss" :  id_loss, "ood_test_loss" : ood_loss, "ood_test_acc" : ood_acc}
    wandb.log(log_dict)
    wandb.finish()
    
    
    
