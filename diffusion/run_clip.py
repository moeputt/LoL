import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain, combinations
import os
from utils.aligner import ModelAligner
import numpy as np
import pandas as pd
import argparse
import wandb
from tqdm import tqdm
from models.models import *
from models.transformer import Transformer
from utils.utils import AlignedModelDataset, get_equiv_shapes, get_standard_shapes
from utils.utils import svd, get_uvs_from_file, get_tensors_from_file
from utils.utils import train_clip, valid_clip, test_clip

def load_data(mode = 'standard', to_keep = None, verbose = False):
    all_data = []
    for i, model in tqdm(enumerate(os.listdir(dir)), desc="Loading Data"):
        if 'model' not in model or ((to_keep != None) and (model not in to_keep)):
            continue
        dir_with_images = os.path.join(dir, model, 'score')
        
        label = (torch.tensor(torch.load(dir_with_images)).reshape(1) - 27.4153)/4.1134
        path = os.path.join(dir, model, 'unet')
        if mode == 'standard':
            weights = get_uvs_from_file(path, do_svd)
        elif mode =='tensor':
            weights = get_tensors_from_file(path)
            
        all_data.append((weights, label))
    return all_data
    
class model_dataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'standard', to_keep = None):
        if mode == 'standard':
            self.data = load_data(mode = mode, to_keep = to_keep, verbose = args.verbose)
        else:
            self.data = [model for model in os.listdir(dir) if 'model' in model]
        self.to_keep = to_keep
            
        self.mode = mode
    def __getitem__(self, idx):
        if self.mode == 'standard':
            return self.data[idx]
        else:
            model = self.to_keep[idx]
            dir_with_images = os.path.join(dir, model, 'score')
            label = (torch.tensor(torch.load(dir_with_images)).reshape(1) - 27.4153)/4.1134 #subtract mean, divide by std
            path = os.path.join(dir, model, 'unet')
            d = get_tensors_from_file(path)  
            return d, label
    def __len__(self):
        return len(self.to_keep)




import argparse
parser = argparse.ArgumentParser(description='Properties for ResNets for CIFAR10 in pytorch')
if __name__ == '__main__':
    # '/workspace/peft/examples/lora_dreambooth/models_4'
    device = torch.device('cuda')

    
    from torchvision import datasets, transforms
    parser.add_argument('--train_directory', default="/workspace/peft/examples/lora_dreambooth/models_4", type=str,
                        help='Path of directory to train on?')
    parser.add_argument('--model_type', default=0, type=int,
                        help='0:GLNet, 1: MLP+Mul, 2: MLP, 3: MLP+SVD, 4: MLP+Align, 5: Transformer')
    parser.add_argument('--epochs', default=3, type=int, 
                    help='number of epochs to train for')
    parser.add_argument('--batch_size', default=8, type=int, 
                    help='batch size for optimization')
    parser.add_argument('--n_layers', default=1, type=int, 
                    help='number of layers for GLNet (Usually 1 works)')
    parser.add_argument('--hidden_dim', default=64, type=int, 
                    help='hidden dimension of MLP')
    parser.add_argument('--lr', default=.001, type=float, 
                    help='learning rate to use for optimization')
    parser.add_argument('--verbose', default=True, type=bool, 
                    help='verbose?')
    
    args = parser.parse_args()
    #----------------------------------------------__#
    dir = args.train_directory
    
    do_tensor = (args.model_type == 1)
    
    do_svd = (args.model_type ==3)
    #----------------------------------------------__#

    wandb.init(
        project="LOL-CLIP",
        config = args
    )

    
    train_pt = torch.load('splits/train.pt')
    valid_pt = torch.load('splits/valid.pt')
    test_pt = torch.load('splits/test.pt')
    train_set = model_dataset(mode = 'tensor' if do_tensor else 'standard', to_keep = train_pt)
    valid_set = model_dataset(mode = 'tensor' if do_tensor else 'standard', to_keep = valid_pt)
    test_set = model_dataset(mode = 'tensor' if do_tensor else 'standard', to_keep = test_pt)
    
    if args.model_type == 4:
        ind = np.random.randint(len(train_set))
        x = train_set[ind][0]
        model_aligner = ModelAligner([(u.cpu(), v.cpu()) for u,v in x])
        train_set = AlignedModelDataset(model_aligner, train_set, batch_size = 10 )
        valid_set = AlignedModelDataset(model_aligner, valid_set, batch_size = 10 )
        test_set = AlignedModelDataset(model_aligner, test_set, batch_size = 10 )
    
    print(f"FINISHED LOADING TRAINING DATA FOR {dir}")
    num_input_layers = len(train_set[0][0])
    num_pred = 1
    if args.model_type == 0:
        point = train_set[0][0]
        ns, ms = get_equiv_shapes(point)
        model = GLInvariantMLP(
            ns, ms, num_input_layers, 1, n_layers = args.n_layers, hidden_dim_equiv=16, hidden_dim_inv =   args.hidden_dim, clip = True).to(device)
        
    elif args.model_type == 1:
        model = BaselineNet(46755840, args.hidden_dim, 1).to(device)
        
    elif args.model_type in (2,4):
        point = train_set[0][0]
        ns, ms = get_standard_shapes(point)
        model = SimpleNet(ns, ms, args.hidden_dim, 1).to(device)
    elif args.model_type == 5:
        point = train_set[0][0]
        ns,ms = get_standard_shapes(point)
        model = Transformer(ns, ms, num_pred, d_model = args.hidden_dim, num_layers = args.n_layers).to(device)
        print(model)
    elif args.model_type == 3:
        num_inputs = train_set[0][0].shape[-1]
        
        model = BaselineNet(num_inputs, args.hidden_dim, 1).to(device)
    else:
        assert False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = .0)
    train_clip(model, device, train_set, optimizer, args.epochs, args.batch_size)
    
    loss, r, tau = test_clip(model, device, test_set)
    
    log_dict = {"test_loss" : loss, "test_r" :  r, "test_tau" : tau}
    wandb.log(log_dict)
    wandb.finish()
    
    
    
