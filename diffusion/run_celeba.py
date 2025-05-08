from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.numpy import load_file
from itertools import chain, combinations
import os
from utils.aligner import ModelAligner
import numpy as np
import pandas as pd
import argparse
import wandb
from torchvision import datasets, transforms
from tqdm import tqdm
from models.models import *
from models.transformer import Transformer
from utils.utils import AlignedModelDataset, get_equiv_shapes, get_standard_shapes
from utils.utils import svd, get_uvs_from_file, get_tensors_from_file
from utils.utils import train, valid, test


device = torch.device('cuda')

def load_data(mode = 'standard', to_keep = None, verbose = False):
    
    a = []
    for i, model in tqdm(enumerate(os.listdir(dir)), desc="Loading Data"):

        if 'model' not in model or ((to_keep != None) and (model not in to_keep)):
            continue
        num = int(model.split('_')[-1])
        dir_with_images = os.path.join(args.img_directory, 'celeb_' + str(num))
        imgs = [i for i in os.listdir(dir_with_images) if 'jpg' in i]
        inds = [int(s.split('.jpg')[0]) - 1 for s in imgs]
        rows_to_keep = df.loc[inds]
        label = (torch.tensor(np.asarray(rows_to_keep.iloc[:,:].mean(axis = 0))) >= 0 )
        
        path = os.path.join(dir, model, 'unet')
        if mode == 'standard':
            d = get_uvs_from_file(path, do_svd)
        elif mode =='tensor':
            d = get_tensors_from_file(path)
            
        a.append((d, label))
    return a
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
            num = int(model.split('_')[-1])
            dir_with_images = os.path.join(args.img_directory, 'celeb_' + str(num))
            imgs = [i for i in os.listdir(dir_with_images) if 'jpg' in i]
            inds = [int(s.split('.jpg')[0]) - 1 for s in imgs]
            to_keep = df.loc[inds]
            label = (torch.tensor(np.asarray(to_keep.iloc[:,:].mean(axis = 0))) >= 0 )
            
            path = os.path.join(dir, model, 'unet')
            
            d = get_tensors_from_file(path)
                
            return d, label
    def __len__(self):
        return len(self.to_keep)

import argparse
parser = argparse.ArgumentParser(description='Properties for ResNets for CIFAR10 in pytorch')
if __name__ == '__main__':
    df = pd.read_csv('list_attr_celeba.csv')
    df = df[['Male', 'Young', 'Wearing_Lipstick', 'Big_Lips', 'No_Beard']]
    parser.add_argument('--train_directory', default="/workspace/peft/examples/lora_dreambooth/models_4", type=str,
                        help='Path of directory to train on?')
    parser.add_argument('--img_directory', default="/workspace/peft/examples/lora_dreambooth/celeb_organized", type=str,
                        help='Path of directory to train on?')

    parser.add_argument('--model_type', default=0, type=int,
                        help='0:GLNet, 1: MLP+Mul, 2: MLP, 3: MLP+SVD, 4: MLP+Align, 5: Transformer')
    parser.add_argument('--epochs', default=12, type=int,
                    help='number of epochs to train for')
    parser.add_argument('--batch_size', default=100, type=int,
                    help='batch size for optimization')
    parser.add_argument('--n_layers', default=1, type=int,
                    help='number of layers for GLNet (usually 1 works)')
    parser.add_argument('--hidden_dim', default=16, type=int,
                    help='hidden dimension of MLP')
    parser.add_argument('--lr', default=.001, type=float, 
                    help='learning rate to use for optimization')
    parser.add_argument('--verbose', default=True, type=bool, 
                    help='verbose?')
    args = parser.parse_args()

    #----------------------------------------------#
    dir = args.train_directory

    
    do_tensor = (args.model_type == 1)
    
    do_svd = (args.model_type ==3)

    num_pred = 5

    #----------------------------------------------#
    

    wandb.init(
        # set the wandb project where this run will be logged
        project="LOL-Celeba",
        config = args
        # track hyperparameters and run metadata
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
    
    if args.model_type == 0:
        point = train_set[0][0]
        ns, ms = get_equiv_shapes(point)
        model = GLInvariantMLP(ns, ms, num_input_layers, num_pred, n_layers = args.n_layers, hidden_dim_equiv=64, hidden_dim_inv =   args.hidden_dim, clip = False).to(device)
    elif args.model_type == 1:
        model = BaselineNet(46755840, args.hidden_dim, num_pred).to(device)
    elif args.model_type in (2,4):
        point = train_set[0][0]
        ns, ms = get_standard_shapes(point)
        model = SimpleNet(ns, ms, args.hidden_dim, num_pred).to(device)
    elif args.model_type == 5:
        point = train_set[0][0]
        ns,ms = get_standard_shapes(point)
        model = Transformer(ns, ms, num_pred, d_model = args.hidden_dim, num_layers = args.n_layers).to(device)
        print(model)
    elif args.model_type == 3:
        num_inputs = train_set[0][0].shape[-1]
        model = BaselineNet(num_inputs, args.hidden_dim, num_pred).to(device)
    else:
        assert False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = .0)
    train(model, device, train_set,  optimizer, args.epochs, args.batch_size, valid_set = valid_set)
    
    loss, acc = test(model, device, test_set, num_pred = num_pred)
    train_loss, train_acc = test(model, device, train_set, num_pred=num_pred)
    log_dict = {"test_loss" : loss, "test_acc" :  acc, "train_loss" : train_loss}
    wandb.log(log_dict)
    wandb.finish()
    
    
    
