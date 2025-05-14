import os
import time
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_utils import FlattenWeights, MultiplyLoraFlatten, PairWeights, SingularValues, OrthogAlignFlatten, RandomLoraGPTDataset, PairFast
from models_metanet import GLInvariantMLP as SlowGLInvariantMLP
from fast_metanet import GLInvariantMLP as FastGLInvariantMLP

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    elif isinstance(x, list):
        x = [(u.to(device), v.to(device)) for (u,v) in x]
    else:
        raise ValueError("Unexpected x data type")
    return x

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if args.transform == "flatten":
        transform = FlattenWeights()
    elif args.transform == "mult_flatten":
        transform = MultiplyLoraFlatten(device=device)
    elif args.transform == "pair":
        transform = PairWeights()
    elif args.transform == "pair_fast":
        transform = PairFast()
    elif args.transform == "svals":
        transform = SingularValues()
    elif args.transform == "align":
        transform = OrthogAlignFlatten(device=device)
    else:
        raise ValueError("Invalid transform")
    
    dtype = torch.float32
    dataset = RandomLoraGPTDataset(size=512, rank=args.rank, idx=args.idx, transform=transform, dtype=dtype)

    if args.transform == "align":
        dataset[0]

    in_dim = dataset.dim_weights
    hidden_dim = args.hidden_dim
    out_dim = 1
    print("Num examples:", len(dataset))

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)

    if args.only_load:
        full_transform_time = 0
        for x, transform_time in tqdm(loader, total=len(loader)):
            full_transform_time += transform_time.sum().item()
        print(f"Transform time: {full_transform_time:.6f} seconds for {100} items")
        print("Done loading data, exiting")
        exit()


    if args.model == "mlp":
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(args.num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),  nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        model = nn.Sequential(*layers)
    elif "fast_gl_mlp" in args.model:
        n = dataset[0][0].shape[1]
        model = FastGLInvariantMLP(n, n_input_layers=dataset.num_layers, rank=args.rank, out_dim=out_dim, n_layers=args.num_layers, hidden_dim_equiv=args.hidden_dim, hidden_dim_inv=hidden_dim)
    elif "gl_mlp" in args.model:
        ns, ms = [], []
        for u, v in dataset[0][0]:
            ns.append(len(u))
            ms.append(len(v))

        model = SlowGLInvariantMLP(ns, ms, n_input_layers=dataset.num_layers, rank=args.rank, out_dim=out_dim, n_layers=args.num_layers, hidden_dim_equiv=args.hidden_dim, hidden_dim_inv=hidden_dim, k=4, d=32, p=0.0, pool_mean=False, head_type=1)
    else:
        raise ValueError("Invalid model (metanet)")

    model = model.to(dtype).to(device)

    #print(model)
    print("Num metanet params:", sum(p.numel() for p in model.parameters()))

    model.eval()
    total_forward_time = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, total=len(loader)):
            x = to_device(x, device)
            start_time = time.time()
            _ = model(x)
            elapsed_time = time.time() - start_time
            total_forward_time += elapsed_time

    print(f"Total forward pass time: {total_forward_time:.6f} seconds for {len(dataset)} items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metanet forward pass timing')
    parser.add_argument('--model', type=str, default='gl_mlp')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--transform', type=str, default="flatten")
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--only_load', type=int, default=0, help="if true, only load the data, then exit")
    args = parser.parse_args()
    
    print("#"*80)
    print("Model:", args.model)
    print("Transform:", args.transform)
    print("Rank:", args.rank)
    print("Idx:", args.idx)
    print("#"*80)

    main(args)
