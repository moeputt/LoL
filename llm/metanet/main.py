import os
import random
import time
import argparse
import numpy as np
import csv
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_utils import LoraDataset, FlattenWeights, MultiplyLoraFlatten, PairWeights, SingularValues, OrthogAlignFlatten
from models_metanet import GLInvariantMLP
from model_transformer import Transformer

def get_bs(x):
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    elif isinstance(x, list):
        # when using PairWeights
        return x[0][0].shape[0]
    else:
        raise ValueError("Unexpected x data type")

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    elif isinstance(x, list):
        # when using PairWeights
        x = [(u.to(device), v.to(device)) for (u,v) in x]
    else:
        raise ValueError("Unexpected x data type")
    return x

def train(loader, model, criterion, optimizer, device):
    model.train()
    full_loss = 0
    num_items = 0
    for x, y in loader:
        bs = get_bs(x)
        x, y = to_device(x, device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            out = model(x)
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        full_loss += bs*loss.item()
        num_items += bs

    full_loss = full_loss / num_items
    return full_loss

class Tester():
    def __init__(self, name, loader, criterion, task_type, device, verbose=True):
        self.name = name
        self.loader = loader
        self.criterion = criterion
        self.task_type = task_type
        self.verbose = verbose
        self.device = device
    
    @torch.no_grad
    def __call__(self, model):
        model.eval()
        full_loss = 0
        num_items = 0
        all_out = []
        all_y = []
        for x, y in self.loader:
            bs = get_bs(x)
            x, y = to_device(x, self.device), y.to(self.device)
            with torch.autocast(device_type="cuda"):
                out = model(x)
                loss = self.criterion(out, y)
            full_loss += bs*loss.item()
            num_items += bs
            all_out.append(out.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())

        full_loss = full_loss / num_items
        result = {"loss": full_loss}
        # compute and print metrics
        all_out = np.concatenate(all_out, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        if self.task_type == "regression":
            r2 = r2_score(all_y, all_out)
            tau = kendalltau(all_y, all_out).statistic
            result["r2"], result["tau"] = r2, tau
            if self.verbose:
                print(f"{self.name} | loss: {loss:.5f} | R2: {r2:.4f} | tau: {tau:.4f}")
        elif self.task_type == "multilabel_binary":
            preds = all_out > 0
            acc = (preds == all_y.astype(bool)).mean()
            result["acc"] = acc
            if self.verbose:
                print(f"{self.name} | loss: {loss:.5f} | acc: {acc:.4f}")
        return result

    @torch.no_grad
    def time_forward(self, model):
        model.eval()
        num_repeats = 100
        for x, y in self.loader:
            bs = get_bs(x)
            x, y = to_device(x, self.device), y.to(self.device)
            out = model(x)
            time_lst = []
            for _ in range(num_repeats):
                start_time = time.time()
                out = model(x)
                time_lst.append(time.time() - start_time)
            print(f"Forward time (bs {bs}): {np.mean(time_lst):.6f}+-{np.std(time_lst):.6f}")
            break


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if args.transform == "flatten":
        transform = FlattenWeights()
    elif args.transform == "mult_flatten":
        transform = MultiplyLoraFlatten(device=device)
    elif args.transform == "pair":
        transform = PairWeights()
    elif args.transform == "svals":
        transform = SingularValues()
    elif args.transform == "align":
        transform = OrthogAlignFlatten(device=device)
    else:
        raise ValueError("Invalid transform")
    
    path = os.path.join(args.root, args.dataset)
    dataset = LoraDataset(root = path, target=args.target, transform=transform)

    if args.transform == "align":
        # save a random template to align weights to
        dataset[random.choice(range(1000))]
        

    if args.target in ("eval_loss", "arc-c-test"):
        task_type = "regression"
        criterion = nn.MSELoss()
        out_dim = 1
    elif "filter_sources" in args.target:
        task_type = "multilabel_binary"
        criterion = nn.BCEWithLogitsLoss()
        # for filter_sources_top, only take top 3 datasets
        if "top" in args.target:
            out_dim = 3
        else:
            out_dim = 19
    else:
        raise ValueError("Invalid target")

    in_dim = dataset.dim_weights
    hidden_dim = args.hidden_dim
    
    n = min(len(dataset), args.max_size)
    dataset = Subset(dataset, range(n))
    print("Num examples:", len(dataset))
    train_prop = .8
    val_prop   = .1
    test_prop  = .1
    train_num = round(train_prop * n)
    val_num   = round(val_prop * n)
    train_dataset = Subset(dataset, range(train_num))
    val_dataset   = Subset(dataset, range(train_num, train_num+val_num))
    test_dataset  = Subset(dataset, range(train_num+val_num, n))

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=2*args.bs, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=2*args.bs, shuffle=False)
    print(f"Size Train|Val|Test: {len(train_dataset)}|{len(val_dataset)}|{len(test_dataset)}")


    
    if args.model == "linear":
        model = nn.Linear(in_dim, out_dim)
    elif args.model == "mlp":
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(args.dropout), nn.ReLU()]
        for _ in range(args.num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(args.dropout), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        model = nn.Sequential(*layers)
    elif "gl_mlp" in args.model:
        # can be "gl_mlp" or "gl_mlp_mean"
        pool_mean = "mean" in args.model
        if "head2" in args.model:
            head_type = 2
        else:
            head_type = 1

        if "1layer" in args.model:
            args.num_layers = 1

        ns, ms = [], []
        for u, v in dataset[0][0]:
            ns.append(len(u))
            if u.ndim == 2:
                ms.append(len(v))
            else:
                ms.append(len(v[0]))

        model = GLInvariantMLP(ns, ms, n_input_layers=56, rank=4, out_dim=out_dim, n_layers=args.num_layers, hidden_dim_equiv=16, hidden_dim_inv=hidden_dim, k=4, d=32, p=args.dropout, pool_mean=pool_mean, head_type=head_type)
    elif args.model == "transformer":
        ns, ms = [], []
        for u, v in dataset[0][0]:
            ns.append(u.numel())
            ms.append(v.numel())
        model = Transformer(ns, ms, out_dim, d_model=hidden_dim, num_heads=8, num_layers=args.num_layers, d_ff=hidden_dim, max_seq_length=56, dropout=args.dropout)
    else:
        raise ValueError("Invalid model (metanet)")
    model = model.to(device)

    print(model)
    print("Num metanet params:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    train_tester = Tester("Train", train_loader, criterion, task_type, device)
    val_tester =   Tester("Val",   val_loader,   criterion, task_type, device)
    test_tester =  Tester("Test",  test_loader,  criterion, task_type, device, verbose=False)

    # test forward pass time:
    train_tester.time_forward(model)
    
    best_val_loss = float('inf')
    best_test_result = {}
    best_epoch = -1
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"### Epoch {epoch} ###")
        train(train_loader, model, criterion, optimizer, device)
        train_tester(model)
        val_result = val_tester(model)
        val_loss = val_result['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_result = val_result
            best_test_result = test_tester(model)
            best_epoch = epoch
        print(f"Elapsed: {time.time() - start_time:.3f}")

    print(f"### Best Val Results: Epoch {best_epoch} ###")
    for k, v in best_val_result.items():
        print(k, v)

    print(f"### Test Results: Epoch {best_epoch} ###")
    for k, v in best_test_result.items():
        print(k, v)
    print(args)
    
    return best_val_result, best_test_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metanet training')
    parser.add_argument('--bs', type=int, default=32,)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--target', type=str, default="eval_loss")
    parser.add_argument('--transform', type=str, default="flatten")
    parser.add_argument('--max_size', type=int, default=10000, help='max dataset size')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to do it')
    parser.add_argument('--root', type=str, default="lora_data/")
    parser.add_argument('--dataset', type=str, default="llama_arc_data")
    args = parser.parse_args()

    def reduce_dict(lst):
        """ given list of dicts,
            compute dict that has same shared keys,
            and each value is (mean, std) of values across that key
        """
        summary = {k: [] for k in lst[0]}
        for d in lst:
            for k in summary:
                summary[k].append(d[k])
        for k, v in summary.items():
            summary[k] = (np.mean(v), np.std(v))
        return summary

    vals = []
    tests = []
    for repeat in range(args.repeats):
        best_val_result, best_test_result = main(args)
        vals.append(best_val_result), tests.append(best_test_result)
    best_val_result = reduce_dict(vals)
    best_test_result = reduce_dict(tests)

    path = "results/"
    os.makedirs(path, exist_ok=True)
    filename = f"{path}{args.target}_{args.transform}_{args.model}_{args.dataset}.csv"
    print("Saving to", filename)
    file_exists = os.path.isfile(filename)
    header = ["lr", "repeats", "val_loss"]
    row = [args.lr, args.repeats, best_val_result['loss']]
    if "r2" in best_val_result:
        header.append("val_r2")
        header.append("test_r2")
        row.append(best_val_result['r2'])
        row.append(best_test_result['r2'])
    if "acc" in best_val_result:
        header.append("val_acc")
        header.append("test_acc")
        row.append(best_val_result['acc'])
        row.append(best_test_result['acc'])
    with open(filename, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
