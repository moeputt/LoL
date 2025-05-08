import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from tqdm import tqdm
from safetensors.numpy import load_file
def get_standard_shapes(point):
    ns = []
    ms = []
    for u, v in point:
        ns.append(u.nelement())
        ms.append(v.nelement())
    return ns, ms
def to_cuda(ls):
    if  type(ls) != list:
        return ls.cuda()
    else:
        return [(u.cuda(), v.cuda()) for u,v in ls]




def train_clip(model, device, train_set, optimizer, epochs, batch_size=100):
    model.train()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last=True)
    loss_fn = nn.MSELoss()
    for epoch in tqdm(range(epochs), desc = "Training"):
        tot_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = to_cuda(data)
            optimizer.zero_grad()
            output = model(data)
            target = (target).float().to(device)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        tqdm.write(f"EPOCH {epoch}: Loss {tot_loss/len(train_loader)}")
def valid_clip(model, device, valid_set, num_pred = 1):
    model.eval()
    test_loss = 0
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 8, shuffle = True)
    loss_fn = nn.MSELoss(reduction = 'sum')
    with torch.no_grad():
    
        for data, target in valid_loader:
            data = to_cuda(data)
            output = model(data)
            target = (target).float().to(device)
            test_loss += loss_fn(output, target)

    test_loss /= len(valid_loader.dataset) * num_pred

    print('\Valid set: Average loss: {:.4f}'.format(
        test_loss))
    return test_loss
def test_clip(model, device,  test_set, num_pred = 1):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 8, shuffle = True)
    loss_fn = nn.MSELoss(reduction = 'sum')
    all_outs = [] 
    with torch.no_grad():
        for data, target in test_loader:
            data = to_cuda(data)
            output = model(data)
            target = target.float().to(device)
            test_loss += loss_fn(output, target)
              # get the index of the max log-probability
            correct += F.sigmoid(output).round().eq(target).sum().item()
            all_outs.append([output.cpu().numpy(), target.cpu().numpy()])
    test_loss /= len(test_loader.dataset) * num_pred
    tau, r = (calculate_kendall_spearman(all_outs))
    print('\nTest set: Average loss: {:.4f}, R^2: {:.4f}, T: {:.4f}\n'.format(
        test_loss, r, tau))
    return test_loss, r, tau






def train(model, device, train_set, optimizer, epochs, batch_size=100, valid_set = None):
    model.train()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last=True)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(epochs), desc = "Training"):
        tot_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = to_cuda(data)
            optimizer.zero_grad()
            output = model(data)
            target = (target).float().to(device)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        tqdm.write(f"EPOCH {epoch}: Loss {tot_loss/len(train_loader)}")
        valid(model, device, valid_set)
def valid(model, device, valid_set, num_pred = 10):
    model.eval()
    test_loss = 0
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 8, shuffle = True)
    correct = 0
    loss_fn = nn.BCEWithLogitsLoss(reduction = 'sum')
    with torch.no_grad():
    
        for data, target in valid_loader:
            data = to_cuda(data)
            output = model(data)
            target = (target).float().to(device)
            
            test_loss += loss_fn(output, target)
            pred = F.sigmoid(output).round()  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
    test_loss /= len(valid_loader.dataset) * num_pred

    print('\Valid set: Average loss: {:.4f}, Acc : {:.4f}, {:.4f}/{:4f}\n'.format(
        test_loss, correct/(len(valid_set)*num_pred), correct, len(valid_set) * num_pred))
    return test_loss
def test(model, device, test_set, num_pred = 1):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 8, shuffle = True)
    loss_fn = nn.BCEWithLogitsLoss(reduction = 'sum')
    all_outs = [] 
    with torch.no_grad():
        for data, target in test_loader:
            data = to_cuda(data)
            output = model(data)
            target = target.float().to(device)
            test_loss += loss_fn(output, target)
              # get the index of the max log-probability
            correct += F.sigmoid(output).round().eq(target).sum().item()
            all_outs.append([output.cpu().numpy(), target.cpu().numpy()])
    test_loss /= len(test_loader.dataset) * num_pred
    print('\nTest set: Average loss: {:.4f}, Acc : {:.4f}, {:.4f}/{:4f}\n'.format(
        test_loss, correct/(len(test_set)*num_pred), correct, (len(test_set)*num_pred)))
    return test_loss, correct/(len(test_set)*num_pred)



from scipy import stats
def calculate_kendall_spearman(all_outs):
    outs = []
    targets = []
    for out, target in all_outs:
        outs.append(out)
        targets.append(target)
    outs = np.concatenate(outs, axis = 0).flatten()
    targets = np.concatenate(targets, axis = 0).flatten()
    return stats.kendalltau(outs, targets).correlation, stats.spearmanr(outs, targets).correlation
class AlignedModelDataset(torch.utils.data.Dataset):
    def __init__(self, aligner, dataset, batch_size = 1):
        self.aligner = aligner
        self.dataset = dataset
        self.data = []
        loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)
        
        for index, (data, target) in enumerate(loader):
            if index%10 == 0:
                print(index * batch_size)
            aligned_data = self.aligner(data)
            b = len(target)
            for i in range(b):
                new_uv = [(u[i].cpu(), v[i].cpu()) for u, v in aligned_data]
                self.data.append((new_uv, target[i]))
                
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
u_std, v_std = [0.0046, 0.0235]
def svd(a,b):
    R_1 = torch.linalg.qr(a).R
    R_2 = torch.linalg.qr(b).R
    return torch.linalg.svd(R_1@ R_2.T).S
def get_uvs_from_file(path, do_svd = False):
    new_path = os.path.join(path, 'adapter_model.safetensors')
    if not os.path.exists(new_path):
        return None
    uvs = []
    tensors =  list(load_file(new_path).values())
    for i in range(len(tensors)//2):

        t1 = torch.tensor(tensors[2*i+1]) 
        t2 = torch.tensor(tensors[2*i])
        if do_svd:
            uvs.append(svd(t1,t2.T))
        else:
            uvs.append((t1,t2.T))
    if do_svd:
        return torch.cat(uvs, dim = -1)
    else:
        return uvs
def get_tensors_from_file(path):
    new_path = os.path.join(path, 'adapter_model.safetensors')
    if not os.path.exists(new_path):
        return None
    tensors =  list(load_file(new_path).values())
    uvs = []
    for i in range(len(tensors)//2):
        t1 = torch.tensor(tensors[2*i+1]).cuda()
        t2 = torch.tensor(tensors[2*i]).cuda()
        t3 = t1 @ t2
        uvs.append(t3.flatten())
    return torch.cat(uvs)

def get_equiv_shapes(point):
    ns = []
    ms = []
    for u, v in point:
        ns.append(len(u))
        if len(u.shape) == 2:
            ms.append(len(v))
        else:
            ms.append(len(v[0]))
    return ns, ms