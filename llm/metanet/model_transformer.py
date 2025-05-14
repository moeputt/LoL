import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class Embedding(nn.Module):
    def __init__(self, ns, ms, d_model = 1024):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.ns = ns
        self.ms = ms
        for (n,m) in zip(ns, ms):
            self.mlps.append(MLP(n+m, d_model, d_model))
    def forward(self, uvs):
        tot = [] #b x (n+m) x 4
        for (u,v), mlp in zip(uvs, self.mlps):
            uv = torch.cat([u,v],dim = 1).flatten(start_dim = 1)
            tot.append(mlp(uv))
        tot = torch.stack(tot, dim = 1)
        return tot
class ReverseEmbedding(nn.Module):
    def __init__(self, input_dims, dim = 2048):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.input_dims = input_dims
        for input_dim in input_dims:
            self.mlps.append(MLP(dim, dim*2, input_dim))
    def forward(self, tensors):
        tot = []
        uvs = tensors.swapaxes(0,1)
        for uv, mlp in zip(uvs, self.mlps):
            tot.append(mlp(uv))
        return tot
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = MLP(d_model, d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, ns, ms, out_dim, d_model = 1024, num_heads = 8, num_layers = 6, d_ff = 1024, max_seq_length = 64, dropout = 0):
        super(Transformer, self).__init__()
        self.embedding = Embedding(ns, ms, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_layer = MLP(d_model * max_seq_length, 1024, out_dim)
        
        
    def create_positional_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x, mask = None):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        x = x.flatten(start_dim = 1)
        return self.final_layer(x)
