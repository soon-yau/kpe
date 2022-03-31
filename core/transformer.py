import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class PyTorchTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ffwd_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.Sequential(*[torch.nn.TransformerEncoderLayer(\
                        d_model, num_heads, ffwd_dim, batch_first=True) for _ in range(num_layers)])
            
    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)


class SelfAttention(nn.Module):
    '''
        input = q, k, v each has dim (d_model)
        output = dim (d_model//head)
    '''
    def __init__(self, d_model, d_out):
        super().__init__()
        self.query = nn.Linear(d_model, d_out)
        self.key = nn.Linear(d_model, d_out)
        self.value = nn.Linear(d_model, d_out)
        
    def forward(self, q, k, v, mask=None):
        '''
        mask is 1-d array, where 0 is to mask out unwanted tokens
        '''
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        
        scale = key.size(-1) ** 0.5
        key_T = rearrange(key, 'b h w -> b w h')
        
        attn = torch.bmm(query, key_T)/scale
        
        if mask is not None:
            #mask = rearrange(mask, 'm -> () () m').expand(attn.size())
            attn = attn.masked_fill(mask==0, -float("Inf"))
            
        attn = F.softmax(attn, dim = -1)
        
        outputs = torch.bmm(attn, value)
        return outputs

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.0):        
        super().__init__()        
        assert d_model%num_heads == 0, \
                f'd_model({d_model}) must be multiples of num_heads({num_heads})'
        
        d_attn =  d_model//num_heads
        self.attns = nn.ModuleList([SelfAttention(d_model, d_attn) for _ in range(num_heads)])
        self.linear = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.Dropout(dropout))
        
        
    def forward(self, q, k, v, mask=None):
        
        outputs = torch.cat([layer(q, k, v, mask) for layer in self.attns], dim=-1)
        
        return self.linear(outputs)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffwd_dim, attn_dropout=0.0, ffwd_dropout=0.0):
        super().__init__()
        self.attn = MultiHeadedAttention(d_model, num_heads, attn_dropout)
        self.ffwd = nn.Sequential(
                        nn.Linear(d_model, ffwd_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(ffwd_dropout),
                        nn.Linear(ffwd_dim, d_model))
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffwd_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = self.attn_norm(self.attn(x, x, x, mask) + x)
        
        x = self.ffwd_norm(self.ffwd(x) + x)
        
        return x    
    
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ffwd_dim, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(\
                        [EncoderLayer(d_model, num_heads, ffwd_dim, dropout, dropout) \
                                     for _ in range(num_layers)])
            
    def forward(self, x, mask=None):

        batch_size, seq_len, _= x.size()

        if mask is not None:
            mask = rearrange(mask, 'm -> () () m').expand(batch_size, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)
        return x

