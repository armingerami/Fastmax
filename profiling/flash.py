import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale


class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,
        device,
        group_size = 1,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        dropout = 0.,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = False,
        laplace_attn_fn = False,
        reduce_group_non_causal_attn = True,
        global_ = True,
        local_ = True,
        conv = False,
        kernel = 63,
        square=False,
        castling = False,
        softmax = False,
        quad = False,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        self.attn_fn = ReLUSquared()

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb

        # norm

        self.dropout = nn.Dropout(dropout)

        # whether to reduce groups in non causal linear attention

        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # projections

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2, device = device),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim, device = device),
            nn.SiLU()
        )

        self.to_out = nn.Linear(hidden_dim, dim, device = device)

        self.global_ = global_
        self.local_ = local_ 
        self.square = square

        self.conv = conv
        if conv:
            res_kernel_size = kernel
            self.dwconv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=1,
            )
            self.conv_mask = torch.ones((1, 1, res_kernel_size, 1), dtype=torch.float32)
            self.conv_mask[:, :, res_kernel_size // 2 + 1 :, :] = 0.0
        self.castling = castling
        self.softmax = softmax
        self.quad = quad
    def forward(
        self,
        x,
        *,
        mask = None
    ):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # prenorm

        normed_x = x

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
        v = v[:,0,:,:]
        qk = self.to_qk(normed_x)

        # offset and scale
        v0 = v

        quad_q, lin_q, quad_k, lin_k = qk.unbind(1)

        # mask out linear attention keys



        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (n g) d -> b n g d', g = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))


        # calculate quadratic attention output

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        attn = self.attn_fn(sim)
        attn = self.dropout(attn)


        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

            # exclusive cumulative sum along group dimension

            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)

            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
        else:
            context_einsum_eq = 'b d e' if self.reduce_group_non_causal_attn else 'b g d e'
            lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k, v) / n
            lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

        # gate
        out = None 
        if self.global_:
            out = lin_attn_out
        if self.local_:
            out = quad_attn_out if out is None else quad_attn_out + lin_attn_out
        
        if self.conv:
            if self.conv_mask.device != out.device:
                self.conv_mask = self.conv_mask.to(out.device)
            self.dwconv.weight.data *= self.conv_mask
            conv_out = self.dwconv(v0.unsqueeze(1)) 
            if self.castling:
                out = 0.5 * v0 + 1.0 / math.pi * out 
                out = out / out.norm(dim=-1, keepdim=True)
                out += conv_out.squeeze(1)
            else:
                out = out + conv_out.squeeze(1)

        out = gate * (out)

        # projection out and residual

        return self.to_out(out) + x