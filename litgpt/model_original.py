# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from litgpt.config import Config

from torch import cuda

# import fastmax_cuda

# class FASTMultiHeadAttention_Function(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q,k,v, drop_noise, mask = False, dropout = 0.0, normalize = False, temperature = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0, p=2):
#         # print(q.get_device())
#         if q.get_device() == -1: 
#             ctx.save_for_backward(q,k,v,q)
#             return q

#         b = 0
#         if len(q.shape) == 4:
#             b = q.shape[0]
#             q = q.reshape((q.shape[0]*q.shape[1],q.shape[2],q.shape[3])) # (b,h,n,d) -> (b*h,n,d)
#             k = k.reshape((k.shape[0]*k.shape[1],k.shape[2],k.shape[3])) # (b,h,n,d) -> (b*h,n,d)
#             v = v.reshape((v.shape[0]*v.shape[1],v.shape[2],v.shape[3])) # (b,h,n,d) -> (b*h,n,d)
#             drop_noise = drop_noise.reshape((drop_noise.shape[0]*drop_noise.shape[1],drop_noise.shape[2],drop_noise.shape[3])) # (b,h,n,d) -> (b*h,n,d)
#         elif len(q.shape) != 3: print("q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d).")


#         q = q.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
#         k = k.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
#         v = v.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
#         drop_noise = drop_noise.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        
#         o = fastmax_cuda.forwardpass(q,k,v,drop_noise,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
#         ctx.save_for_backward(q,k,v,o)
#         ctx.mask = mask
#         ctx.p = p
#         ctx.b = b
#         ctx.t = temperature
#         ctx.a0 = a0
#         ctx.a1 = a1
#         ctx.a2 = a2
#         o = o[:,:,:q.shape[2]]
#         o = o.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
#         if b != 0: o = o.reshape((b,int(o.shape[0]/b),o.shape[1],o.shape[2])) # (b*h,n,d) -> (b,h,n,d)
#         return o


#     @staticmethod
#     def backward(ctx, grad_output):
#         q,k,v,o = ctx.saved_tensors
#         if q.get_device() == -1: 
#             return q, q, q, None, None, None, None, None, None, None, None, None, None

#         mask = ctx.mask
#         p = ctx.p
#         b = ctx.b
#         t = ctx.t
#         a0 = ctx.a0
#         a1 = ctx.a1
#         a2 = ctx.a2

#         if(b != 0): grad_output = grad_output.reshape((grad_output.shape[0]*grad_output.shape[1],grad_output.shape[2],grad_output.shape[3])).contiguous()
#         grad_output = grad_output.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
#         gradq, gradk, gradv = fastmax_cuda.backwardpass(q,k,v,o,grad_output,mask,a0,a1,a2,p)

#         gradq = gradq.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
#         gradk = gradk.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
#         gradv = gradv.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)

#         if(b != 0):
#           gradq = gradq.reshape((b,int(gradq.shape[0]/b),gradq.shape[1],gradq.shape[2])).contiguous()
#           gradk = gradk.reshape((b,int(gradk.shape[0]/b),gradk.shape[1],gradk.shape[2])).contiguous()
#           gradv = gradv.reshape((b,int(gradv.shape[0]/b),gradv.shape[1],gradv.shape[2])).contiguous()
        
#         return gradq, gradk/t, gradv, None, None, None, None, None, None, None, None, None, None
    
   
# def fastmax_function(q, k, v, mask=0, dropout_rate = 0.0, normalize=0, temperature=1, a0=1,a1=1,a2=0.5,lim=1,p=2, create_attn_matrix = 0):
#     """
#     Input: query, key, and value matrices (b, h, n, d)
#         b: batch size
#         h: number of heads
#         n: number of tokens
#         d: dimension per attention head (d = d_model / h)
#     mask: boolean indicating whether to apply causal masking
#     temperature: Hyperparameter to control the standard deviation of <q, k>; stdev(<q, k>) = 1/temperature
#         Stdev of <q, k> is important in general with attention, but even more so when using a taylor
#         expansion to approximate an exponential because the error increases with the stdev of <q, k>.
#         In normal attention, stdev equates to the "temperature" of the softmax function, and with a
#         taylor approximation, higher temperature also means we drift further from the true softmax.
#         For positive inputs, this drifting error actually lowers the temperature, and for negative inputs
#         it raises the temperature.
#     Output: The result of Attention matrix * Value (b, h, n, d)
#     """
#     if create_attn_matrix == 0:
#         if normalize == 1:
#             temperature = 1
#             qn = torch.linalg.norm(q, dim = 3)
#             kn = torch.linalg.norm(k, dim = 3)
#             q = lim*q/torch.linalg.norm(qn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
#             k = lim*k/torch.linalg.norm(kn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
#         else:
#             temperature = temperature*math.sqrt(q.shape[3])
#             temperature = 1
#         temperature2 = temperature*temperature

#         # Prepare the quadratic terms with respect to k and q:
#         if p == 2:
#             # Prepare the quadratic terms with respect to k and q:
#             k2 = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
#             k2 = k2.flatten(-2)                     # (b, h, n, d*d)
#             q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
#             q2 = q2.flatten(-2)                     # (b, h, n, d*d)
#             drop_attn = torch.nn.Dropout(p=dropout_rate)
#             k2 = drop_attn(k2)
#             q2 = drop_attn(q2)

#             if mask == 0:
#                 first_term = a0*torch.sum(v,-2)  # (b, h, d)

#                 second_term = a1*torch.matmul(k.swapaxes(-2,-1),v)/temperature  # (b, h, d, d)

#                 third_term = a2*torch.matmul(k2.swapaxes(-2,-1),v)/temperature2  # (b, h, d^2, d)

#                 div1 = a0*torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
#                 div2 = a1*torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)
#                 div3 = a2*torch.sum(k2,-2).unsqueeze(-1) # (b, h, d^2, 1)

#                 ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
#                 ans3 = torch.matmul(q2,third_term)  # (b, h, n, d)
#                 div2 = torch.matmul(q,div2)/(temperature) # (b, h, n, 1)
#                 div3 = torch.matmul(q2,div3)/(temperature2) # (b, h, n, 1)

#                 ans = ans2+ans3 # (b, h, n, d)
#                 ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
#                 div = div2+div3 # (b, h, n, d)
#                 div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
#                 ans = ans/div # (b, h, n, d)

#             else:
#                 first = a0*torch.cumsum(v,2) # (b, h, n, d)
#                 second = a1*torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/temperature # (b, h, n, d)
#                 third = a2*torch.einsum("bhij,bhijk -> bhik",[q2,torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k2,v]),2)])/temperature2 # (b, h, n, d)

#                 kcs = torch.cumsum(k,-2) # (b, h, n, d)
#                 k2cs = torch.cumsum(k2,-2) # (b, h, n, d^2)
#                 div1 = a0*torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
#                 div2 = a1*torch.einsum("bhij,bhij -> bhi",[q,kcs])/temperature # (b, h, n)
#                 div3 = a2*torch.einsum("bhij,bhij -> bhi",[q2,k2cs])/temperature2 # (b, h, n)
#                 div = (div1 + div2 + div3).unsqueeze(-1) # (b, h, n, 1)

#                 ans = first + second + third # (b, h, n, d)
#                 ans /= div # (b, h, n, d)
            
#         # Taylor series with constant and linear terms:
#         elif p == 1:
#             drop_attn = torch.nn.Dropout(p=dropout_rate)
#             k = drop_attn(k)
#             q = drop_attn(q)
#             if mask is None or not mask:
#                 first_term = a0*torch.sum(v,-2)  # (b, h, d)
#                 second_term = a1*torch.matmul(k.swapaxes(-2,-1),v)/temperature  # (b, h, d, d)

#                 div1 = a0*torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
#                 div2 = a1*torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)

#                 ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
#                 div2 = torch.matmul(q,div2)/(temperature) # (b, h, n, 1)

#                 ans = ans2 # (b, h, n, d)
#                 ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
#                 div = div2 # (b, h, n, d)
#                 div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
#                 ans = ans/div # (b, h, n, d)

#             else:
#                 first = a0*torch.cumsum(v,2) # (b, h, n, d)
#                 second = a1*torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/temperature # (b, h, n, d)

#                 kcs = torch.cumsum(k,-2) # (b, h, n, d)
#                 div1 = a0*torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
#                 div2 = a1*torch.einsum("bhij,bhij -> bhi",[q,kcs])/temperature # (b, h, n)
#                 div = (div1 + div2).unsqueeze(-1) # (b, h, n, 1)

#                 ans = first + second # (b, h, n, d)
#                 ans /= div # (b, h, n, d)
        
#         else:
#             raise ValueError(f"p must be 1 or 2, got: {p}")
#         return ans

#     else:
#         # temperature = temperature*math.sqrt(q.shape[3])
#         temperature2 = temperature*temperature

#         k2 = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
#         k2 = k2.flatten(-2)                     # (b, h, n, d*d)
#         q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
#         q2 = q2.flatten(-2)    
#         attn = a0 + a1*torch.matmul(q, torch.swapaxes(k, -2, -1))/temperature + a2*torch.matmul(q2, torch.swapaxes(k2, -2, -1))/temperature2
#         if mask is not None:
#             attn = torch.where(mask == 0, 0, attn)
#         attn /= (torch.sum(attn, axis=3)).unsqueeze(-1)
#         ans = torch.matmul(attn,v)
#         return ans, attn

# class FASTMultiHeadAttention(torch.nn.Module):
#     def __init__(self, use_custom_gradient = True):
#         super(FASTMultiHeadAttention, self).__init__()
#         self.use_custom_gradient = use_custom_gradient

#     def forward(self, q,k,v,drop_noise=None, mask = False, dropout = 0.0, normalize = False, temperature = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0,p=2):
#         if self.use_custom_gradient: return FASTMultiHeadAttention_Function.apply(q,k,v,drop_noise,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
#         else: return fastmax_function(q,k,v,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)


# fastmax_custom = FASTMultiHeadAttention()

class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}."
                             " This is likely because the input text exceeds the supported context length of this model.")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)  # (b, t, vocab_size)
        if self.config.final_logit_softcapping is not None:
            x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
        return x

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)
        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and
            block_idx % config.sliding_window_layer_placing == 0
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
            sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None, print_flag = False
    ) -> torch.Tensor:
        
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
            scores = q @ k.mT * scale
            scores = (
                torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            )
            if mask is None:
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
            )
        if print_flag:
            print(y.shape)
        return y.transpose(1, 2)

        # if(mask is None): mask_fastmax = False
        # else: mask_fastmax = True
        # dropout = 0.2 # between 0 and 1
        # normalize = True
        # temperature = 1.0
        # a0 = 1.0
        # a1 = 1.0
        # a2 = 0.5
        # lim = 1.0
        # p = 1

        # drop_noise = torch.normal(0,1,size=(q.shape),dtype=q.dtype,device=q.device).contiguous()
        # output = fastmax_custom(q,k,v,drop_noise,mask_fastmax,dropout,normalize,temperature,a0,a1,a2,lim,p)
        # return output.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def batched_index_select(t, dim, idx):
    """index_select for batched index and unbatched t"""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
    dims = [dim] + list(range(res.dim()))
    del dims[dim + 1]
    res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def batched_index_copy_(t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)
        t_indexed_dim = t.size(dim)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        n = k.size(0)
        k = batched_index_copy_(self.k[:n, ...], -2, input_pos, k)
        v = batched_index_copy_(self.v[:n, ...], -2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)




# # Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# """Full definition of a decoder-only transformer-based language model, all of it in this single file.

# Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
# https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
# """

# import math
# from typing import Any, Optional, Tuple

# import torch
# import torch.nn as nn
# from typing_extensions import Self

# from litgpt.config import Config


# class GPT(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         assert config.padded_vocab_size is not None
#         self.config = config

#         self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
#         self.transformer = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
#         self.max_seq_length = self.config.block_size
#         self.mask_cache: Optional[torch.Tensor] = None

#     @property
#     def max_seq_length(self) -> int:
#         return self._max_seq_length

#     @max_seq_length.setter
#     def max_seq_length(self, value: int) -> None:
#         """
#         When doing inference, the sequences used might be shorter than the model's context length.
#         This allows setting a smaller number to avoid allocating unused memory
#         """
#         if value > self.config.block_size:
#             raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}."
#                              " This is likely because the input text exceeds the supported context length of this model.")
#         self._max_seq_length = value
#         if not hasattr(self, "cos"):
#             # first call
#             cos, sin = self.rope_cache()
#             self.register_buffer("cos", cos, persistent=False)
#             self.register_buffer("sin", sin, persistent=False)
#         # override
#         elif value != self.cos.size(0):
#             self.cos, self.sin = self.rope_cache(device=self.cos.device)
#         # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
#         # if the kv cache is expected

#     def reset_parameters(self) -> None:
#         # Trigger resetting the rope-cache
#         self.cos, self.sin = self.rope_cache(device=self.cos.device)

#     def _init_weights(self, module: nn.Module) -> None:
#         """Meant to be used with `gpt.apply(gpt._init_weights)`."""
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
#         T = idx.size(1)
#         if self.max_seq_length < T:
#             raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

#         if input_pos is not None:  # use the kv cache
#             cos = batched_index_select(self.cos, 0, input_pos)
#             sin = batched_index_select(self.sin, 0, input_pos)
#             if self.mask_cache is None:
#                 raise TypeError("You need to call `gpt.set_kv_cache()`")
#             mask = batched_index_select(self.mask_cache, 2, input_pos)
#             if mask.dim() > 4:
#                 # the mask cache has a batch dim of 1 in addition to the one
#                 # we get if input_pos has a batch dimension
#                 mask = mask.squeeze(1)
#         else:
#             cos = self.cos[:T]
#             sin = self.sin[:T]
#             mask = None

#         x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
#         if self.config.scale_embeddings:
#             x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

#         for block in self.transformer.h:
#             x = block(x, cos, sin, mask, input_pos)
#         x = self.transformer.ln_f(x)
#         x = self.lm_head(x)  # (b, t, vocab_size)
#         if self.config.final_logit_softcapping is not None:
#             x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
#         return x

#     @classmethod
#     def from_name(cls, name: str, **kwargs: Any) -> Self:
#         return cls(Config.from_name(name, **kwargs))

#     def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         return build_rope_cache(
#             seq_len=self.max_seq_length,
#             n_elem=self.config.rope_n_elem,
#             device=device,
#             condense_ratio=self.config.rope_condense_ratio,
#             base=self.config.rope_base,
#         )

#     def set_kv_cache(
#         self,
#         batch_size: int,
#         max_seq_length: Optional[int] = None,
#         rope_cache_length: Optional[int] = None,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         if rope_cache_length is None:
#             rope_cache_length = self.cos.size(-1)

#         if max_seq_length is None:
#             max_seq_length = self.max_seq_length

#         # initialize the kv cache for all blocks
#         for block in self.transformer.h:
#             block.attn.kv_cache = block.attn.build_kv_cache(
#                 batch_size, max_seq_length, rope_cache_length, device, dtype,
#             )

#         if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
#             # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
#             # for the kv-cache support (only during inference), we only create it in that situation
#             self.mask_cache = build_mask_cache(max_seq_length, device)

#     def clear_kv_cache(self) -> None:
#         self.mask_cache = None
#         for block in self.transformer.h:
#             block.attn.kv_cache = None


# class Block(nn.Module):
#     def __init__(self, config: Config, block_idx: int) -> None:
#         super().__init__()
#         if not config.parallel_residual and config.shared_attention_norm:
#             raise NotImplementedError(
#                 "No checkpoint amongst the ones we support uses this configuration"
#                 " (non-parallel residual and shared attention norm)."
#             )

#         self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
#         self.attn = CausalSelfAttention(config, block_idx)
#         self.post_attention_norm = (
#             config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
#         )
#         self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
#         self.mlp = config.mlp_class(config)
#         self.post_mlp_norm = (
#             config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
#         )

#         self.config = config

#     def forward(
#         self,
#         x: torch.Tensor,
#         cos: torch.Tensor,
#         sin: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         input_pos: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Non-parallel residual       Parallel residual
#            ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
#            │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
#            │  norm_1                │  norm_1  ───────►    norm_2
#            │  ↓                     │  ↓                   ↓
#            │  attn                  │  attn                MLP
#            │  ↓                     │  ↓                   ↓
#            |  post_attn_norm        |  post_attn_norm      post_mlp_norm
#            |  ↓                     |  ↓                   ↓
#         ┌─ └► +                     └► + ◄─────────────────┘
#         |     ↓
#         │     norm_2
#         │     ↓
#         │     MLP
#         │     ↓
#         |     post_mlp_norm
#         |     ↓
#         └───► +
#         """

#         x_normed = self.norm_1(x)
#         attention_output = self.attn(x_normed, cos, sin, mask, input_pos)
#         attention_output = self.post_attention_norm(attention_output)

#         if self.config.parallel_residual:
#             x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
#             x = self.mlp(x_normed) + attention_output + x
#         else:
#             x = attention_output + x
#             x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
#         return x


# class CausalSelfAttention(nn.Module):
#     def __init__(self, config: Config, block_idx: int) -> None:
#         super().__init__()
#         shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
#         # key, query, value projections for all heads, but in a batch
#         self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
#         # output projection
#         # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
#         self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
#         # disabled by default
#         self.kv_cache: Optional[KVCache] = None
#         self.apply_sliding_window_attention = (
#             config.sliding_window_size is not None and
#             block_idx % config.sliding_window_layer_placing == 0
#         )

#         self.config = config

#     def forward(
#         self,
#         x: torch.Tensor,
#         cos: torch.Tensor,
#         sin: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         input_pos: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

#         qkv = self.attn(x)

#         # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
#         q_per_kv = self.config.n_head // self.config.n_query_groups
#         total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
#         qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
#         qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

#         # split batched computation into three
#         q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

#         # maybe repeat k and v if for the non multi-head attention cases
#         # training: flash attention requires it
#         # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
#         if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
#             k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
#             v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

#         q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
#         k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
#         v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

#         q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
#         k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
#         q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
#         k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

#         if input_pos is not None:
#             if not isinstance(self.kv_cache, KVCache):
#                 raise TypeError("You need to call `gpt.set_kv_cache()`")
#             k, v = self.kv_cache(input_pos, k, v)

#         if self.apply_sliding_window_attention:
#             """
#                   Global Window              Sliding window             Sliding window
#                   attention mask      +            bias          =      attention mask
#             ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
#             │ True False False False │  │ True  True  True True │  │ True  False False False │
#             │ True True  False False │  │ True  True  True True │  │ True  True  False False │
#             │ True True  True  False │  │ False True  True True │  │ False True  True  False │
#             │ True True  True  True  │  │ False False True True │  │ False False True  True  │
#             └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
#             """
#             if mask is None:
#                 mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
#                 mask.masked_fill_(mask.bool(), float("-inf"))
#             sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
#             sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
#             mask += sliding_window_bias

#         y = self.scaled_dot_product_attention(q, k, v, mask)

#         y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

#         # output projection
#         return self.proj(y)

#     def scaled_dot_product_attention(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

#         # with softcapping we cannot use SDPA
#         if self.config.attention_logit_softcapping is not None:
#             scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
#             scores = q @ k.mT * scale
#             scores = (
#                 torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
#             )
#             if mask is None:
#                 mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
#                 mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
#             scores = scores + mask
#             scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
#             y = scores @ v
#         else:
#             y = torch.nn.functional.scaled_dot_product_attention(
#                 q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
#             )
#         return y.transpose(1, 2)

#     def build_kv_cache(
#         self,
#         batch_size: int,
#         max_seq_length: int,
#         rope_cache_length: Optional[int] = None,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> "KVCache":
#         heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
#         v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
#         if rope_cache_length is None:
#             if self.config.rotary_percentage != 1.0:
#                 raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
#             k_shape = v_shape
#         else:
#             k_shape = (
#                 batch_size,
#                 heads,
#                 max_seq_length,
#                 rope_cache_length + self.config.head_size - self.config.rope_n_elem,
#             )
#         return KVCache(k_shape, v_shape, device=device, dtype=dtype)


# class GptNeoxMLP(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
#         self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

#         self.config = config

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc(x)
#         x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
#         return self.proj(x)


# class LLaMAMLP(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
#         self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
#         self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

#         self.config = config

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_fc_1 = self.fc_1(x)
#         x_fc_2 = self.fc_2(x)
#         x = torch.nn.functional.silu(x_fc_1) * x_fc_2
#         return self.proj(x)


# class GemmaMLP(LLaMAMLP):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_fc_1 = self.fc_1(x)
#         x_fc_2 = self.fc_2(x)
#         x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
#         return self.proj(x)


# class LLaMAMoE(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
#         self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

#         self.config = config

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
#         See also figure 1 in https://arxiv.org/abs/2211.15841
#         """
#         B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
#         x = x.view(-1, C)  # (B*T, C)
#         router = self.gate(x)  # (B*T, n_expert)
#         probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
#         probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
#         masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
#         masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
#         y = torch.zeros_like(x)  # (B*T, C)
#         for mask, expert in zip(masks, self.experts):
#             token_idx, expert_idx = torch.where(mask)
#             y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
#         return y.view(B, T, C)


# def build_rope_cache(
#     seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Enhanced Transformer with Rotary Position Embedding.

#     Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
#     transformers/rope/__init__.py. MIT License:
#     https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
#     """
#     # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
#     theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

#     # Create position indexes `[0, 1, ..., seq_len - 1]`
#     seq_idx = torch.arange(seq_len, device=device) / condense_ratio

#     # Calculate the product of position index and $\theta_i$
#     idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

#     return torch.cos(idx_theta), torch.sin(idx_theta)


# def batched_index_select(t, dim, idx):
#     """index_select for batched index and unbatched t"""
#     if idx.dim() == 1:
#         return torch.index_select(t, dim, idx)

#     *batch_shape, idx_size = idx.shape
#     res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
#     # split out single batch idx
#     res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
#     # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
#     dims = [dim] + list(range(res.dim()))
#     del dims[dim + 1]
#     res = res.permute(dims)
#     # unflatten batch dims
#     res = res.view(*batch_shape, *res.shape[1:])
#     return res


# def batched_index_copy_(t, dim, idx, val):
#     """Index copy for batched t, idx, val"""

#     if t.device.type == "mps":
#         # Normalize negative dimensions
#         if dim < 0:
#             dim = t.dim() + dim
#         if idx.dim() == 1:
#             idx_shape = [1] * val.dim()
#             idx_shape[dim] = -1
#             idx_expanded = idx.view(*idx_shape)
#             idx_expanded = idx_expanded.expand_as(val)
#             t.scatter_(dim, idx_expanded, val)
#             return t

#         elif idx.dim() == 2:
#             assert dim != 0, "Cannot index the batch dimension"
#             batch_size = idx.size(0)
#             idx_size = idx.size(1)
#             assert batch_size == t.size(0) == val.size(0)

#             idx_shape = [batch_size] + [1] * (val.dim() - 1)
#             idx_shape[dim] = idx_size
#             idx_expanded = idx.view(*idx_shape)
#             idx_expanded = idx_expanded.expand_as(val)

#             t.scatter_(dim, idx_expanded, val)
#             return t
#         else:
#             raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

#     else:
#         if idx.dim() == 1:
#             return t.index_copy_(dim, idx, val)

#         assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
#         assert dim != 0, f"cannot index batch dim"
#         batch_size, idx_size = idx.shape
#         assert batch_size == t.size(0)
#         assert batch_size == val.size(0)
#         t_indexed_dim = t.size(dim)

#         # if we can view the batch and indexed dimensions together, we could
#         # do index trickery. This is, sadly, not the case for kvcache so we
#         # fall back to for loop
#         for i in range(batch_size):
#             unbatched_dim = dim if dim < 0 else dim - 1
#             t[i].index_copy_(unbatched_dim, idx[i], val[i])
#         return t


# def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
#     head_size = x.size(-1)
#     x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
#     x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
#     rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
#     if cos.dim() > 1:
#         # batch dimensions must align
#         # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
#         # we count from back because all of apply_rope does
#         cos = cos.unsqueeze(-3)
#         sin = sin.unsqueeze(-3)

#     roped = (x * cos) + (rotated * sin)
#     return roped.to(dtype=x.dtype)


# class KVCache(nn.Module):
#     def __init__(
#         self,
#         k_shape: Tuple[int, int, int, int],
#         v_shape: Tuple[int, int, int, int],
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         super().__init__()
#         self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
#         self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

#     def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # move the buffer to the activation dtype for when AMP is used
#         self.k = self.k.to(k.dtype)
#         self.v = self.v.to(v.dtype)
#         # update the cache
#         n = k.size(0)
#         k = batched_index_copy_(self.k[:n, ...], -2, input_pos, k)
#         v = batched_index_copy_(self.v[:n, ...], -2, input_pos, v)
#         return k, v

#     def reset_parameters(self) -> None:
#         torch.nn.init.zeros_(self.k)
#         torch.nn.init.zeros_(self.v)


# def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
#     ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
#     return torch.tril(ones).unsqueeze(0).unsqueeze(0)


# class RMSNorm(torch.nn.Module):
#     """Root Mean Square Layer Normalization.

#     Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
#     https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
#     """

#     def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.ones(size))
#         self.eps = eps
#         self.dim = dim
#         self.add_unit_offset = add_unit_offset

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         dtype = x.dtype
#         x = x.float()
#         # NOTE: the original RMSNorm paper implementation is not equivalent
#         norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
#         x_normed = x * torch.rsqrt(norm_x + self.eps)
#         weight = (1 + self.weight) if self.add_unit_offset else self.weight
#         return (x_normed * weight.float()).to(dtype=dtype)

#     def reset_parameters(self) -> None:
#         torch.nn.init.ones_(self.weight)
