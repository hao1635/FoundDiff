
### 两个条件,c是第四层特征cross-attention，t=p+t,
import copy
import math
import os, glob, shutil
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import wandb

import Augmentor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from datasets.get_dataset import dataset
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm

#from data.mayo16_dataset import Mayo16Dataset
from data.pdf_dataset import PDFDataset
import ipdb

from . import util
from .util import get_logger
#from .mamba import VSSBlock,SS2D,CAB
from .emamba2 import SS2D

from .model_clipiqa import load
# from .model_clipiqalora3 import CLIPIQA
from .model_clipiqa3 import CLIPIQA

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

import lpips



ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions

def make_dir(path, refresh=False):
    
    """ function for making directory (to save results). """
    
    try: os.mkdir(path)
    except: 
        if(refresh): 
            shutil.rmtree(path)
            os.mkdir(path)

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules





def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g



# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered



    

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class TransposedAttention(nn.Module):
    def __init__(self, dim, heads, bias=False):
        super(TransposedAttention, self).__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x,c=None):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x,c=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=256, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        b,c,H,W=x.shape
        x = rearrange(x, 'b c h w -> b (h w) c') #b,l,c

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out =self.to_out(out)

        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

def NonLinearity(inplace=False):
    return nn.SiLU(inplace)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(time_emb_dim, dim_out * 2)
        # ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        #self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # compress_ratio=4
        # num_feat=input_channel
        # self.cab = nn.Sequential(
        #     nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
        #     nn.GELU(),
        #     nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
        #     ChannelAttention(num_feat, squeeze_factor=16)
        # )

    def forward(self, x, time_emb=None):

        scale_shift = None
        # if exists(self.mlp) and exists(time_emb):
        #     time_emb = self.mlp(time_emb)
        #     time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        #     scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        #h = self.block2(h)

        return h + self.res_conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

class Mamba_block(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, d_state,expand,dropout,cross=False,time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mamba = SS2D(d_model=hidden_size, d_state=d_state,expand=expand,dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * hidden_size, bias=True)
        )
        self.cross=cross
        heads=hidden_size//32
        self.attn_blk = CrossAttention(hidden_size) if cross else TransposedAttention(hidden_size,heads)
        #self.mlp=Mlp(in_features=512, hidden_features=hidden_size, act_layer=approx_gelu, drop=0)

        #zero adain
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(self, x,c,t):
        B,C,H,W=x.shape
    
        x=x.permute(0,2,3,1) # b,h,w,c

        #ipdb.set_trace()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        #ipdb.set_trace()
        x = x + gate_msa.unsqueeze(1).unsqueeze(1) * self.mamba(modulate(self.norm1(x), shift_msa, scale_msa),c)
        x = x + gate_mlp.unsqueeze(1).unsqueeze(1) * self.attn_blk(modulate(self.norm2(x), shift_mlp, scale_mlp).permute(0, 3, 1, 2).contiguous(),c).permute(0, 2, 3, 1).contiguous()
        return x.permute(0, 3, 1, 2)

# class Mamba_block(nn.Module):
#     """
#     A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, d_state,expand,dropout):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size)
#         self.mamba = SS2D(d_model=hidden_size, d_state=d_state,expand=expand,dropout=dropout)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         self.adaLN_modulation1 = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(512, 3 * hidden_size, bias=True)
#         )
#         self.adaLN_modulation2 = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(512, 3 * hidden_size, bias=True)
#         )
#         self.attn_blk = LinearAttention(hidden_size)
#         #self.mlp=Mlp(in_features=512, hidden_features=hidden_size, act_layer=approx_gelu, drop=0)

#     def forward(self, x, c,t):
#         B,C,H,W=x.shape
    
#         x=x.permute(0,2,3,1) # b,h,w,c

#         shift_msa, scale_msa, gate_msa = self.adaLN_modulation1(t).chunk(3, dim=1)

#         shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation2(t).chunk(3, dim=1)
#         #ipdb.set_trace()
#         x = x + gate_msa.unsqueeze(1).unsqueeze(1) * self.mamba(modulate(self.norm1(x), shift_msa, scale_msa))
#         x = x + gate_mlp.unsqueeze(1).unsqueeze(1) * self.attn_blk(modulate(self.norm2(x), shift_mlp, scale_mlp).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
#         return x.permute(0, 3, 1, 2)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        input_condition=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + channels * \
            (1 if self_condition else 0) + channels * \
            (1 if condition else 0) + channels * (1 if input_condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
        
# class Unet(nn.Module):
#     def __init__(
#         self,
#         dim,
#         init_dim=None,
#         out_dim=None,
#         dim_mults=(1, 2, 4, 8),
#         channels=1,
#         self_condition=False,
#         resnet_block_groups=8,
#         learned_variance=False,
#         learned_sinusoidal_cond=False,
#         random_fourier_features=False,
#         learned_sinusoidal_dim=16,
#         condition=False,
#         input_condition=False
#     ):
#         super().__init__()

#         # determine dimensions

#         self.channels = channels
#         self.self_condition = self_condition
#         input_channels = channels + channels * \
#             (1 if self_condition else 0) + channels * \
#             (1 if condition else 0) + channels * (1 if input_condition else 0)

#         init_dim = default(init_dim, dim)
#         self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

#         #ipdb.set_trace()
#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         block_klass = partial(ResnetBlock, groups=resnet_block_groups)

#         # time embeddings

#         time_dim = dim * 4

#         self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

#         if self.random_or_learned_sinusoidal_cond:
#             sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
#                 learned_sinusoidal_dim, random_fourier_features)
#             fourier_dim = learned_sinusoidal_dim + 1
#         else:
#             sinu_pos_emb = SinusoidalPosEmb(dim)
#             fourier_dim = dim

#         self.time_mlp = nn.Sequential(
#             sinu_pos_emb,
#             nn.Linear(fourier_dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim)
#         )


#         condition=True
#         if condition:
#             self.clip_model=load('RN50', 'cpu')
#             for param in self.clip_model.parameters():
#                 param.requires_grad = False

#             clipiqa=CLIPIQA(model_type='clipiqa+')
            
#             #state_dict=torch.load('/mnt/miah203/zhchen/DN_foundation/dose_checkpoints/Dose_CLIP_rnc_sup/100_trainloss_8.623530.pth', map_location='cpu')

#             #state_dict=torch.load('/mnt/miah203/zhchen/DN_foundation/dose_checkpoints/Dose_CLIP_rnc_sup2_mayo/90_trainloss_10.010777.pth', map_location='cpu')
#             state_dict=torch.load('/mnt/miah203/zhchen/DN_foundation/dose_checkpoints/Dose_CLIP_rnc_sup2_mayo_newhead/95_trainloss_10.010643.pth', map_location='cpu')

#             #state_dict=torch.load('/mnt/miah203/zhchen/DN_foundation/dose_checkpoints/Dose_CLIP3/175_trainloss_0.100316.pth', map_location='cpu')
#             #state_dict=torch.load('/mnt/miah203/zhchen/DN_foundation/dose_checkpoints/Dose_CLIP/100_trainloss_0.144409.pth', map_location='cpu')
#             #state_dict=torch.load('/data/zhchen/DN_foundation/dose_checkpoints/clipiqa+_ab_lora3/80_trainloss_0.007333.pth', map_location='cpu')
#             clipiqa.load_state_dict(state_dict, strict=True)
#             self.dose_encoder=clipiqa
#             for param in self.dose_encoder.parameters():
#                 param.requires_grad = False
#             self.dose_encoder.eval()


#             # hidden_size=512
#             # self.t_embedder = TimestepEmbedder(hidden_size)

#             context_dim=1024
#             self.prompt = nn.Parameter(torch.rand(1, time_dim))
#             self.text_mlp = nn.Sequential(
#                 nn.Linear(context_dim, time_dim), NonLinearity(),
#                 nn.Linear(time_dim, time_dim))
#             self.prompt_mlp = nn.Linear(time_dim, time_dim)

#         # layers

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)
#         base_d_state=4


#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             # self.downs.append(nn.ModuleList([
#             #     block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#             #     block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#             #     Residual(PreNorm(dim_in, LinearAttention(dim_in))),
#             #     Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
#             #         dim_in, dim_out, 3, padding=1)
#             # ]))
            
#             if ind==0:
#                 d_state=base_d_state
#             else:
#                 d_state=int(base_d_state * 2 ** ind)
#             self.downs.append(nn.ModuleList([
#                 #block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                 block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                 # Mamba_block(hidden_size=dim_in, d_state=d_state,expand=2.0,dropout=0,cross=False if ind < 2 else True,time_emb_dim=time_dim),
#                 Mamba_block(hidden_size=dim_in, d_state=d_state,expand=2.0,dropout=0,cross=False,time_emb_dim=time_dim),
#                 #Residual(PreNorm(dim_in, SS2D(d_model=dim_in, d_state=base_d_state,expand=2.0,dropout=0))),
#                 Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
#                     dim_in, dim_out, 3, padding=1)
#             ]))

#         mid_dim = dims[-1]

#         self.mid_block = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#         self.mid_attn=Mamba_block(hidden_size=mid_dim, d_state=int(base_d_state * 2 ** 3),expand=2.0,dropout=0,cross=False,time_emb_dim=time_dim)
#         #self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
#             is_last = ind == (len(in_out) - 1)

#             # self.ups.append(nn.ModuleList([
#             #     block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#             #     block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#             #     Residual(PreNorm(dim_out, LinearAttention(dim_out))),
#             #     Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
#             #         dim_out, dim_in, 3, padding=1)
#             # ]))
            
#             if (3-ind)==0:
#                 d_state=base_d_state
#             else:
#                 d_state=int(base_d_state * 2 ** (3-ind))
            
#             self.ups.append(nn.ModuleList([
#                 #block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#                 block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#                 # Mamba_block(hidden_size=dim_out, d_state=d_state,expand=2.0,dropout=0,cross=False if ind > 1 else True,time_emb_dim=time_dim),
#                 Mamba_block(hidden_size=dim_out, d_state=d_state,expand=2.0,dropout=0,cross=False,time_emb_dim=time_dim),
#                 #Residual(PreNorm(dim_in, SS2D(d_model=dim_in, d_state=base_d_state,expand=2.0,dropout=0))),
#                 Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
#                     dim_out, dim_in, 3, padding=1)
#             ]))

#         default_out_dim = channels * (1 if not learned_variance else 2)
#         self.out_dim = default(out_dim, default_out_dim)

#         self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
#         #self.final_attn_block = Mamba_block(hidden_size=dim, d_state=base_d_state,expand=2.0,dropout=0,cross=False,time_emb_dim=time_dim),
#         self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

#     def forward(self, x,time, x_self_cond=None):
        
#         if self.self_condition:
#             x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
#             x = torch.cat((x_self_cond, x), dim=1)
        
#         #ipdb.set_trace()
#         _,dose_embedding,context_embedding= self.dose_encoder(x[:,1,:,:].unsqueeze(1).repeat(1,3,1,1))
#         c=context_embedding.unsqueeze(1)
#         #dose_embedding,context_embedding= self.dose_encoder.clip_model.encode_image(x[:,1,:,:].unsqueeze(1).repeat(1,3,1,1),pos_embedding=False)
#         #c=self.dose_encoder.attnpool2d(context_embedding).unsqueeze(1)

#         #context_embedding= self.dose_encoder.clip_model.visual(x[:,1,:,:].unsqueeze(1).repeat(1,3,1,1),return_token=True, pos_embedding=False)
#         #c= torch.flatten(F.adaptive_avg_pool2d(context_embedding, (1, 1)),1).unsqueeze(1)
#         #ipdb.set_trace()
#         x = self.init_conv(x)
#         r = x.clone()

#         t = self.time_mlp(time)
#         #t = self.t_embedder(time)
        
#         prompt_embedding = torch.softmax(self.text_mlp(dose_embedding), dim=1) * self.prompt
#         prompt_embedding = self.prompt_mlp(prompt_embedding)

#         t = t + prompt_embedding

#         h = []
#         for res_block, attn, downsample in self.downs:

#             x = attn(x,c,t) 
#             x = res_block(x)
            
#             h.append(x)

#             x = downsample(x)

#         x = self.mid_block(x)
#         x = self.mid_attn(x,c,t)

        
#         for res_block, attn, upsample in self.ups:

#             x = torch.cat((x, h.pop()), dim=1)
#             x = res_block(x)
#             x = attn(x,c,t)

#             x = upsample(x)

#         x = torch.cat((x, r), dim=1)

#         x = self.final_res_block(x)

#         #ipdb.set_trace()
#         #x = self.final_attn_block(x,c,t)

#         return self.final_conv(x)


class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=False,
        input_condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise"
    ):
        super().__init__()
        self.condition = condition
        self.input_condition = input_condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions
        if self.num_unet == 2:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)
            self.unet1 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)
        elif self.num_unet == 1:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)

    def forward(self, x,time, x_self_cond=None):
        if self.num_unet == 2:
            if self.test_res_or_noise == "res_noise":
                return self.unet0(x,time[0], x_self_cond=x_self_cond), self.unet1(x, time[1], x_self_cond=x_self_cond)
            elif self.test_res_or_noise == "res":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), 0
            elif self.test_res_or_noise == "noise":
                return 0, self.unet1(x,time[1], x_self_cond=x_self_cond)
        elif self.num_unet == 1:
            if self.objective == 'pred_res_noise':
                # num_unet=2
                pass
            elif self.objective == 'pred_x0_noise':
                # num_unet=2
                pass
            elif self.objective == "pred_noise":
                time = time[1]
            elif self.objective == "pred_res":
                time = time[0]
            return [self.unet0(x, time, x_self_cond=x_self_cond)]

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise="None"
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask
        self.test_res_or_noise = test_res_or_noise


        # self.context_encoder, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        #                                                 cache_dir='/data/zhchen/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # for param in self.context_encoder.parameters():
        #     param.requires_grad = False
        # self.context_align=nn.Sequential(
        #                     nn.SiLU(),
        #                     nn.Linear(512, 2 * 512, bias=True),
        #                     nn.SiLU(),  
        #                     nn.Linear(2 * 512, 4 * 512, bias=True),
        #                     nn.SiLU(),  
        #                     nn.Linear(4 * 512, 4 * 512, bias=True),
        #                     nn.SiLU(),
        #                     nn.Linear(4 * 512, 2 * 512, bias=True),
        #                     nn.SiLU(),
        #                     nn.Linear(2 * 512, 512, bias=True),
        #                 )

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        convert_to_ddim=True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = 0
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = 0
        else:
            alphas = gen_coefficients(timesteps, schedule="decreased")
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

        self.perceploss = lpips.LPIPS(net='alex') # best forward scores
        for param in self.perceploss.parameters():
            param.requires_grad = False

    def init(self):
        timesteps = 1000

        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = alphas[1]
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = betas2[1]

            # adjust
            # alphas = gen_coefficients(timesteps, schedule="average", ratio=0)
            # alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            # alphas_cumsum_prev = F.pad(
            #     alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])

            # betas2 = gen_coefficients(
            #     timesteps, schedule="average", sum_scale=self.sum_scale, ratio=0)
            # betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
            # betas2_cumsum_prev = F.pad(
            #     betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])
        else:
            alphas = gen_coefficients(timesteps, schedule="average", ratio=1)
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale, ratio=3)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(
                alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])
            betas2_cumsum_prev = F.pad(
                betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                      self.betas_cumsum[t]*self.num_timesteps],
                                  x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, img,time_cond, x_input_condition, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            #ipdb.set_trace()
            if type == "use_pred_noise":
                img = img - alpha*pred_res + sigma2.sqrt()*noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                    (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                    sigma2.sqrt()*noise
            if not last:
                img_list.append(img)


        
        # for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
        #     time_cond = torch.full(
        #         (batch,), time, device=device, dtype=torch.long)
        #     self_cond = x_start if self.self_condition else None
        #     preds = self.model_predictions(
        #         x_input, img, time_cond, x_input_condition, self_cond)

        #     pred_res = preds.pred_res
        #     pred_noise = preds.pred_noise
        #     x_start = preds.pred_x_start

        #     if time_next < 0:
        #         img = x_start
        #         if not last:
        #             img_list.append(img)
        #         continue

        #     alpha_cumsum = self.alphas_cumsum[time]
        #     alpha_cumsum_next = self.alphas_cumsum[time_next]
        #     alpha = alpha_cumsum-alpha_cumsum_next

        #     betas2_cumsum = self.betas2_cumsum[time]
        #     betas2_cumsum_next = self.betas2_cumsum[time_next]
        #     betas2 = betas2_cumsum-betas2_cumsum_next
        #     betas = betas2.sqrt()
        #     betas_cumsum = self.betas_cumsum[time]
        #     betas_cumsum_next = self.betas_cumsum[time_next]
        #     sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
        #     sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
        #         betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

        #     if eta == 0:
        #         noise = 0
        #     else:
        #         noise = torch.randn_like(img)

        #     if type == "use_pred_noise":
        #         img = img - (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
        #             pred_noise + sigma2.sqrt()*noise
        #     elif type == "use_x_start":
        #         img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
        #             (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
        #             (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
        #             sigma2.sqrt()*noise

        #     if not last:
        #         img_list.append(img)

        
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    # @torch.no_grad()
    # def ddim_sample(self, x_input, shape, last=True):
    #     if self.input_condition:
    #         x_input_condition = x_input[1]
    #     else:
    #         x_input_condition = 0
    #     x_input = x_input[0]

    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
    #         0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    #     # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = torch.linspace(-1, total_timesteps - 1,
    #                            steps=sampling_timesteps + 1)
    #     times = list(reversed(times.int().tolist()))
    #     # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    #     time_pairs = list(zip(times[:-1], times[1:]))

    #     if self.condition:
    #         img = x_input+math.sqrt(self.sum_scale) * \
    #             torch.randn(shape, device=device)
    #         input_add_noise = img
    #     else:
    #         img = torch.randn(shape, device=device)

    #     x_start = None
    #     type = "use_pred_noise"

    #     if not last:
    #         img_list = []

    #     eta = 0

    #     for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
    #         time_cond = torch.full(
    #             (batch,), time, device=device, dtype=torch.long)
    #         self_cond = x_start if self.self_condition else None
    #         preds = self.model_predictions(
    #             x_input, img, time_cond, x_input_condition, self_cond)

    #         pred_res = preds.pred_res
    #         pred_noise = preds.pred_noise
    #         x_start = preds.pred_x_start

    #         if time_next < 0:
    #             img = x_start
    #             if not last:
    #                 img_list.append(img)
    #             continue

    #         alpha_cumsum = self.alphas_cumsum[time]
    #         alpha_cumsum_next = self.alphas_cumsum[time_next]
    #         alpha = alpha_cumsum-alpha_cumsum_next

    #         betas2_cumsum = self.betas2_cumsum[time]
    #         betas2_cumsum_next = self.betas2_cumsum[time_next]
    #         betas2 = betas2_cumsum-betas2_cumsum_next
    #         # betas2 = 1-(1-betas2_cumsum)/(1-betas2_cumsum_next)
    #         betas = betas2.sqrt()
    #         betas_cumsum = self.betas_cumsum[time]
    #         betas_cumsum_next = self.betas_cumsum[time_next]
    #         sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
    #         sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
    #             betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

    #         if eta == 0:
    #             noise = 0
    #         else:
    #             noise = torch.randn_like(img)

    #         if type == "use_pred_noise":
    #             img = img - alpha*pred_res - \
    #                 (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
    #                 pred_noise + sigma2.sqrt()*noise
    #         elif type == "use_x_start":
    #             img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
    #                 (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
    #                 (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
    #                 sigma2.sqrt()*noise

    #         if not last:
    #             img_list.append(img)

    #     if self.condition:
    #         if not last:
    #             img_list = [input_add_noise]+img_list
    #         else:
    #             img_list = [input_add_noise, img]
    #         return unnormalize_to_zero_to_one(img_list)
    #     else:
    #         if not last:
    #             img_list = img_list
    #         else:
    #             img_list = [img]
    #         return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs,t, noise=None):
        if isinstance(imgs, list):  # Condition
            if self.input_condition:
                x_input_condition = imgs[2]
            else:
                x_input_condition = 0
            x_input = imgs[1]
            x_start = imgs[0]  # gt = imgs[0], input = imgs[1]
        else:  # Generation
            x_input = 0
            x_start = imgs

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_input, x,t, x_input_condition if self.input_condition else 0).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in,
                               [self.alphas_cumsum[t]*self.num_timesteps,
                                   self.betas_cumsum[t]*self.num_timesteps],
                               x_self_cond)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)

            pred_res = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            for i in range(len(model_out)):
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                #ipdb.set_trace()
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean() #+ self.perceploss(x_input-model_out[i].repeat(1,3,1,1),x_start.repeat(1,3,1,1))
                loss_list.append(loss)
            return loss_list

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if self.input_condition and self.input_condition_mask:
            img[0] = normalize_to_neg_one_to_one(img[0])
            img[1] = normalize_to_neg_one_to_one(img[1])
        else:
            img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)

# trainer class

from accelerate.utils import DistributedDataParallelKwargs


class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='.results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
        equalizeHist=False,
        crop_patch=False,
        generation=False,
        num_unet=2,
        checkpoint_folder=None,
        is_train=True,
        train_logger=None
    ):
        super().__init__()
        self.opt=opt

        #ipdb.set_trace()
        self.checkpoint_folder=checkpoint_folder
        self.results_folder=checkpoint_folder+'/sample'
        make_dir(self.results_folder)

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if fp16 else 'no',
            kwargs_handlers=[kwargs]
        )
        #self.accelerator = Accelerator()

        self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.num_unet = num_unet

        self.use_wandb=False

        #ipdb.set_trace()
        # if self.condition:
        #     if len(folder) == 3:
        #         self.condition_type = 1
        #         # test_input
        #         ds = dataset(folder[-1], self.image_size,
        #                      augment_flip=False, convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
        #         trian_folder = folder[0:2]

        #         self.sample_dataset = ds
        #         self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
        #                                                                        pin_memory=True, num_workers=4)))  # cpu_count()

        #         ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
        #                      convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
        #         self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
        #                         shuffle=True, pin_memory=True, num_workers=4)))
        #     elif len(folder) == 4:
        #         self.condition_type = 2
        #         # test_gt+test_input
        #         ds = dataset(folder[2:4], self.image_size,
        #                      augment_flip=False, convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
        #         trian_folder = folder[0:2]

        #         self.sample_dataset = ds
        #         self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
        #                                                                        pin_memory=True, num_workers=4)))  # cpu_count()

        #         ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
        #                      convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
        #         self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
        #                         shuffle=True, pin_memory=True, num_workers=4)))
        #     elif len(folder) == 6:
        #         self.condition_type = 3
        #         # test_gt+test_input
        #         ds = dataset(folder[3:6], self.image_size,
        #                      augment_flip=False, convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
        #         trian_folder = folder[0:3]

        #         self.sample_dataset = ds
        #         self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
        #                                                                        pin_memory=True, num_workers=4)))  # cpu_count()

        #         ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
        #                      convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
        #         self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
        #                         shuffle=True, pin_memory=True, num_workers=4)))
        # else:
        #     self.condition_type = 0
        #     trian_folder = folder

        #     ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
        #                  convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)

        #     self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
        #                     shuffle=True, pin_memory=True, num_workers=4)))

        self.condition_type = 2

        val_dataset = PDFDataset(phase='test')
        #val_dataset = Mayo16Dataset(phase='test')
        self.sample_dataset = val_dataset
        self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                        pin_memory=True, num_workers=4)))  # cpu_count()

        ds =PDFDataset(phase='train512')  
        #ds =Mayo16Dataset(phase='train')  
        self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=4)))
        # optimizer

        # self.opt = Adam(diffusion_model.parameters(),
        #                 lr=train_lr, betas=adam_betas)
        if self.num_unet == 1:
            # self.opt0 = RAdam(diffusion_model.parameters(),
            #                   lr=train_lr, weight_decay=0.0)
            self.opt0 = Adam(diffusion_model.parameters(),
                            lr=train_lr, betas=adam_betas)
        elif self.num_unet == 2:
            self.opt0 = RAdam(
                diffusion_model.model.unet0.parameters(), lr=train_lr, weight_decay=0.0)
            self.opt1 = RAdam(
                diffusion_model.model.unet1.parameters(), lr=train_lr, weight_decay=0.0)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            #self.set_results_folder(results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        if self.num_unet == 1:
            self.model, self.opt0 = self.accelerator.prepare(
                self.model, self.opt0)
        elif self.num_unet == 2:
            self.model, self.opt0, self.opt1 = self.accelerator.prepare(
                self.model, self.opt0, self.opt1)
        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        if self.num_unet == 1:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        elif self.num_unet == 2:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'opt1': self.opt1.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        torch.save(data, self.results_folder +'/'+ f'model-{milestone}.pt')

    def load(self, milestone):
        path = Path(self.results_folder + '/'+f'model-{milestone}.pt')

        if path.exists():
            data = torch.load(
                str(path), map_location=self.device)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']
            if self.num_unet == 1:
                self.opt0.load_state_dict(data['opt0'])
            elif self.num_unet == 2:
                self.opt0.load_state_dict(data['opt0'])
                self.opt1.load_state_dict(data['opt1'])
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))

        # self.ema.to(self.device)

    def train(self):

        accelerator = self.accelerator

        self.train_logger = get_logger(self.checkpoint_folder+'/train.log')
        if self.use_wandb:
            wandb.init(project='Foundation_DN',name='rddm_dn')

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                if self.num_unet == 1:
                    total_loss = [0]
                elif self.num_unet == 2:
                    total_loss = [0, 0]
                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        for i in range(self.num_unet):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            total_loss[i] = total_loss[i] + loss[i].item()

                    for i in range(self.num_unet):
                        self.accelerator.backward(loss[i])

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                if self.num_unet == 1:
                    self.opt0.step()
                    self.opt0.zero_grad()
                elif self.num_unet == 2:
                    self.opt0.step()
                    self.opt0.zero_grad()
                    self.opt1.step()
                    self.opt1.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.sample(milestone)

                        if self.step != 0 and self.step > self.save_and_sample_every*10*4 and self.step % (self.save_and_sample_every*10) == 0:
                            self.save(milestone)
                            results_folder = self.results_folder
                            gen_img = self.checkpoint_folder+'/test_timestep_10_' + \
                                str(milestone)+"_pt"
                            #self.set_results_folder(gen_img)
                            self.results_folder=gen_img
                            make_dir(gen_img)
                            self.test(last=True, FID=True)
                            if self.use_wandb:
                                wandb.log({ 'epoch_test_psnr':np.mean(self.test_running_psnr),
                                            'epoch_test_ssim':np.mean(self.test_running_ssim),
                                            'epoch_test_rmse':np.mean(self.test_running_rmse),
                                            'test_iters':self.step % (self.save_and_sample_every*10)})  
                            self.train_logger.info('Iters: [{}/{}], test_psnr: {:.4f}, test_ssim: {:.4f},test_rmse:{:.4f}'.format(self.step ,self.train_num_steps, np.mean(self.test_running_psnr), np.mean(self.test_running_ssim), np.mean(self.test_running_rmse)))   
                            os.system(
                                "python fid_and_inception_score.py "+gen_img)
                            #self.set_results_folder(results_folder)
                            self.results_folder=results_folder

                if self.num_unet == 1:
                    pbar.set_description(f'loss_unet0: {total_loss[0]:.6f}')
                    if self.use_wandb:
                        wandb.log({ "train_loss_unet0": total_loss[0]} )
                elif self.num_unet == 2:
                    pbar.set_description(
                        f'loss_unet0: {total_loss[0]:.6f},loss_unet1: {total_loss[1]:.6f}')
                    if self.use_wandb:
                        wandb.log({ "train_loss_unet0": total_loss[0],
                                    "train_loss_unet1": total_loss[1]})
                pbar.update(1)

        accelerator.print('training complete')

    def sample(self, milestone, last=True, FID=False):
        self.ema.ema_model.eval()

        with torch.no_grad():
            batches = self.num_samples
            if self.condition_type == 0:
                x_input_sample = [0]
                show_x_input_sample = []
            elif self.condition_type == 1:
                x_input_sample = [next(self.sample_loader).to(self.device)]
                show_x_input_sample = x_input_sample
            elif self.condition_type == 2:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]
            elif self.condition_type == 3:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]

            all_images_list = show_x_input_sample + \
                list(self.ema.ema_model.sample(
                    x_input_sample, batch_size=batches, last=last))

            all_images = torch.cat(all_images_list, dim=0)
            all_images=torch.clip(all_images*3000-1000,-160,240)
            all_images=(all_images+160)/400

            if last:
                nrow = int(math.sqrt(self.num_samples))
            else:
                nrow = all_images.shape[0]

            if FID:
                for i in range(batches):
                    file_name = f'sample-{milestone}.png'
                    utils.save_image(
                        all_images_list[0][i].unsqueeze(0), os.path.join(self.results_folder, file_name), nrow=1)
                    milestone += 1
                    if milestone >= self.total_n_samples:
                        break
            else:
                file_name = f'sample-{milestone}.png'
                #ipdb.set_trace()
                utils.save_image(all_images, self.results_folder +'/'+ file_name, nrow=nrow)
            print("sampe-save "+file_name)
        return milestone

    def test(self, sample=False, last=True, FID=False):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1)
            i = 0

            self.test_running_psnr = []
            self.test_running_ssim = []
            self.test_running_rmse = []

            for items in tqdm(loader):
                if self.condition:
                    file_name = self.sample_dataset.load_name(
                        i, sub_dir=self.sub_dir)
                else:
                    file_name = f'{i}.png'
                i += 1

                with torch.no_grad():
                    batches = self.num_samples

                    if self.condition_type == 0:
                        x_input_sample = [0]
                        show_x_input_sample = []
                    elif self.condition_type == 1:
                        x_input_sample = [items.to(self.device)]
                        show_x_input_sample = x_input_sample
                    elif self.condition_type == 2:
                        #ipdb.set_trace()
                        x_input_sample = [item.to(self.device)
                                          for item in items]
                        show_x_input_sample = x_input_sample
                        y=x_input_sample[0]
                        x_input_sample = x_input_sample[1:]

                    elif self.condition_type == 3:
                        x_input_sample = [item.to(self.device)
                                          for item in items]
                        show_x_input_sample = x_input_sample
                        x_input_sample = x_input_sample[1:]

                    if sample:
                        all_images_list = show_x_input_sample + \
                            list(self.ema.ema_model.sample(
                                x_input_sample, batch_size=batches))
                    else:
                        all_images_list = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last))
                        y_pred=all_images_list[-1]
                        all_images_list = [all_images_list[-1]]
                        if self.crop_patch:
                            k = 0
                            for img in all_images_list:
                                pad_size = self.sample_dataset.get_pad_size(i)
                                _, _, h, w = img.shape
                                img = img[:, :, 0:h -
                                          pad_size[0], 0:w-pad_size[1]]
                                all_images_list[k] = img
                                k += 1

                        #ipdb.set_trace()
                        psnr=util.compute_psnr(y_pred,y)
                        ssim=util.compute_ssim(y_pred,y)
                        rmse=util.compute_rmse(y_pred,y)
                        self.test_running_psnr.append(psnr.detach().cpu().numpy())
                        self.test_running_ssim.append(ssim.detach().cpu().numpy())
                        self.test_running_rmse.append(rmse.detach().cpu().numpy())
                        print('(psnr: %.4f, ssim: %.4f,rmse:.%.4f) ' % (psnr, ssim,rmse))

                all_images = torch.cat(all_images_list, dim=0)


                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]

                all_images = torch.cat(all_images_list, dim=0)

                # all_images=torch.clip(all_images*3000-1000,-160,240)
                # all_images=(all_images+160)/400

                if 'quarter' in file_name:
                    image_file_name=file_name.split('.')[0]+'.png'
                else:
                    image_file_name=file_name.split('.')[0]+'.'+file_name.split('.')[1]+'.png'

                # utils.save_image(all_images,
                #     self.results_folder +'/'+ image_file_name, nrow=nrow)

                if not self.opt.is_train:
                    npy_name=self.results_folder +'/'+file_name[:-4]
                    np.save(npy_name,all_images.detach().cpu().numpy().reshape(512,512))
                    print("test-save "+file_name)

            #ipdb.set_trace()
            length=290
            ab_psnr=self.test_running_psnr[:290*4]
            ab_ssim=self.test_running_ssim[:290*4]
            ab_rmse=self.test_running_rmse[:290*4]
            self.train_logger.info('(ab average mean: psnr: %.4f, ssim: %.4f,rmse: %.4f)' % (np.mean(ab_psnr), np.mean(ab_ssim),np.mean(ab_rmse)))
            for i in range(4):
                dose_test_psnr= np.mean(ab_psnr[int(i*length):int((i+1)*length)])
                dose_test_ssim= np.mean(ab_ssim[int(i*length):int((i+1)*length)])
                dose_test_rmse= np.mean(ab_rmse[int(i*length):int((i+1)*length)])
                self.train_logger.info('(ab————dose: %2d,average: psnr: %.4f, ssim: %.4f,rmse: %.4f)' % (i,dose_test_psnr, dose_test_ssim,dose_test_rmse))
                

            length=637
            lung_psnr=self.test_running_psnr[290*4:290*4+637*4]
            lung_ssim=self.test_running_ssim[290*4:290*4+637*4]
            lung_rmse=self.test_running_rmse[290*4:290*4+637*4]
            self.train_logger.info('(lung average mean: psnr: %.4f, ssim: %.4f,rmse: %.4f)' % (np.mean(lung_psnr), np.mean(lung_ssim),np.mean(lung_rmse)))
            for i in range(4):
                dose_test_psnr= np.mean(lung_psnr[int(i*length):int((i+1)*length)])
                dose_test_ssim= np.mean(lung_ssim[int(i*length):int((i+1)*length)])
                dose_test_rmse= np.mean(lung_rmse[int(i*length):int((i+1)*length)])
                self.train_logger.info('(lung————dose: %2d,average: psnr: %.4f, ssim: %.4f,rmse: %.4f)' % (i,dose_test_psnr, dose_test_ssim,dose_test_rmse))

            length=159
            head_psnr=self.test_running_psnr[-159*4:]
            head_ssim=self.test_running_ssim[-159*4:]
            head_rmse=self.test_running_rmse[-159*4:]
            self.train_logger.info('(head average mean: psnr: %.4f, ssim: %.4f,rmse: %.4f)' % (np.mean(head_psnr), np.mean(head_ssim),np.mean(head_rmse)))
            for i in range(4):
                dose_test_psnr= np.mean(head_psnr[int(i*length):int((i+1)*length)])
                dose_test_ssim= np.mean(head_ssim[int(i*length):int((i+1)*length)])
                dose_test_rmse= np.mean(head_rmse[int(i*length):int((i+1)*length)])
                self.train_logger.info('(head————dose: %2d,average: psnr: %.4f, ssim: %.4f,rmse: %.4f)' % (i,dose_test_psnr, dose_test_ssim,dose_test_rmse))

            self.train_logger.info('test_psnr: {:.4f}, test_ssim: {:.4f},test_rmse:{:.4f}'.format(np.mean(self.test_running_psnr), np.mean(self.test_running_ssim), np.mean(self.test_running_rmse)))   
                 
        else:
            if FID:
                self.total_n_samples = 50000
                img_id = len(glob.glob(f"{self.results_folder}/*"))
                n_rounds = (self.total_n_samples -
                            img_id) // self.num_samples+1
            else:
                n_rounds = 100
            for i in range(n_rounds):
                if FID:
                    i = img_id
                img_id = self.sample(i, last=last, FID=FID)
        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)
