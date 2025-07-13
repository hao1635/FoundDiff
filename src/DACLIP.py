import hashlib
import os
import urllib
import warnings
from tqdm import tqdm
from typing import Tuple, Union, List
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoTokenizer
import torch
import torch.nn as nn
import ipdb

import open_clip

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spacial_dim = spacial_dim
        self.embed_dim = embed_dim

    def forward(self, x, return_token=False, pos_embedding=False):
        #ipdb.set_trace()
        n, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        if pos_embedding:
            positional_embedding_resize = F.interpolate(self.positional_embedding.unsqueeze(
                0).unsqueeze(0), size=(x.size(0), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize[:, None, :].to(x.dtype)  # (HW+1)NC

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        if return_token:
            return x[0], x[1:]
        else:
            return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        self.feature_dim_list = [width, width * 4, width * 8, width * 16, width * 32]

        embed_dim = width * 32  # the ResNet feature dimension

        #input_resolution=512
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)


    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward_features(self, x, return_token=False, pos_embedding=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x1 = self.avgpool(x)
            return x1,x
        
        x = x.type(self.conv1.weight.dtype)
        x,x_ = stem(x)
        feat_list = [x_]
        x = self.layer1(x)
        feat_list += [x]
        x = self.layer2(x)
        feat_list += [x]
        x = self.layer3(x)
        feat_list += [x]
        x = self.layer4(x)
        feat_list += [x]
        return feat_list

    def forward(self, x, return_token=False, pos_embedding=False):
        #ipdb.set_trace()
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if return_token:
            x, tokens = self.attnpool(x, return_token, pos_embedding)
            return x, tokens
        else:
            out = self.attnpool(x, return_token, pos_embedding)
            return out


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_token=False, pos_embedding=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if pos_embedding:
            positional_embedding_resize = F.interpolate(self.positional_embedding.unsqueeze(
                0).unsqueeze(0), size=(x.size(1), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        token = self.ln_post(x[:, 1:, :])

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        if return_token:
            return x, token
        else:
            return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, pos_embedding):
        return self.visual(image.type(self.dtype), pos_embedding=pos_embedding)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, pos_embedding=False, text_features=None):
        #ipdb.set_trace()
        image_features = self.encode_image(image, pos_embedding)
        if text_features is None:
            text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, image_features


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith(
            "visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2]
                            for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    #ipdb.set_trace()
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

from collections import OrderedDict
import collections.abc
from itertools import repeat
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision.transforms as T

#from pyiqa.utils.download_util import load_file_from_url

import math
import os
import requests
from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm
from urllib.parse import urlparse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file
# --------------------------------------------
# IQA utils
# --------------------------------------------


def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """Convert distribution prediction to mos score.
    For datasets with detailed score labels, such as AVA

    Args:
        dist_score (tensor): (*, C), C is the class number

    Output:
        mos_score (tensor): (*, 1)
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score


def random_crop(input_list, crop_size, crop_num):
    if not isinstance(input_list, collections.abc.Sequence):
        input_list = [input_list]

    b, c, h, w = input_list[0].shape
    ch, cw = to_2tuple(crop_size)

    if min(h, w) <= crop_size:
        scale_factor = (crop_size + 1) / min(h, w)
        input_list = [
            F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            for x in input_list
        ]
        b, c, h, w = input_list[0].shape

    crops_list = [[] for i in range(len(input_list))]
    for i in range(crop_num):
        sh = np.random.randint(0, h - ch + 1)
        sw = np.random.randint(0, w - cw + 1)
        for j in range(len(input_list)):
            crops_list[j].append(input_list[j][..., sh : sh + ch, sw : sw + cw])

    for i in range(len(crops_list)):
        crops_list[i] = torch.stack(crops_list[i], dim=1).reshape(
            b * crop_num, c, ch, cw
        )

    if len(crops_list) == 1:
        crops_list = crops_list[0]
    return crops_list


def clip_preprocess_tensor(x: torch.Tensor, model):
    """clip preprocess function with tensor input.

    NOTE: Results are slightly different with original preprocess function with PIL image input, because of differences in resize function.
    """
    # Bicubic interpolation
    x = (x * 255).byte()
    x = T.functional.resize(
        x,
        model.visual.input_resolution,
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    )
    # Center crop
    x = T.functional.center_crop(x, model.visual.input_resolution)
    x = x.float() / 255.0
    x = T.functional.normalize(x, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
    return x


# --------------------------------------------
# Common utils
# --------------------------------------------


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_pretrained_network(net, model_path, strict=True, weight_keys=None):
    if model_path.startswith("https://") or model_path.startswith("http://"):
        model_path = load_file_from_url(model_path)
    print(f"Loading pretrained model {net.__class__.__name__} from {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    if weight_keys is not None:
        state_dict = state_dict[weight_keys]
    state_dict = clean_state_dict(state_dict)
    net.load_state_dict(state_dict, strict=strict)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    r"""Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0.
        kwargs (dict): Other arguments for initialization function.

    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


r"""CLIP-IQA metric, proposed by

Exploring CLIP for Assessing the Look and Feel of Images.
Jianyi Wang Kelvin C.K. Chan Chen Change Loy.
AAAI 2023.

Ref url: https://github.com/IceClear/CLIP-IQA
Re-implmented by: Chaofeng Chen (https://github.com/chaofengc) with the following modification:
    - We assemble multiple prompts to improve the results of clipiqa model.

"""



# from pyiqa.utils.registry import ARCH_REGISTRY
# from pyiqa.archs.arch_util import load_file_from_url
# from pyiqa.archs.arch_util import load_pretrained_network

import clip



IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


default_model_urls = {
    'clipiqa+': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIP-IQA+_learned_prompts-603f3273.pth',
    'clipiqa+_rn50_512': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIPIQA+_RN50_512-89f5d940.pth',
    'clipiqa+_vitL14_512': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIPIQA+_ViTL14_512-e66488f2.pth',
}


class PromptLearner(nn.Module):
    """
    Disclaimer:
        This implementation follows exactly the official codes in: https://github.com/IceClear/CLIP-IQA. We have no idea why some tricks are implemented like this, which include
            1. Using n_ctx prefix characters "X"
            2. Appending extra "." at the end
            3. Insert the original text embedding at the middle
    """

    def __init__(self, clip_model, n_ctx=16) -> None:
        super().__init__()

        # For the following codes about prompts, we follow the official codes to get the same results
        prompt_prefix = " ".join(["X"] * n_ctx) + ' '
        #init_prompts = [prompt_prefix + 'Low dose CT image of', prompt_prefix + 'High dose CT image of']
        init_prompts = [prompt_prefix + 'high dose CT image..', prompt_prefix + 'low dose CT noisy image..']
        with torch.no_grad():
            txt_token = clip.tokenize(init_prompts)
            self.tokenized_prompts = txt_token
            init_embedding = clip_model.token_embedding(txt_token)

        init_ctx = init_embedding[:, 1: 1 + n_ctx]
        self.ctx = nn.Parameter(init_ctx)

        self.n_ctx = n_ctx

        self.n_cls = len(init_prompts)
        self.name_lens = [3, 3]  # hard coded length, which does not include the extra "." at the end

        self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def get_prompts_with_middel_class(self,):

        ctx = self.ctx.to(self.token_prefix)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        half_n_ctx = self.n_ctx // 2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i: i + 1, :, :]
            class_i = self.token_suffix[i: i + 1, :name_len, :]
            suffix_i = self.token_suffix[i: i + 1, name_len:, :]
            ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

    def forward(self, clip_model):
        prompts = self.get_prompts_with_middel_class()
        # self.get_prompts_with_middel_class
        x = prompts + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection

        return x

# class PromptLearner(nn.Module):
#     """
#     Disclaimer:
#         This implementation follows exactly the official codes in: https://github.com/IceClear/CLIP-IQA. We have no idea why some tricks are implemented like this, which include
#             1. Using n_ctx prefix characters "X"
#             2. Appending extra "." at the end
#             3. Insert the original text embedding at the middle
#     """

#     def __init__(self, clip_model, n_ctx=16) -> None:
#         super().__init__()
#         #ipdb.set_trace()
#         # For the following codes about prompts, we follow the official codes to get the same results
#         prompt_prefix = " ".join(["X"] * n_ctx) + ' '
#         init_prompts = [prompt_prefix + 'noise-free image..', prompt_prefix + 'noisy image..']
#         #init_prompts = [prompt_prefix + 'noise-free image..', prompt_prefix + 'noisy image..']
#         #init_prompts = [prompt_prefix + 'acceptable noise level', prompt_prefix + 'average noise level',prompt_prefix + 'suboptimal noise level',prompt_prefix + 'Unacceptable noise level']
#         #'acceptable noise level', 'average noise level','suboptimal noise level','Unacceptable noise level'
#         with torch.no_grad():
#             txt_token = clip.tokenize(init_prompts)
#             self.tokenized_prompts = txt_token
#             init_embedding = clip_model.token_embedding(txt_token)

#         init_ctx = init_embedding[:, 1: 1 + n_ctx]
#         self.ctx = nn.Parameter(init_ctx)

#         self.n_ctx = n_ctx

#         self.n_cls = len(init_prompts)
#         #self.name_lens = [3, 3,3,3]  # hard coded length, which does not include the extra "." at the end
#         self.name_lens = [3, 3]
#         self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

#     def get_prompts_with_middel_class(self,):
#         #ipdb.set_trace()
#         ctx = self.ctx.to(self.token_prefix)
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

#         half_n_ctx = self.n_ctx // 2
#         prompts = []
#         for i in range(self.n_cls):
#             name_len = self.name_lens[i]
#             prefix_i = self.token_prefix[i: i + 1, :, :]
#             class_i = self.token_suffix[i: i + 1, :name_len, :]
#             suffix_i = self.token_suffix[i: i + 1, name_len:, :]
#             ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
#             ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
#             prompt = torch.cat(
#                 [
#                     prefix_i,     # (1, 1, dim)
#                     ctx_i_half1,  # (1, n_ctx//2, dim)
#                     class_i,      # (1, name_len, dim)
#                     ctx_i_half2,  # (1, n_ctx//2, dim)
#                     suffix_i,     # (1, *, dim)
#                 ],
#                 dim=1,
#             )
#             prompts.append(prompt)
#         prompts = torch.cat(prompts, dim=0)
#         return prompts

#     def forward(self, clip_model):
#         #ipdb.set_trace()
#         prompts = self.get_prompts_with_middel_class()

#         self.get_prompts_with_middel_class
#         x = prompts + clip_model.positional_embedding.type(clip_model.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = clip_model.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = clip_model.ln_final(x).type(clip_model.dtype)
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)]  @ clip_model.text_projection

#         return x

class Pub_PromptLearner(nn.Module):
    """
    Disclaimer:
        This implementation follows exactly the official codes in: https://github.com/IceClear/CLIP-IQA. We have no idea why some tricks are implemented like this, which include
            1. Using n_ctx prefix characters "X"
            2. Appending extra "." at the end
            3. Insert the original text embedding at the middle
    """

    def __init__(self, clip_model, n_ctx=16) -> None:
        super().__init__()
        #ipdb.set_trace()
        # For the following codes about prompts, we follow the official codes to get the same results
        prompt_prefix = " ".join(["X"] * n_ctx) + ' '
        init_prompts = [prompt_prefix + 'noise-free image..', prompt_prefix + 'noisy image..']
        with torch.no_grad():
            txt_token = clip.tokenize(init_prompts)
            self.tokenized_prompts = txt_token
            init_embedding = clip_model.text_model.embeddings.token_embedding(txt_token)

        init_ctx = init_embedding[:, 1: 1 + n_ctx]
        self.ctx = nn.Parameter(init_ctx)

        self.n_ctx = n_ctx

        self.n_cls = len(init_prompts)
        self.name_lens = [3, 3]  # hard coded length, which does not include the extra "." at the end

        self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("position_ids", torch.arange(77).expand((1, -1)))

    def get_prompts_with_middel_class(self,):

        ctx = self.ctx.to(self.token_prefix)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        half_n_ctx = self.n_ctx // 2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i: i + 1, :, :]
            class_i = self.token_suffix[i: i + 1, :name_len, :]
            suffix_i = self.token_suffix[i: i + 1, name_len:, :]
            ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

    def forward(self, clip_model):
        prompts = self.get_prompts_with_middel_class()

        #ipdb.set_trace()
        x = prompts + clip_model.text_model.embeddings.position_embedding(self.position_ids).type(clip_model.dtype)

        x=clip_model.text_model.encoder(x)[0]
        x=clip_model.text_model.final_layer_norm(x)
        x=clip_model.text_projection(x)

        # self.get_prompts_with_middel_class
        # x = prompts + clip_model.positional_embedding.type(clip_model.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = clip_model.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = clip_model.ln_final(x).type(clip_model.dtype)
        # # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)]  @ clip_model.text_projection

        return x


class CLIPIQA(nn.Module):
    def __init__(self,
                 model_type='clipiqa',
                 backbone='RN50',
                 pretrained=True,
                 pos_embedding=False,
                 ) -> None:
        super().__init__()

        self.clip_model = load(backbone, 'cpu') # avoid saving clip weights
        # Different from original paper, we assemble multiple prompts to improve performance
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        # tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        # self.prompt_pairs=tokenizer(['Noise-free  CT scan', 'noisy CT scan'], truncation=True, max_length=77, return_length=True,
        #                                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt").input_ids

#         self.prompt_pairs = clip.tokenize([
# #             'Good image', 'bad image',
# #             'Sharp image', 'blurry image',
# #             'sharp edges', 'blurry edges',
# #             'High resolution image', 'low resolution image',
#             'Noise-free image', 'noisy image',
#         ])

        self.model_type = model_type
        self.pos_embedding = pos_embedding
        if 'clipiqa+' in model_type:
            self.prompt_learner = PromptLearner(self.clip_model)

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        for p in self.clip_model.transformer.parameters():
            p.requires_grad = False
        
        if pretrained and 'clipiqa+' in model_type:
            if model_type == 'clipiqa+' and backbone == 'RN50':
                self.prompt_learner.ctx.data = torch.load(load_file_from_url(default_model_urls['clipiqa+']))
            elif model_type in default_model_urls.keys():
                load_pretrained_network(self, default_model_urls[model_type], True, 'params')
            else:
                raise(f'No pretrained model for {model_type}')

        #self.attnpool2d=AttentionPool2d(spacial_dim=512//32, embed_dim=2048, num_heads=32, output_dim= 1024)
        self.head1 = nn.Sequential(       
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024)
            )
        self.head2 = nn.Sequential(       
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 256)
            )
    def forward(self, x):
        # preprocess image
        #ipdb.set_trace()
        #x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        #ipdb.set_trace()
        clip_model = self.clip_model.to(x)

        if self.model_type == 'clipiqa':
            prompts = self.prompt_pairs.to(x.device)
            logits_per_image, logits_per_text = clip_model(x, prompts, pos_embedding=self.pos_embedding)
        elif 'clipiqa+' in self.model_type:
            learned_prompt_feature = self.prompt_learner(clip_model)
            text_features=learned_prompt_feature

            features = clip_model.encode_image(x, pos_embedding=False)
            image_features = self.head1(features)

            content_features=self.head2(features)
            content_features=F.normalize(content_features, dim=1)

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

        # probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 4).softmax(dim=-1).squeeze(1)
        # return probs
        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        return probs[..., 0].mean(dim=1, keepdim=True),image_features,content_features
        
#     def forward(self, x):
#         self.prompt_pairs = clip.tokenize([
# #             'Good image', 'bad image',
# #             'Sharp image', 'blurry image',
# #             'sharp edges', 'blurry edges',
# #             'High resolution image', 'low resolution image',
#                'Noise-free CT image of', 'noisy CT image of',
#               # 'High dose CT image', 'Low dose CT image',
#               # 'excellent visibility of small structures', 'unacceptable visibility of small structures',
#             #'completely diagnostic cofidence', 'poor diagnostic cofidence',
#             #'acceptable noise level', 'average noise level','suboptimal noise level','Unacceptable noise level'
#         ])
#         # preprocess image
#         x = (x - self.default_mean.to(x)) / self.default_std.to(x)
#         #ipdb.set_trace()
#         clip_model = self.clip_model.to(x)

#         if self.model_type == 'clipiqa':
#             prompts = self.prompt_pairs.to(x.device)
#             logits_per_image, logits_per_text = clip_model(x, prompts, pos_embedding=self.pos_embedding)
#         elif 'clipiqa+' in self.model_type:
#             #ipdb.set_trace()
#             learned_prompt_feature = self.prompt_learner(clip_model)
#             logits_per_image, logits_per_text = clip_model(
#                 x, None, text_features=learned_prompt_feature, pos_embedding=self.pos_embedding)

#         # probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 4).softmax(dim=-1).squeeze(1)

#         # return probs
#         probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
#         return probs[..., 0].mean(dim=1, keepdim=True)



