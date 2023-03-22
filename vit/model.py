import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
import wget
import numpy as np
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from collections import OrderedDict
from functools import partial # partial allows to add more *args to the function

class MLPBlock(nn.Module):
    """ Feed Forward module.

    Parameters
    ----------
    d_size : int - embedding dimension from config class
    mlp_size : int - expansion dimension in mlp module
    p_mlp : float - mlp dropout rate
    bias : bool = True - bias in the linear layers

    Attributes
    ----------
    linear_1 : nn.Linear - first linear projection
    act_layer : nn.GELU - GELU activation layer
    linear_2 : nn.Linear - second linear projection
    drop_1 : nn.Dropout - dropout layer for first projection
    drop_2 : nn.Dropout - dropout for second projection
    """

    def __init__(
        self,
        d_size,
        mlp_size,
        p_mlp,
        bias = True
        ):

        super().__init__()

        self.linear_1 = nn.Linear(d_size, mlp_size, bias=bias)
        self.act_layer = nn.GELU()
        self.linear_2 = nn.Linear(mlp_size, d_size, bias=bias)
        self.drop_1 = nn.Dropout(p_mlp)
        self.drop_2 = nn.Dropout(p_mlp)
        
    def forward(self, x):
        """ Forward function.
        Parameters
        ----------
        x : Tensor - input image with shape (batch, n_patches + cls_token, d_size)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + cls_token, d_size)
        """
        x = self.drop_1(self.act_layer(self.linear_1(x)))

        return self.drop_2(self.linear_2(x))

class EncoderBlock(nn.Module):
    """ Encoder Block with Multi-Head Attention module, MLPBlock and Layer Norms. 

    Parameters
    ----------
    d_size : int - embedding dimension from config class
    n_heads : int - number of heads
    mlp_size : int - expansion dimension in mlp module
    p_att : float - attention dropout rate
    p_mlp : float - mlp dropout rate
    layer_norm : nn.Module - normalization layer

    eps : float - a value added in denominator of Layer Norm for numerical stability

    Attributes
    ----------
    mha : nn.Module - Multi-Head Attention module
    mlp : nn.Module - MLP module
    ln1 : nn.LayerNorm - layer normalization 1
    ln2 : nn.LayerNorm - layer normalization 2

    """

    def __init__(
        self,
        d_size,
        n_heads,
        mlp_size,
        p_att,
        p_mlp,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)
        ):

        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim = d_size,
            num_heads = n_heads,
            dropout = p_att,
            batch_first = True
        )

        self.mlp = MLPBlock(
            d_size = d_size,
            mlp_size = mlp_size,
            p_mlp = p_mlp
        )

        self.ln_1 = norm_layer(d_size)

        self.ln_2 = norm_layer(d_size)

        self.dropout = nn.Dropout(p_mlp)
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, n_patches + cls_token, d_size)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + cls_token, d_size)
        """

        f = self.ln_1(x)
        f, _ = self.self_attention(f, f, f, need_weights=False)
        f = self.dropout(f)
        f = x + f

        y = self.ln_2(f)
        y = self.mlp(y)
        return f + y 

class Encoder(nn.Module):
    """ Encoder with position embeddings and all other encoder blocks.

    Parameters
    ----------
    seq_length : int - the number of patches
    n_layers : int - number of encoder blocks
    d_size : int - embedding dimension from config class
    n_heads : int - number of heads
    mlp_size : int - expansion dimension in mlp module
    p_att : float - attention dropout rate
    p_mlp : float - mlp dropout rate
    p_emb : float - embedding dropout rate
    eps : float - a value added in denominator of Layer Norm for numerical stability

    Attributes
    ----------
    pos_emb : nn.Parameter - positional encodings
    dropout : nn.Dropout - dropout layer applied on the output of input with postional embeddings
    layers : nn.Sequential - collection of encoder blocks
    ln : nn.LayerNorm - layer normalization at the output 
    """

    def __init__(
        self,
        seq_length,
        n_layers,
        n_heads,
        d_size,
        mlp_size,
        p_att,
        p_mlp,
        p_emb,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, d_size).normal_(std=0.02))
        self.dropout = nn.Dropout(p_emb)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(n_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                    d_size,
                    n_heads,
                    mlp_size,
                    p_att,
                    p_mlp,
                    norm_layer
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(d_size)

    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, seq_len, d_size)

        Outputs
        -------
        Tensor - with shape (batch, seq_len, d_size)
        """

        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        x = x + self.pos_embedding
        return self.ln(self.layers(self.dropout(x)))


class VisionTransformer(nn.Module):
    """ ViT architecture with Embeddings, Encoder Blocks and Classification head. 

    Parameters
    ----------
    config : class - configuration class with model specifications
    in_channels : int - number of input channels, default 3
    
    Attributes
    ----------
    class_token : nn.Parameter - learnable classification token 
    seq_length : int - the number of patches
    encoder : nn.Module - collection of encoder layers and postional encodings
    heads : nn.Sequential - collection of heads which are placed on the output of encoder
    conv_proj : nn.Conv2d - first projection convolution 
    d_size : int - embedding size
    patch_size : int - size of the patch

    """

    def __init__(
        self,
        config,
        in_channels = 3
        ):

        super().__init__()

        norm_layer = nn.LayerNorm

        self.conv_proj = nn.Conv2d(
                in_channels = in_channels,
                out_channels = config.d_size,
                kernel_size = config.patch_size,
                stride = config.patch_size
                )
        

        seq_length = (config.image_size // config.patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.d_size))
        seq_length += 1
        
        self.encoder = Encoder(
            seq_length,
            config.n_layers,
            config.n_heads,
            config.d_size,
            config.mlp_size,
            config.p_att,
            config.p_mlp,
            config.p_emb,
            norm_layer
        )
    
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        if config.rep_size is None:
            heads_layers["head"] = nn.Linear(config.d_size, config.n_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(config.d_size, config.rep_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(config.rep_size, config.n_classes)

        self.heads = nn.Sequential(heads_layers)

        self.conv_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=config.d_size,
            kernel_size = config.patch_size,
            stride = config.patch_size
        )

        self.d_size = config.d_size
        self.patch_size = config.patch_size

    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, in_channels, height, width)

        Outputs
        -------
        Tensor - with shape (batch, n_classes)

        """
        n, c, h, w = x.shape
        n_h, n_w = h // self.patch_size, w // self.patch_size
        x = self.conv_proj(x)
        x = x.reshape(n, self.d_size, n_h * n_w)
        x = x.permute(0, 2, 1)

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = x[:,0]
        x = self.heads(x)
    
        return x

## config class
class Config:
    image_size : int = 224
    patch_size : int = 16
    n_layers : int = 12
    d_size : int = 768
    mlp_size : int = 3072
    n_heads : int = 12
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1
    rep_size : int = None
    n_classes : int = 1000
    url : str = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'

