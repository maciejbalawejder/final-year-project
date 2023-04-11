import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math

def conv2dnorm(in_channels, 
           out_channels, 
           kernel_size, 
           stride, 
           groups = 1, 
           act : Optional[Callable[..., nn.Module]] = nn.SiLU(), 
           layer_norm : Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
           bias = None
           ):

            """Convolution Block with Batch Normalization and Activation Function
            Parameters
            ----------
                in_channels : int 
                    number of input channels
                out_channels : int
                    number of output channels
                kernel_size : int
                    int value of kernel 
                    
                stride : int
                    stride size
                groups : int
                    number of groups the input channels are split into
                act : torch.nn.Module
                    defines if there is activation function
                layer_norm : torch.nn.Module
                    normalization layer
                bias : bool
                    defines if convolution has bias parameter
                
            Returns
            -------
                ret : nn.ModuleList()
                    list with convolution, norm layer and activation function
            """
            padding = kernel_size // 2
            layer_norm = layer_norm(out_channels)
            if act == None:
                act = nn.Identity()

            return nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups), layer_norm, act])

class SeBlock(nn.Module):
    """Squeeze-and-Excitation Block
    Parameters
    ----------
        in_channels : int
            number of input channels
        squeeze_channels : int
            number of channels to squeeze to
    Attributes
    ----------
        avg_pool : nn.AdaptiveAvgPool2d 
            global pooling operation(squeeze) that brings down spatial dimensions to (1,1) for all channels
        fc1 : nn.Conv2d
            first linear/convolution layer that brings down the number of channels the reduction space
        fc2 : nn.Conv2d
            second linear/convolution layer that brings up the number of channels the input space(excitation)
        silu : nn.SiLU
            silu activation function
        sigmoid : nn.Sigmoid
            sigmoid activation function
    """

    def __init__(
                self, 
                in_channels, 
                squeeze_channels
                ):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _scale(self, x):
        """Scale pass.
        Parameters
        ----------
            x : torch.Tensor
                Input tensor with shape (batch_size, in_channels, height, width).
        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).
        """
        scale = self.avg_pool(x)
        scale = self.activation(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return scale
        

    def forward(self, x) -> Tensor:
        """Forward pass.
        Parameters
        ----------
            x : torch.Tensor
                Input tensor with shape (batch_size, in_channels, height, width).
        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).
        """
        scale = self._scale(x)

        return scale * x

class MBConv(nn.Module):
    """MBConv Block
    Parameters
    ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            int value of kernel 
        stride : int
            single number
        exp : int
            expansion ratio

    Attributes
    ----------
        res_connection : bool
            defines whether use residual connection or not
        block : nn.Sequential
            collection of expansion operation, depthwise conv, seblock and reduction convolution
    """

    def __init__(
                self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                exp
                ):

        super().__init__()

        self.res_connection = in_channels == out_channels and stride == 1
        layers: List[nn.Module] = []


        # expand
        exp_channels = adjust_channels(in_channels, exp)
        if exp > 1:
            layers.append(conv2dnorm(in_channels, exp_channels, 1, 1))

        # depthwise
        layers.append(conv2dnorm(exp_channels, exp_channels, kernel_size, stride, groups=exp_channels))

        # squeeze and excitation
        squeeze_channels = max(1, in_channels//4)
        layers.append(SeBlock(exp_channels, squeeze_channels))

        # projection
        layers.append(conv2dnorm(exp_channels, out_channels, 1, 1, act=None))
        

        self.block = nn.Sequential(*layers)


    def forward(self, x) -> Tensor:
        """Forward pass.
        Parameters
        ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, in_channels, height, width).
        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).
        """
        f = self.block(x)

        if self.res_connection:
            f = x + f

        return f

class EfficientNet(nn.Module):
    """EfficientNet architecture
    
    Parameters
    ----------
        config : class
            class with configurations
        in_channels : int 
            number of input channels, default 3 for image
        out_channels : int
            number of output channels, default 1000 classes for imagenet
        p_drop : float
            dropout probability
        
    Attributes
    ----------
        add_pretrain_head : bool
            contains add add_pretrain_head parameters
        blocks : nn.ModuleList
            container for all stages with MBConv and FusedMbConv
        stem : ConvBlock
            first convolution block in the network
        final_conv : ConvBlock
            final convolution block before classification head
        avgpool : nn.AdaptiveAvgPool2d
            adaptive average pooling that squeezes all the channels
        head : nn.Sequential
            the final layers of the model with Dropout and classification layer
        
    
    """

    def __init__(
            self, 
            config,
            in_channels=3,
            out_channels=1000,
            p_drop=0.5
            ):

        super().__init__()

        layers: List[nn.Module] = []


        
        # first layer
        layers.append(conv2dnorm(in_channels, config.first_conv_out_channels, 3, 2))

        # inverted resiudal blocks
        stage_block_id = 0
        for s in config.layer_settings:
            n_layers, name, in_channels, out_channels, kernel_size, stride, exp = s

            stage: List[nn.Module] = []
            block = MBConv

            for _ in range(n_layers):
                if stage:
                    in_channels = out_channels
                    stride = 1
                
                stage.append(block(in_channels, out_channels, kernel_size, stride, exp))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))
        
        # last layers
        layers.append(conv2dnorm(
            in_channels = config.last_conv_in_channels,
            out_channels = config.last_conv_out_channels,
            kernel_size = 1,
            stride = 1
        ))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_drop, inplace=True),
            nn.Linear(config.last_conv_out_channels, out_channels),
        )



    def forward(self, x):
        """Forward pass.
        Parameters
        ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, in_channels, height, width).
        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels).
        """

        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

## Utils

class Base:
    first_conv_out_channels : int = 32
    last_conv_in_channels : int = 320
    last_conv_out_channels : int = 1280
    n_blocks : int = 40
    n_classes : int = 1000
    layer_settings : List = [
            [1, "mb", 32, 16, 3, 1, 1],
            [2, "mb", 16, 24, 3, 2, 6],
            [2, "mb", 24, 40, 5, 2, 6],
            [3, "mb", 40, 80, 3, 2, 6],
            [3, "mb", 80, 112, 5, 1, 6],
            [4, "mb", 112, 192, 5, 2, 6],
            [1, "mb", 192, 320, 3, 1, 6]
    ]

configs = {
    'B0': (1.0, 1.0, 224, 0.2, "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"),
    'B1': (1.0, 1.1, 240, 0.2, "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth"),
    'B2': (1.1, 1.2, 260, 0.3, "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth"),
    'B3': (1.2, 1.4, 300, 0.3, "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth"),
    'B4': (1.4, 1.8, 380, 0.4, "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"),
    'B5': (1.6, 2.2, 456, 0.4, "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth"),
    'B6': (1.8, 2.6, 528, 0.5, "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth"),
    'B7': (2.0, 3.1, 600, 0.5, "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth"),
    'B8': (2.2, 3.6, 672, 0.5, "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth"),
}

def get_config(name : str = "B4"):
    assert name in configs.keys(), "Wrong name of the configuration."
    width_mult, depth_mult, img_size, p_drop, url = configs[name]
    base = Base()
    base.first_conv_out_channels = adjust_channels(base.first_conv_out_channels, width_mult)
    base.last_conv_out_channels = adjust_channels(base.last_conv_out_channels, width_mult)
    base.last_conv_in_channels = adjust_channels(base.last_conv_in_channels, width_mult)
    n_blocks = 0
    for i, layer in enumerate(base.layer_settings):
        base.layer_settings[i][0] = adjust_depth(base.layer_settings[i][0], depth_mult)
        base.layer_settings[i][2] = adjust_channels(base.layer_settings[i][2], width_mult)
        base.layer_settings[i][3] = adjust_channels(base.layer_settings[i][3], width_mult)
        n_blocks += base.layer_settings[i][0]
    base.n_blocks = n_blocks
    return base, url, p_drop, img_size

def adjust_channels(channels : int, width_mult : float, min_value : int = None) -> int:
    v = channels * width_mult
    divisor = 8

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def adjust_depth(n_layers : int, depth_mult : float) -> int: 
    return int(math.ceil(n_layers * depth_mult)) # round to the next int
