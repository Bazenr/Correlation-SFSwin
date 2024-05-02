# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnet import DoubleResNet, DoubleResNet_src  # 一个普通改版, 一个上一篇论文的双ResNet
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin_src import SwinTransformer  # 原版
from .swin_double import DoubleSwinTransformer  # 
from .swin_SrcSKNet import SrcSKNet_DoubleSwinTransformer
from .swin_DivideSKNet import DivideSKNet_DoubleSwinTransformer
from .swin_Correlation import Corr_SwinTransformer
from .swin_Correlation_DivideSKNet import Corr_DivideSKNet_SwinTransformer
from .swin_Correlation_SrcSKNet import Corr_SrcSKNet_SwinTransformer
from .swin_Correlation_SF_SKNet import Corr_SF_SKNet_SwinTransformer
# from .swin import Corr_3_DivideSKNet_SwinTransformer

from .trident_resnet import TridentResNet

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'DoubleResNet', 'DoubleResNet_src',
    'PyramidVisionTransformer',
    'SwinTransformer',
    'DoubleSwinTransformer',
    'SrcSKNet_DoubleSwinTransformer',
    'DivideSKNet_DoubleSwinTransformer',
    'Corr_SwinTransformer',
    'Corr_DivideSKNet_SwinTransformer',
    # 'Corr_3_DivideSKNet_SwinTransformer',
    'Corr_SrcSKNet_SwinTransformer',
    'Corr_SF_SKNet_SwinTransformer'
]
