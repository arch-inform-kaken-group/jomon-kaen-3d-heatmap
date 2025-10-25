import torch
import torch.nn as nn
import numpy as np


def initialize_sparse_heads(model):
    print("Initializing sparse prediction heads")
    if hasattr(model, 'emotion_head'):
        for module in model.emotion_head.modules():
            if isinstance(module, (nn.ConvTranspose3d, nn.Conv3d)):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                print(
                    f"Xavier normal initialization for emotion: {module._get_name}"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.5)
                    print("Emotion head bias initialized to 0.5")

    if hasattr(model, 'heatmap_head'):
        for module in model.heatmap_head.modules():
            if isinstance(module, (nn.ConvTranspose3d, nn.Conv3d)):
                nn.init.xavier_normal_(module.weight, gain=0.2)
                print(
                    f"Xavier normal initialization for heatmap: {module._get_name}"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.5)
                    print("Heatmap haad bias initialized to 0.5")


def ConvConvEncoder(in_dim, out_dim):
    return nn.Sequential(nn.Conv3d(in_dim,
                                   out_dim,
                                   3,
                                   1,
                                   1,
                                   bias=False),
                         nn.BatchNorm3d(out_dim),
                         nn.ReLU(inplace=True),
                         nn.Conv3d(out_dim,
                                   out_dim,
                                   3,
                                   1,
                                   1,
                                   bias=False),
                         nn.BatchNorm3d(out_dim),
                         nn.ReLU(inplace=True))


def UpSampleDecoder(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim,
                           out_dim,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True))


# Pass through skip block after each upsampling
def SkipBlock(upsample_out_dim, skip_channels_dim):
    return nn.Sequential(
        nn.Conv3d(upsample_out_dim + skip_channels_dim,
                  upsample_out_dim,
                  kernel_size=3,
                  padding=1,
                  bias=False),
        nn.BatchNorm3d(upsample_out_dim),
        nn.ReLU(inplace=True))


def ExpertBlock_PersonalityBlock(bottleneck_dim):
    hidden_dim = max(4, bottleneck_dim // 8)  # Small model, 64 // 8 = 8

    return nn.Sequential(nn.AdaptiveAvgPool3d(1),
                         nn.Conv3d(bottleneck_dim,
                                   hidden_dim,
                                   1),
                         nn.ReLU(),
                         nn.Conv3d(hidden_dim,
                                   bottleneck_dim,
                                   1),
                         nn.Softmax(dim=1))
