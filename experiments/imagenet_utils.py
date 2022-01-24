"""Utility functions for ImageNet experiments."""
import torch


class IMModel(torch.nn.Module):

    def __init__(self, base_model, mode):
        super(IMModel, self).__init__()

        self.base_model = base_model
        self.mode = mode
        self.mechanism = f"m_{mode}"
    
    def forward(self, x):
        x = self.base_model.backbone(x)
        x = getattr(self.base_model, self.mechanism)(x)
        return x


class AverageEnsembleModel(torch.nn.Module):

    def __init__(self, base_model):
        super(AverageEnsembleModel, self).__init__()

        self.base_model = base_model
    
    def forward(self, x):
        x = self.base_model.backbone(x)
        x_shape = self.base_model.m_shape(x)
        x_texture = self.base_model.m_texture(x)
        x_bg = self.base_model.m_bg(x)
        x = (x_shape + x_texture + x_bg) / 3.0

        return x