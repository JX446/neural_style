import torch
import torch.nn as nn
import torch.nn.functional as F

# Scale gradients in the backward pass
class ScaleGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, strength):
        ctx.strength = strength
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output / (torch.norm(grad_output, keepdim=True) + 1e-8)
        return grad_input * ctx.strength * ctx.strength, None

# Gram Matrix calculation
class GramMatrix(nn.Module):
    def forward(self, input):
        B, C, H, W = input.size()
        # Flatten HxW for each channel
        features = input.view(B, C, H * W)
        # Batch matrix multiplication to get Gram matrix per image
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (C * H * W)

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, strength=1.0, normalize=False):
        super().__init__()
        self.strength = strength
        self.normalize = normalize
        self.loss = None
        self.target = None
        self.mode = 'NONE'

    def forward(self, input):
        if self.mode == 'capture':
            # Store target for later loss computation
            self.target = input.detach()
        elif self.mode == 'loss':
            loss = F.mse_loss(input, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = loss * self.strength
        return input

# Style Loss
class StyleLoss(nn.Module):
    def __init__(self, strength=1.0, normalize=False):
        super().__init__()
        self.strength = strength
        self.normalize = normalize
        self.loss = None
        self.target = None
        self.G = None
        self.blend_weight = None
        self.mode = 'NONE'
        self.gram = GramMatrix()  # reuse GramMatrix instance

    def forward(self, input):
        self.G = self.gram(input)
        if self.mode == 'capture':
            if self.blend_weight is None:
                self.target = self.G.detach()
            elif self.target is None or self.target.nelement() == 0:
                self.target = self.G.detach() * self.blend_weight
            else:
                self.target = self.target + self.G.detach() * self.blend_weight
        elif self.mode == 'loss':
            loss = F.mse_loss(self.G, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = loss * self.strength
        return input

# Total Variation Loss
class TVLoss(nn.Module):
    def __init__(self, strength=1e-3):
        super().__init__()
        self.strength = strength
        self.loss = None
        self.x_diff = None
        self.y_diff = None

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) +
                                     torch.sum(torch.abs(self.y_diff)))
        return input

# Divide weights by channel size
def normalize_weights(content_losses, style_losses):
    for n, i in enumerate(content_losses):
        i.strength = i.strength / max(i.target.size())
    for n, i in enumerate(style_losses):
        i.strength = i.strength / max(i.target.size())