import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


### Normalized cross-correlation, implementation from the excellent GitHub repository by Bob de Vos (AmsterdamUMC):
### https://github.com/BDdeVos/TorchIR


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


class NCC(_Loss):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.forward = self.metric

    def ncc(self, x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of similar shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        stablestd = StableStd.apply
        std = stablestd(x1) * stablestd(x2)
        ncc = cc / (std + e)
        return ncc

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return -self.ncc(fixed, warped).mean()
