"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        # the learnable bias of shape (1, out_features). 
        # The values should be initialized with the Kaiming Uniform initialize with fan_in = out_features. 
        # Note the difference in fan_in choice, due to their relative sizes.
        # When initializing the bias with shape (1, out_features), 
        # if you incorrectly set fan_in to 1, the bound calculation will be incorrect, leading to improperly scaled initialization values.
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # (32, 28, 28, 3) -> (32, 28 * 28 * 3), which is (32, 2352).
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        log_sum_exp = ops.logsumexp(logits, (1,))
        one_hot_Iy = init.one_hot(logits.shape[1], y)   # init.onehot()调用了y.numpy转换, y可为一维或scalar
        z_y_sum = (logits * one_hot_Iy).sum()
        return (-z_y_sum + log_sum_exp.sum()) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size = x.shape[0]
            batch_mean = x.sum((0, )) / batch_size
            # NOTE array with shape (4, ) is considered as a row, so it can be brcsto (2, 4) and cannot be brcsto (4, 2)
            x_minus_mean = x - batch_mean.broadcast_to(x.shape)
            batch_var = (x_minus_mean ** 2).sum((0,)) / batch_size
            batch_std = (batch_var + self.eps) ** 0.5
            normalized_x = x_minus_mean / batch_std.broadcast_to(x.shape)
            
            # 记得调用data，以从计算图中detach，防止计算图过长
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            
            return self.weight.broadcast_to(x.shape) * normalized_x + self.bias.broadcast_to(x.shape)
        else:
            # NOTE no momentum here!
            normalized_x = (x - self.running_mean) / ((self.running_var + self.eps) ** 0.5)
            # NOTE testing time also need self.weight and self.bias
            return self.weight.broadcast_to(x.shape) * normalized_x + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, requires_grad=True))
        # NOTE bias initialized to 0!!!
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        
        # NOTE need reshape, because (4, ) can brcsto (2, 4) but (4, ) cannot brcsto (4, 2)
        mean = (x.sum(axes=(1, )) / feature_size)   \
                .reshape((batch_size, 1))
        x_minus_mean = x - mean.broadcast_to(x.shape)
        x_std = (
                ( (x_minus_mean ** 2).sum(axes=(1, )) / feature_size )  \
                .reshape((batch_size, 1))   \
                + self.eps  \
                ) ** 0.5
        # NOTE need manual broadcast_to!!!
        normalized_x = x_minus_mean / x_std.broadcast_to(x.shape)
        
        return self.weight.broadcast_to(x.shape) * normalized_x + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
