from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ## BEGIN YOUR SOLUTION
    
        # max_z_original = array_api.max(Z, self.axes, keepdims=True) 
        # max_z_reduce = array_api.max(Z, self.axes)
        # return array_api.log(array_api.sum(array_api.exp(Z - max_z_original), self.axes)) + max_z_reduce 
        
        # keepdims!!!
        max_z_original = array_api.max(Z, self.axes, keepdims=True) 
        max_z_reduce = array_api.max(Z, self.axes)
        return array_api.log(
            array_api.sum(array_api.exp(Z - max_z_original), self.axes)
            ) + max_z_reduce 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ## BEGIN YOUR SOLUTION
        # Retrieve the input tensor
        z = node.inputs[0]
        # Compute max(Z) for numerical stability, keeping dimensions for broadcasting
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        # Compute exp(Z - max(Z))
        exp_z = exp(z - max_z)
        # Compute sum(exp(Z - max(Z))) along the specified axes
        sum_exp_z = summation(exp_z, self.axes)
        # Gradient of log-sum-exp with respect to sum(exp(Z - max(Z)))
        grad_sum_exp_z = out_grad / sum_exp_z
        # Ensure grad_sum_exp_z is broadcastable to the shape of z
        expand_shape = list(z.shape)
        if self.axes is not None:
            for axis in self.axes:
                expand_shape[axis] = 1
        else:
            for axis in range(len(expand_shape)):
                expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        # Return the gradient with respect to the input tensor == the gradient with respect to (the input tensor Z - max_z)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

