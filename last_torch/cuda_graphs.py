"""Optional CUDA Graph execution for a fixed lattice input shape.

Capture one wrapper per (batch size, frame length, label length) bucket. Capture
cost and graph memory are paid once per wrapper; this module does not maintain
an unbounded global cache. Input values and parameter values may change between
calls. Shapes, strides, dtypes, devices, and requires_grad flags must stay fixed.
"""

import torch
from torch import nn


def _signature(tensor):
    return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device,
            tensor.requires_grad)


class _LatticeCall(nn.Module):
    def __init__(self, lattice):
        super().__init__()
        self.lattice = lattice

    def forward(self, frames, num_frames, labels, num_labels):
        return self.lattice(frames, num_frames, labels, num_labels)


class _CapturedLattice(nn.Module):
    def __init__(self, call, signatures):
        super().__init__()
        self.call = call
        self.signatures = signatures

    def forward(self, frames, num_frames, labels, num_labels):
        args = (frames, num_frames, labels, num_labels)
        if tuple(map(_signature, args)) != self.signatures:
            raise ValueError('CUDA Graph inputs must match the captured shapes, '
                             'strides, dtypes, devices, and requires_grad flags; '
                             'capture a separate wrapper for this input bucket')
        return self.call(*args)


def make_graphed_lattice(lattice, sample_args, *, num_warmup_iters=3):
    """Capture lattice forward and backward without changing lattice.forward.

    Args:
        lattice: CUDA RecognitionLattice with capture-compatible weight functions.
        sample_args: Tuple (frames, num_frames, labels, num_labels), all CUDA
            tensors on one device. Enable frames.requires_grad if the encoder
            needs frame gradients. Parameters are included automatically.
        num_warmup_iters: Warmup iterations before capture, including backward.

    Returns:
        An nn.Module sharing lattice's parameters. Its output and gradient buffers
        are reused on replay: clone results that must outlive the next call, and
        finish backward before the next forward on the same wrapper. Optimizers
        may update parameters in place; replacing parameters or adding modules
        after capture is unsupported. Higher-order differentiation is unsupported.
        Use a separate wrapper for each shape bucket and do not replay a wrapper
        concurrently. Custom weight functions must obey PyTorch capture rules.
    """
    if not isinstance(sample_args, tuple) or len(sample_args) != 4:
        raise ValueError('sample_args must be a tuple of four CUDA tensors')
    if not all(isinstance(x, torch.Tensor) and x.is_cuda for x in sample_args):
        raise ValueError('sample_args must contain only CUDA tensors')
    if len({x.device for x in sample_args}) != 1:
        raise ValueError('sample_args must be on the same CUDA device')
    with torch.cuda.device(sample_args[0].device):
        # JointWeightFn creates its projections lazily. Register every parameter
        # before make_graphed_callables collects the module parameter surface.
        with torch.no_grad():
            lattice(*sample_args)
        call = _LatticeCall(lattice)
        call.train(lattice.training)
        # Own the static inputs so even callers reusing their sample buffers
        # follow the same input-copy path as a normal encoder training loop.
        static_args = tuple(x.detach().clone().requires_grad_(x.requires_grad)
                            for x in sample_args)
        captured = torch.cuda.make_graphed_callables(
            call, static_args, num_warmup_iters=num_warmup_iters)
    return _CapturedLattice(captured, tuple(map(_signature, sample_args)))
