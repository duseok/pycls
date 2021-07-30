import numpy as np
import torch
import torch.nn.functional as F
from pycls.core.config import cfg
from torch.nn.parameter import Parameter
from torch.quantization.fake_quantize import FakeQuantizeBase


class ShiftScaleQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s):
        return torch.exp2(torch.log2(s).round())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FakeQuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, quant_x: torch.Tensor):
        ctx.save_for_backward(x - quant_x)
        return quant_x.clone().detach_()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        diff = ctx.saved_tensors[0]
        grad = 2 * diff.div_(diff.numel())
        return grad, grad_output


class ShiftFakeQuantize(FakeQuantizeBase):
    scale: Parameter
    zero_point: torch.Tensor
    bitwidth: torch.Tensor

    def __init__(self, observer, quant_min=0, quant_max=255, **observer_kwargs):
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        self.activation_post_process = observer(**observer_kwargs)
        assert (
            torch.iinfo(self.activation_post_process.dtype).min <= quant_min
        ), "quant_min out of bound"
        assert (
            quant_max <= torch.iinfo(self.activation_post_process.dtype).max
        ), "quant_max out of bound"
        self.scale = Parameter(torch.tensor([np.log(np.exp(1) - 1)], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0.0]))
        self.register_buffer(
            "bitwidth",
            torch.tensor([np.log2(quant_max - quant_min + 1)], dtype=torch.int),
        )
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme

    def forward(self, X: torch.Tensor):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = (
                _scale.to(self.scale.device),
                _zero_point.to(self.zero_point.device),
            )
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)
            self.scale.data.fill_(torch.log(torch.exp(_scale[0]) - 1))

        if self.fake_quant_enabled[0] == 1:
            s = ShiftScaleQuant.apply(F.softplus(self.scale))
            Y = torch._fake_quantize_learnable_per_tensor_affine(
                X, s, self.zero_point, self.quant_min, self.quant_max, 1.0
            )
            if cfg.QUANTIZATION.QAT.ENABLE_QUANTIZATION_LOSS:
                Y = FakeQuantFunc.apply(X, Y)
        return Y

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self):
        scale = torch.exp2(torch.log2(F.softplus(self.scale)).round())
        return (
            "fake_quant_enabled={}, observer_enabled={}, "
            "quant_min={}, quant_max={}, dtype={}, qscheme={}, "
            "scale={}, zero_point={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.quant_min,
                self.quant_max,
                self.dtype,
                self.qscheme,
                scale,
                self.zero_point,
            )
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(ShiftFakeQuantize, self)._save_to_state_dict(
            destination, prefix, keep_vars
        )
        destination[prefix + "scale"] = self.scale.data
        destination[prefix + "zero_point"] = self.zero_point
        destination[prefix + "bitwidth"] = self.bitwidth

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "zero_point", "bitwidth"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "zero_point":
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.data.fill_(val)
                    elif name == "zero_point":
                        self.zero_point.copy_(val)
                    else:
                        assert name == "bitwidth"
                        self.bitwidth.copy_(val)
            elif strict:
                missing_keys.append(key)
        super(ShiftFakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
