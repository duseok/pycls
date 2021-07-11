from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.quantization.observer import HistogramObserver, MinMaxObserver

if TYPE_CHECKING:
    from typing import Tuple


class MinMaxShiftObserver(MinMaxObserver):
    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ):
        if qscheme != torch.per_tensor_symmetric:
            raise NotImplemented("Currently only support for 'per_tensor_symetric'")
        super(MinMaxShiftObserver, self).__init__(
            dtype, qscheme, reduce_range, quant_min, quant_max, factory_kwargs
        )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)

        val = get_symmetric_shift_val(min_val_cur, max_val_cur)
        min_val = torch.min(torch.tensor(-val), self.min_val)
        max_val = torch.max(torch.tensor(val), self.max_val)

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class MovingAvgMinMaxShiftObserver(MinMaxObserver):
    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ):
        if qscheme != torch.per_tensor_symmetric:
            raise NotImplemented("Currently only support for 'per_tensor_symetric'")
        super(MovingAvgMinMaxShiftObserver, self).__init__(
            dtype, qscheme, reduce_range, quant_min, quant_max, factory_kwargs
        )
        self.averaging_constant = averaging_constant
        self._min_val_raw = torch.Tensor(0)
        self._max_val_raw = torch.Tensor(0)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        if self._min_val_raw.numel() == 0 or self._max_val_raw.numel() == 0:
            self._min_val_raw = min_val_cur
            self._max_val_raw = max_val_cur
        else:
            self._min_val_raw = self._min_val_raw + self.averaging_constant * (
                min_val_cur - self._min_val_raw
            )
            self._max_val_raw = self._max_val_raw + self.averaging_constant * (
                max_val_cur - self._max_val_raw
            )
        val = get_symmetric_shift_val(self._min_val_raw, self._max_val_raw)
        min_val = torch.min(torch.tensor(-val), self.min_val)
        max_val = torch.max(torch.tensor(val), self.max_val)

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class HistogramShiftObserver(HistogramObserver):
    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        factory_kwargs=None,
    ):
        if qscheme != torch.per_tensor_symmetric:
            raise NotImplemented("Currently only support for 'per_tensor_symetric'")
        super(HistogramShiftObserver, self).__init__(
            bins, upsample_rate, dtype, qscheme, reduce_range, factory_kwargs
        )

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density.to("cpu:0") * norm

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float("inf") and max_val == float("-inf")
        if is_uninitialized or same_values:
            min_val_cur, max_val_cur = torch._aminmax(x)

            val = get_symmetric_shift_val(min_val_cur, max_val_cur, False)
            min_val = torch.min(torch.tensor(-val), self.min_val)
            max_val = torch.max(torch.tensor(val), self.max_val)

            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            assert (
                min_val.numel() == 1 and max_val.numel() == 1
            ), "histogram min/max values must be scalar."
            torch.histc(
                x, self.bins, min=int(min_val), max=int(max_val), out=self.histogram
            )
        else:
            new_min, new_max = torch._aminmax(x)
            val = get_symmetric_shift_val(new_min, new_max, False)
            new_min = torch.min(torch.tensor(-val), self.min_val)
            new_max = torch.max(torch.tensor(val), self.max_val)

            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            (
                combined_min,
                combined_max,
                downsample_rate,
                start_idx,
            ) = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            assert (
                combined_min.numel() == 1 and combined_max.numel() == 1
            ), "histogram min/max values must be scalar."
            combined_histogram = torch.histc(
                x, self.bins, min=int(combined_min), max=int(combined_max)
            )
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins,
                )

            self.histogram.resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        bin_idx = 0
        norm_min = float("inf")

        min_idx = 0
        width = self.bins // 4
        while bin_idx < self.bins // 2 and width >= 1:
            norm = self._compute_quantization_error(bin_idx, self.bins - bin_idx - 1)

            if norm < norm_min:
                norm_min = norm
                min_idx = bin_idx
            bin_idx += width
            width //= 2

        new_min = self.min_val + bin_width * min_idx
        new_max = self.min_val + bin_width * (self.bins - min_idx - 1)
        return new_min, new_max


def get_symmetric_shift_val(min_val_cur: int, max_val_cur: int, close_val: bool = True):
    max_abs = max(abs(min_val_cur), abs(max_val_cur))
    if max_abs != 0:
        value = int(torch.floor(torch.log2(max_abs / 255)))
        delta = abs(max_abs - 255 * (2 ** value))
        while 255 * (2 ** value) < max_abs:
            value += 1
            delta = abs(max_abs - 255 * (2 ** value))
        if not close_val or delta > abs(max_abs - 255 * (2 ** (value + 1))):
            value += 1
        return 255 * (2 ** value)
    return 0
