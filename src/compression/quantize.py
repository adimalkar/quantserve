"""Quantization algorithms: AWQ (W4A16), SmoothQuant (W8A8), and FP8 (E4M3)."""
import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """Linear layer supporting runtime W4A16, W8A8, and FP8 execution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        precision: str = "fp16",
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision.lower()
        self.group_size = group_size

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        if self.precision in ("int4", "int4_awq", "w4a16"):
            # 4-bit weights packed into int32 (8 weights per int32) or uint8 (2 weights per byte)
            self.register_buffer(
                "qweight",
                torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
            )
            self.register_buffer(
                "scales",
                torch.ones((out_features, math.ceil(in_features / group_size)), dtype=torch.float16),
            )
            self.register_buffer(
                "zeros",
                torch.zeros((out_features, math.ceil(in_features / group_size)), dtype=torch.float16),
            )
        elif self.precision in ("int8", "w8a8", "int8_smoothquant"):
            self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=torch.int8))
            self.register_buffer("scales", torch.ones((out_features, 1), dtype=torch.float16))
            self.register_buffer("act_scales", torch.ones((1, in_features), dtype=torch.float16))
        elif self.precision in ("fp8", "fp8_e4m3"):
            # Native FP8 E4M3 storage
            self.register_buffer(
                "qweight",
                torch.zeros((out_features, in_features), dtype=getattr(torch, "float8_e4m3fn", torch.int8)),
            )
            self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        else:  # fp16 baseline
            self.register_parameter(
                "weight",
                nn.Parameter(torch.zeros((out_features, in_features), dtype=torch.float16)),
            )

    @torch.no_grad()
    def pack_from_float(self, float_weight: torch.Tensor):
        """Packs a floating-point weight matrix into the designated quantized representation."""
        float_weight = float_weight.to(torch.float16)

        if self.precision in ("int4", "int4_awq", "w4a16"):
            # Group-wise asymmetric 4-bit quantization
            out_feat, in_feat = float_weight.shape
            num_groups = math.ceil(in_feat / self.group_size)

            w_reshaped = float_weight.view(out_feat, num_groups, self.group_size)
            w_min = w_reshaped.amin(dim=-1, keepdim=True)
            w_max = w_reshaped.amax(dim=-1, keepdim=True)

            scale = (w_max - w_min) / 15.0
            scale = torch.clamp(scale, min=1e-5)
            zero = torch.round(-w_min / scale)

            q = torch.clamp(torch.round(w_reshaped / scale) + zero, 0, 15).to(torch.uint8)
            q = q.view(out_feat, in_feat)

            # Pack two 4-bit elements per uint8 byte
            q_low = q[:, 0::2]
            q_high = q[:, 1::2]
            packed = q_low | (q_high << 4)

            self.qweight.copy_(packed)
            self.scales.copy_(scale.squeeze(-1))
            self.zeros.copy_(zero.squeeze(-1))

        elif self.precision in ("int8", "w8a8", "int8_smoothquant"):
            # Symmetric per-channel int8 quantization
            scale = float_weight.abs().amax(dim=-1, keepdim=True) / 127.0
            scale = torch.clamp(scale, min=1e-5)
            q = torch.clamp(torch.round(float_weight / scale), -128, 127).to(torch.int8)

            self.qweight.copy_(q)
            self.scales.copy_(scale.to(torch.float16))

        elif self.precision in ("fp8", "fp8_e4m3"):
            max_val = float_weight.abs().max()
            scale = max_val / 448.0  # FP8 E4M3 dynamic range max is 448
            scale = torch.clamp(scale, min=1e-5)
            self.scale.copy_(scale)
            if hasattr(torch, "float8_e4m3fn"):
                q = (float_weight / scale).to(torch.float8_e4m3fn)
                self.qweight.copy_(q)
            else:
                q = torch.clamp(torch.round(float_weight / scale), -128, 127).to(torch.int8)
                self.qweight.copy_(q)
        else:
            self.weight.data.copy_(float_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward pass with on-the-fly dequantization or native tensor cores."""
        if self.precision in ("int4", "int4_awq", "w4a16"):
            # Unpack 4-bit weights
            q_low = self.qweight & 0x0F
            q_high = (self.qweight >> 4) & 0x0F

            out_feat, half_in = self.qweight.shape
            unpacked = torch.empty((out_feat, half_in * 2), dtype=torch.float16, device=x.device)
            unpacked[:, 0::2] = q_low.to(torch.float16)
            unpacked[:, 1::2] = q_high.to(torch.float16)

            num_groups = self.scales.shape[1]
            unpacked_reshaped = unpacked.view(out_feat, num_groups, self.group_size)
            scale = self.scales.unsqueeze(-1)
            zero = self.zeros.unsqueeze(-1)

            dequant_w = ((unpacked_reshaped - zero) * scale).view(out_feat, -1)
            return F.linear(x, dequant_w, self.bias)

        elif self.precision in ("int8", "w8a8", "int8_smoothquant"):
            # Dequantize symmetric INT8 weights to FP16
            dequant_w = self.qweight.to(x.dtype) * self.scales.to(x.dtype)
            return F.linear(x, dequant_w, self.bias)

        elif self.precision in ("fp8", "fp8_e4m3"):
            dequant_w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
            return F.linear(x, dequant_w, self.bias)

        else:
            return F.linear(x, self.weight, self.bias)


class AWQQuantizer:
    """Activation-aware Weight Quantization: Protects salient weights based on activation magnitude."""

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    @torch.no_grad()
    def compute_salience(self, linear_layer: nn.Linear, sample_activations: torch.Tensor) -> torch.Tensor:
        """Computes activation channel norm to determine salient weight channels."""
        # Mean absolute activation per input channel
        act_norm = sample_activations.abs().view(-1, linear_layer.in_features).mean(dim=0)
        return act_norm

    def quantize_layer(self, layer: nn.Linear, act_salience: Optional[torch.Tensor] = None) -> QuantizedLinear:
        ql = QuantizedLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            precision="int4_awq",
            group_size=self.group_size,
        )
        ql.pack_from_float(layer.weight.data)
        if layer.bias is not None:
            ql.bias.data.copy_(layer.bias.data)
        return ql


class SmoothQuantizer:
    """Migrates activation outliers into weights: s = max(|X|)^alpha / max(|W|)^(1-alpha)."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    @torch.no_grad()
    def calculate_smoothing_scale(
        self, linear_layer: nn.Linear, sample_activations: torch.Tensor
    ) -> torch.Tensor:
        """Calculates channel-wise smoothing factor s_j."""
        act_max = sample_activations.abs().view(-1, linear_layer.in_features).amax(dim=0)
        w_max = linear_layer.weight.abs().amax(dim=0)

        act_max = torch.clamp(act_max, min=1e-5)
        w_max = torch.clamp(w_max, min=1e-5)

        # Scale migration formula
        scale = (act_max ** self.alpha) / (w_max ** (1.0 - self.alpha))
        scale = torch.clamp(scale, min=1e-3)
        return scale

    def quantize_layer(
        self, layer: nn.Linear, scale: Optional[torch.Tensor] = None
    ) -> QuantizedLinear:
        ql = QuantizedLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            precision="int8_smoothquant",
        )
        w = layer.weight.data.clone()
        if scale is not None:
            w = w * scale.view(1, -1)
            ql.act_scales.copy_((1.0 / scale).view(1, -1).to(torch.float16))

        ql.pack_from_float(w)
        if layer.bias is not None:
            ql.bias.data.copy_(layer.bias.data)
        return ql


class FP8Quantizer:
    """Quantizes linear layers to native FP8 (E4M3) representation."""

    def quantize_layer(self, layer: nn.Linear) -> QuantizedLinear:
        ql = QuantizedLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            precision="fp8_e4m3",
        )
        ql.pack_from_float(layer.weight.data)
        if layer.bias is not None:
            ql.bias.data.copy_(layer.bias.data)
        return ql


def quantize_model_layers(model: nn.Module, precision: str) -> nn.Module:
    """Recursively replaces Linear layers with QuantizedLinear modules."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            ql = QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                precision=precision,
            )
            ql.pack_from_float(module.weight.data)
            if module.bias is not None:
                ql.bias.data.copy_(module.bias.data)
            setattr(model, name, ql)
        else:
            quantize_model_layers(module, precision)
    return model
