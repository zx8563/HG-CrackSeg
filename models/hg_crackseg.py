"""Hessian-guided crack segmentation."""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
ModelOutput = Dict[str, Union[Tensor, Dict[str, Tensor]]]


class AttentionGate(nn.Module):

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, bias=True), nn.Sigmoid())
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(
        self, g: Tensor, x: Tensor, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        attention = self.psi(self.leaky_relu(g1 + x1))
        gated = x * attention
        if return_attention:
            return (gated, attention)
        return gated


class IdentityAttentionGate(nn.Module):

    def forward(
        self, g: Tensor, x: Tensor, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        del g
        if return_attention:
            attention = torch.ones(
                x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device
            )
            return (x, attention)
        return x


class DifferentiableHessian(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
        max_kernel_size: int = 25,
        residual_scale: float = 0.03,
        dark_gate_sharpness: float = 8.0,
        eps: float = 1e-06,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError("DifferentiableHessian requires at least three RGB channels.")
        if max_kernel_size % 2 != 1:
            raise ValueError("max_kernel_size must be odd.")
        if not sigmas:
            raise ValueError("At least one Hessian scale is required.")
        self.in_channels = int(in_channels)
        self.sigmas = tuple((float(sigma) for sigma in sigmas))
        self.num_scales = len(self.sigmas)
        self.max_kernel_size = int(max_kernel_size)
        self.padding = self.max_kernel_size // 2
        self.residual_scale = float(residual_scale)
        self.dark_gate_sharpness = float(dark_gate_sharpness)
        self.eps = float(eps)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("gray_weight", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        (kxx, kyy, kxy) = self._build_multiscale_hessian_kernels()
        self.register_buffer("base_gxx", kxx)
        self.register_buffer("base_gyy", kyy)
        self.register_buffer("base_gxy", kxy)
        self.delta_gxx = nn.Parameter(torch.zeros_like(kxx))
        self.delta_gyy = nn.Parameter(torch.zeros_like(kyy))
        self.delta_gxy = nn.Parameter(torch.zeros_like(kxy))
        self.beta_raw = nn.Parameter(self._inv_softplus(torch.tensor(0.5)))
        self.c_raw = nn.Parameter(self._inv_softplus(torch.tensor(0.12)))

    @staticmethod
    def _inv_softplus(x: Tensor) -> Tensor:
        return torch.log(torch.exp(x) - 1.0)

    def _center_pad_to_max(self, kernel: Tensor) -> Tensor:
        kernel_size = kernel.shape[-1]
        output = torch.zeros(self.max_kernel_size, self.max_kernel_size, dtype=torch.float32)
        start = (self.max_kernel_size - kernel_size) // 2
        output[start : start + kernel_size, start : start + kernel_size] = kernel.float()
        return output

    def _single_scale_kernels(self, sigma: float) -> Tuple[Tensor, Tensor, Tensor]:
        kernel_size = int(2 * round(3.0 * sigma) + 1)
        kernel_size = max(7, kernel_size)
        kernel_size = min(kernel_size, self.max_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2
        grid = torch.arange(-half, half + 1, dtype=torch.float32)
        (y, x) = torch.meshgrid(grid, grid, indexing="ij")
        sigma2 = sigma**2
        gauss = torch.exp(-(x**2 + y**2) / (2.0 * sigma2))
        gauss = gauss / (gauss.sum() + 1e-12)
        gxx = (x**2 / sigma**4 - 1.0 / sigma2) * gauss * sigma2
        gyy = (y**2 / sigma**4 - 1.0 / sigma2) * gauss * sigma2
        gxy = x * y / sigma**4 * gauss * sigma2

        def normalize_kernel(kernel: Tensor) -> Tensor:
            kernel = kernel - kernel.mean()
            return kernel / (kernel.abs().sum() + 1e-12)

        return (normalize_kernel(gxx), normalize_kernel(gyy), normalize_kernel(gxy))

    def _build_multiscale_hessian_kernels(self) -> Tuple[Tensor, Tensor, Tensor]:
        kxx_list = []
        kyy_list = []
        kxy_list = []
        for sigma in self.sigmas:
            (gxx, gyy, gxy) = self._single_scale_kernels(sigma)
            kxx_list.append(self._center_pad_to_max(gxx))
            kyy_list.append(self._center_pad_to_max(gyy))
            kxy_list.append(self._center_pad_to_max(gxy))
        kxx = torch.stack(kxx_list, dim=0).unsqueeze(1)
        kyy = torch.stack(kyy_list, dim=0).unsqueeze(1)
        kxy = torch.stack(kxy_list, dim=0).unsqueeze(1)
        return (kxx, kyy, kxy)

    def _to_gray_01(self, x: Tensor) -> Tensor:
        rgb = x[:, :3, :, :]
        rgb01 = rgb * self.rgb_std + self.rgb_mean
        rgb01 = rgb01.clamp(0.0, 1.0)
        return (rgb01 * self.gray_weight).sum(dim=1, keepdim=True)

    def _constrained_kernel(self, base: Tensor, delta: Tensor) -> Tensor:
        delta = delta - delta.mean(dim=(2, 3), keepdim=True)
        return base + self.residual_scale * delta

    def _hessian_response(self, gray: Tensor) -> Tensor:
        wxx = self._constrained_kernel(self.base_gxx, self.delta_gxx)
        wyy = self._constrained_kernel(self.base_gyy, self.delta_gyy)
        wxy = self._constrained_kernel(self.base_gxy, self.delta_gxy)
        dxx = F.conv2d(gray, wxx, padding=self.padding)
        dyy = F.conv2d(gray, wyy, padding=self.padding)
        dxy = F.conv2d(gray, wxy, padding=self.padding)
        trace = dxx + dyy
        difference = dxx - dyy
        discriminant = torch.sqrt(difference * difference + 4.0 * dxy * dxy + self.eps)
        lambda_a = 0.5 * (trace - discriminant)
        lambda_b = 0.5 * (trace + discriminant)
        swap = lambda_a.abs() > lambda_b.abs()
        lambda_small = torch.where(swap, lambda_b, lambda_a)
        lambda_large = torch.where(swap, lambda_a, lambda_b)
        beta = F.softplus(self.beta_raw) + self.eps
        c_value = F.softplus(self.c_raw) + self.eps
        ratio = lambda_small.abs() / (lambda_large.abs() + self.eps)
        structure_energy = lambda_small * lambda_small + lambda_large * lambda_large
        line_likeness = torch.exp(-(ratio * ratio) / (2.0 * beta * beta))
        structure = 1.0 - torch.exp(-structure_energy / (2.0 * c_value * c_value))
        dark_gate = torch.sigmoid(self.dark_gate_sharpness * lambda_large)
        vessel = line_likeness * structure * dark_gate
        return vessel.clamp(0.0, 1.0)

    def forward(self, x: Tensor, return_aux: bool = False) -> Union[Tensor, Dict[str, Tensor]]:
        gray = self._to_gray_01(x)
        scale_responses = self._hessian_response(gray).contiguous()
        if return_aux:
            return {
                "gray": gray,
                "scale_responses": scale_responses,
                "stacked_response": scale_responses,
            }
        return scale_responses


class ConvBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class LargeKernelResidualBlock(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.proj(x)
        x = self.norm(x)
        x = self.ffn(x)
        return x + shortcut


class ASPP(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        dilations = (1, 6, 12, 18)
        self.aspp1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=dilations[0], bias=False
        )
        self.aspp2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilations[1],
            dilation=dilations[1],
            bias=False,
        )
        self.aspp3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilations[2],
            dilation=dilations[2],
            bias=False,
        )
        self.aspp4 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilations[3],
            dilation=dilations[3],
            bias=False,
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        (height, width) = (x.size(2), x.size(3))
        x5 = x5.expand(-1, -1, height, width).contiguous()
        x_concat = torch.cat((x1, x2, x3, x4, x5), dim=1).contiguous()
        return self.out_conv(x_concat)


class HGCrackSeg(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        base_c: int = 32,
        classes: int = 2,
        use_hessian_branch: bool = True,
        edge_guidance_levels: int = 2,
        use_large_kernel: bool = True,
        use_aspp: bool = True,
        use_attention_gate: bool = True,
        hessian_sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
        hessian_max_kernel_size: int = 25,
        hessian_residual_scale: float = 0.03,
        hessian_dark_gate_sharpness: float = 8.0,
    ) -> None:
        super().__init__()
        if in_channels < 3 and use_hessian_branch:
            raise ValueError("use_hessian_branch=True requires at least three input channels.")
        if base_c < 2 or base_c % 2 != 0:
            raise ValueError(f"base_c must be an even integer >= 2, got {base_c}.")
        if classes < 1:
            raise ValueError("classes must be at least 1.")
        if edge_guidance_levels not in (2, 4):
            raise ValueError("edge_guidance_levels currently supports only 2 or 4.")
        self.in_channels = int(in_channels)
        self.base_c = int(base_c)
        self.classes = int(classes)
        self.use_hessian_branch = bool(use_hessian_branch)
        self.edge_guidance_levels = int(edge_guidance_levels) if self.use_hessian_branch else 0
        self.use_large_kernel = bool(use_large_kernel)
        self.use_aspp = bool(use_aspp)
        self.use_attention_gate = bool(use_attention_gate)
        if self.use_hessian_branch:
            self.hessian = DifferentiableHessian(
                in_channels=in_channels,
                sigmas=hessian_sigmas,
                max_kernel_size=hessian_max_kernel_size,
                residual_scale=hessian_residual_scale,
                dark_gate_sharpness=hessian_dark_gate_sharpness,
            )
            edge_in_c = in_channels + self.hessian.num_scales
            self.edge_enc1 = ConvBlock(edge_in_c, base_c)
            self.edge_down = nn.MaxPool2d(2)
            self.edge_enc2 = ConvBlock(base_c, base_c * 2)
            if self.edge_guidance_levels == 4:
                self.edge_enc3 = ConvBlock(base_c * 2, base_c * 4)
                self.edge_enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.down1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3_conv = ConvBlock(base_c * 2, base_c * 4)
        self.down3_large_kernel = (
            LargeKernelResidualBlock(base_c * 4) if self.use_large_kernel else nn.Identity()
        )
        self.pool3 = nn.MaxPool2d(2)
        self.down4_conv = ConvBlock(base_c * 4, base_c * 8)
        self.down4_large_kernel = (
            LargeKernelResidualBlock(base_c * 8) if self.use_large_kernel else nn.Identity()
        )
        self.pool4 = nn.MaxPool2d(2)
        self.neck = (
            ASPP(base_c * 8, base_c * 8) if self.use_aspp else ConvBlock(base_c * 8, base_c * 8)
        )
        skip1_channels = base_c
        skip2_channels = base_c * 2
        skip3_channels = base_c * 4
        skip4_channels = base_c * 8
        if self.use_hessian_branch:
            skip1_channels += base_c
            skip2_channels += base_c * 2
            if self.edge_guidance_levels == 4:
                skip3_channels += base_c * 4
                skip4_channels += base_c * 8
        gate_cls = AttentionGate if self.use_attention_gate else IdentityAttentionGate
        self.up4 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.ag4 = (
            gate_cls(F_g=base_c * 4, F_l=skip4_channels, F_int=base_c * 4)
            if self.use_attention_gate
            else gate_cls()
        )
        self.dec4 = ConvBlock(base_c * 4 + skip4_channels, base_c * 4)
        self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.ag3 = (
            gate_cls(F_g=base_c * 2, F_l=skip3_channels, F_int=base_c * 2)
            if self.use_attention_gate
            else gate_cls()
        )
        self.dec3 = ConvBlock(base_c * 2 + skip3_channels, base_c * 2)
        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.ag2 = (
            gate_cls(F_g=base_c, F_l=skip2_channels, F_int=base_c)
            if self.use_attention_gate
            else gate_cls()
        )
        self.dec2 = ConvBlock(base_c + skip2_channels, base_c)
        self.up1 = nn.ConvTranspose2d(base_c, base_c // 2, kernel_size=2, stride=2)
        self.ag1 = (
            gate_cls(F_g=base_c // 2, F_l=skip1_channels, F_int=base_c // 2)
            if self.use_attention_gate
            else gate_cls()
        )
        self.dec1 = ConvBlock(base_c // 2 + skip1_channels, base_c // 2)
        self.prob_head = nn.Sequential(
            nn.Conv2d(base_c // 2, base_c // 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_c // 2, classes, kernel_size=1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, module in self.named_modules():
            if name.startswith("hessian"):
                continue
            if "psi.0" in name and isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                continue
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def get_architecture_config(self) -> Dict[str, Union[int, bool, str]]:
        return {
            "in_channels": self.in_channels,
            "base_c": self.base_c,
            "classes": self.classes,
            "use_hessian_branch": self.use_hessian_branch,
            "edge_guidance_levels": self.edge_guidance_levels,
            "use_large_kernel": self.use_large_kernel,
            "use_aspp": self.use_aspp,
            "use_attention_gate": self.use_attention_gate,
        }

    @staticmethod
    def _validate_spatial_size(x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Input must be [B,C,H,W], got shape {tuple(x.shape)}.")
        (height, width) = x.shape[-2:]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Input height and width must be divisible by 16; got H={height}, W={width}."
            )

    def _apply_gate(
        self, gate: nn.Module, decoder_feature: Tensor, skip_feature: Tensor, return_aux: bool
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if return_aux:
            (gated, attention) = gate(g=decoder_feature, x=skip_feature, return_attention=True)
            return (gated, attention)
        gated = gate(g=decoder_feature, x=skip_feature, return_attention=False)
        return (gated, None)

    def forward(self, x: Tensor, return_aux: bool = False) -> ModelOutput:
        self._validate_spatial_size(x)
        aux: Dict[str, Tensor] = {}
        e1: Optional[Tensor] = None
        e2: Optional[Tensor] = None
        e3: Optional[Tensor] = None
        e4: Optional[Tensor] = None
        if self.use_hessian_branch:
            if return_aux:
                hessian_aux = self.hessian(x, return_aux=True)
                assert isinstance(hessian_aux, dict)
                h_feat = hessian_aux["stacked_response"]
                aux["hessian_gray"] = hessian_aux["gray"]
                aux["hessian_scale_responses"] = hessian_aux["scale_responses"]
                aux["hessian_stacked_response"] = h_feat
            else:
                h_feat = self.hessian(x)
                assert torch.is_tensor(h_feat)
            edge_in = torch.cat([x, h_feat], dim=1).contiguous()
            e1 = self.edge_enc1(edge_in)
            e2 = self.edge_enc2(self.edge_down(e1))
            if self.edge_guidance_levels == 4:
                e3 = self.edge_enc3(self.edge_down(e2))
                e4 = self.edge_enc4(self.edge_down(e3))
            if return_aux:
                aux["edge_feature_1"] = e1
                aux["edge_feature_2"] = e2
                if e3 is not None:
                    aux["edge_feature_3"] = e3
                if e4 is not None:
                    aux["edge_feature_4"] = e4
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3_large_kernel(self.down3_conv(self.pool2(d2)))
        d4 = self.down4_large_kernel(self.down4_conv(self.pool3(d3)))
        neck = self.neck(self.pool4(d4))
        u4 = self.up4(neck)
        skip4 = torch.cat([d4, e4], dim=1).contiguous() if e4 is not None else d4
        (x4, attention4) = self._apply_gate(self.ag4, u4, skip4, return_aux)
        u4 = self.dec4(torch.cat([u4, x4], dim=1).contiguous())
        u3 = self.up3(u4)
        skip3 = torch.cat([d3, e3], dim=1).contiguous() if e3 is not None else d3
        (x3, attention3) = self._apply_gate(self.ag3, u3, skip3, return_aux)
        u3 = self.dec3(torch.cat([u3, x3], dim=1).contiguous())
        u2 = self.up2(u3)
        skip2 = torch.cat([d2, e2], dim=1).contiguous() if e2 is not None else d2
        (x2, attention2) = self._apply_gate(self.ag2, u2, skip2, return_aux)
        u2 = self.dec2(torch.cat([u2, x2], dim=1).contiguous())
        u1 = self.up1(u2)
        skip1 = torch.cat([d1, e1], dim=1).contiguous() if e1 is not None else d1
        (x1, attention1) = self._apply_gate(self.ag1, u1, skip1, return_aux)
        u1 = self.dec1(torch.cat([u1, x1], dim=1).contiguous())
        body_logits = self.prob_head(u1)
        output: ModelOutput = {"body_logits": body_logits}
        if return_aux:
            aux.update(
                {
                    "encoder_feature_1": d1,
                    "encoder_feature_2": d2,
                    "encoder_feature_3": d3,
                    "encoder_feature_4": d4,
                    "neck_feature": neck,
                    "decoder_feature_4": u4,
                    "decoder_feature_3": u3,
                    "decoder_feature_2": u2,
                    "decoder_feature_1": u1,
                    "final_feature": u1,
                }
            )
            if attention1 is not None:
                aux["attention_1"] = attention1
            if attention2 is not None:
                aux["attention_2"] = attention2
            if attention3 is not None:
                aux["attention_3"] = attention3
            if attention4 is not None:
                aux["attention_4"] = attention4
            output["aux"] = aux
        return output
