"""SplitFedZip: Learned Compression for Data Transfer Reduction in Split-Federated Learning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer

try:
    from compressai.entropy_models import EntropyBottleneck as _CompressAIEntropyBottleneck
    from compressai.layers import GDN as _CompressAIGDN

except ImportError as exc:
    raise ImportError(
        "Paper-faithful SplitFedZip requires CompressAI. "
        "Install compressai instead of using the fallback entropy model."
    ) from exc


PAPER_CODEC_CHANNELS = 64
PAPER_MODEL_LR = 1e-4
PAPER_CODEC_LR = 1e-4
PAPER_LAMBDA = 1.0
PAPER_SCHEMES = ("F", "FG")
PAPER_GRADIENT_STRATEGIES = ("paper",)


def _make_gdn(channels: int, *, inverse: bool = False) -> nn.Module:
    return _CompressAIGDN(channels, inverse=inverse)


def _make_entropy_bottleneck(channels: int) -> nn.Module:
    return _CompressAIEntropyBottleneck(channels)


def _reference_nhw(reference_input: Tensor) -> float:
    if reference_input.ndim < 4:
        raise ValueError("reference_input must have shape [N, C, H, W]")
    return float(reference_input.shape[0] * reference_input.shape[-2] * reference_input.shape[-1])


def entropy_bpp_loss(likelihoods: Tensor, reference_input: Tensor) -> Tensor:
    if not isinstance(likelihoods, torch.Tensor):
        raise TypeError("likelihoods must be a torch.Tensor")
    normalizer = _reference_nhw(reference_input)
    safe_likelihoods = likelihoods.clamp_min(1e-9)
    return torch.log(safe_likelihoods).sum() / (-math.log(2.0) * normalizer)


def multiclass_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    if logits.ndim < 4:
        raise ValueError("logits must have shape [N, C, H, W]")
    if logits.shape[1] == 1:
        probabilities = torch.sigmoid(logits)
        if target.ndim == logits.ndim - 1:
            target_tensor = target.unsqueeze(1)
        else:
            target_tensor = target
        target_tensor = target_tensor.to(dtype=probabilities.dtype)
    else:
        probabilities = torch.softmax(logits, dim=1)
        if target.shape == logits.shape:
            target_tensor = target.to(dtype=probabilities.dtype)
        else:
            class_target = target
            if class_target.ndim == logits.ndim and class_target.shape[1] == 1:
                class_target = class_target[:, 0]
            if class_target.ndim != logits.ndim - 1:
                raise ValueError("target must contain class indices or one-hot masks")
            class_target = class_target.long().clamp(min=0, max=logits.shape[1] - 1)
            target_tensor = F.one_hot(class_target, num_classes=logits.shape[1]).movedim(-1, 1).to(dtype=probabilities.dtype)
    probabilities = probabilities.flatten(2)
    target_tensor = target_tensor.flatten(2)
    intersection = (probabilities * target_tensor).sum(dim=-1)
    denominator = probabilities.sum(dim=-1) + target_tensor.sum(dim=-1)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


@dataclass(frozen=True)
class EntropyCompressionResult:
    reconstructed: Tensor
    likelihoods: Tensor
    bpp_loss: Tensor
    mse_loss: Tensor
    total_loss: Tensor


@dataclass(frozen=True)
class PaperEntropyStepResult:
    scheme: str
    gradient_strategy: str
    total_loss: float
    dice_loss: float
    s1_feature_bpp: float
    s2_feature_bpp: float
    s1_feature_mse: float
    s2_feature_mse: float
    s2_gradient_bpp: float
    s1_gradient_bpp: float
    s2_gradient_mse: float
    s1_gradient_mse: float
    aux_loss: float


class PaperEntropyCodec(nn.Module):
    def __init__(self, channels: int = PAPER_CODEC_CHANNELS) -> None:
        super().__init__()
        self.channels = int(channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=5, stride=2, padding=2),
            _make_gdn(self.channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=5, stride=2, padding=2),
            _make_gdn(self.channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=5, stride=2, padding=2),
        )
        self.entropy_bottleneck = _make_entropy_bottleneck(self.channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            _make_gdn(self.channels, inverse=True),
            nn.ConvTranspose2d(self.channels, self.channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            _make_gdn(self.channels, inverse=True),
            nn.ConvTranspose2d(self.channels, self.channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.encoder(x)
        y_hat, likelihoods = self.entropy_bottleneck(y)
        reconstructed = self.decoder(y_hat)
        if reconstructed.shape[-2:] != x.shape[-2:]:
            reconstructed = F.interpolate(reconstructed, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return reconstructed, likelihoods

    def compress_with_metrics(self, x: Tensor, reference_input: Tensor, lambda_weight: float) -> EntropyCompressionResult:
        reconstructed, likelihoods = self(x)
        bpp_loss = entropy_bpp_loss(likelihoods, reference_input)
        mse_loss = F.mse_loss(reconstructed, x)
        total_loss = bpp_loss + float(lambda_weight) * mse_loss
        return EntropyCompressionResult(
            reconstructed=reconstructed,
            likelihoods=likelihoods,
            bpp_loss=bpp_loss,
            mse_loss=mse_loss,
            total_loss=total_loss,
        )

    def aux_loss(self) -> Tensor:
        if hasattr(self.entropy_bottleneck, "aux_loss"):
            return self.entropy_bottleneck.aux_loss()
        return next(self.parameters()).new_zeros(())

    def aux_parameters(self) -> Iterable[nn.Parameter]:
        if hasattr(self.entropy_bottleneck, "aux_parameters"):
            return self.entropy_bottleneck.aux_parameters()
        return ()


class PaperEntropyFrontEnd(nn.Module):
    def __init__(self, in_channels: int = 1, feature_channels: int = PAPER_CODEC_CHANNELS) -> None:
        super().__init__()
        self.skip_block = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.split_block = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        enc1 = self.skip_block(x)
        activation = self.split_block(enc1)
        return enc1, activation


class PaperEntropyServerModel(nn.Module):
    def __init__(self, feature_channels: int = PAPER_CODEC_CHANNELS) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class PaperEntropyBackEnd(nn.Module):
    def __init__(self, feature_channels: int = PAPER_CODEC_CHANNELS, num_classes: int = 2) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, num_classes, kernel_size=1),
        )

    def forward(self, enc1: Tensor, activation: Tensor) -> Tensor:
        upsampled = self.upsample(activation)
        if upsampled.shape[-2:] != enc1.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([enc1, upsampled], dim=1)
        return self.head(merged)


def _trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


class PaperEntropySplitLearning:
    def __init__(
        self,
        *,
        front_end: Optional[nn.Module] = None,
        server_model: Optional[nn.Module] = None,
        back_end: Optional[nn.Module] = None,
        codec_s1_feature: Optional[PaperEntropyCodec] = None,
        codec_s2_feature: Optional[PaperEntropyCodec] = None,
        codec_s2_gradient: Optional[PaperEntropyCodec] = None,
        codec_s1_gradient: Optional[PaperEntropyCodec] = None,
        input_channels: int = 1,
        feature_channels: int = PAPER_CODEC_CHANNELS,
        num_classes: int = 2,
        scheme: str = "FG",
        gradient_strategy: str = "paper",
        lambda_weight: float = PAPER_LAMBDA,
        model_lr: float = PAPER_MODEL_LR,
        codec_lr: float = PAPER_CODEC_LR,
        task_loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        model_optimizer: Optional[Optimizer] = None,
        codec_optimizers: Optional[Dict[str, Optimizer]] = None,
        aux_optimizers: Optional[Dict[str, Optional[Optimizer]]] = None,
    ) -> None:
        normalized_scheme = str(scheme).upper()
        if normalized_scheme not in PAPER_SCHEMES:
            raise ValueError(f"scheme must be one of {PAPER_SCHEMES}")
        normalized_strategy = str(gradient_strategy).lower()
        if normalized_strategy not in PAPER_GRADIENT_STRATEGIES:
            raise ValueError(f"gradient_strategy must be one of {PAPER_GRADIENT_STRATEGIES}")

        self.scheme = normalized_scheme
        self.gradient_strategy = normalized_strategy
        self.lambda_weight = float(lambda_weight)
        self.front_end = front_end if front_end is not None else PaperEntropyFrontEnd(in_channels=input_channels, feature_channels=feature_channels)
        self.server_model = server_model if server_model is not None else PaperEntropyServerModel(feature_channels=feature_channels)
        self.back_end = back_end if back_end is not None else PaperEntropyBackEnd(feature_channels=feature_channels, num_classes=num_classes)
        self.codec_s1_feature = codec_s1_feature if codec_s1_feature is not None else PaperEntropyCodec(channels=feature_channels)
        self.codec_s2_feature = codec_s2_feature if codec_s2_feature is not None else PaperEntropyCodec(channels=feature_channels)
        self.codec_s2_gradient = codec_s2_gradient if codec_s2_gradient is not None else (PaperEntropyCodec(channels=feature_channels) if self.scheme == "FG" else None)
        self.codec_s1_gradient = codec_s1_gradient if codec_s1_gradient is not None else (PaperEntropyCodec(channels=feature_channels) if self.scheme == "FG" else None)
        self.task_loss_fn = task_loss_fn if task_loss_fn is not None else multiclass_dice_loss

        model_parameters = _trainable_parameters(self.front_end) + _trainable_parameters(self.server_model) + _trainable_parameters(self.back_end)
        self.model_optimizer = model_optimizer if model_optimizer is not None else Adam(model_parameters, lr=float(model_lr))

        self.codec_modules: Dict[str, PaperEntropyCodec] = {
            "codec1": self.codec_s1_feature,
            "codec2": self.codec_s2_feature,
        }
        if self.scheme == "FG":
            assert self.codec_s2_gradient is not None
            assert self.codec_s1_gradient is not None
            self.codec_modules["codec3"] = self.codec_s2_gradient
            self.codec_modules["codec4"] = self.codec_s1_gradient

        if codec_optimizers is None:
            self.codec_optimizers = {
                name: Adam(_trainable_parameters(codec), lr=float(codec_lr))
                for name, codec in self.codec_modules.items()
            }
        else:
            self.codec_optimizers = dict(codec_optimizers)

        if aux_optimizers is None:
            self.aux_optimizers = {
                name: self._build_default_aux_optimizer(codec, codec_lr)
                for name, codec in self.codec_modules.items()
            }
        else:
            self.aux_optimizers = dict(aux_optimizers)

    @classmethod
    def paper_configuration(
        cls,
        *,
        scheme: str = "FG",
        gradient_strategy: str = "paper",
        lambda_weight: float = PAPER_LAMBDA,
        input_channels: int = 1,
        feature_channels: int = PAPER_CODEC_CHANNELS,
        num_classes: int = 2,
    ) -> "PaperEntropySplitLearning":
        return cls(
            scheme=scheme,
            gradient_strategy=gradient_strategy,
            lambda_weight=lambda_weight,
            input_channels=input_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
        )

    def _build_default_aux_optimizer(self, codec: PaperEntropyCodec, codec_lr: float) -> Optional[Optimizer]:
        aux_params = list(codec.aux_parameters())
        if not aux_params:
            return None
        return Adam(aux_params, lr=float(codec_lr))

    def _zero_main_optimizers(self) -> None:
        self.model_optimizer.zero_grad(set_to_none=True)
        for optimizer in self.codec_optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def _step_main_optimizers(self) -> None:
        self.model_optimizer.step()
        for optimizer in self.codec_optimizers.values():
            optimizer.step()

    def _step_aux_optimizers(self) -> float:
        aux_total = 0.0
        for name, codec in self.codec_modules.items():
            optimizer = self.aux_optimizers.get(name)
            if optimizer is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            aux_loss = codec.aux_loss()
            aux_loss.backward()
            optimizer.step()
            aux_total += float(aux_loss.detach().item())
        return aux_total

    def _assign_gradients(self, parameters: Sequence[nn.Parameter], gradients: Sequence[Optional[Tensor]], *, add: bool = False) -> None:
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            detached = gradient.detach()
            if add and parameter.grad is not None:
                parameter.grad = parameter.grad + detached
            else:
                parameter.grad = detached.clone()

    def _feature_forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, EntropyCompressionResult, Tensor, EntropyCompressionResult, Tensor]:
        enc1, activation_s1 = self.front_end(x)
        rd_s1 = self.codec_s1_feature.compress_with_metrics(activation_s1, x, self.lambda_weight)
        activation_s2 = self.server_model(rd_s1.reconstructed)
        rd_s2 = self.codec_s2_feature.compress_with_metrics(activation_s2, x, self.lambda_weight)
        logits = self.back_end(enc1, rd_s2.reconstructed)
        dice = self.task_loss_fn(logits, y)
        return enc1, rd_s1, activation_s2, rd_s2, dice

    def _training_step_f(self, x: Tensor, y: Tensor) -> PaperEntropyStepResult:
        self._zero_main_optimizers()
        _enc1, rd_s1, _activation_s2, rd_s2, dice = self._feature_forward(x, y)
        total_loss = rd_s1.total_loss + rd_s2.total_loss + self.lambda_weight * dice
        total_loss.backward()
        self._step_main_optimizers()
        aux_total = self._step_aux_optimizers()
        return PaperEntropyStepResult(
            scheme="F",
            gradient_strategy=self.gradient_strategy,
            total_loss=float(total_loss.detach().item()),
            dice_loss=float(dice.detach().item()),
            s1_feature_bpp=float(rd_s1.bpp_loss.detach().item()),
            s2_feature_bpp=float(rd_s2.bpp_loss.detach().item()),
            s1_feature_mse=float(rd_s1.mse_loss.detach().item()),
            s2_feature_mse=float(rd_s2.mse_loss.detach().item()),
            s2_gradient_bpp=0.0,
            s1_gradient_bpp=0.0,
            s2_gradient_mse=0.0,
            s1_gradient_mse=0.0,
            aux_loss=aux_total,
        )

    def _training_step_fg_code(self, x: Tensor, y: Tensor) -> PaperEntropyStepResult:
        raise RuntimeError("Not part of the SplitFedZip paper algorithm. Use gradient_strategy='paper'.")

    def _training_step_fg_paper(self, x: Tensor, y: Tensor) -> PaperEntropyStepResult:
        assert self.codec_s2_gradient is not None
        assert self.codec_s1_gradient is not None

        self._zero_main_optimizers()
        _enc1, rd_s1, activation_s2, rd_s2, dice = self._feature_forward(x, y)

        s1_feature_loss = rd_s1.total_loss
        s2_feature_task_loss = rd_s2.total_loss + self.lambda_weight * dice

        grad_s2 = torch.autograd.grad(
            s2_feature_task_loss,
            activation_s2,
            retain_graph=True,
            create_graph=False,
        )[0].detach()

        rd_s3 = self.codec_s2_gradient.compress_with_metrics(
            grad_s2,
            x,
            self.lambda_weight,
        )

        grad_s2_received = rd_s3.reconstructed.detach()

        grad_s1 = torch.autograd.grad(
            activation_s2,
            rd_s1.reconstructed,
            grad_outputs=grad_s2_received,
            retain_graph=True,
            create_graph=False,
        )[0].detach()

        rd_s4 = self.codec_s1_gradient.compress_with_metrics(
            grad_s1,
            x,
            self.lambda_weight,
        )

        grad_s1_received = rd_s4.reconstructed.detach()

        fe_params = _trainable_parameters(self.front_end)
        server_params = _trainable_parameters(self.server_model)
        be_params = _trainable_parameters(self.back_end)
        codec1_params = _trainable_parameters(self.codec_s1_feature)
        codec2_params = _trainable_parameters(self.codec_s2_feature)
        codec3_params = _trainable_parameters(self.codec_s2_gradient)
        codec4_params = _trainable_parameters(self.codec_s1_gradient)

        be_and_codec2 = be_params + codec2_params
        be_and_codec2_grads = torch.autograd.grad(
            s2_feature_task_loss,
            be_and_codec2,
            retain_graph=True,
            allow_unused=True,
        )

        server_grads = torch.autograd.grad(
            activation_s2,
            server_params,
            grad_outputs=grad_s2_received,
            retain_graph=True,
            allow_unused=True,
        )

        fe_and_codec1 = fe_params + codec1_params
        fe_and_codec1_grads = torch.autograd.grad(
            rd_s1.reconstructed,
            fe_and_codec1,
            grad_outputs=grad_s1_received,
            retain_graph=True,
            allow_unused=True,
        )

        codec1_direct_grads = torch.autograd.grad(
            s1_feature_loss,
            codec1_params,
            retain_graph=True,
            allow_unused=True,
        )

        grad_codec_grads = torch.autograd.grad(
            rd_s3.total_loss + rd_s4.total_loss,
            codec3_params + codec4_params,
            allow_unused=True,
        )

        self._assign_gradients(be_and_codec2, be_and_codec2_grads)
        self._assign_gradients(server_params, server_grads)
        self._assign_gradients(fe_and_codec1, fe_and_codec1_grads)
        self._assign_gradients(codec1_params, codec1_direct_grads, add=True)
        self._assign_gradients(codec3_params + codec4_params, grad_codec_grads)

        total_loss = s1_feature_loss + s2_feature_task_loss + rd_s3.total_loss + rd_s4.total_loss
        self._step_main_optimizers()
        aux_total = self._step_aux_optimizers()
        return PaperEntropyStepResult(
            scheme="FG",
            gradient_strategy="paper",
            total_loss=float(total_loss.detach().item()),
            dice_loss=float(dice.detach().item()),
            s1_feature_bpp=float(rd_s1.bpp_loss.detach().item()),
            s2_feature_bpp=float(rd_s2.bpp_loss.detach().item()),
            s1_feature_mse=float(rd_s1.mse_loss.detach().item()),
            s2_feature_mse=float(rd_s2.mse_loss.detach().item()),
            s2_gradient_bpp=float(rd_s3.bpp_loss.detach().item()),
            s1_gradient_bpp=float(rd_s4.bpp_loss.detach().item()),
            s2_gradient_mse=float(rd_s3.mse_loss.detach().item()),
            s1_gradient_mse=float(rd_s4.mse_loss.detach().item()),
            aux_loss=aux_total,
        )

    def training_step(self, x: Tensor, y: Tensor) -> PaperEntropyStepResult:
        if self.scheme == "F":
            return self._training_step_f(x, y)
        return self._training_step_fg_paper(x, y)


__all__ = [
    "PAPER_CODEC_CHANNELS",
    "PAPER_MODEL_LR",
    "PAPER_CODEC_LR",
    "PAPER_LAMBDA",
    "PAPER_SCHEMES",
    "PAPER_GRADIENT_STRATEGIES",
    "EntropyCompressionResult",
    "PaperEntropyStepResult",
    "PaperEntropyCodec",
    "PaperEntropyFrontEnd",
    "PaperEntropyServerModel",
    "PaperEntropyBackEnd",
    "PaperEntropySplitLearning",
    "entropy_bpp_loss",
    "multiclass_dice_loss",
]