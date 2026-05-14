from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import SGD, Optimizer

PAPER_BATCH_SIZE = 64
PAPER_NUM_CLASSES = 43
PAPER_LEARNING_RATE = 0.0002
PAPER_MOMENTUM = 0.9

PAPER_FORWARD_NO_AE_BYTES = 3_212_639
PAPER_FORWARD_AE_BYTES = 263_520
PAPER_BACKWARD_GRAD_BYTES = 3_212_639
PAPER_STATUS_BYTES = 561

PAPER_CLIENT_FORWARD_FLOPS = 445_956_096
PAPER_ENCODER_FLOPS = 134_020_096
PAPER_CLIENT_BACKWARD_FLOPS = 891_912_192
PAPER_CLIENT_BASELINE_FLOPS = 1_337_868_288
PAPER_CLIENT_AE_NO_SKIP_FLOPS = 1_471_888_384


class PaperAutoencoderClient(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PaperAutoencoderEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.network(activation)


class PaperAutoencoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 16, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 64, kernel_size=1, stride=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class PaperAutoencoderServer(nn.Module):
    def __init__(self, num_classes: int = PAPER_NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(activation))


@dataclass(frozen=True)
class PaperAutoencoderStepResult:
    loss: float
    gradient_sent: bool
    client_updated: bool
    server_updated: bool
    split_shape: Tuple[int, ...]
    latent_shape: Tuple[int, ...]
    transmitted_forward_bytes: int
    transmitted_backward_bytes: int


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def paper_epoch_traffic_bytes(
    num_batches: int,
    dismissal_rate: float,
    *,
    forward_bytes: int = PAPER_FORWARD_AE_BYTES,
    backward_grad_bytes: int = PAPER_BACKWARD_GRAD_BYTES,
    status_bytes: int = PAPER_STATUS_BYTES,
) -> int:
    rho = max(0.0, min(1.0, float(dismissal_rate)))
    total = (
        float(num_batches) * float(forward_bytes)
        + float(num_batches) * (1.0 - rho) * float(backward_grad_bytes)
        + float(num_batches) * rho * float(status_bytes)
    )
    return int(round(total))


def paper_client_total_flops(dismissal_rate: float) -> int:
    rho = max(0.0, min(1.0, float(dismissal_rate)))
    total = float(PAPER_CLIENT_FORWARD_FLOPS) + float(PAPER_ENCODER_FLOPS) + (1.0 - rho) * float(PAPER_CLIENT_BACKWARD_FLOPS)
    return int(round(total))


def average_client_weights(client_models: Sequence[nn.Module]) -> Dict[str, torch.Tensor]:
    if not client_models:
        raise ValueError("client_models must not be empty")
    state_dicts = [model.state_dict() for model in client_models]
    averaged: Dict[str, torch.Tensor] = {}
    for key, reference in state_dicts[0].items():
        if torch.is_floating_point(reference):
            stacked = torch.stack([state[key].detach().to(dtype=torch.float32) for state in state_dicts], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=reference.dtype)
        else:
            averaged[key] = reference.detach().clone()
    return averaged


def broadcast_client_weights(client_models: Sequence[nn.Module], shared_state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    if not client_models:
        raise ValueError("client_models must not be empty")
    state = average_client_weights(client_models) if shared_state is None else {key: value.detach().clone() for key, value in shared_state.items()}
    for model in client_models:
        model.load_state_dict(state, strict=True)
    return state


def synchronize_clients(client_models: Sequence[nn.Module]) -> Dict[str, torch.Tensor]:
    return broadcast_client_weights(client_models)


class PaperAutoencoderSplitLearning:
    def __init__(
        self,
        *,
        client: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        server: Optional[nn.Module] = None,
        num_classes: int = PAPER_NUM_CLASSES,
        gradient_threshold: float = 0.0,
        learning_rate: float = PAPER_LEARNING_RATE,
        momentum: float = PAPER_MOMENTUM,
        freeze_autoencoder_during_training: bool = True,
        client_optimizer: Optional[Optimizer] = None,
        server_optimizer: Optional[Optimizer] = None,
        autoencoder_optimizer: Optional[Optimizer] = None,
    ) -> None:
        self.client = client if client is not None else PaperAutoencoderClient()
        self.encoder = encoder if encoder is not None else PaperAutoencoderEncoder()
        self.decoder = decoder if decoder is not None else PaperAutoencoderDecoder()
        self.server = server if server is not None else PaperAutoencoderServer(num_classes=num_classes)
        self.gradient_threshold = float(gradient_threshold)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.freeze_autoencoder_during_training = bool(freeze_autoencoder_during_training)
        if self.freeze_autoencoder_during_training:
            freeze_module(self.encoder)
            freeze_module(self.decoder)
        self.client_optimizer = client_optimizer if client_optimizer is not None else SGD(
            self.client.parameters(), lr=self.learning_rate, momentum=self.momentum
        )
        self.server_optimizer = server_optimizer if server_optimizer is not None else SGD(
            self.server.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

        ae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.autoencoder_optimizer = autoencoder_optimizer if autoencoder_optimizer is not None else SGD(
            ae_params, lr=self.learning_rate, momentum=self.momentum
        )

        self.loss_fn = nn.NLLLoss()
        self.reconstruction_loss_fn = nn.MSELoss()

    @classmethod
    def paper_configuration(
        cls,
        *,
        gradient_threshold: float = 0.0,
        num_classes: int = PAPER_NUM_CLASSES,
        learning_rate: float = PAPER_LEARNING_RATE,
        momentum: float = PAPER_MOMENTUM,
        autoencoder_is_pretrained: bool = False,
    ) -> "PaperAutoencoderSplitLearning":
        return cls(
            num_classes=num_classes,
            gradient_threshold=gradient_threshold,
            learning_rate=learning_rate,
            momentum=momentum,
            freeze_autoencoder_during_training=autoencoder_is_pretrained,
        )

    def autoencoder_forward(self, activation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(activation)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def autoencoder_reconstruction_loss(self, activation: torch.Tensor) -> torch.Tensor:
        _, reconstructed = self.autoencoder_forward(activation)
        return torch.mean((activation - reconstructed) ** 2)

    def train_autoencoder_step(self, x: torch.Tensor) -> float:
        self.client.eval()
        self.encoder.train()
        self.decoder.train()

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(True)

        self.autoencoder_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            split_activation = self.client(x)

        latent = self.encoder(split_activation)
        reconstructed = self.decoder(latent)

        reconstruction_loss = self.reconstruction_loss_fn(reconstructed, split_activation)
        reconstruction_loss.backward()
        self.autoencoder_optimizer.step()

        return float(reconstruction_loss.item())

    def freeze_autoencoder(self) -> None:
        freeze_module(self.encoder)
        freeze_module(self.decoder)
        self.freeze_autoencoder_during_training = True

    def load_autoencoder_weights(self, encoder_state: Dict[str, torch.Tensor], decoder_state: Dict[str, torch.Tensor]) -> None:
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(decoder_state, strict=True)
        self.freeze_autoencoder()

    def should_freeze_autoencoder(self, reconstruction_loss: float, threshold: float) -> bool:
        return float(reconstruction_loss) <= float(threshold)

    def should_send_gradient(self, loss_value: float) -> bool:
        return float(loss_value) > self.gradient_threshold

    def warmup_step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, PaperAutoencoderStepResult]:
        if self.freeze_autoencoder_during_training:
            raise RuntimeError(
                "warmup_step() is for the AE-training phase. "
                "Do not call it after freeze_autoencoder() or load_autoencoder_weights()."
            )

        ae_loss = self.train_autoencoder_step(x)

        self.freeze_autoencoder_during_training = True
        result = self.training_step(x, y)
        self.freeze_autoencoder_during_training = False

        self.encoder.train()
        self.decoder.train()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(True)

        return ae_loss, result

    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> PaperAutoencoderStepResult:
        if not self.freeze_autoencoder_during_training:
            raise RuntimeError(
                "training_step() expects a trained/frozen autoencoder. "
                "Call train_autoencoder_step(...) during warmup, then freeze_autoencoder()."
            )

        self.client.train()
        self.server.train()
        self.encoder.eval()
        self.decoder.eval()

        self.client_optimizer.zero_grad(set_to_none=True)
        self.server_optimizer.zero_grad(set_to_none=True)

        split_activation = self.client(x)
        with torch.no_grad():
            latent = self.encoder(split_activation)
            reconstructed = self.decoder(latent)

        split_proxy = reconstructed.detach().requires_grad_(True)
        log_probs = self.server(split_proxy)
        loss = self.loss_fn(log_probs, y)
        loss.backward()
        self.server_optimizer.step()

        loss_value = float(loss.item())
        gradient_sent = self.should_send_gradient(loss_value)
        client_updated = False
        backward_bytes = PAPER_STATUS_BYTES
        if gradient_sent:
            split_gradient = split_proxy.grad.detach()
            self.client_optimizer.zero_grad(set_to_none=True)
            split_activation.backward(split_gradient)
            self.client_optimizer.step()
            client_updated = True
            backward_bytes = PAPER_BACKWARD_GRAD_BYTES
        else:
            self.client_optimizer.zero_grad(set_to_none=True)

        return PaperAutoencoderStepResult(
            loss=loss_value,
            gradient_sent=gradient_sent,
            client_updated=client_updated,
            server_updated=True,
            split_shape=tuple(int(dim) for dim in split_activation.shape),
            latent_shape=tuple(int(dim) for dim in latent.shape),
            transmitted_forward_bytes=PAPER_FORWARD_AE_BYTES,
            transmitted_backward_bytes=backward_bytes,
        )


__all__ = [
    "PAPER_BATCH_SIZE",
    "PAPER_NUM_CLASSES",
    "PAPER_LEARNING_RATE",
    "PAPER_MOMENTUM",
    "PAPER_FORWARD_NO_AE_BYTES",
    "PAPER_FORWARD_AE_BYTES",
    "PAPER_BACKWARD_GRAD_BYTES",
    "PAPER_STATUS_BYTES",
    "PAPER_CLIENT_FORWARD_FLOPS",
    "PAPER_ENCODER_FLOPS",
    "PAPER_CLIENT_BACKWARD_FLOPS",
    "PAPER_CLIENT_BASELINE_FLOPS",
    "PAPER_CLIENT_AE_NO_SKIP_FLOPS",
    "PaperAutoencoderClient",
    "PaperAutoencoderEncoder",
    "PaperAutoencoderDecoder",
    "PaperAutoencoderServer",
    "PaperAutoencoderStepResult",
    "PaperAutoencoderSplitLearning",
    "freeze_module",
    "average_client_weights",
    "broadcast_client_weights",
    "synchronize_clients",
    "paper_epoch_traffic_bytes",
    "paper_client_total_flops",
]