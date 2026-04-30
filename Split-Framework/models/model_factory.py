"""Model factory built on imported torchvision and transformers backbones."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, efficientnet_b0, resnet18


def _cfg_get(parse, key, default=None):
    if isinstance(parse, dict):
        return parse.get(key, default)
    try:
        value = parse[key]
    except Exception:
        value = getattr(parse, key, default)
    return default if value is None else value


def _normalize_model_name(name: str) -> str:
    model_key = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "resnet18": "resnet18",
        "densenet121": "densenet121",
        "densenet_121": "densenet121",
        "mobilenet_v3_small": "densenet121",
        "mobilenetv3small": "densenet121",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnetb0": "efficientnet_b0",
        "bilstm": "bilstm",
        "bi_lstm": "bilstm",
        "bigru": "bilstm",
        "bi_gru": "bilstm",
        "bert_tiny": "bert_tiny",
        "berttiny": "bert_tiny",
        "bert_tiny_uncased": "bert_tiny",
    }
    if model_key not in aliases:
        raise ValueError(
            f"Unknown model '{name}'. Supported: resnet18, densenet121, efficientnet_b0, bigru, bert_tiny"
        )
    return aliases[model_key]


def _normalize_dataset_name(name: str) -> str:
    dataset_key = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "cifar10": "cifar10",
        "cifar_10": "cifar10",
        "cifar100": "cifar100",
        "cifar_100": "cifar100",
        "ag_news": "ag_news",
        "agnews": "ag_news",
    }
    if dataset_key not in aliases:
        raise ValueError("Supported datasets are: cifar10, cifar100, ag_news")
    return aliases[dataset_key]


def _infer_num_classes(parse) -> int:
    dataset = _normalize_dataset_name(_cfg_get(parse, "dataset", ""))
    if dataset == "cifar100":
        return 100
    if dataset == "ag_news":
        return 4
    return 10


def _require_bert_tiny_modules():
    try:
        from transformers import BertConfig, BertModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BERT-tiny models require transformers, but it is not installed in the active environment. "
            "Install transformers to run the bert_tiny model."
        ) from exc
    return BertConfig, BertModel


def _build_bert_tiny_backbone() -> nn.Module:
    BertConfig, BertModel = _require_bert_tiny_modules()
    try:
        return BertModel.from_pretrained("prajjwal1/bert-tiny")
    except Exception:
        return BertModel(
            BertConfig(
                vocab_size=30522,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=512,
            )
        )


def _build_bert_extended_attention_mask(attention_mask: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
    extended_attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_dtype)
    return (1.0 - extended_attention_mask) * torch.finfo(hidden_dtype).min


def _run_bert_layer_range(
    encoder_layers: nn.ModuleList,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    start_layer: int,
    end_layer: int,
) -> torch.Tensor:
    if start_layer >= end_layer:
        return hidden_states

    extended_attention_mask = _build_bert_extended_attention_mask(attention_mask, hidden_states.dtype)
    padding_mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

    for layer_idx in range(start_layer, end_layer):
        hidden_states = encoder_layers[layer_idx](
            hidden_states,
            attention_mask=extended_attention_mask,
        )[0]
        hidden_states = hidden_states * padding_mask

    return hidden_states


def _build_resnet18_backbone(num_classes: int) -> nn.Module:
    backbone = resnet18(weights=None, num_classes=num_classes)
    backbone.maxpool = nn.Identity()
    return backbone


def _build_densenet121_backbone(num_classes: int) -> nn.Module:
    return densenet121(weights=None, num_classes=num_classes)


def _build_efficientnet_b0_backbone(num_classes: int) -> nn.Module:
    return efficientnet_b0(weights=None, num_classes=num_classes)


class DenseNet121SuffixClassifier(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.features = nn.Sequential(
            backbone.features.denseblock2,
            backbone.features.transition2,
            backbone.features.denseblock3,
            backbone.features.transition3,
            backbone.features.denseblock4,
            backbone.features.norm5,
        )
        self.classifier = backbone.classifier

    def forward(self, hidden_states):
        hidden_states = self.features(hidden_states)
        hidden_states = F.relu(hidden_states, inplace=True)
        hidden_states = F.adaptive_avg_pool2d(hidden_states, (1, 1))
        hidden_states = torch.flatten(hidden_states, 1)
        return self.classifier(hidden_states)


def _split_feature_backbone(backbone: nn.Module) -> tuple[nn.Module, nn.Module]:
    split_index = len(backbone.features) // 2
    return (
        nn.Sequential(*backbone.features[:split_index]),
        nn.Sequential(*backbone.features[split_index:], backbone.avgpool, nn.Flatten(1), backbone.classifier),
    )


class BiGRUEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int, embed_dim: int = 256, chunk_size: int = 4):
        super().__init__()
        self.pad_token_id = int(pad_token_id)
        self.chunk_size = max(1, int(chunk_size))
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token_id)

    def forward(self, input_ids):
        input_ids = input_ids.long()
        embedded = self.embedding(input_ids)
        if self.chunk_size <= 1:
            return embedded

        batch_size, sequence_length, embed_dim = embedded.shape
        remainder = sequence_length % self.chunk_size
        if remainder:
            pad_steps = self.chunk_size - remainder
            embedded = F.pad(embedded, (0, 0, 0, pad_steps))

        mask = embedded.abs().sum(dim=-1, keepdim=True).ne(0).to(dtype=embedded.dtype)
        reduced_steps = embedded.size(1) // self.chunk_size
        embedded = embedded.view(batch_size, reduced_steps, self.chunk_size, embed_dim)
        mask = mask.view(batch_size, reduced_steps, self.chunk_size, 1)
        valid_chunks = mask.sum(dim=2)
        token_counts = valid_chunks.clamp(min=1.0)
        pooled = (embedded * mask).sum(dim=2) / token_counts
        return pooled * valid_chunks.ne(0).to(dtype=pooled.dtype)


class BiGRUClassifierHead(nn.Module):
    def __init__(self, hidden_size: int = 256, num_classes: int = 4, embed_dim: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.pre_classifier = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, embedded):
        mask = embedded.abs().sum(dim=-1).ne(0)
        sequence_output, _ = self.encoder(embedded)
        lengths = mask.sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(sequence_output.size(0), device=sequence_output.device)
        forward_last = sequence_output[batch_indices, lengths - 1, : self.encoder.hidden_size]
        backward_first = sequence_output[:, 0, self.encoder.hidden_size :]
        pooled = torch.cat([forward_last, backward_first], dim=-1)
        pooled = self.pre_classifier(pooled)
        pooled = F.relu(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class BertTinyPrefixEncoder(nn.Module):
    def __init__(self, pad_token_id: int, split_layer_count: int = 1):
        super().__init__()
        backbone = _build_bert_tiny_backbone()
        self.pad_token_id = int(pad_token_id)
        self.embeddings = backbone.embeddings
        self.encoder_layers = backbone.encoder.layer
        self.split_layer_count = int(split_layer_count)

    def forward(self, input_ids):
        input_ids = input_ids.long()
        attention_mask = input_ids.ne(self.pad_token_id).long()
        hidden_states = self.embeddings(input_ids)
        return _run_bert_layer_range(
            self.encoder_layers,
            hidden_states,
            attention_mask,
            start_layer=0,
            end_layer=self.split_layer_count,
        )


class BertTinySuffixClassifier(nn.Module):
    def __init__(self, num_classes: int, split_layer_count: int = 1):
        super().__init__()
        backbone = _build_bert_tiny_backbone()
        self.encoder_layers = backbone.encoder.layer
        self.pooler = backbone.pooler
        self.dropout = nn.Dropout(float(backbone.config.hidden_dropout_prob))
        self.classifier = nn.Linear(int(backbone.config.hidden_size), num_classes)
        self.split_layer_count = int(split_layer_count)
        self.total_layer_count = int(backbone.config.num_hidden_layers)

    def forward(self, hidden_states):
        attention_mask = hidden_states.abs().sum(dim=-1).ne(0).long()
        hidden_states = _run_bert_layer_range(
            self.encoder_layers,
            hidden_states,
            attention_mask,
            start_layer=self.split_layer_count,
            end_layer=self.total_layer_count,
        )
        pooled = self.pooler(hidden_states)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class SplitModelCombiner(nn.Module):
    def __init__(self, client_model: nn.Module, server_model: nn.Module):
        super().__init__()
        self.client_model = client_model
        self.server_model = server_model

    def forward(self, inputs):
        return self.server_model(self.client_model(inputs))


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    dataset_name: str


class model_factory:
    def __init__(self, parse):
        self.parse = parse
        self.model_name = _normalize_model_name(_cfg_get(parse, "model", ""))
        self.dataset_name = _normalize_dataset_name(_cfg_get(parse, "dataset", ""))
        self.variant_name = str(_cfg_get(parse, "variants_type", "vanilla") or "vanilla").strip().lower()
        self.num_classes = _infer_num_classes(parse)
        self._validate_requested_pair()

    def _validate_requested_pair(self):
        valid_pairs = {
            ModelSpec("resnet18", "cifar10"),
            ModelSpec("densenet121", "cifar10"),
            ModelSpec("efficientnet_b0", "cifar100"),
            ModelSpec("bilstm", "ag_news"),
            ModelSpec("bert_tiny", "ag_news"),
        }
        if ModelSpec(self.model_name, self.dataset_name) not in valid_pairs:
            raise ValueError(
                "Supported model/dataset pairs are only: "
                "(resnet18, cifar10), (densenet121, cifar10), "
                "(efficientnet_b0, cifar100), (bilstm, ag_news), (bert_tiny, ag_news)."
            )

    def _text_model_args(self):
        vocab_size = int(_cfg_get(self.parse, "text_vocab_size", 30522) or 30522)
        pad_token_id = int(_cfg_get(self.parse, "text_pad_token_id", 0) or 0)
        return vocab_size, pad_token_id

    def _build_vision_complete_model(self):
        if self.model_name == "resnet18":
            return _build_resnet18_backbone(num_classes=self.num_classes)
        if self.model_name == "densenet121":
            return _build_densenet121_backbone(num_classes=self.num_classes)
        if self.model_name == "efficientnet_b0":
            return _build_efficientnet_b0_backbone(num_classes=self.num_classes)
        raise ValueError(f"Unsupported vision model '{self.model_name}'")

    def _build_split_models(self):
        if self.model_name == "resnet18":
            backbone = _build_resnet18_backbone(num_classes=self.num_classes)
            return (
                nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.layer1),
                nn.Sequential(backbone.layer2, backbone.layer3, backbone.layer4, backbone.avgpool, nn.Flatten(1), backbone.fc),
            )

        if self.model_name == "densenet121":
            backbone = _build_densenet121_backbone(num_classes=self.num_classes)
            return (
                nn.Sequential(
                    backbone.features.conv0,
                    backbone.features.norm0,
                    backbone.features.relu0,
                    backbone.features.pool0,
                    backbone.features.denseblock1,
                    backbone.features.transition1,
                ),
                DenseNet121SuffixClassifier(backbone),
            )

        if self.model_name == "efficientnet_b0":
            backbone = _build_efficientnet_b0_backbone(num_classes=self.num_classes)
            return _split_feature_backbone(backbone)

        if self.model_name == "bilstm":
            vocab_size, pad_token_id = self._text_model_args()
            return (
                BiGRUEncoder(vocab_size=vocab_size, pad_token_id=pad_token_id),
                BiGRUClassifierHead(num_classes=self.num_classes),
            )

        if self.model_name == "bert_tiny":
            _, pad_token_id = self._text_model_args()
            return (
                BertTinyPrefixEncoder(pad_token_id=pad_token_id),
                BertTinySuffixClassifier(num_classes=self.num_classes),
            )

        raise ValueError(f"Unsupported model '{self.model_name}'")

    def model_complete(self, _parse=None):
        if self.model_name in {"resnet18", "densenet121", "efficientnet_b0"}:
            return self._build_vision_complete_model()
        if self.model_name == "bilstm":
            return SplitModelCombiner(*self._build_split_models())
        if self.model_name == "bert_tiny":
            return SplitModelCombiner(*self._build_split_models())
        raise ValueError(f"Unsupported model '{self.model_name}'")

    def create(self):
        if self.variant_name in {"fedavg", "fedprox"}:
            return self.model_complete(self.parse), nn.Identity()

        if self.variant_name == "central":
            return nn.Identity(), self.model_complete(self.parse)

        return self._build_split_models()
