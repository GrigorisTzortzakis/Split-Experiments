"""Datasets: load CIFAR-10, CIFAR-100, and AG_NEWS for imported split backbones."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100

from runtime.exports.log import Log

from .Partition_Data import get_partition_callable


def _cfg_get(parse, key, default=None):
    if isinstance(parse, dict):
        return parse.get(key, default)
    try:
        value = parse[key]
    except Exception:
        value = getattr(parse, key, default)
    return default if value is None else value


def _normalize_dataset_name(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "cifar10": "cifar10",
        "cifar_10": "cifar10",
        "cifar100": "cifar100",
        "cifar_100": "cifar100",
        "ag_news": "ag_news",
        "agnews": "ag_news",
    }
    if key not in aliases:
        raise ValueError("Supported datasets are: cifar10, cifar100, ag_news")
    return aliases[key]


def _normalize_model_name(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "bilstm": "bilstm",
        "bi_lstm": "bilstm",
        "bigru": "bilstm",
        "bi_gru": "bilstm",
        "bert_tiny": "bert_tiny",
        "berttiny": "bert_tiny",
        "bert_tiny_uncased": "bert_tiny",
        "agnews_bilstm": "bilstm",
        "agnews_bert_tiny": "bert_tiny",
    }
    return aliases.get(key, key)


def _require_bert_tiny_tokenizer():
    try:
        from transformers import BertTokenizerFast
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BERT-tiny tokenization requires transformers, but it is not installed in the active environment. "
            "Install transformers to run bert_tiny on ag_news."
        ) from exc
    return BertTokenizerFast


class SimpleVocab:
    def __init__(self, stoi, default_token: str = "<unk>"):
        self.stoi = stoi
        self.default_index = int(stoi[default_token])

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def set_default_index(self, index: int):
        self.default_index = int(index)


def _basic_english_tokenize(text: str):
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def _build_simple_vocab_from_records(records):
    counter = Counter()
    for _label, text in records:
        counter.update(_basic_english_tokenize(text))

    stoi = {"<pad>": 0, "<unk>": 1}
    for token, _count in counter.most_common():
        if token not in stoi:
            stoi[token] = len(stoi)
    vocab = SimpleVocab(stoi)
    vocab.set_default_index(stoi["<unk>"])
    return vocab


def _read_arrow_records(arrow_path: Path):
    table = ipc.open_stream(pa.memory_map(str(arrow_path), "r")).read_all()
    labels = table.column("label").to_pylist()
    texts = table.column("text").to_pylist()
    label_offset = 1 if labels and min(int(label) for label in labels) == 0 else 0
    return [(int(label) + label_offset, str(text)) for label, text in zip(labels, texts)]


def _resolve_ag_news_cache_dir(root: str) -> Path:
    root_path = Path(str(root)).resolve()

    direct_candidates = [
        root_path / "ag_news",
        root_path / "ag_news" / "default" / "0.0.0",
    ]

    for candidate in direct_candidates:
        if not candidate.exists():
            continue
        if (candidate / "ag_news-train.arrow").exists() and (candidate / "ag_news-test.arrow").exists():
            return candidate
        subdirs = [item for item in candidate.iterdir() if item.is_dir()]
        for subdir in subdirs:
            if (subdir / "ag_news-train.arrow").exists() and (subdir / "ag_news-test.arrow").exists():
                return subdir

    raise FileNotFoundError(
        "Could not find AG News Arrow files. Expected ag_news-train.arrow and ag_news-test.arrow under the framework data directory, typically datasets/downloads/ag_news/default/0.0.0/."
    )


class TruncatedTorchvisionDataset(data.Dataset):
    def __init__(self, parse, dataset_name: str, dataidxs=None, train: bool = True, transform=None):
        self.parse = parse
        self.dataset_name = _normalize_dataset_name(dataset_name)
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = None
        self.log = Log(self.__class__.__name__, parse)
        self.download = _cfg_get(parse, "download", False)
        self.root = str((_cfg_get(self.parse, "dataDir", "")).rstrip("/"))
        self.data, self.target = self._build()

    def _dataset_class(self):
        if self.dataset_name == "cifar10":
            return CIFAR10
        if self.dataset_name == "cifar100":
            return CIFAR100
        raise ValueError(f"Unsupported vision dataset_name: {self.dataset_name}")

    def _build(self):
        dataset_cls = self._dataset_class()
        ds = dataset_cls(self.root, self.train, self.transform, self.target_transform, self.download)
        data_arr = ds.data
        target_arr = np.asarray(ds.targets, dtype=np.int64)

        if self.dataidxs is not None:
            data_arr = data_arr[self.dataidxs]
            target_arr = target_arr[self.dataidxs]

        return data_arr, torch.as_tensor(target_arr, dtype=torch.long)

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class AGNewsTokenizedDataset(data.Dataset):
    def __init__(self, records, encode_text, dataidxs=None):
        self.records = records if dataidxs is None else [records[int(idx)] for idx in dataidxs]
        self.encode_text = encode_text

    def __getitem__(self, index):
        label, text = self.records[index]
        input_ids = self.encode_text(text)
        target = torch.tensor(int(label) - 1, dtype=torch.long)
        return input_ids, target

    def __len__(self):
        return len(self.records)


def _pad_text_batch(batch, pad_token_id: int):
    inputs, labels = zip(*batch)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=int(pad_token_id))
    label_tensor = torch.stack(labels)
    return padded_inputs, label_tensor


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _cifar_transforms(dataset_name: str):
    stats = {
        "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    }
    mean, std = stats[dataset_name]
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(16),
    ])
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, valid_transform


class TorchvisionDatasetController:
    def __init__(self, parse, dataset_name: str, transform=None):
        self.parse = parse
        self.dataset_name = _normalize_dataset_name(dataset_name)
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.log = Log(self.__class__.__name__, parse)

        self.root = parse["dataDir"]
        self.target = None
        self.batch_size = parse["batch_size"]

        self.train = parse["train"] if parse["train"] is not None else True
        self.download = parse["download"] if parse["download"] is not None else False
        self.model_name = _normalize_model_name(_cfg_get(parse, "model", ""))
        self.max_text_length = int(_cfg_get(parse, "max_text_length", 256) or 256)
        self._tokenizer = None
        self._basic_tokenizer = None
        self._vocab = None
        self._train_records = None
        self._test_records = None

    def _transforms(self):
        if self.dataset_name in {"cifar10", "cifar100"}:
            return _cifar_transforms(self.dataset_name)
        return None, None

    def prepare_model_metadata(self):
        if self.dataset_name != "ag_news":
            return
        if self.model_name == "bert_tiny":
            self._get_tokenizer()
            return
        if self.model_name == "bilstm":
            self._get_bilstm_vocab()
            return
        raise ValueError(f"Unsupported AG News text model '{self.model_name}'")

    def _get_tokenizer(self):
        if self.model_name != "bert_tiny":
            raise ValueError(f"BERT-tiny tokenizer requested for unsupported model '{self.model_name}'")
        if self._tokenizer is None:
            BertTokenizerFast = _require_bert_tiny_tokenizer()
            self._tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
            self.parse["text_vocab_size"] = int(self._tokenizer.vocab_size)
            self.parse["text_pad_token_id"] = int(self._tokenizer.pad_token_id)
        return self._tokenizer

    def _get_bilstm_vocab(self):
        if self.model_name != "bilstm":
            raise ValueError(f"BiLSTM vocabulary requested for unsupported model '{self.model_name}'")
        if self._vocab is None:
            train_records, _test_records = self._load_ag_news_records()
            self._basic_tokenizer = _basic_english_tokenize
            self._vocab = _build_simple_vocab_from_records(train_records)
            self.parse["text_vocab_size"] = int(len(self._vocab))
            self.parse["text_pad_token_id"] = int(self._vocab["<pad>"])
        return self._vocab, self._basic_tokenizer

    def _text_encoder(self):
        if self.model_name == "bert_tiny":
            tokenizer = self._get_tokenizer()

            def _encode_text(text):
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
                return encoded["input_ids"].squeeze(0)

            return _encode_text

        if self.model_name == "bilstm":
            vocab, basic_tokenizer = self._get_bilstm_vocab()
            pad_token_id = int(self.parse["text_pad_token_id"])
            unk_token_id = int(vocab["<unk>"])

            def _encode_text(text):
                token_ids = [int(vocab[token]) for token in basic_tokenizer(text)]
                if not token_ids:
                    token_ids = [unk_token_id]
                token_ids = token_ids[: self.max_text_length]
                if len(token_ids) < self.max_text_length:
                    token_ids.extend([pad_token_id] * (self.max_text_length - len(token_ids)))
                return torch.tensor(token_ids, dtype=torch.long)

            return _encode_text

        raise ValueError(f"Unsupported AG News text model '{self.model_name}'")

    def _load_ag_news_records(self):
        if self._train_records is None or self._test_records is None:
            cache_dir = _resolve_ag_news_cache_dir(self.root)
            self._train_records = _read_arrow_records(cache_dir / "ag_news-train.arrow")
            self._test_records = _read_arrow_records(cache_dir / "ag_news-test.arrow")
        return self._train_records, self._test_records

    def loadData(self):
        self.log.info(self.parse.dataDir)
        if self.dataset_name in {"cifar10", "cifar100"}:
            train_transform, test_transform = self._transforms()
            train_ds = TruncatedTorchvisionDataset(parse=self.parse, dataset_name=self.dataset_name, transform=train_transform)
            test_ds = TruncatedTorchvisionDataset(
                parse=self.parse,
                dataset_name=self.dataset_name,
                transform=test_transform,
                train=False,
            )
            return train_ds.data, train_ds.target, test_ds.data, test_ds.target

        self._text_encoder()
        train_records, test_records = self._load_ag_news_records()
        y_train = np.asarray([int(label) - 1 for label, _text in train_records], dtype=np.int64)
        y_test = np.asarray([int(label) - 1 for label, _text in test_records], dtype=np.int64)
        X_train = np.arange(len(train_records), dtype=np.int64)
        X_test = np.arange(len(test_records), dtype=np.int64)
        return X_train, y_train, X_test, y_test

    def partition_data(self):
        partition_method = get_partition_callable(self.parse)
        self.log.info(partition_method)
        return partition_method(self.loadData)

    def get_dataloader(self, dataidxs=None):
        collate_fn = None
        if self.dataset_name in {"cifar10", "cifar100"}:
            transform_train, transform_test = self._transforms()
            train_ds = TruncatedTorchvisionDataset(
                parse=self.parse,
                dataset_name=self.dataset_name,
                transform=transform_train,
                dataidxs=dataidxs,
            )
            test_ds = TruncatedTorchvisionDataset(
                parse=self.parse,
                dataset_name=self.dataset_name,
                transform=transform_test,
                train=False,
            )
        else:
            encode_text = self._text_encoder()
            train_records, test_records = self._load_ag_news_records()
            train_ds = AGNewsTokenizedDataset(
                records=train_records,
                encode_text=encode_text,
                dataidxs=dataidxs,
            )
            test_ds = AGNewsTokenizedDataset(
                records=test_records,
                encode_text=encode_text,
            )
            collate_fn = lambda batch: _pad_text_batch(batch, self.parse["text_pad_token_id"])

        train_dl = data.DataLoader(
            dataset=train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        test_dl = data.DataLoader(
            dataset=test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        return train_dl, test_dl

    def load_partition_data(self, process_id):
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        class_num = len(np.unique(y_train))
        variant_name = str(self.parse["variants_type"] or "").lower()

        active_client_number = int(self.parse["client_number"])

        if variant_name == "central":
            if isinstance(X_train, tuple):
                train_data_num = int(X_train[0].shape[0])
            elif isinstance(X_train, torch.utils.data.dataset.Subset):
                train_data_num = len(X_train)
            else:
                train_data_num = int(X_train.shape[0])
        else:
            selected_partition_count = active_client_number
            train_data_num = sum(len(net_dataidx_map[r]) for r in range(selected_partition_count))

        if process_id == 0:
            if variant_name == "central":
                local_data_num = int(train_data_num)
                train_data_global, test_data_global = self.get_dataloader()
                self.log.info(
                    "central server uses full dataset with local_sample_number = %d"
                    % local_data_num
                )
            else:
                train_data_global, test_data_global = self.get_dataloader()
                local_data_num = 0
            self.log.info("train_dl_global number = " + str(len(train_data_global)))
            self.log.info("test_dl_global number = " + str(len(test_data_global)))
            train_data_local = None
            test_data_local = None
        else:
            dataidxs = net_dataidx_map[process_id - 1]
            local_data_num = len(dataidxs)
            self.log.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
            train_data_local, test_data_local = self.get_dataloader(dataidxs)
            self.log.info(
                "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
                % (process_id, len(train_data_local), len(test_data_local))
            )
            train_data_global = None
            test_data_global = None

        self.parse["trainloader"] = train_data_local
        self.parse["testloader"] = test_data_local
        self.parse["train_data_num"] = train_data_num
        self.parse["train_data_global"] = train_data_global
        self.parse["test_data_global"] = test_data_global
        self.parse["local_data_num"] = local_data_num
        self.parse["class_num"] = class_num

        return (
            train_data_num,
            train_data_global,
            test_data_global,
            local_data_num,
            train_data_local,
            test_data_local,
            class_num,
        )

