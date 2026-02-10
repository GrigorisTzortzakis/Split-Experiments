"""Datasets: load MNIST/CIFAR-10, transforms, and truncated variants."""

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST

from runtime.log import Log

from .Partition_Data import get_partition_callable


class TruncatedTorchvisionDataset(data.Dataset):
    def __init__(
        self,
        parse,
        dataset_name: str,
        dataidxs=None,
        train: bool = True,
        transform=None,
    ):
        self.parse = parse
        self.dataset_name = dataset_name
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = None

        self.log = Log(self.__class__.__name__, parse)
        self.download = parse["download"] if parse["download"] is not None else False

        if dataset_name in {"mnist", "cifar10"}:
            # Root is the downloads folder; torchvision will create MNIST/ or CIFAR10/ inside it.
            self.root = str((self.parse["dataDir"]).rstrip("/"))
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        self.data, self.target = self._build()

    def _build(self):
        if self.dataset_name == "mnist":
            ds = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
            data_arr = ds.data
            target_arr = np.array(ds.targets)

            if self.dataidxs is not None:
                data_arr = data_arr[self.dataidxs]
                target_arr = target_arr[self.dataidxs]

            return data_arr, target_arr

        ds = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        data_arr = ds.data
        target_arr = np.array(ds.targets)

        if self.dataidxs is not None:
            data_arr = data_arr[self.dataidxs]
            target_arr = target_arr[self.dataidxs]

        # Preserve previous behavior: CIFAR targets stored as torch.LongTensor.
        target_arr = torch.Tensor(target_arr).long()
        return data_arr, target_arr

    def __getitem__(self, index):
        if self.dataset_name == "mnist":
            img, target = self.data[index], int(self.target[index])
            img = Image.fromarray(img.numpy(), mode="L")
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


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


def _mnist_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform, transform


def _cifar10_transforms():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


class TorchvisionDatasetController:
    def __init__(self, parse, dataset_name: str, transform=None):
        self.parse = parse
        self.dataset_name = dataset_name
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.log = Log(self.__class__.__name__, parse)

        self.root = parse["dataDir"]
        self.target = None
        self.batch_size = parse["batch_size"]

        self.train = parse["train"] if parse["train"] is not None else True
        self.download = parse["download"] if parse["download"] is not None else False

    def _transforms(self):
        if self.dataset_name == "mnist":
            return _mnist_transforms()
        if self.dataset_name == "cifar10":
            return _cifar10_transforms()
        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def loadData(self):
        self.log.info(self.parse.dataDir)
        train_transform, test_transform = self._transforms()

        train_ds = TruncatedTorchvisionDataset(parse=self.parse, dataset_name=self.dataset_name, transform=train_transform)
        test_ds = TruncatedTorchvisionDataset(
            parse=self.parse,
            dataset_name=self.dataset_name,
            transform=test_transform,
            train=False,
        )

        X_train, y_train = train_ds.data, train_ds.target
        X_test, y_test = test_ds.data, test_ds.target

        return X_train, y_train, X_test, y_test

    def partition_data(self):
        partition_method = get_partition_callable(self.parse)
        self.log.info(partition_method)
        return partition_method(self.loadData)

    def get_dataloader(self, dataidxs=None):
        dl_obj = TruncatedTorchvisionDataset
        transform_train, transform_test = self._transforms()

        train_ds = dl_obj(parse=self.parse, dataset_name=self.dataset_name, transform=transform_train, dataidxs=dataidxs)
        test_ds = dl_obj(parse=self.parse, dataset_name=self.dataset_name, transform=transform_test, train=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        return train_dl, test_dl

    def load_partition_data(self, process_id):
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        class_num = len(np.unique(y_train))

        partition_client_number = self.parse["partition_client_number"]
        if partition_client_number is None:
            partition_client_number = self.parse["client_number"]
        partition_client_number = int(partition_client_number)

        active_client_number = int(self.parse["client_number"])

        if self.dataset_name == "mnist":
            # If we created more partitions than active MPI clients, only the first N partitions
            # are actually used (clients are ranks 1..N and pick partition (rank-1)).
            if partition_client_number != active_client_number:
                train_data_num = sum(len(net_dataidx_map[r]) for r in range(active_client_number))
            else:
                train_data_num = sum(len(net_dataidx_map[r]) for r in net_dataidx_map.keys())
        else:
            train_data_num = sum([len(net_dataidx_map[r]) for r in range(partition_client_number)])

        if process_id == 0:
            train_data_global, test_data_global = self.get_dataloader()
            self.log.info("train_dl_global number = " + str(len(train_data_global)))
            self.log.info("test_dl_global number = " + str(len(test_data_global)))
            train_data_local = None
            test_data_local = None
            local_data_num = 0
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
