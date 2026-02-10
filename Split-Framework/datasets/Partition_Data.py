"""Datasets: partitioning logic for split/federated experiments."""

import logging

import numpy as np
import torch

from runtime.log import Log


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts


class DataPartitioner:
    def __init__(self, parse):
        self.parse = parse
        self.log = Log(self.__class__.__name__, parse)

    def _partition_client_number(self) -> int:
        partition_n = self.parse["partition_client_number"]
        if partition_n is None:
            partition_n = self.parse["client_number"]
        return int(partition_n)

    def get(self):
        partition_method = self.parse["partition_method"]
        if partition_method == "homo":
            return self.homo
        if partition_method == "hetero":
            return self.hetero
        if partition_method in {"alpha0", "extreme_noniid", "disjoint_labels"}:
            return self.disjoint_labels
        if partition_method == "base_on_class":
            return self.base_on_class
        if partition_method == "base_on_attribute":
            return self.base_on_attribute
        if partition_method == "vertical":
            return self.vertical
        if partition_method == "base_on_class_intersection":
            return self.base_on_class_intersection
        raise NameError(f"Unknown partition_method: {partition_method}")

    def disjoint_labels(self, load_data):
        """Extreme non-IID partition (Dirichlet alpha->0 limit): no label overlap across clients.

        Each class is assigned to exactly one client. For n_clients <= n_classes, labels are split
        into contiguous groups (e.g., for MNIST 10 classes and 3 clients: [0,1,2,3], [4,5,6], [7,8,9]).
        All samples are kept (no downsampling).
        """
        n_nets = self._partition_client_number()
        self.log.info("disjoint_labels (alpha0/extreme_noniid)")
        X_train, y_train, X_test, y_test = load_data()

        unique_labels = np.unique(y_train)
        K = len(unique_labels)
        if n_nets > K:
            raise ValueError(
                f"disjoint_labels requires client_number <= num_classes; got client_number={n_nets}, num_classes={K}"
            )

        # Deterministic label assignment: contiguous groups by label id.
        # Assumes labels are integer-like and comparable (true for MNIST/CIFAR/etc in this repo).
        labels_sorted = np.sort(unique_labels)
        label_groups = np.array_split(labels_sorted, n_nets)
        class_per_client = [list(map(int, grp.tolist())) for grp in label_groups]
        self.log.info("class_per_client: {}".format(class_per_client))

        net_dataidx_map = {}
        for client_id, labels in enumerate(class_per_client):
            idxs = []
            for label in labels:
                idxs.extend(np.where(y_train == label)[0].tolist())
            np.random.shuffle(idxs)
            net_dataidx_map[client_id] = idxs

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def homo(self, load_data):
        if self.parse["variants_type"] == "TaskAgnostic":
            return self.task_agnostic_homo(load_data)

        self.log.info("homo")
        X_train, y_train, X_test, y_test = load_data()

        if isinstance(X_train, tuple):
            n_train = X_train[0].shape[0]
        elif isinstance(X_train, torch.utils.data.dataset.Subset):
            n_train = len(X_train)
        else:
            n_train = X_train.shape[0]

        idxs = np.random.permutation(n_train)
        n_nets = self._partition_client_number()
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        traindata_cls_counts = None
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def hetero(self, load_data):
        self.log.info("hetero")
        X_train, y_train, X_test, y_test = load_data()

        # Raw Dirichlet partition (paper-style):
        # For each class k, sample proportions across clients ~ Dirichlet(alpha,...,alpha),
        # then split the class-k indices accordingly. No rejection sampling, no capacity mask,
        # and no post-hoc equal-size truncation.
        K = len(np.unique(y_train))
        n_nets = self._partition_client_number()
        alpha = float(self.parse["partition_alpha"])
        idx_batch = [[] for _ in range(n_nets)]

        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for client_id, part in enumerate(np.split(idx_k, split_points)):
                idx_batch[client_id].extend(part.tolist())

        net_dataidx_map = {}
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        # Make the log human-readable (cast numpy scalars to plain int).
        cls_counts_clean = {
            int(client): {int(k): int(v) for k, v in cls.items()}
            for client, cls in traindata_cls_counts.items()
        }
        self.log.info("traindata_cls_counts: {}".format(cls_counts_clean))

        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def base_on_class(self, load_data):
        n_nets = self._partition_client_number()
        self.log.info("base_on_class")
        X_train, y_train, X_test, y_test = load_data()
        K = len(np.unique(y_train))

        if n_nets > K:
            mod = n_nets % K
            div = n_nets // K
            client_per_class = []
            client_idx = 0
            for _ in range(K):
                num = div + 1 if mod > 0 else div
                client_per_class.append([i for i in range(client_idx, client_idx + num)])
                client_idx += num
                mod -= 1

            net_dataidx_map = {}
            for k in range(K):
                client_list = client_per_class
                idxs = np.where(y_train == k)[0]
                np.random.shuffle(idxs)
                batch_idxs = np.array_split(idxs, len(client_list))
                net_dataidx_map = {i: batch_idxs[i] for i in range(len(client_list))}

            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
            return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

        mod = K % n_nets
        class_idx = 0
        div = K // n_nets
        class_per_client = []
        net_dataidx_map = {}

        for _ in range(n_nets):
            num = div + 1 if mod > 0 else div
            class_per_client.append([i for i in range(class_idx, class_idx + num)])
            class_idx += num
            mod -= 1

        self.log.info("{}".format(class_per_client))
        for i in range(n_nets):
            batch_idxs = []
            for k in class_per_client[i]:
                idx_k = np.where(y_train == k)[0]
                batch_idxs.extend(list(idx_k))
            net_dataidx_map[i] = batch_idxs

        min_len = min(len(net_dataidx_map[i]) for i in range(n_nets))
        for i in range(n_nets):
            if len(net_dataidx_map[i]) > min_len:
                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = net_dataidx_map[i][0:min_len]

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def base_on_attribute(self, load_data):
        n_nets = self._partition_client_number()
        self.log.info("base_on_class")
        X_train, y_train, X_test, y_test = load_data()
        attribute = int(self.parse["partition_method_attributes"])
        attribute_label = np.unique(X_train[:, attribute])
        K = len(attribute_label)

        self.log.info("attribute_label: {}".format(attribute_label))
        net_dataidx_map = {}

        if n_nets > K:
            mod = n_nets % K
            div = n_nets // K
            client_per_class = []
            client_idx = 0
            for _ in range(K):
                num = div + 1 if mod > 0 else div
                client_per_class.append([i for i in range(client_idx, client_idx + num)])
                client_idx += num
                mod -= 1

            for k in range(0, -K, -1):
                client_list = client_per_class[-k]
                client_first = client_list[0]
                idxs = np.where(X_train[:, attribute] == k)[0]
                np.random.shuffle(idxs)
                batch_idxs = np.array_split(idxs, len(client_list))
                for i in client_list:
                    net_dataidx_map[i] = batch_idxs[i - client_first]

            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
            return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

        mod = K % n_nets
        class_idx = 0
        div = K // n_nets
        class_per_client = []

        for _ in range(n_nets):
            num = div + 1 if mod > 0 else div
            class_per_client.append([attribute_label[i] for i in range(class_idx, class_idx + num)])
            class_idx += num
            mod -= 1

        self.log.info("class_per_client : {}".format(class_per_client))
        for i in range(n_nets):
            batch_idxs = []
            for k in class_per_client[i]:
                idx_k = np.where(X_train[:, attribute] == k)[0]
                batch_idxs.extend(list(idx_k))
            net_dataidx_map[i] = batch_idxs

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

        min_len = min(len(net_dataidx_map[i]) for i in range(n_nets))
        for i in range(n_nets):
            if len(net_dataidx_map[i]) > min_len:
                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = net_dataidx_map[i][0:min_len]

        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def base_on_class_intersection(self, load_data):
        n_nets = self._partition_client_number()
        class_number_per_client = self.parse["class_number_per_client"]
        self.log.info("base_on_class_intersection")
        X_train, y_train, X_test, y_test = load_data()
        K = len(np.unique(y_train))

        if class_number_per_client * n_nets < K:
            return self.base_on_class(load_data)

        mod = K % n_nets
        class_idx = 0
        div = K // n_nets

        class_per_client = []
        net_dataidx_map = {}
        for i in range(n_nets):
            num = div + 1 if mod > 0 else div
            class_per_client.append([i % K for i in range(class_idx, class_idx + class_number_per_client)])
            class_idx += num
            mod -= 1

        self.log.info("{}".format(class_per_client))
        for i in range(n_nets):
            batch_idxs = []
            for k in class_per_client[i]:
                idx_k = np.where(y_train == k)[0]
                batch_idxs.extend(list(idx_k))
            net_dataidx_map[i] = batch_idxs

        min_len = min(len(net_dataidx_map[i]) for i in range(n_nets))
        for i in range(n_nets):
            if len(net_dataidx_map[i]) > min_len:
                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = net_dataidx_map[i][0:min_len]

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def vertical(self, load_data):
        self.log.info("vertical")
        X_train, y_train, X_test, y_test = load_data()
        n_train = X_train.shape[0]
        n_nets = self._partition_client_number()
        net_dataidx_map = {i: [j for j in range(n_train)] for i in range(n_nets)}

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def task_agnostic_homo(self, load_data):
        self.log.info("task_agnostic_homo")
        X_train, y_train, X_test, y_test = load_data()
        n_train = X_train.shape[0]
        idxs = np.random.permutation(n_train)

        dataset_cur, cur_client_num = self.judge_client_dataset()
        logging.info("dataset_cur: {}, cur_client_num: {}".format(dataset_cur, cur_client_num))

        batch_idxs = np.array_split(idxs, cur_client_num)
        if dataset_cur == 0:
            net_dataidx_map = {i: batch_idxs[i] for i in range(cur_client_num)}
        else:
            pre = self.parse["client_split"][dataset_cur - 1]
            net_dataidx_map = {pre + i: batch_idxs[i] for i in range(cur_client_num)}

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def judge_client_dataset(self):
        client_num = self.parse["client_split"][0]
        for i in range(len(self.parse["client_split"])):
            if self.parse["rank"] <= self.parse["client_split"][i]:
                return i, client_num
            client_num = self.parse["client_split"][i + 1] - self.parse["client_split"][i]


def get_partition_callable(parse):
    return DataPartitioner(parse).get()
