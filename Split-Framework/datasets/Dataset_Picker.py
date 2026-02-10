"""Datasets: factory for selecting/building datasets and loaders."""

from runtime.log import Log

from .Dataset_Loader import TorchvisionDatasetController


class DatasetFactory:
    def __init__(self, parse):
        self.parse = parse
        self.log = Log(self.__class__.__name__, parse)

    def _normalize_dataset_name(self, name: str) -> str:
        return str(name).lower().strip()

    def _get_controller(self, dataset_name: str):
        dataset_name = self._normalize_dataset_name(dataset_name)
        if dataset_name not in ("mnist", "cifar10"):
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Split-Framework keeps only: mnist, cifar10"
            )
        return TorchvisionDatasetController(self.parse, dataset_name=dataset_name)

    def factory(self):
        self.log.info(self.parse["dataset"])

        if isinstance(self.parse["dataset"], list):
            dataset_cur, cur_client_num = self.judge_client_dataset()
            item = self.parse["dataset"][dataset_cur]
            return self._get_controller(item)

        dataset_name = getattr(self.parse, "dataset", self.parse["dataset"])
        return self._get_controller(dataset_name)

    def judge_client_dataset(self):
        client_num = self.parse["client_split"][0]
        for i in range(len(self.parse["client_split"])):
            if self.parse["rank"] <= self.parse["client_split"][i]:
                return i, client_num
            client_num = self.parse["client_split"][i + 1] - self.parse["client_split"][i]


# Backwards-compatible alias (kept so experiments/main.py can stay stable if needed).
class datasetFactory(DatasetFactory):
    pass
