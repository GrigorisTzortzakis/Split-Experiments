"""Manager for the `central` algorithm variant."""

from __future__ import annotations

import time

import torch

from runtime.exports.log import Log


class ServerManager:
    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer
        self.log = Log(self.__class__.__name__, args)
        self.comm_log = Log("CommBreakdown", args)

    def run(self):
        for epoch in range(self.trainer.MAX_EPOCH_PER_NODE):
            self.trainer.epoch_count = epoch
            epoch_start = time.perf_counter()

            self.trainer.train_mode()
            for batch in self.trainer.dataloader:
                self.trainer.run_batch(batch)
                self.trainer.write_log()

            train_acc = self.trainer.correct / max(1, self.trainer.total)
            train_loss = self.trainer.val_loss

            self.trainer.eval_mode()
            with torch.no_grad():
                for batch in self.trainer.dataloader:
                    self.trainer.run_batch(batch)
            self.trainer.write_log()

            val_acc = self.trainer.correct / max(1, self.trainer.total)
            val_loss = self.trainer.val_loss
            epoch_time = time.perf_counter() - epoch_start

            self.comm_log.info(
                "epoch_summary rank={} node_type=server epoch={} train_acc={} train_loss={} val_acc={} val_loss={} "
                "raw_acts_bytes=0 quantized_acts_bytes=0 acts_metadata_bytes=0 raw_grads_bytes=0 quantized_grads_bytes=0 "
                "grads_metadata_bytes=0 acts_compression=1 grads_compression=1 total_compression=1 acts_quant_time=0 "
                "grads_quant_time=0 send_time=0 recv_time=0 epoch_time={}".format(
                    self.trainer.rank,
                    epoch,
                    train_acc,
                    train_loss,
                    val_acc,
                    val_loss,
                    epoch_time,
                )
            )

