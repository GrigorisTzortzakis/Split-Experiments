# Split Experiments – MNIST Split Learning

This mini-project runs a true split-learning pipeline where a resource-constrained client trains the first part of LeNet-5 on MNIST and streams smashed activations to a resource-rich server. The server completes the forward/backward pass, enforces stricter limits, and returns gradients plus training stats. The client shows live progress and, after training, reports precision, recall, F1, accuracy, and full confusion matrices saved to disk.

## Requirements

Create or activate a Python environment that already has PyTorch with CUDA/MPS support if you plan to use a GPU. Install the remaining dependencies:

```bash
pip install torchvision websockets psutil matplotlib
```

Torch installs vary per platform; follow https://pytorch.org/get-started/locally/ for the command that matches your OS and accelerator.

## Enforced resource limits

Both scripts accept `--cpu-seconds` and `--memory-mb` arguments. When running on macOS/Linux (POSIX systems), they set `RLIMIT_CPU` and `RLIMIT_AS`, which are hard limits when the kernel allows it. The client defaults to CPU execution and you can give it a small budget (for example, `--cpu-seconds 600 --memory-mb 1500`). The server automatically picks CUDA → MPS → CPU and can be given a larger budget (for example, `--cpu-seconds 3600 --memory-mb 8192`).

If the OS does not expose the `resource` module (e.g., Windows) **or** the kernel refuses to shrink `RLIMIT_AS` (common on recent macOS builds), the scripts warn and fall back to sampled enforcement: RSS is tracked every log interval and the process aborts once it crosses the configured limit.

## Running the demo

1. **Start the server** (ideally on a GPU box):

   ```bash
   cd "Split Experiments"
   python server.py \
     --host 0.0.0.0 \
     --port 8765 \
     --cut-layer 1 \
     --learning-rate 0.01 \
     --cpu-seconds 3600 \
     --memory-mb 8192
   ```

   The server logs its device choice, honors the hard limits, and prints periodic utilization statistics.

2. **Start the client** (can be another shell or machine):

   ```bash
   cd "Split Experiments"
   python client.py \
     --host <server-ip> \
     --port 8765 \
     --device cpu \
     --batch-size 128 \
     --num-epochs 5 \
     --cpu-seconds 600 \
     --memory-mb 1500 \
     --output-dir .
   ```

   Replace `<server-ip>` with the machine where the server is running. The client downloads MNIST (in `./data` by default), shows running loss/accuracy with CPU/RAM usage, and limits itself with the provided budgets.

## Outputs

- Real-time logs describe training progress for both roles.
- After training, the client evaluates on the full MNIST test split using the split pipeline and saves:
  - `metrics.json` – accuracy plus macro/micro precision, recall, F1, per-class stats, and an embedded resource/limits report for client + server (including per-core-normalized CPU percentages).
  - `confusion_matrix.pt` – raw tensor for further analysis.
  - `confusion_matrix_counts.png` – raw counts heatmap with annotations (only the counts variant per latest request).
  - `resource_report.json` – structured limits + usage snapshot for the client as well as the server summary returned during finalization.
  - `client_checkpoint.pt` – client shard weights + optimizer state to skip retraining when resuming.
  - `server_checkpoint.pt` and `server_resources.json` – emitted by the server when the client sends the finalization message.

These outputs match the requested metrics suite and confusion-matrix visuals.

## Notes

- The client always runs its shard on the user-selected device (`cpu` default) to simulate constrained hardware, while the server automatically leverages GPU if available.
- `--max-batches` can throttle epochs for quick smoke tests.
- Increase `--num-workers` on the client if the host has enough CPU cores and RAM headroom.
- The websocket protocol is JSON + base64 tensors, so it can be proxied or secured as needed.
- Supply `--resume-client-checkpoint` and/or `--resume-checkpoint` on the server to warm-start from previously saved weights; by default checkpoints are written into the repo root so you can rerun with minimal configuration.
