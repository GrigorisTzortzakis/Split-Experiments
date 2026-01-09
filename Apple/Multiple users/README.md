# Split Experiments – MNIST Split Learning (Multi-Client)

This mini-project runs a true split-learning pipeline where clients train the first part of a LeNet-style network on MNIST and stream smashed activations to a server. The server completes the forward/backward pass, returns gradients, and enforces paper-style turn-based training with baton passing of the client-front weights across clients.

Model code lives in `split_models.py` and is imported by both `client.py` and `server.py`.

## Requirements

Create or activate a Python environment that already has PyTorch with CUDA/MPS support if you plan to use a GPU. Install the remaining dependencies:

```bash
pip install torchvision websockets psutil matplotlib
```

Optional (MLflow tracking/UI):

```bash
pip install mlflow
```

Torch installs vary per platform; follow https://pytorch.org/get-started/locally/ for the command that matches your OS and accelerator.

## Enforced resource limits

Both scripts accept `--cpu-seconds` and `--memory-mb` arguments. When running on macOS/Linux (POSIX systems), they set `RLIMIT_CPU` and `RLIMIT_AS`, which are hard limits when the kernel allows it. The client defaults to CPU execution and you can give it a small budget (for example, `--cpu-seconds 600 --memory-mb 1500`). The server automatically picks CUDA → MPS → CPU and can be given a larger budget (for example, `--cpu-seconds 3600 --memory-mb 8192`).

If the OS does not expose the `resource` module (e.g., Windows) **or** the kernel refuses to shrink `RLIMIT_AS` (common on recent macOS builds), the scripts warn and fall back to sampled enforcement: RSS is tracked every log interval and the process aborts once it crosses the configured limit.

## Running the demo

1. **Start the server**:

   ```bash
   cd "Split Experiments/Multiple users"
   python server.py \
     --host 0.0.0.0 \
     --port 8765 \
     --cut-layer 1 \
     --learning-rate 0.01 \
     --device auto \
     --cpu-seconds 3600 \
     --memory-mb 8192
   ```

   The server logs its device choice, honors the hard limits, and prints periodic utilization statistics.

2. **Start clients** (repeat for each client id):

   ```bash
   cd "Split Experiments/Multiple users"
   python client.py \
     --host <server-ip> \
     --port 8765 \
     --device cpu \
     --batch-size 128 \
     --num-epochs 5 \
     --num-clients 2 \
     --client-id 0 \
     --turn-batches 1 \
     --cpu-seconds 600 \
     --memory-mb 1500 \
     --output-dir .
   ```

  Replace `<server-ip>` with the machine where the server is running. By default the client does not download MNIST; pass `--download` if you want it to fetch MNIST when missing.

## MLflow

To track runs in MLflow, enable it on the server:

```bash
python server.py --mlflow --mlflow-uri file:./mlruns --mlflow-experiment split-learning
```

Then open the UI (from the same folder):

```bash
mlflow ui --backend-store-uri file:./mlruns
```

MLflow logging happens when the server writes the combined report (end of training). This creates a local `./mlruns/` folder.

## Outputs

- Real-time logs describe training progress for both roles.
- After training, the **server** writes a single combined report and aggregate artifacts in the working directory:
  - `metrics.json` – combined multi-client report (server resources + per-client metrics + aggregate metrics)
  - `confusion_matrix.pt` – aggregated confusion matrix tensor
  - `confusion_matrix_counts.png` – aggregated confusion matrix heatmap

These outputs match the requested metrics suite and confusion-matrix visuals.

## Notes

- The server supports `--device cpu|cuda|mps|auto`. On Apple Silicon, use `--device mps`.
- For paper-style equivalence, set `--momentum 0.0` on both server and clients.
- `--max-batches` can throttle epochs for quick smoke tests.
- Increase `--num-workers` on the client if the host has enough CPU cores and RAM headroom.
- The websocket protocol is JSON + base64 tensors, so it can be proxied or secured as needed.
- Supply `--resume-client-checkpoint` and/or `--resume-checkpoint` on the server to warm-start from previously saved weights; by default checkpoints are written into the repo root so you can rerun with minimal configuration.
