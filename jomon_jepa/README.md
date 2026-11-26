## INSTALL

```Bash
cd /path/to/jomon_jepa
uv sync
source .venv/bin/activate
```

Faster Triton kernels
```Bash
sudo apt-get install -y build-essential python3.12-dev
```

Open3D requirements for Ubuntu
```Bash
sudo apt-get update && sudo apt-get install -y libusb-1.0-0 libgomp1 libgl1
sudo apt-get install -y ffmpeg
```

## TRAIN

Monitoring
```Bash
uv run tensorboard --logdir ./runs_jomon_jepa --port 6006 --bind_all
```

1. WSL2 Smoke Test (2 epochs on a small subset, Next-Concept model)

  ```Bash
  uv run python3 train.py \
    --run_mode wsl_test \
    --country japan \
    --artifact_type pottery \
    --inputs object \
    --outputs gaze emotion text \
    --text_decoder next_concept
  ```

2. 10 Epoch full test

  ```Bash
  uv run python3 train.py \
    --run_mode dgx_full \
    --epochs 10 \
    --batch_size 4 \
    --devices 1 \
    --country japan \
    --artifact_type pottery \
    --inputs object \
    --outputs gaze emotion text \
    --text_decoder slm
  ```

3. WSL2 Cross-Modal Conditioning Test (['object', 'gaze'] -> ['emotion', 'text'] with SLM)

  ```Bash
  uv run python3 train.py \
    --run_mode wsl_test \
    --country malaysia \
    --artifact_type dogu \
    --inputs object gaze \
    --outputs emotion text \
    --text_decoder slm
  ```

4. DGX Full Training Run (Multi-GPU)

  ```Bash
  uv run python3 train.py \
    --run_mode dgx_full \
    --country japan \
    --artifact_type all \
    --inputs object \
    --outputs gaze emotion text \
    --text_decoder next_concept \
    --devices 4
  ```

5. Evaluation + Temperature Search

  ```Bash
  uv run python3 eval.py \
    --checkpoint ./checkpoints_jomon_jepa/last.ckpt \
    --run_mode dgx_full \
    --eval_country japan \
    --eval_artifact_type pottery
  ```