1. WSL2 Smoke Test (2 epochs on a small subset, Next-Concept model)
```Bash
python train.py \
  --run_mode wsl_test \
  --country japan \
  --artifact_type pottery \
  --inputs object \
  --outputs gaze emotion text \
  --text_decoder next_concept
```

2. WSL2 Cross-Modal Conditioning Test (['object', 'gaze'] -> ['emotion', 'text'] with SLM)
```Bash
python train.py \
  --run_mode wsl_test \
  --country malaysia \
  --artifact_type dogu \
  --inputs object gaze \
  --outputs emotion text \
  --text_decoder slm
```

3. DGX Full Training Run (Multi-GPU)
```Bash
python train.py \
  --run_mode dgx_full \
  --country all \
  --artifact_type all \
  --inputs object \
  --outputs gaze emotion text \
  --text_decoder next_concept \
  --devices 4
```

4. Evaluation + Temperature Search
```Bash
python eval.py \
  --checkpoint ./checkpoints_jomon_jepa/last.ckpt \
  --run_mode dgx_full \
  --country japan \
  --artifact_type pottery \
  --inputs object \
  --outputs gaze emotion text
```