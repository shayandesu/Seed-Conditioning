# Environment Setup
```
conda env create -f environment.yaml
conda activate lb
```

# Evaluation

## Qwen3-4B

- Seed-Conditioned (Random number prefix)
  ```
  python aime.py --no_tqdm -t 0.6 -m Qwen/Qwen3-4B --tensor_parallel_size N # N = Number of Available 
  ```
- No Conditioning
  ```
  python aime.py --no_tqdm -t 0.6 -m Qwen/Qwen3-4B --disable_seed --tensor_parallel_size N # N = Number of Available GPUs
  ```

## Qwen3-8B

- Seed-Conditioned (Random number prefix)
  ```
  python aime.py --no_tqdm -t 0.6 -m Qwen/Qwen3-8B --tensor_parallel_size N # N = Number of Available 
  ```
- No Conditioning
  ```
  python aime.py --no_tqdm -t 0.6 -m Qwen/Qwen3-8B --disable_seed --tensor_parallel_size N # N = Number of Available GPUs
  ```
