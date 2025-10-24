# Environment Setup
```
conda env create -f environment.yaml
conda activate lb
```

# Evaluation

- Seed-Conditioned (Random number prefix)
  ```
  python aime.py --no_tqdm -t 0.6 -m Qwen/Qwen3-8B
  ```
- No Conditioning
  ```
  python aime.py --no_tqdm -t 0.6 -m Qwen/Qwen3-8B --disable_seed
  ```

# Seed-Conditioning