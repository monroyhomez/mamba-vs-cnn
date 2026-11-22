# Environment

Two conda environments were used on an NVIDIA RTX 5000 Ada (CUDA 12, sm_89):

- `environment_tesis_v3.yml` / `requirements_tesis_v3.txt` — CNNs, VMamba-T, EfficientVMamba-S.
- `environment_vim_s.yml` / `requirements_vim_s.txt` — Vim-S, which needs the older
  `mamba_ssm 1.1.1` and `causal_conv1d` builds shipped inside the hustvl/Vim repository
  (replace `/PATH/TO/` with your local clone).

The selective-scan CUDA kernels of each Mamba repository must be compiled for your GPU
architecture, following the instructions of the official repositories.
