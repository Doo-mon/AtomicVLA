# Installation Guide (conda)

This guide walks through setting up the `openpi` conda environment for **AtomicVLA**.

## Prerequisites

| Requirement | Version |
|---|---|
| OS | Linux (x86_64) |
| NVIDIA Driver | >= 570.x |
| GPU | NVIDIA H200 / A100 / H100 (or similar Ampere/Hopper) |
| CUDA Toolkit | 11.8 (PyTorch) and 12.x (JAX) |
| Conda | Miniconda or Anaconda |

## Step 1: Create Conda Environment

```bash
conda create -n openpi python=3.11.9 -y
conda activate openpi
```

## Step 2: Install PyTorch (CUDA 11.8)

PyTorch must be installed with the `cu118` variant from the PyTorch wheel index:

```bash
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Install JAX (CUDA 12)

JAX uses the CUDA 12 plugin, which is compatible alongside PyTorch's CUDA 11.8 wheels:

```bash
pip install jax==0.5.3 jaxlib==0.5.3 jax-cuda12-pjrt==0.5.3 jax-cuda12-plugin==0.5.3
```

## Step 4: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Since PyTorch and JAX were already installed in Steps 2-3, pip will skip them when processing `requirements.txt`. If you see version conflicts, ensure you ran Steps 2-3 first.

## Step 5: Install the Project

Install `openpi` and `openpi-client` in editable mode:

```bash
pip install -e .
pip install -e packages/openpi-client
```

### Optional: Install LeRobot from Source

The environment includes [LeRobot](https://github.com/huggingface/lerobot) pinned to a specific commit:

```bash
pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

### Optional: Install dlimp from Source

```bash
pip install git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a
```

## Step 6: Verify Installation

Run the following to verify that all key frameworks are working correctly:

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)

import jax
print('JAX:', jax.__version__)
print('JAX devices:', jax.devices())

import flax
print('Flax:', flax.__version__)

import transformers
print('Transformers:', transformers.__version__)
"
```

Expected output (on an 8-GPU machine):

```
PyTorch: 2.6.0+cu118
CUDA available: True
CUDA version: 11.8
JAX: 0.5.3
JAX devices: [CudaDevice(id=0), ..., CudaDevice(id=7)]
Flax: 0.10.2
Transformers: 4.48.1
```

## Troubleshooting

### JAX does not detect GPUs

Make sure the CUDA 12 runtime libraries are accessible. You can set:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
```

Or verify with:

```bash
python -c "import jax; print(jax.devices())"
```

If only `CpuDevice` is shown, reinstall the JAX CUDA plugin:

```bash
pip install --force-reinstall jax-cuda12-pjrt==0.5.3 jax-cuda12-plugin==0.5.3
```

### PyTorch CUDA version mismatch

Ensure you installed the `+cu118` variant. Check with:

```bash
python -c "import torch; print(torch.version.cuda)"
```

It should print `11.8`. If not, uninstall and reinstall:

```bash
pip uninstall torch torchaudio torchvision -y
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### XLA memory issues during training

Set the following environment variable to allow JAX to use most of GPU memory:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```
