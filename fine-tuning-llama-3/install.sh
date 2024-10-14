# Install script (Ubuntu 22.04)
# Authors: Fluid Numerics LLC
#          Garrett Byrd             (garrett@fluidnumerics.com)
#          Dr. Joseph Schoonover    (joe@fluidnumerics.com)

# Use should already have ROCM 6.1 and conda installed
# and a conda environment created with pip installed

# PyTorch
# https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# BitsandBytes
# main repository: https://github.com/bitsandbytes-foundation/bitsandbytes
# AMD ROCm repository: https://github.com/ROCm/bitsandbytes
git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -S . -DBNB_ROCM_ARCH="gfx1100"
make
pip install .
cd ..

# Hugging Face Libraries
# https://huggingface.co/docs
# https://huggingface.co/docs/transformers/index
# https://huggingface.co/docs/datasets/index
# https://huggingface.co/docs/peft/index
# https://huggingface.co/docs/evaluate/index
# https://huggingface.co/docs/trl/index
pip install transformers datasets peft evaluate trl

# scikit-learn (for loss function)
# https://scikit-learn.org/stable/
pip install scikit-learn

# Install Jupyter
pip install jupyterlab