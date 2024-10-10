# Authors: Fluid Numerics LLC
#          Garrett Byrd             (garrett@fluidnumerics.com)
#          Dr. Joseph Schoonover    (joe@fluidnumerics.com)

# User must have git-lfs installed
# https://git-lfs.com/

# Get Llama-3-8B (requires permission/authentication)
# https://ai.meta.com/blog/meta-llama-3/

# Hugging Face username
HF_USERNAME=
# Hugging Face authorization token
AUTH_TOKEN=

git clone https://${HF_USERNAME}:${AUTH_TOKEN}@huggingface.co/meta-llama/Meta-Llama-3-8B

# Get MetaMathQA dataset
# https://meta-math.github.io/
git clone https://huggingface.co/datasets/meta-math/MetaMathQA

# You can also download these via the Python API
# https://huggingface.co/docs/hub/en/models-downloading