# Install script (Ubuntu 22.04)
# Authors: Fluid Numerics LLC
#          Garrett Byrd             (garrett@fluidnumerics.com)
#          Dr. Joseph Schoonover    (joe@fluidnumerics.com)

# install ROCm
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME # Adding current user to Video, Render groups. See prerequisites.
wget https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
sudo apt install ./amdgpu-install_6.1.60101-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms
sudo apt install rocm
echo "Please reboot system for all settings to take effect."


# install conda (miniconda) 
# https://docs.anaconda.com/free/miniconda/ 
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh   
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh


# set up conda environment                                                                                
conda create -n llama-2-env
conda activate llama-2-env
conda install pip
pip install jupyterlab


# install pytorch
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0


# install transformers
pip install transformers accelerate
