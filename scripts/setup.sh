#!/bin/bash

# Create Python virtual environment
python3 -m venv multi-lingual

# Activate the virtual environment
source multi-lingual/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
pip install -r requirements.txt

# Change directory to 'peft'
cd peft

# Update and install git-lfs
sudo apt update
sudo apt install -y git-lfs

# Initialize git-lfs and clone the dataset
git lfs install
git clone https://huggingface.co/datasets/Themira/xnli_layalign

# chmod +x setup.sh
