#!/bin/bash
# chmod +x setup.sh
# Exit immediately if any command fails
set -e

# Create virtual environment
python3 -m venv layalign

# Activate virtual environment
source layalign/bin/activate

# Install PyTorch with specific CUDA version
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

# Change to 'peft' directory
cd peft

# Install editable package with training extras
pip install -e ".[train]"