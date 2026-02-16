#!/bin/bash

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${OS}"
esac

echo "Detected OS: ${machine}"

if [ "$machine" == "Mac" ]; then
    echo "Installing for macOS..."
    conda create -n evogym_env python=3.8 cmake make "clang_osx-64" "clangxx_osx-64" -y
elif [ "$machine" == "Linux" ]; then
    echo "Installing for Linux..."
    conda create -n evogym_env python=3.8 cmake make gxx_linux-64 gcc_linux-64 -y
else
    echo "Warning: Unsupported OS ${machine}. Attempting generic install..."
    conda create -n evogym_env python=3.8 cmake make -y
fi

eval "$(conda shell.bash hook)"
conda activate evogym_env

# Downgrade build tools for compatibility
pip install "pip<24"
pip install "setuptools<66"

# Install Core Dependencies
pip install numpy matplotlib gym==0.21.0 imageio pyyaml torch torchvision

# Install Example Dependencies (stable-baselines3, etc.)
pip install -r evogym/requirements.txt

# Build EvoGym
cd evogym
pip install .
cd ..

echo "Setup complete. Run 'conda activate evogym_env' to start."