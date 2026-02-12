#!/bin/bash
# Install conda environment
conda env create -f environment.yml
conda activate evogym_env

# Build EvoGym
cd evogym
python setup.py install
cd ..