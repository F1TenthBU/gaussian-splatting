#!/bin/bash

# Load required modules
module load ninja
module load miniconda
module load cuda/11.8

# Export work directory for conda packages installation
export WRK_DIR=/projectnb/path_you_figured_out

# Activate conda env
conda activate venv
