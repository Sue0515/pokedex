#!/bin/bash

# Exit on error
set -e

# Create a virtual environment called "pokedex"
python -m venv pokedex

# Activate the environment
source pokedex/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
