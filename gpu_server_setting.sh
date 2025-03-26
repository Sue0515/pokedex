#!/bin/bash

# Exit on error
set -e

# Create a virtual environment called "pokedex"
python -m venv pokedex

# Activate the environment
source pokedex/bin/activate

# Install dependencies
pip install -r requirements.txt

accelerate config default