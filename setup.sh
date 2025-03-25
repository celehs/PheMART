#!/bin/bash
# abort entire script on error
set -e

python3 -m venv pheMART_env 
source pheMART_env/bin/activate 
pip install -r requirements.txt
