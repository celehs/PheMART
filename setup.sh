#!/bin/bash
# abort entire script on error
set -e

module load gcc/6.2.0
module load conda2/4.2.13
conda create -n tf2 python=3.7.4
source activate tf2
pip install tensorflow==2.5.2
pip install pandas
pip install numpy
pip install click
pip install scipy
pip install scikit-learn
pip install configparser
