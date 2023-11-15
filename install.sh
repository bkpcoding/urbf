#!/bin/bash

echo "Installing custom Stable-Baselines3 fork ..."
# unzip utils/stable_baselines3.tar.gz -d utils/stable_baselines3
tar -xvf utils/stable_baselines3.tar.gz -C utils

pip install -e ./utils/stable_baselines3
echo "Installing the regression and rl_maze packages in developer mode .."
pip install -e ./src/regression
pip install -e ./src/rl_maze

echo "Install and setup jupyter notebook extensions ..."
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable --py --sys-prefix qgrid

echo "Finished"
