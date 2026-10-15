#!/usr/bin/env bash
# Runs the full pipeline. Usage: bash run_all.sh   (from the repository root)
set -e
export MPLBACKEND=Agg
cd "$(dirname "$0")/Code"
python plot_data.py
python KNN_optimal.py
python predict_x2_x4_x5.py
python predict_x4_x5.py
echo "Done. Plots and accuracy tables are in ../Output"
