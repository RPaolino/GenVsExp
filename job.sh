#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate graphgps2

python synthetic.py --dataset er --task sum_basis_C4 --pe basis_C12