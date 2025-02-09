#!/bin/bash

# Initialisation de la variable MASTER_PORT
export MASTER_PORT=29501
export MASTER_ADDR="insa-11293"

export WORLD_SIZE=2
export RANK=0
export NODE_RANK=0

# Exécuter un script Python
python3 ./training.py
