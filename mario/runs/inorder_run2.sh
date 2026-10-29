#!/bin/bash
cd /p/projects/cadence-arcade/mario
mkdir -p runs/inorder
export MARIO_CAP_MIN=38 MARIO_SENSES=senses
../.venv-mario/Scripts/python.exe levels.py --out runs/inorder --until 19:25 --workers 5 --start-brain fresh > runs/inorder/stdout.txt 2>&1
echo "EXIT $?" >> runs/inorder/stdout.txt
