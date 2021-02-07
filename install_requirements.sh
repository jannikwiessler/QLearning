#!/bin/bash
# ------------------------------------------------------------------
# [Jannik Wiessler] installation script for requirements
# ------------------------------------------------------------------
FILE=/qLearning
if test -f "$FILE"; then
    echo "$FILE exists."
fi

python -m venv
pip install -r requirements.txt