#!/bin/bash
# ------------------------------------------------------------------
# [Jannik Wiessler] installation script for venv and requirements
# this is only for linux !
# ------------------------------------------------------------------
FILE=./Scripts # check if repo is already venv
if ! test -f "$FILE"; then
    python -m venv .
fi
./Scripts/pip install -r requirements.txt
