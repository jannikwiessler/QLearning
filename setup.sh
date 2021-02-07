#!/bin/bash
# ------------------------------------------------------------------
# [Jannik Wiessler] installation script for venv and requirements
# this is only for linux !
# ------------------------------------------------------------------
FILE=./Scripts # check if repo is already venv
if ! test -f "$FILE"; then
    python3 -m venv .
fi
./Scripts/pip3 install -r requirements.txt
