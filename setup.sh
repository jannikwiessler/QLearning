#!/bin/bash
# ------------------------------------------------------------------
# [Jannik Wiessler] installation script for venv and requirements
# this is only for linux !
# ------------------------------------------------------------------
FILE=./Scripts # check if repo is already venv
if ! test -f "$FILE"; then
    apt-get install -y python3-venv
    python3 -m venv .
fi
./bin/pip3 install -r requirements.txt
