#!/bin/bash

rm -rf .venv

python3 -m venv .venv

pip install -r requirements.txt

# maybe download data here

