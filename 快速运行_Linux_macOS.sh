#!/usr/bin/env bash
set -e
python3 -m pip install -r requirements.txt
python3 boredom_sim.py --quick --output-dir boredom_outputs
