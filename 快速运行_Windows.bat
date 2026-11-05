@echo off
python -m pip install -r requirements.txt
python boredom_sim.py --quick --output-dir boredom_outputs
pause
