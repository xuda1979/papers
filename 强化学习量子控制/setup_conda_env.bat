@echo off
echo Creating Conda environment 'rl_quantum_control'...

call conda create -n rl_quantum_control python=3.9 -y

echo Activating environment...
call conda activate rl_quantum_control

echo Installing requirements...
pip install -r rl_quantum_control\requirements.txt

echo Setup complete!
echo To activate the environment, run: conda activate rl_quantum_control
pause
