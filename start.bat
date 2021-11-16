@echo off
:: Path to the Anaconda3 root directory
set root=C:\ProgramData\Anaconda3
:: Name of the conda environment to use. Set to %root% if the base environment should be used.
set conda_env=returntypes
call %root%\Scripts\activate.bat %conda_env%
python %~dp0predictor.py --port 10000