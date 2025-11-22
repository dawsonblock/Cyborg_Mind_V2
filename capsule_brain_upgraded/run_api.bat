@echo off
REM Activate virtual environment and run the Capsule Brain API on Windows
IF NOT EXIST venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
python -m capsule_brain.api.app