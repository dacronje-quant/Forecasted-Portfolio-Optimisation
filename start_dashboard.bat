@echo off
rem Start only the dashboard (after run_backtest.bat has set everything up once).
cd /d "%~dp0"
title Forecasted Portfolio Optimisation - dashboard
if not exist .venv\Scripts\python.exe (
  echo First run run_backtest.bat to set up Python and the packages.
  pause
  exit /b 1
)
if not exist outputs mkdir outputs
echo Opening the dashboard at http://localhost:8501 - keep this window open while you use it.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8501"
.venv\Scripts\python.exe -m streamlit run app\dashboard.py --server.headless true --browser.gatherUsageStats false > outputs\dashboard_log.txt 2>&1
