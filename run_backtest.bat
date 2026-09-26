@echo off
setlocal EnableExtensions
rem One-click setup + backtest + dashboard for Windows.
rem Installs Python 3.12 for your user account if it is missing, creates .venv,
rem installs requirements, runs the backtest and this month's portfolio, then opens the dashboard.
cd /d "%~dp0"
title Forecasted Portfolio Optimisation
if not exist outputs mkdir outputs
set LOG=outputs\run_log.txt
echo [%time%] start> %LOG%
echo STARTED> outputs\run_status.txt

call :findpy
if not defined PY (
  echo Python not found - installing Python 3.12 for your user account...
  echo [%time%] installing python>> %LOG%
  echo INSTALLING_PYTHON> outputs\run_status.txt
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements >> %LOG% 2>&1
  call :findpy
)
if not defined PY (
  echo winget unavailable or failed - downloading the installer from python.org...
  echo [%time%] python.org installer>> %LOG%
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -OutFile $env:TEMP\py312.exe" >> %LOG% 2>&1
  "%TEMP%\py312.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 >> %LOG% 2>&1
  call :findpy
)
if not defined PY goto :nopy
echo [%time%] python: %PY%>> %LOG%

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  echo SETUP> outputs\run_status.txt
  "%PY%" -m venv .venv >> %LOG% 2>&1 || goto :fail
)
echo Installing packages (first run takes a few minutes)...
echo INSTALLING_PACKAGES> outputs\run_status.txt
.venv\Scripts\python.exe -m pip install --upgrade pip >> %LOG% 2>&1
.venv\Scripts\python.exe -m pip install -r requirements.txt >> %LOG% 2>&1 || goto :fail
.venv\Scripts\python.exe -m pip install -e . >> %LOG% 2>&1 || goto :fail

echo Running backtest on live Yahoo Finance data...
echo BACKTEST> outputs\run_status.txt
.venv\Scripts\python.exe -m fpo backtest --model gbm --out outputs > outputs\backtest_log.txt 2>&1 || goto :fail
type outputs\backtest_log.txt
echo Building this month's portfolio for 50 000 SEK...
echo RECOMMEND> outputs\run_status.txt
.venv\Scripts\python.exe -m fpo recommend --model gbm --budget 50000 --out outputs > outputs\recommend_log.txt 2>&1 || goto :fail
type outputs\recommend_log.txt

:dashboard
echo DASHBOARD> outputs\run_status.txt
echo [%time%] starting dashboard>> %LOG%
rem Skip Streamlit's first-run email prompt
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)
echo.
echo Opening the dashboard at http://localhost:8501 - keep this window open while you use it.
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
.venv\Scripts\python.exe -m streamlit run app\dashboard.py --server.headless true --browser.gatherUsageStats false > outputs\dashboard_log.txt 2>&1
exit /b 0

:findpy
set PY=
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "C:\Program Files\Python313\python.exe" "C:\Program Files\Python312\python.exe" "C:\Program Files\Python311\python.exe" "C:\Program Files\Python310\python.exe") do (
  if not defined PY if exist %%P set "PY=%%~P"
)
if defined PY exit /b 0
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
exit /b 0

:nopy
echo PYTHON_INSTALL_FAILED> outputs\run_status.txt
echo Could not install Python automatically. See outputs\run_log.txt
pause
exit /b 1

:fail
echo FAILED> outputs\run_status.txt
echo Something went wrong - details are in the outputs folder.
pause
exit /b 1
