@echo off
rem Ingest UCF-Crime: scores, decomposition and detection mAP for the eleven
rem checkpoints, one process per arm.
rem
rem One process per arm is deliberate: the GPU occasionally raises a sticky
rem CUDA error, and a fresh process keeps a bad arm from killing the batch.
rem Each arm takes about a minute; the whole file about twelve.
rem
rem Put the DSANet checkout at third_party\DSANet and its weights in
rem third_party\DSANet\model, then run this from anywhere.
setlocal
cd /d "%~dp0.."
if "%PYTHON%"=="" set PYTHON=python
set PYTHONPATH=%CD%\third_party\DSANet\src;%CD%\third_party\DSANet\_win_shim;%CD%
if not exist results mkdir results

call :arm base 234
call :arm base 555
call :arm base 789
call :arm motion6 555
call :arm motion6 789
call :arm motion6shuf5678 234
call :arm attr 234
call :arm cons0 234
call :arm core8g1 234
call :arm core14txt 234
call :arm res 234
echo done: results\map11.json and results\scores\scores_ucf_*.npz
goto :eof

:arm
echo === %~1 seed %~2 ===
"%PYTHON%" cafe\cafe_map11.py --arm %~1 --seed %~2 --root "%CD%\third_party\DSANet" 2>>results\ingest_ucf.log
if errorlevel 1 (
  echo     retry %~1 %~2 after a failed run
  "%PYTHON%" cafe\cafe_map11.py --arm %~1 --seed %~2 --root "%CD%\third_party\DSANet" 2>>results\ingest_ucf.log
)
goto :eof
