@echo off
rem Recompute the detection mAP column with the reference implementation shared by
rem UCF-Crime papers, plus an independent reimplementation in the same process,
rem plus the frame-level AUC and AP.
rem
rem The two implementations agree to 0.0 over all 5 IoU thresholds and all arms;
rem that identity is what the paper cites when it calls the column "the published
rem protocol".
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
echo done: results\official12.json
goto :eof

:arm
echo === %~1 seed %~2 ===
"%PYTHON%" cafe\cafe_official12.py --arm %~1 --seed %~2 2>>results\ingest_official.log
if errorlevel 1 (
  echo     retry %~1 %~2 after a failed run
  "%PYTHON%" cafe\cafe_official12.py --arm %~1 --seed %~2 2>>results\ingest_official.log
)
goto :eof
