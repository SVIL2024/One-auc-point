@echo off
rem Ingest XD-Violence: scores, decomposition and localization metrics for the
rem four arms.  Reads XD features from third_party\DSANet\features, writes
rem results\xd_scores.json and results\xdstack_*.npz.
rem
rem The XD localization column is NOT computed here: the reference implementation
rem hardcodes UCF-Crime's 14 classes and XD-Violence has 6, so the repo's dmAP
rem raises IndexError on this benchmark.  cafe_tloc.py is the replacement.
setlocal
cd /d "%~dp0.."
if "%PYTHON%"=="" set PYTHON=python
set PYTHONPATH=%CD%\third_party\DSANet\src;%CD%\third_party\DSANet\_win_shim;%CD%
if not exist results mkdir results

for %%A in (base:789 base:234 core8g1:234 res:234) do call :arm %%~A
echo done: results\xd_scores.json
goto :eof

:arm
for /f "tokens=1,2 delims=:" %%X in ("%~1") do (
  echo === %%X seed %%Y ===
  "%PYTHON%" cafe\dsanet_xd_dump.py --arms "%%X %%Y" --root "%CD%\third_party\DSANet" 2>>results\ingest_xd.log
  if errorlevel 1 (
    echo     retry %%X %%Y
    "%PYTHON%" cafe\dsanet_xd_dump.py --arms "%%X %%Y" --root "%CD%\third_party\DSANet" 2>>results\ingest_xd.log
  )
)
goto :eof
