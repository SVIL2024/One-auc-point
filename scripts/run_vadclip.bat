@echo off
rem Ingest the released VadCLIP weights and decompose them exactly like our own
rem arms.  This is the external anchor: someone else's training run, someone
rem else's score convention.
rem
rem Put the VadCLIP checkout at third_party\VadCLIP and the released weights at
rem third_party\VadCLIP\model\vadclip_model_ucf.pth.  About a minute per dataset,
rem most of it feature extraction.
setlocal
cd /d "%~dp0.."
if "%PYTHON%"=="" set PYTHON=python
set PYTHONPATH=%CD%\third_party\VadCLIP\_win_shim;%CD%\third_party\VadCLIP\src;%CD%
if not exist results mkdir results

echo === UCF-Crime anchor ===
"%PYTHON%" cafe\cafe_vadclip.py --ds ucf --root "%CD%\third_party\VadCLIP" --out "%CD%\results" 2>>results\ingest_vadclip.log
if errorlevel 1 echo UCF anchor failed, see results\ingest_vadclip.log

if exist "%CD%\third_party\VadCLIP\model\vadclip_model_xd.pth" (
  echo === XD-Violence anchor ===
  "%PYTHON%" cafe\cafe_vadclip.py --ds xd --root "%CD%\third_party\VadCLIP" --out "%CD%\results" --skip-map 2>>results\ingest_vadclip.log
  if errorlevel 1 echo XD anchor failed, see results\ingest_vadclip.log
)
echo done: results\vadclip.json and results\scores\scores_*_vadclip.npz
