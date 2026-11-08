@echo off
rem Live viewer: a brain plays Mario in one window, its activity shows in a second window.
rem   watch-brain.bat                          LivyA, 12 levels (1-1 to 3-4), 5 tries each
rem   watch-brain.bat LivyB                    another brain (runs\<name>\<name>_ck0200.npz)
rem   watch-brain.bat LivyA --first 2-1 --last 2-1 --tries 3 --speed 0.5
rem Keys (click a window first): space pause, n next level, q quit, + / - speed.
cd /d P:\projects\cadence-arcade\mario\coldstart
set NAME=%1
if "%NAME%"=="" set NAME=LivyA
shift
set PY=P:\projects\cadence-arcade\.venv-concepts\Scripts\python.exe
if "%NAME:~0,6%"=="Fabius" set PY=P:\projects\cadence-arcade\.venv-095\Scripts\python.exe
if "%NAME:~0,5%"=="Cato2" set PY=P:\projects\cadence-arcade\.venv-095\Scripts\python.exe
%PY% live.py --brain runs\%NAME%\%NAME%_ck0200.npz --name %NAME% %1 %2 %3 %4 %5 %6 %7 %8 %9
pause
