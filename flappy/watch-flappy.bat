@echo off
rem Corvus plays Flappy Bird: game + brain on the main monitor. Keys: space pause, f fast, q quit.
rem   watch-flappy.bat          no learning
rem   watch-flappy.bat --learn  the teacher keeps correcting it while it plays
cd /d P:\projects\cadence-arcadelappy
.venv\Scripts\python.exe watch.py %*
