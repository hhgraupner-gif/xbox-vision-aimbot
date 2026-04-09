@echo off
chcp 65001 >nul
title Warzone Audio
cd /d %USERPROFILE%\Downloads
python competitive_audio.py
