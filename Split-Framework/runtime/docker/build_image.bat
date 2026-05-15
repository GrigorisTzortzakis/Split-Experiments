@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "FRAMEWORK_DIR=%SCRIPT_DIR%..\.."

docker build -t split-framework-runner:latest -f "%SCRIPT_DIR%Dockerfile" "%FRAMEWORK_DIR%"