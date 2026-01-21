@echo off
setlocal

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv not found. Installing...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "UV_BIN=%USERPROFILE%\.local\bin"
  if exist "%UV_BIN%\uv.exe" (
    set "PATH=%UV_BIN%;%PATH%"
  )
)

uv --version >nul 2>nul
if errorlevel 1 (
  echo uv is still not available. Please restart your shell and rerun this script.
  exit /b 1
)

uv python install 3.12
if errorlevel 1 exit /b 1
uv sync
if errorlevel 1 exit /b 1

if "%~1"=="" (
  echo Starting server at http://127.0.0.1:8000/
  uv run python manage.py runserver
) else (
  uv run python manage.py %*
)
