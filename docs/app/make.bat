@ECHO OFF

pushd %~dp0
set BUILDER=%1
if "%BUILDER%"=="" set BUILDER=html

uv run --group docs sphinx-build -M %BUILDER% . ..\_build -W --keep-going %SPHINXOPTS% %O%
set EXITCODE=%ERRORLEVEL%

popd
exit /b %EXITCODE%
