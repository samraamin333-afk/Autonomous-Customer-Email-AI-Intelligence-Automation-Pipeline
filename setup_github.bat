@echo off
setlocal

echo ======================================================================
echo    GitHub Repository Setup Helper - Customer Email AI Pipeline
echo ======================================================================
echo.

:: 1. Check if git user name/email are configured
git config user.name >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Git author identity is not yet set.
    set /p GIT_NAME="Enter your GitHub Name / Username: "
    set /p GIT_EMAIL="Enter your GitHub Email: "
    git config --global user.name "%GIT_NAME%"
    git config --global user.email "%GIT_EMAIL%"
    echo [OK] Git author identity configured.
    echo.
)

:: 2. Stage and commit files
echo [*] Staging all files and assets...
git add .
git commit -m "feat: complete autonomous customer email AI intelligence pipeline" >nul 2>&1
git branch -M main

echo.
echo [*] Almost done! Next step:
echo     1. Go to https://github.com/new in your browser
echo     2. Create a new repository named: customer-email-ai-pipeline
echo        (Do NOT check 'Add README' or 'Add .gitignore')
echo     3. Copy the repository HTTPS or SSH URL
echo.

set /p REPO_URL="Paste your GitHub repository URL (e.g. https://github.com/username/customer-email-ai-pipeline.git): "

if "%REPO_URL%"=="" (
    echo [!] No URL entered. Run this script again when your GitHub repo is created.
    pause
    exit /b 1
)

echo.
echo [*] Linking remote origin...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo [*] Pushing code to GitHub main branch...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo   [SUCCESS] Your project is now live on GitHub!
    echo   URL: %REPO_URL%
    echo ======================================================================
) else (
    echo.
    echo [!] Push failed. Please verify your repository URL and GitHub credentials.
)

echo.
pause
