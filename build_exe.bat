@echo off
REM Windows Build Script for FDEF Dependency Visualizer
REM Creates a standalone executable using PyInstaller

echo ===============================================
echo FDEF Dependency Visualizer - Windows Build
echo ===============================================

REM Check if PyInstaller is available
python -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo Error: PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Clean previous builds
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" del "*.spec"

echo.
echo Building standalone executable...
echo This may take 5-10 minutes...
echo.

REM Build the executable
pyinstaller --clean -y ^
  --name fdef_dependency_visualizer ^
  --add-data "resources;resources" ^
  --add-data "parser;parser" ^
  --add-data "ocr_enhanced;ocr_enhanced" ^
  --collect-all streamlit ^
  --collect-all pyvis ^
  --collect-all networkx ^
  --collect-all pandas ^
  --collect-all numpy ^
  --collect-all opencv-python ^
  --collect-all pytesseract ^
  --collect-all pymupdf ^
  --hidden-import=streamlit.web.cli ^
  --hidden-import=streamlit.runtime.scriptrunner.magic_funcs ^
  --hidden-import=pyvis.network ^
  --hidden-import=networkx ^
  --hidden-import=pandas ^
  --hidden-import=fitz ^
  --hidden-import=cv2 ^
  --hidden-import=pytesseract ^
  --exclude-module=pytest ^
  --exclude-module=IPython ^
  --exclude-module=jupyter ^
  --windowed ^
  --onefile ^
  ui/app.py

if %errorlevel% equ 0 (
    echo.
    echo ===============================================
    echo Build completed successfully!
    echo ===============================================
    echo.
    echo Executable location: dist\fdef_dependency_visualizer.exe
    echo File size: 
    dir dist\fdef_dependency_visualizer.exe | find ".exe"
    echo.
    echo To distribute:
    echo 1. Copy dist\fdef_dependency_visualizer.exe to target machine
    echo 2. Double-click to run
    echo 3. Web interface will open at http://localhost:8501
    echo.
    echo No Python installation required on target machine!
    echo ===============================================
) else (
    echo.
    echo ===============================================
    echo Build failed! Check the error messages above.
    echo ===============================================
    echo.
    echo Common solutions:
    echo 1. Ensure all dependencies are installed: pip install -r requirements.txt
    echo 2. Run from project root directory
    echo 3. Check that Python and PyInstaller are working
    echo.
    echo For help, see docs/TROUBLESHOOTING.md
    echo ===============================================
)

pause