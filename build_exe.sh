#!/bin/bash

# Unix/Linux/macOS Build Script for FDEF Dependency Visualizer
# Creates a standalone executable using PyInstaller

echo "==============================================="
echo "FDEF Dependency Visualizer - Unix/Linux Build"
echo "==============================================="

# Check if PyInstaller is available
python3 -c "import PyInstaller" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyInstaller not found. Installing..."
    pip3 install pyinstaller
fi

# Clean previous builds
rm -rf build dist *.spec

echo ""
echo "Building standalone executable..."
echo "This may take 5-10 minutes..."
echo ""

# Build the executable
pyinstaller --clean -y \
  --name fdef_dependency_visualizer \
  --add-data "resources:resources" \
  --add-data "parser:parser" \
  --add-data "ocr_enhanced:ocr_enhanced" \
  --collect-all streamlit \
  --collect-all pyvis \
  --collect-all networkx \
  --collect-all pandas \
  --collect-all numpy \
  --collect-all opencv-python \
  --collect-all pytesseract \
  --collect-all pymupdf \
  --hidden-import=streamlit.web.cli \
  --hidden-import=streamlit.runtime.scriptrunner.magic_funcs \
  --hidden-import=pyvis.network \
  --hidden-import=networkx \
  --hidden-import=pandas \
  --hidden-import=fitz \
  --hidden-import=cv2 \
  --hidden-import=pytesseract \
  --exclude-module=pytest \
  --exclude-module=IPython \
  --exclude-module=jupyter \
  --windowed \
  --onefile \
  ui/app.py

if [ $? -eq 0 ]; then
    echo ""
    echo "==============================================="
    echo "Build completed successfully!"
    echo "==============================================="
    echo ""
    echo "Executable location: dist/fdef_dependency_visualizer"
    echo "File size: $(du -h dist/fdef_dependency_visualizer | cut -f1)"
    echo ""
    echo "To distribute:"
    echo "1. Copy dist/fdef_dependency_visualizer to target machine"
    echo "2. Make executable: chmod +x fdef_dependency_visualizer"
    echo "3. Run: ./fdef_dependency_visualizer"
    echo "4. Web interface will open at http://localhost:8501"
    echo ""
    echo "No Python installation required on target machine!"
    echo "==============================================="
    
    # Make the executable file executable
    chmod +x dist/fdef_dependency_visualizer
    
else
    echo ""
    echo "==============================================="
    echo "Build failed! Check the error messages above."
    echo "==============================================="
    echo ""
    echo "Common solutions:"
    echo "1. Ensure all dependencies are installed: pip3 install -r requirements.txt"
    echo "2. Run from project root directory"
    echo "3. Check that Python3 and PyInstaller are working"
    echo "4. On macOS, you may need: brew install python-tk"
    echo "5. On Linux, you may need: sudo apt-get install python3-tk"
    echo ""
    echo "For help, see docs/TROUBLESHOOTING.md"
    echo "==============================================="
fi