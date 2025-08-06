# FDEF Signal Analyzer

**Comprehensive Function Design and Engineering Flow Analysis for Automotive PDFs**

A production-ready tool for extracting signal dependencies from automotive engineering documents using advanced computer vision, mathematical parsing, and template matching techniques.

![FDEF Analyzer Demo](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## 🚀 Features

### Core Capabilities
- **📄 Advanced PDF Processing**: High-quality OCR with automotive document optimizations
- **👁️ Computer Vision**: Arrow and block detection using template matching and contour analysis
- **🧮 Mathematical Parsing**: SymPy-powered equation extraction and dependency analysis
- **🚗 Automotive Templates**: Specialized pattern matching for A2L, AUTOSAR, CAN/LIN signals
- **🕸️ Dependency Mapping**: Intelligent graph construction combining multiple analysis methods
- **🌐 Interactive Visualization**: Professional HTML visualizations with Plotly

### Analysis Methods
1. **Computer Vision Analysis**
   - Multi-orientation arrow detection
   - Functional block identification
   - Spatial relationship mapping
   - Template matching with confidence scoring

2. **Mathematical Expression Parsing**
   - Symbolic computation with SymPy
   - Automotive function recognition
   - Lookup table extraction
   - Variable dependency analysis

3. **Domain Template Matching**
   - A2L characteristic/measurement patterns
   - AUTOSAR component recognition
   - CAN/LIN signal identification
   - Diagnostic trouble code detection

4. **Intelligent Graph Construction**
   - Multi-source evidence fusion
   - Cross-reference resolution
   - Signal name normalization
   - Confidence-weighted relationships

## 📋 Requirements

### System Dependencies
- **Tesseract OCR** - Text extraction from images
- **Poppler Utils** - PDF to image conversion
- **Python 3.8+** - Core runtime environment

### Python Packages
See `requirements.txt` for complete list. Key dependencies:
- `opencv-python` - Computer vision operations
- `sympy` - Mathematical expression parsing
- `networkx` - Graph analysis and algorithms
- `pytesseract` - OCR interface
- `pdf2image` - PDF processing
- `plotly` - Interactive visualizations

## 🛠️ Installation

### Quick Setup (Recommended)
```bash
# Clone or download the FDEF analyzer files
cd fdef-analyzer

# Run the automated setup script
python setup.py

# Test the installation
python fdef_analyzer.py --help
```

### Manual Installation

#### 1. System Dependencies

**Windows (with Chocolatey):**
```powershell
choco install tesseract poppler -y
```

**macOS (with Homebrew):**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils -y
```

**CentOS/RHEL:**
```bash
sudo yum install tesseract poppler-utils -y
```

#### 2. Python Environment
```bash
# Create virtual environment (recommended)
python -m venv fdef-env
source fdef-env/bin/activate  # On Windows: fdef-env\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

#### 3. Verification
```bash
python setup.py --config-only  # Creates sample config
python -c "from fdef_analyzer import FdefAnalyzer; print('✅ Installation successful')"
```

## 🚀 Quick Start

### Basic Usage
```bash
# Analyze a PDF document
python fdef_analyzer.py document.pdf

# Focus analysis on a specific signal
python fdef_analyzer.py document.pdf --target-signal Engine_RPM

# Specify output directory
python fdef_analyzer.py document.pdf --output results/

# Use custom configuration
python fdef_analyzer.py document.pdf --config custom_config.json
```

### Example Output
```
🚀 Starting FDEF analysis...
📖 Phase 1: PDF Loading and Preprocessing
   Loaded 15 pages from PDF
👁️ Phase 2: Computer Vision Analysis
   Detected 23 arrows and 18 blocks
🧮 Phase 3: Mathematical Equation Parsing
   Parsed 12 equations and 3 lookup tables
🚗 Phase 4: Automotive Template Matching
   Found 8 template matches
🕸️ Phase 5: Dependency Graph Construction
   Built graph with 34 nodes and 28 edges
📊 Phase 6: Statistics Collection
✅ Analysis complete in 12.3 seconds

📊 ANALYSIS SUMMARY
=====================================
📄 Document: vehicle_control_fdef.pdf
🔍 Signals detected: 34
🕸️ Dependencies mapped: 28
⏱️ Analysis time: 12.3 seconds
📊 Average confidence: 78.5%

📤 Results exported to: fdef_analysis_output
```

## 📁 Project Structure

```
fdef-analyzer/
├── 📄 README.md                 # This documentation
├── 📋 requirements.txt          # Python dependencies
├── ⚙️ setup.py                  # Installation script
├── 🔧 fdef_config.json         # Sample configuration
│
├── 🧠 Core Analysis Modules:
│   ├── 📖 core_io.py            # PDF loading and OCR
│   ├── 👁️ cv_arrows.py          # Computer vision detection
│   ├── 🧮 sympy_parser.py       # Mathematical parsing
│   ├── 🚗 domain_templates.py   # Automotive templates
│   ├── 🕸️ graph_builder.py      # Dependency graph construction
│   └── 🌐 export_html.py        # HTML visualization
│
├── 🎯 Main Application:
│   └── 📱 fdef_analyzer.py      # Main orchestrator
│
├── 📊 Output Examples:
│   ├── 🌐 signal_network.html   # Interactive visualization
│   ├── 📋 analysis_summary.html # Summary dashboard
│   ├── 📄 detailed_report.json  # Complete analysis data
│   └── 📝 extracted_signals.txt # Signal list
│
└── 🧪 Tests & Examples:
    ├── 📚 examples/              # Example documents
    ├── 🧪 tests/                 # Unit tests
    └── 📖 docs/                  # Additional documentation
```

## ⚙️ Configuration

### Configuration File Format
The analyzer uses JSON configuration files. Create with:
```bash
python setup.py --config-only
```

### Key Configuration Sections

#### PDF Processing
```json
{
  "pdf_processing": {
    "dpi": 200,                    // Image resolution for conversion
    "enhance_images": true,        // Apply image enhancement
    "max_pages": null              // Limit pages (null = all)
  }
}
```

#### Computer Vision
```json
{
  "computer_vision": {
    "arrow_detection_threshold": 0.6,     // Arrow detection sensitivity
    "block_detection_threshold": 0.7,     // Block detection sensitivity
    "enable_template_matching": true,     // Use template matching
    "enable_line_detection": true,        // Use Hough line detection
    "enable_contour_analysis": true       // Use contour analysis
  }
}
```

#### Mathematical Parsing
```json
{
  "equation_parsing": {
    "confidence_threshold": 0.5,          // Minimum equation confidence
    "enable_automotive_functions": true,   // Recognize automotive functions
    "parse_lookup_tables": true           // Extract lookup tables
  }
}
```

## 📊 Understanding Results

### Interactive Network Visualization
The main output is an interactive HTML network visualization showing:

- **🔵 Input Signals** (Sea Green) - External inputs to the system
- **🔴 Output Signals** (Crimson) - System outputs
- **🟦 Intermediate Signals** (Steel Blue) - Internal processing signals
- **🟡 Parameters** (Goldenrod) - Configuration parameters

**Edge Types:**
- **Blue lines** - Direct mathematical relationships
- **Orange lines** - Conditional dependencies
- **Green lines** - Lookup table relationships
- **Red lines** - Rate-limited connections

### Analysis Reports

#### 1. Summary Dashboard (`analysis_summary.html`)
High-level metrics and performance indicators

#### 2. Detailed Report (`detailed_analysis_report.json`)
Complete analysis data including:
- All detected signals with confidence scores
- Mathematical equations and their parsing results
- Template matches and extracted data
- Graph structure and relationships

#### 3. Signal List (`extracted_signals.txt`)
Tab-separated list of all signals:
```
Signal_Name          Type          Confidence    Sources
Engine_RPM          input         0.920         equation_parsing,cv_detection
Vehicle_Speed       output        0.856         template_matching
Throttle_Position   input         0.743         cv_detection,text_extraction
```

## 🔧 Advanced Usage

### Programmatic API
```python
from fdef_analyzer import FdefAnalyzer

# Create analyzer with custom config
config = {
    "pdf_processing": {"dpi": 300},
    "output": {"save_intermediate_results": True}
}

analyzer = FdefAnalyzer(config=config)

# Perform analysis
results = analyzer.analyze_pdf("document.pdf", target_signal="Engine_RPM")

# Export results
exported_files = analyzer.export_results("output_dir/", target_signal="Engine_RPM")

# Get detailed signal analysis
signal_info = analyzer.get_signal_analysis("Engine_RPM")
print(f"Signal confidence: {signal_info['analysis_confidence']}")
print(f"Dependencies: {signal_info['dependencies']}")
```

### Custom Analysis Pipeline
```python
from core_io import PdfLoader
from cv_arrows import CvArrowDetector
from sympy_parser import SymPyEquationParser
from graph_builder import DependencyGraphBuilder

# Load PDF
loader = PdfLoader("document.pdf", dpi=200)
pages = loader.load_pages()

# Computer vision analysis
cv_detector = CvArrowDetector()
arrows = cv_detector.detect_arrow_paths(pages[0].processed_image)
blocks = cv_detector.detect_blocks(pages[0].processed_image)

# Equation parsing
parser = SymPyEquationParser()
equations = parser.parse_equations(pages[0].text_content)

# Build dependency graph
builder = DependencyGraphBuilder()
graph = builder.build_comprehensive_graph(arrows, blocks, equations, [], [])
```

## 🧪 Testing and Validation

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Validation Checklist
- ✅ Tesseract OCR functional
- ✅ PDF conversion working
- ✅ Computer vision detection
- ✅ Mathematical parsing
- ✅ Graph construction
- ✅ HTML export generation

## 🐛 Troubleshooting

### Common Issues

#### 1. Tesseract Not Found
```
Error: TesseractNotFoundError
```
**Solution:** Install Tesseract OCR system package
- Windows: Use Chocolatey or manual install
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

#### 2. PDF Conversion Fails
```
Error: PDFInfoNotInstalledError
```
**Solution:** Install Poppler utilities
- Windows: Install poppler-windows
- macOS: `brew install poppler`
- Linux: `sudo apt-get install poppler-utils`

#### 3. Low Signal Detection
**Symptoms:** Few signals detected, low confidence scores
**Solutions:**
- Increase DPI in configuration (try 300)
- Enable image enhancement
- Check PDF quality and text clarity
- Adjust detection thresholds

#### 4. Memory Issues with Large PDFs
**Symptoms:** Out of memory errors, slow processing
**Solutions:**
- Reduce DPI setting
- Limit pages with `max_pages` parameter
- Process in smaller batches
- Increase system RAM

### Performance Optimization

#### For Large Documents (>50 pages)
```json
{
  "pdf_processing": {
    "dpi": 150,                    // Reduce DPI
    "max_pages": 20                // Limit pages
  },
  "computer_vision": {
    "enable_contour_analysis": false  // Disable expensive analysis
  }
}
```

#### For High Accuracy
```json
{
  "pdf_processing": {
    "dpi": 300,                    // Increase DPI
    "enhance_images": true
  },
  "equation_parsing": {
    "confidence_threshold": 0.3    // Lower threshold
  }
}
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd fdef-analyzer

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install in development mode
pip install -r requirements.txt
pip install -e .

# Install development tools
pip install black flake8 pytest pytest-cov
```

### Code Style
- Use `black` for code formatting
- Follow PEP 8 guidelines
- Add type hints for new functions
- Include docstrings for public methods

### Testing
- Add unit tests for new features
- Ensure >80% code coverage
- Test with various PDF formats
- Validate cross-platform compatibility

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** - Computer vision operations
- **SymPy** - Symbolic mathematics
- **NetworkX** - Graph algorithms
- **Tesseract** - OCR engine
- **Plotly** - Interactive visualizations
- **pdf2image** - PDF processing

## 📞 Support

### Documentation
- 📖 [User Guide](docs/user_guide.md)
- 🔧 [API Reference](docs/api_reference.md)
- 🧪 [Testing Guide](docs/testing.md)

### Getting Help
1. Check the troubleshooting section above
2. Review the example configurations
3. Run the validation script: `python setup.py`
4. Check system requirements and dependencies

### Reporting Issues
When reporting issues, please include:
- System information (OS, Python version)
- Complete error messages
- Sample PDF (if possible)
- Configuration used
- Expected vs actual behavior

---

**Built with ❤️ for the automotive engineering community**

*FDEF Signal Analyzer - Making signal dependency analysis accessible and automated*