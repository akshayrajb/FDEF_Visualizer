# FDEF Dependency Visualizer

A comprehensive Python tool for analyzing Mercedes-Benz FDEF (Functional Description) documents and creating interactive dependency graphs for signal tracing.

## 🚀 Features

- **Multi-PDF Processing**: Handle multiple FDEF PDF documents simultaneously
- **Enhanced OCR**: Advanced OCR pipeline for scanned technical diagrams
- **Signal Mapping**: Excel mapping sheet integration (columns C, D, F)
- **Interactive Visualization**: Web-based dependency graphs with click/double-click navigation
- **Executable Packaging**: One-click conversion to standalone EXE
- **Sample Testing**: Comprehensive test suite with sample data

## 📋 Requirements

- Python 3.8+
- Windows/macOS/Linux
- 4GB+ RAM (for OCR processing)

## 🛠️ Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/your-username/fdef-dependency-visualizer.git
cd fdef-dependency-visualizer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Test Core Functionality
```bash
python sample_test.py
```

### 3. Run the Application
```bash
python ui/app.py
```
Open browser to `http://localhost:8501`

### 4. Build Standalone Executable
```bash
# Windows
build_exe.bat

# Unix/Linux/macOS
chmod +x build_exe.sh
./build_exe.sh
```

## 📁 Project Structure

```
fdef-dependency-visualizer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── sample_test.py              # Test core functionality
├── build_exe.bat               # Windows executable builder
├── build_exe.sh                # Unix executable builder
├── .gitignore                  # Git ignore rules
│
├── parser/                     # Core parsing modules
│   ├── __init__.py
│   ├── mapping_loader.py       # Excel mapping sheet reader
│   ├── pdf_parser.py          # PDF text extraction
│   └── dependency_graph.py    # NetworkX graph builder
│
├── ocr_enhanced/               # Enhanced OCR for diagrams
│   ├── __init__.py
│   ├── diagram_ocr.py         # Main OCR pipeline
│   ├── region_detector.py     # Text/diagram separation
│   └── symbol_recognition.py  # Technical symbol detection
│
├── ui/                        # Streamlit web interface
│   ├── __init__.py
│   └── app.py                 # Main application
│
├── resources/                 # Data and cache
│   ├── sample_data/           # Sample test files
│   │   ├── sample_mapping.xlsx
│   │   └── sample_fdef.pdf
│   └── cache/                 # Graph cache storage
│
└── docs/                      # Documentation
    ├── API.md                 # API documentation
    ├── TROUBLESHOOTING.md     # Common issues
    └── EXAMPLES.md            # Usage examples
```

## 🔧 Usage

### Basic Signal Lookup
1. Upload your FDEF PDFs and mapping Excel file
2. Click "Build/Re-build graph"  
3. Enter signal name (e.g., "PT_Rdy")
4. Adjust graph depth and click "Show dependencies"

### Interactive Features
- **Click node**: View signal properties
- **Double-click node**: Expand dependencies
- **Drag**: Pan the graph
- **Scroll**: Zoom in/out

## 📊 Input Format

### Mapping Sheet (Excel)
| Column C | Column D     | Column F      |
|----------|-------------|---------------|
| Line 1   | Network_A   | Internal_A    |
| Line 2   | Network_B   | Internal_B    |

### FDEF Documents
- PDF format (text-searchable or scanned)
- Contains signal definitions and logic
- Supports complex wiring diagrams

## 🧪 Testing

Run the sample test to verify installation:
```bash
python sample_test.py
```

Expected output:
```
✅ Mapping loader test passed
✅ Dependency graph test passed  
✅ PDF parser test passed
✅ Integration test passed
All tests completed successfully!
```

## 🎯 Key Features Explained

### Enhanced OCR Pipeline
- **Preprocessing**: OCRmyPDF with technical document optimization
- **Region Detection**: Computer vision separation of text/diagrams
- **Symbol Recognition**: ML-based detection of logic gates and symbols
- **Connection Tracing**: Graph-based signal flow analysis

### Signal Dependency Tracing
- **Multi-level traversal**: Configurable depth exploration
- **Operator support**: AND, OR, NOT, Switch logic
- **Bidirectional mapping**: Network ↔ Internal name resolution
- **Cache optimization**: Fast repeated queries

## 🔧 Configuration

### OCR Settings (ocr_enhanced/diagram_ocr.py)
```python
OCR_CONFIG = {
    'dpi': 300,
    'optimize': True,
    'pdfa': False,
    'force_ocr': True
}
```

### Graph Settings (parser/dependency_graph.py)
```python
GRAPH_CONFIG = {
    'max_depth': 10,
    'cache_enabled': True,
    'node_limit': 1000
}
```

## 🚨 Troubleshooting

### Common Issues

**"No rules found for signal"**
- Verify signal name exists in mapping sheet
- Check PDF OCR quality with `sample_test.py`

**"Module not found" errors**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**Slow performance**
- Enable graph caching (default)
- Reduce max_depth setting
- Use higher-quality PDF scans

See `docs/TROUBLESHOOTING.md` for detailed solutions.

## 📦 Distribution

### Standalone Executable
The built executable includes:
- All Python dependencies
- OCR libraries and models
- Web interface assets
- Sample data for testing

### System Requirements
- Windows 10+ / macOS 10.14+ / Ubuntu 18.04+
- 100MB disk space
- No Python installation required

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Run tests (`python sample_test.py`)
4. Commit changes (`git commit -am 'Add enhancement'`)
5. Push branch (`git push origin feature/enhancement`)
6. Create Pull Request

## 📝 License

MIT License - see LICENSE file for details.

## 📞 Support

- **Issues**: GitHub Issues tab
- **Documentation**: `docs/` folder
- **Examples**: `docs/EXAMPLES.md`

---

**Ready to trace your FDEF signals? Start with `python sample_test.py`!** 🚀
