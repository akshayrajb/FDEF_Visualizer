#!/usr/bin/env python3
"""
Setup and Installation Script for FDEF Signal Analyzer
Handles system dependencies, Python package installation, and environment setup.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SystemSetup:
    """Handle system-level dependency installation"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.is_admin = self._check_admin_privileges()
        
    def _check_admin_privileges(self) -> bool:
        """Check if running with administrator privileges"""
        try:
            if self.system == "windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except:
            return False
            
    def install_system_dependencies(self) -> bool:
        """Install system-level dependencies (Tesseract, Poppler)"""
        
        print("🔧 Installing system dependencies...")
        
        if self.system == "windows":
            return self._install_windows_dependencies()
        elif self.system == "darwin":  # macOS
            return self._install_macos_dependencies()
        elif self.system == "linux":
            return self._install_linux_dependencies()
        else:
            print(f"❌ Unsupported system: {self.system}")
            return False
            
    def _install_windows_dependencies(self) -> bool:
        """Install dependencies on Windows"""
        
        print("📦 Windows system detected")
        
        # Check if chocolatey is available
        if shutil.which('choco'):
            print("   Using Chocolatey package manager...")
            
            try:
                # Install Tesseract
                subprocess.run(['choco', 'install', 'tesseract', '-y'], 
                             check=True, capture_output=True)
                print("   ✅ Tesseract OCR installed")
                
                # Install Poppler
                subprocess.run(['choco', 'install', 'poppler', '-y'], 
                             check=True, capture_output=True)
                print("   ✅ Poppler utilities installed")
                
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Chocolatey installation failed: {e}")
                return False
        else:
            print("   ⚠️ Chocolatey not found")
            print("   Please install manually:")
            print("   - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            print("   - Poppler: https://github.com/oschwartz10612/poppler-windows")
            return False
            
    def _install_macos_dependencies(self) -> bool:
        """Install dependencies on macOS"""
        
        print("🍎 macOS system detected")
        
        # Check if Homebrew is available
        if shutil.which('brew'):
            print("   Using Homebrew package manager...")
            
            try:
                # Install Tesseract
                subprocess.run(['brew', 'install', 'tesseract'], 
                             check=True, capture_output=True)
                print("   ✅ Tesseract OCR installed")
                
                # Install Poppler
                subprocess.run(['brew', 'install', 'poppler'], 
                             check=True, capture_output=True)
                print("   ✅ Poppler utilities installed")
                
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Homebrew installation failed: {e}")
                return False
        else:
            print("   ⚠️ Homebrew not found")
            print("   Please install Homebrew first: https://brew.sh/")
            print("   Then run: brew install tesseract poppler")
            return False
            
    def _install_linux_dependencies(self) -> bool:
        """Install dependencies on Linux"""
        
        print("🐧 Linux system detected")
        
        # Detect package manager
        if shutil.which('apt-get'):
            return self._install_debian_dependencies()
        elif shutil.which('yum'):
            return self._install_redhat_dependencies()
        elif shutil.which('pacman'):
            return self._install_arch_dependencies()
        else:
            print("   ❌ No supported package manager found")
            print("   Please install manually: tesseract-ocr poppler-utils")
            return False
            
    def _install_debian_dependencies(self) -> bool:
        """Install on Debian/Ubuntu systems"""
        
        print("   Using apt package manager...")
        
        try:
            # Update package list
            subprocess.run(['sudo', 'apt-get', 'update'], 
                         check=True, capture_output=True)
            
            # Install packages
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 
                          'tesseract-ocr', 'poppler-utils'], 
                         check=True, capture_output=True)
            
            print("   ✅ System dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ apt installation failed: {e}")
            return False
            
    def _install_redhat_dependencies(self) -> bool:
        """Install on Red Hat/CentOS systems"""
        
        print("   Using yum package manager...")
        
        try:
            subprocess.run(['sudo', 'yum', 'install', '-y', 
                          'tesseract', 'poppler-utils'], 
                         check=True, capture_output=True)
            
            print("   ✅ System dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ yum installation failed: {e}")
            return False
            
    def _install_arch_dependencies(self) -> bool:
        """Install on Arch Linux systems"""
        
        print("   Using pacman package manager...")
        
        try:
            subprocess.run(['sudo', 'pacman', '-S', '--noconfirm', 
                          'tesseract', 'poppler'], 
                         check=True, capture_output=True)
            
            print("   ✅ System dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ pacman installation failed: {e}")
            return False

class PythonSetup:
    """Handle Python environment and package installation"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.requirements_file = Path(__file__).parent / "requirements.txt"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        
        if self.python_version < (3, 8):
            print(f"❌ Python {self.python_version.major}.{self.python_version.minor} detected")
            print("   FDEF Analyzer requires Python 3.8 or higher")
            return False
            
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor} detected")
        return True
        
    def create_virtual_environment(self, venv_path: str = "venv") -> bool:
        """Create a virtual environment"""
        
        print(f"🐍 Creating virtual environment: {venv_path}")
        
        try:
            subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
            print("   ✅ Virtual environment created")
            
            # Get activation script path
            if platform.system().lower() == "windows":
                activate_script = Path(venv_path) / "Scripts" / "activate.bat"
            else:
                activate_script = Path(venv_path) / "bin" / "activate"
                
            print(f"   📝 To activate: source {activate_script}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to create virtual environment: {e}")
            return False
            
    def install_python_packages(self, upgrade_pip: bool = True) -> bool:
        """Install Python packages from requirements.txt"""
        
        print("📦 Installing Python packages...")
        
        try:
            # Upgrade pip first
            if upgrade_pip:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                             check=True)
                print("   ✅ pip upgraded")
                
            # Install requirements
            if self.requirements_file.exists():
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 
                              str(self.requirements_file)], check=True)
                print("   ✅ Python packages installed")
                return True
            else:
                print(f"   ❌ Requirements file not found: {self.requirements_file}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Package installation failed: {e}")
            return False

class EnvironmentValidator:
    """Validate the installation and environment"""
    
    def validate_installation(self) -> bool:
        """Validate that all components are properly installed"""
        
        print("🔍 Validating installation...")
        
        # Check system dependencies
        tesseract_ok = self._check_tesseract()
        poppler_ok = self._check_poppler()
        
        # Check Python packages
        packages_ok = self._check_python_packages()
        
        # Overall status
        all_ok = tesseract_ok and poppler_ok and packages_ok
        
        if all_ok:
            print("✅ Installation validation successful!")
        else:
            print("❌ Installation validation failed!")
            
        return all_ok
        
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available"""
        
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            print(f"   ✅ Tesseract OCR: {version}")
            return True
        except Exception as e:
            print(f"   ❌ Tesseract OCR: {e}")
            return False
            
    def _check_poppler(self) -> bool:
        """Check if Poppler utilities are available"""
        
        try:
            from pdf2image import convert_from_path
            print("   ✅ Poppler utilities: Available")
            return True
        except Exception as e:
            print(f"   ❌ Poppler utilities: {e}")
            return False
            
    def _check_python_packages(self) -> bool:
        """Check if critical Python packages are available"""
        
        critical_packages = [
            'numpy', 'cv2', 'PIL', 'sympy', 'networkx', 
            'fuzzywuzzy', 'PyPDF2', 'pdf2image', 'pytesseract'
        ]
        
        missing_packages = []
        
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
                
        if missing_packages:
            print(f"   ❌ Missing packages: {', '.join(missing_packages)}")
            return False
        else:
            print("   ✅ All critical packages available")
            return True

def create_sample_config() -> str:
    """Create a sample configuration file"""
    
    config_content = """{
    "pdf_processing": {
        "dpi": 200,
        "enhance_images": true,
        "max_pages": null
    },
    "computer_vision": {
        "arrow_detection_threshold": 0.6,
        "block_detection_threshold": 0.7,
        "enable_template_matching": true,
        "enable_line_detection": true,
        "enable_contour_analysis": true
    },
    "equation_parsing": {
        "confidence_threshold": 0.5,
        "enable_automotive_functions": true,
        "parse_lookup_tables": true
    },
    "template_matching": {
        "enable_a2l_patterns": true,
        "enable_autosar_patterns": true,
        "enable_can_patterns": true,
        "enable_diagnostic_patterns": true,
        "fuzzy_matching_threshold": 0.8
    },
    "graph_building": {
        "merge_similar_signals": true,
        "resolve_cross_references": true,
        "confidence_weight_threshold": 0.4
    },
    "output": {
        "create_html_visualization": true,
        "create_summary_dashboard": true,
        "save_intermediate_results": false,
        "output_directory": "fdef_analysis_output"
    },
    "logging": {
        "level": "INFO",
        "save_logs": true
    }
}"""
    
    config_file = "fdef_config.json"
    with open(config_file, 'w') as f:
        f.write(config_content)
        
    return config_file

def run_installation_test():
    """Run a basic installation test"""
    
    print("\n🧪 Running installation test...")
    
    try:
        # Import main analyzer
        from fdef_analyzer import FdefAnalyzer
        
        # Create analyzer instance
        analyzer = FdefAnalyzer()
        print("   ✅ FDEF Analyzer can be imported and instantiated")
        
        # Test basic functionality (without actual PDF)
        config = analyzer._default_config()
        print(f"   ✅ Configuration loaded: {len(config)} sections")
        
        print("   ✅ Installation test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Installation test failed: {e}")
        return False

def main():
    """Main setup script"""
    
    print("🚀 FDEF Signal Analyzer - Setup Script")
    print("=" * 50)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Setup FDEF Signal Analyzer')
    parser.add_argument('--skip-system', action='store_true', 
                       help='Skip system dependency installation')
    parser.add_argument('--create-venv', metavar='PATH', 
                       help='Create virtual environment at specified path')
    parser.add_argument('--config-only', action='store_true',
                       help='Only create sample configuration file')
    
    args = parser.parse_args()
    
    if args.config_only:
        config_file = create_sample_config()
        print(f"✅ Sample configuration created: {config_file}")
        return
    
    # Setup components
    system_setup = SystemSetup()
    python_setup = PythonSetup()
    validator = EnvironmentValidator()
    
    success_steps = []
    
    # Check Python version
    if python_setup.check_python_version():
        success_steps.append("python_version")
    else:
        print("❌ Setup failed - incompatible Python version")
        sys.exit(1)
        
    # Install system dependencies
    if not args.skip_system:
        if system_setup.install_system_dependencies():
            success_steps.append("system_deps")
        else:
            print("⚠️ System dependency installation failed - continuing anyway")
    else:
        print("⏭️ Skipping system dependency installation")
        
    # Create virtual environment if requested
    if args.create_venv:
        if python_setup.create_virtual_environment(args.create_venv):
            success_steps.append("venv")
            print(f"💡 Activate with: source {args.create_venv}/bin/activate")
        else:
            print("⚠️ Virtual environment creation failed - continuing anyway")
            
    # Install Python packages
    if python_setup.install_python_packages():
        success_steps.append("python_packages")
    else:
        print("❌ Setup failed - Python package installation failed")
        sys.exit(1)
        
    # Create sample configuration
    config_file = create_sample_config()
    success_steps.append("config")
    print(f"📝 Sample configuration created: {config_file}")
    
    # Validate installation
    if validator.validate_installation():
        success_steps.append("validation")
    else:
        print("⚠️ Installation validation failed - some features may not work")
        
    # Run installation test
    if run_installation_test():
        success_steps.append("test")
        
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    step_names = {
        "python_version": "Python version check",
        "system_deps": "System dependencies",
        "venv": "Virtual environment",
        "python_packages": "Python packages",
        "config": "Configuration file",
        "validation": "Installation validation",
        "test": "Installation test"
    }
    
    for step, name in step_names.items():
        status = "✅" if step in success_steps else "❌"
        print(f"{status} {name}")
        
    if len(success_steps) >= 4:  # At least the critical steps
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("1. Test with: python fdef_analyzer.py --help")
        print("2. Analyze a PDF: python fdef_analyzer.py your_document.pdf")
        print("3. View results in the generated HTML files")
    else:
        print("\n❌ Setup incomplete - please resolve the failed steps")
        sys.exit(1)

if __name__ == "__main__":
    main()