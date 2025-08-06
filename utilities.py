#!/usr/bin/env python3
"""
FDEF Analyzer Utilities
Maintenance, troubleshooting, and system validation utilities.
"""

import os
import sys
import subprocess
import platform
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile

def check_system_requirements() -> Dict[str, Any]:
    """
    Comprehensive system requirements check
    """
    
    print("🔍 Checking System Requirements...")
    print("=" * 50)
    
    results = {
        'python': {'status': 'unknown', 'details': {}},
        'tesseract': {'status': 'unknown', 'details': {}},
        'poppler': {'status': 'unknown', 'details': {}},
        'packages': {'status': 'unknown', 'details': {}},
        'system': {'status': 'unknown', 'details': {}},
        'overall': 'unknown'
    }
    
    # Check Python version
    try:
        python_version = sys.version_info
        python_executable = sys.executable
        
        if python_version >= (3, 8):
            results['python']['status'] = 'ok'
            print(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            results['python']['status'] = 'error'
            print(f"❌ Python: {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)")
            
        results['python']['details'] = {
            'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'executable': python_executable,
            'architecture': platform.architecture()[0]
        }
        
    except Exception as e:
        results['python']['status'] = 'error'
        results['python']['details'] = {'error': str(e)}
        print(f"❌ Python check failed: {e}")
    
    # Check Tesseract OCR
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
        
        results['tesseract']['status'] = 'ok'
        results['tesseract']['details'] = {
            'version': str(version),
            'command': tesseract_cmd
        }
        print(f"✅ Tesseract OCR: {version}")
        
        # Test OCR functionality
        test_result = test_tesseract_functionality()
        results['tesseract']['details']['test_result'] = test_result
        
    except Exception as e:
        results['tesseract']['status'] = 'error'
        results['tesseract']['details'] = {'error': str(e)}
        print(f"❌ Tesseract OCR: {e}")
    
    # Check Poppler utilities
    try:
        from pdf2image import convert_from_path
        
        # Try to get poppler version
        poppler_version = get_poppler_version()
        
        results['poppler']['status'] = 'ok'
        results['poppler']['details'] = {
            'version': poppler_version,
            'pdf2image_available': True
        }
        print(f"✅ Poppler: {poppler_version}")
        
        # Test PDF conversion functionality
        test_result = test_poppler_functionality()
        results['poppler']['details']['test_result'] = test_result
        
    except Exception as e:
        results['poppler']['status'] = 'error'
        results['poppler']['details'] = {'error': str(e)}
        print(f"❌ Poppler: {e}")
    
    # Check Python packages
    package_results = check_python_packages()
    results['packages'] = package_results
    
    if package_results['status'] == 'ok':
        print(f"✅ Python packages: {len(package_results['details']['installed'])} installed")
    else:
        missing = package_results['details'].get('missing', [])
        print(f"❌ Python packages: {len(missing)} missing")
    
    # Check system resources
    system_info = check_system_resources()
    results['system'] = system_info
    
    if system_info['status'] == 'ok':
        print(f"✅ System resources: {system_info['details']['memory_mb']:.0f}MB RAM available")
    else:
        print(f"⚠️ System resources: Limited resources detected")
    
    # Overall status
    statuses = [results[key]['status'] for key in ['python', 'tesseract', 'poppler', 'packages']]
    if all(status == 'ok' for status in statuses):
        results['overall'] = 'ok'
        print(f"\n🎉 All requirements satisfied!")
    elif any(status == 'error' for status in statuses):
        results['overall'] = 'error'
        print(f"\n❌ Some requirements not met")
    else:
        results['overall'] = 'warning'
        print(f"\n⚠️ Some requirements have warnings")
    
    return results

def test_tesseract_functionality() -> Dict[str, Any]:
    """Test Tesseract OCR functionality with a simple image"""
    
    try:
        import numpy as np
        import cv2
        import pytesseract
        
        # Create a simple test image with text
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, 'TEST TEXT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Perform OCR
        text = pytesseract.image_to_string(gray).strip()
        
        success = 'TEST' in text.upper()
        
        return {
            'success': success,
            'extracted_text': text,
            'expected_text': 'TEST TEXT'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def test_poppler_functionality() -> Dict[str, Any]:
    """Test Poppler PDF conversion functionality"""
    
    try:
        from pdf2image import convert_from_path
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a simple test PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            c = canvas.Canvas(temp_pdf.name, pagesize=letter)
            c.drawString(100, 750, "Test PDF Content")
            c.save()
            
            # Try to convert PDF to image
            images = convert_from_path(temp_pdf.name, dpi=150, first_page=1, last_page=1)
            
            # Clean up
            os.unlink(temp_pdf.name)
            
            success = len(images) > 0
            
            return {
                'success': success,
                'images_converted': len(images),
                'image_size': images[0].size if images else None
            }
            
    except ImportError:
        # reportlab not available, create minimal test
        try:
            # Just test that pdf2image can be imported
            from pdf2image import convert_from_path
            return {
                'success': True,
                'note': 'Import successful, full test requires reportlab'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_poppler_version() -> str:
    """Get Poppler version information"""
    
    try:
        # Try different commands to get poppler version
        commands = [
            ['pdfinfo', '-v'],
            ['pdftoppm', '-v'],
            ['pdftotext', '-v']
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or 'poppler' in result.stderr.lower():
                    output = result.stderr if result.stderr else result.stdout
                    # Extract version from output
                    for line in output.split('\n'):
                        if 'poppler' in line.lower():
                            return line.strip()
                    return "Available (version unknown)"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
                
        return "Available (version detection failed)"
        
    except Exception:
        return "Unknown"

def check_python_packages() -> Dict[str, Any]:
    """Check availability of required Python packages"""
    
    required_packages = [
        'numpy', 'opencv-python', 'pillow', 'pytesseract',
        'PyPDF2', 'pdf2image', 'sympy', 'networkx',
        'fuzzywuzzy', 'matplotlib', 'plotly'
    ]
    
    optional_packages = [
        'scipy', 'pandas', 'scikit-image', 'scikit-learn',
        'joblib', 'numba', 'tqdm', 'click'
    ]
    
    installed = []
    missing = []
    versions = {}
    
    # Check required packages
    for package in required_packages:
        try:
            # Handle package name mappings
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'pillow':
                import_name = 'PIL'
            elif package == 'PyPDF2':
                import_name = 'PyPDF2'
                
            module = __import__(import_name)
            installed.append(package)
            
            # Try to get version
            version = getattr(module, '__version__', 'unknown')
            versions[package] = version
            
        except ImportError:
            missing.append(package)
    
    # Check optional packages
    optional_installed = []
    for package in optional_packages:
        try:
            __import__(package)
            optional_installed.append(package)
        except ImportError:
            pass
    
    status = 'ok' if not missing else 'error'
    
    return {
        'status': status,
        'details': {
            'installed': installed,
            'missing': missing,
            'optional_installed': optional_installed,
            'versions': versions,
            'total_required': len(required_packages),
            'total_installed': len(installed)
        }
    }

def check_system_resources() -> Dict[str, Any]:
    """Check system resources (memory, disk space)"""
    
    try:
        import psutil
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_mb = memory.available / (1024 * 1024)
        
        # Disk space information
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Determine status
        status = 'ok'
        warnings = []
        
        if memory_mb < 1000:  # Less than 1GB available
            status = 'warning'
            warnings.append('Low available memory')
            
        if disk_free_gb < 1:  # Less than 1GB free
            status = 'warning'
            warnings.append('Low disk space')
            
        if cpu_percent > 90:  # High CPU usage
            status = 'warning'
            warnings.append('High CPU usage')
        
        return {
            'status': status,
            'details': {
                'memory_mb': memory_mb,
                'memory_total_mb': memory.total / (1024 * 1024),
                'disk_free_gb': disk_free_gb,
                'disk_total_gb': disk.total / (1024 * 1024 * 1024),
                'cpu_count': cpu_count,
                'cpu_percent': cpu_percent,
                'warnings': warnings
            }
        }
        
    except ImportError:
        # psutil not available, use basic checks
        try:
            # Basic memory check on Unix systems
            if platform.system() != 'Windows':
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemAvailable:' in line:
                            memory_kb = int(line.split()[1])
                            memory_mb = memory_kb / 1024
                            
                            status = 'ok' if memory_mb > 1000 else 'warning'
                            
                            return {
                                'status': status,
                                'details': {
                                    'memory_mb': memory_mb,
                                    'note': 'Limited system info (psutil not available)'
                                }
                            }
        except:
            pass
            
        return {
            'status': 'unknown',
            'details': {
                'note': 'Cannot determine system resources (psutil not available)'
            }
        }

def diagnose_common_issues() -> Dict[str, Any]:
    """Diagnose common installation and runtime issues"""
    
    print("🔧 Diagnosing Common Issues...")
    print("=" * 40)
    
    issues = []
    
    # Issue 1: Tesseract path problems
    try:
        import pytesseract
        tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
        
        if not shutil.which(tesseract_cmd):
            issues.append({
                'issue': 'Tesseract command not found in PATH',
                'severity': 'error',
                'solution': f'Add Tesseract directory to PATH or set pytesseract.pytesseract.tesseract_cmd',
                'details': f'Current tesseract_cmd: {tesseract_cmd}'
            })
            
    except Exception as e:
        issues.append({
            'issue': 'Tesseract import failed',
            'severity': 'error',
            'solution': 'Install pytesseract package: pip install pytesseract',
            'details': str(e)
        })
    
    # Issue 2: Poppler path problems
    try:
        from pdf2image import convert_from_path
        
        # Try to detect poppler installation
        poppler_commands = ['pdfinfo', 'pdftoppm', 'pdftotext']
        missing_commands = []
        
        for cmd in poppler_commands:
            if not shutil.which(cmd):
                missing_commands.append(cmd)
                
        if missing_commands:
            issues.append({
                'issue': 'Poppler utilities not found in PATH',
                'severity': 'error',
                'solution': 'Install poppler-utils package for your system',
                'details': f'Missing commands: {", ".join(missing_commands)}'
            })
            
    except Exception as e:
        issues.append({
            'issue': 'pdf2image import failed',
            'severity': 'error',
            'solution': 'Install pdf2image package: pip install pdf2image',
            'details': str(e)
        })
    
    # Issue 3: Memory limitations
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        if memory.available < 512 * 1024 * 1024:  # Less than 512MB
            issues.append({
                'issue': 'Low available memory',
                'severity': 'warning',
                'solution': 'Close other applications or reduce PDF processing DPI',
                'details': f'Available: {memory.available / (1024*1024):.0f}MB'
            })
            
    except ImportError:
        pass
    
    # Issue 4: OpenCV issues
    try:
        import cv2
        # Test basic OpenCV functionality
        test_img = cv2.imread('nonexistent.jpg')  # Should return None, not crash
        
    except Exception as e:
        issues.append({
            'issue': 'OpenCV import or functionality issue',
            'severity': 'error',
            'solution': 'Reinstall opencv-python: pip uninstall opencv-python && pip install opencv-python',
            'details': str(e)
        })
    
    # Issue 5: Permission problems
    temp_dir = tempfile.gettempdir()
    try:
        test_file = Path(temp_dir) / 'fdef_test_write.tmp'
        test_file.write_text('test')
        test_file.unlink()
    except Exception as e:
        issues.append({
            'issue': 'File system permission problems',
            'severity': 'warning',
            'solution': 'Check write permissions in temporary directory',
            'details': f'Temp dir: {temp_dir}, Error: {str(e)}'
        })
    
    # Print diagnostic results
    if not issues:
        print("✅ No common issues detected")
    else:
        for issue in issues:
            severity_icon = "❌" if issue['severity'] == 'error' else "⚠️"
            print(f"{severity_icon} {issue['issue']}")
            print(f"   Solution: {issue['solution']}")
            if 'details' in issue:
                print(f"   Details: {issue['details']}")
            print()
    
    return {
        'issues_found': len(issues),
        'issues': issues
    }

def run_performance_benchmark() -> Dict[str, Any]:
    """Run performance benchmark to test system capabilities"""
    
    print("⚡ Running Performance Benchmark...")
    print("=" * 40)
    
    results = {}
    
    # Test 1: Image processing performance
    try:
        import cv2
        import numpy as np
        import time
        
        print("🖼️ Testing image processing performance...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (1000, 1500, 3), dtype=np.uint8)
        
        start_time = time.time()
        
        # Perform typical image operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        processing_time = time.time() - start_time
        
        results['image_processing'] = {
            'time_seconds': processing_time,
            'image_size': test_image.shape,
            'contours_found': len(contours),
            'performance_rating': 'good' if processing_time < 1.0 else 'slow'
        }
        
        print(f"   ✅ Image processing: {processing_time:.2f}s")
        
    except Exception as e:
        results['image_processing'] = {'error': str(e)}
        print(f"   ❌ Image processing test failed: {e}")
    
    # Test 2: Mathematical computation performance
    try:
        import sympy as sp
        import time
        
        print("🧮 Testing mathematical computation performance...")
        
        start_time = time.time()
        
        # Create and manipulate symbolic expressions
        x, y, z = sp.symbols('x y z')
        expr1 = x**2 + y**2 + z**2
        expr2 = sp.sin(x) + sp.cos(y) + sp.tan(z)
        combined = expr1 * expr2
        expanded = sp.expand(combined)
        simplified = sp.simplify(expanded)
        
        computation_time = time.time() - start_time
        
        results['mathematical_computation'] = {
            'time_seconds': computation_time,
            'expression_length': len(str(simplified)),
            'performance_rating': 'good' if computation_time < 2.0 else 'slow'
        }
        
        print(f"   ✅ Mathematical computation: {computation_time:.2f}s")
        
    except Exception as e:
        results['mathematical_computation'] = {'error': str(e)}
        print(f"   ❌ Mathematical computation test failed: {e}")
    
    # Test 3: Graph processing performance
    try:
        import networkx as nx
        import time
        
        print("🕸️ Testing graph processing performance...")
        
        start_time = time.time()
        
        # Create and analyze a graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for i in range(100):
            G.add_node(f"signal_{i}")
            
            if i > 0:
                G.add_edge(f"signal_{i-1}", f"signal_{i}")
                
            # Add some random connections
            if i > 10:
                import random
                target = f"signal_{random.randint(0, i-5)}"
                G.add_edge(f"signal_{i}", target)
        
        # Perform graph analysis
        density = nx.density(G)
        connected_components = list(nx.weakly_connected_components(G))
        shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        
        graph_time = time.time() - start_time
        
        results['graph_processing'] = {
            'time_seconds': graph_time,
            'nodes': len(G.nodes),
            'edges': len(G.edges), 
            'density': density,
            'components': len(connected_components),
            'performance_rating': 'good' if graph_time < 1.0 else 'slow'
        }
        
        print(f"   ✅ Graph processing: {graph_time:.2f}s")
        
    except Exception as e:
        results['graph_processing'] = {'error': str(e)}
        print(f"   ❌ Graph processing test failed: {e}")
    
    # Overall performance rating
    successful_tests = [test for test in results.values() if 'error' not in test]
    
    if successful_tests:
        avg_time = sum(test['time_seconds'] for test in successful_tests) / len(successful_tests)
        
        if avg_time < 1.0:
            overall_rating = 'excellent'
        elif avg_time < 2.0:
            overall_rating = 'good'
        elif avg_time < 5.0:
            overall_rating = 'acceptable'
        else:
            overall_rating = 'slow'
            
        results['overall'] = {
            'rating': overall_rating,
            'average_time': avg_time,
            'successful_tests': len(successful_tests),
            'total_tests': len(results)
        }
        
        print(f"\n📊 Overall performance: {overall_rating.upper()}")
        print(f"   Average test time: {avg_time:.2f}s")
        
    return results

def cleanup_temporary_files():
    """Clean up temporary files created by FDEF analyzer"""
    
    print("🧹 Cleaning up temporary files...")
    
    cleanup_locations = [
        tempfile.gettempdir(),
        Path.cwd() / "fdef_analysis_output",
        Path.cwd() / "*.log",
        Path.cwd() / "*.tmp"
    ]
    
    cleaned_files = []
    
    for location in cleanup_locations:
        if isinstance(location, str):
            location = Path(location)
            
        try:
            if location.is_file():
                location.unlink()
                cleaned_files.append(str(location))
            elif location.is_dir():
                # Look for FDEF-related temp files
                for temp_file in location.glob("fdef_*"):
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_files.append(str(temp_file))
            else:
                # Glob pattern
                for file_path in Path.cwd().glob(str(location.name)):
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_files.append(str(file_path))
                        
        except Exception as e:
            print(f"   ⚠️ Could not clean {location}: {e}")
    
    if cleaned_files:
        print(f"   ✅ Cleaned {len(cleaned_files)} temporary files")
        for file_path in cleaned_files[:5]:  # Show first 5
            print(f"      {Path(file_path).name}")
        if len(cleaned_files) > 5:
            print(f"      ... and {len(cleaned_files) - 5} more")
    else:
        print("   ✅ No temporary files found to clean")

def generate_system_report() -> str:
    """Generate comprehensive system report"""
    
    print("📋 Generating System Report...")
    
    # Gather all information
    system_check = check_system_requirements()
    diagnostics = diagnose_common_issues()
    
    # Try to run benchmark (may be resource intensive)
    try:
        benchmark = run_performance_benchmark()
    except Exception as e:
        benchmark = {'error': f'Benchmark failed: {e}'}
    
    # Create report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture(),
            'processor': platform.processor(),
        },
        'requirements_check': system_check,
        'diagnostics': diagnostics,
        'performance_benchmark': benchmark,
        'recommendations': []
    }
    
    # Generate recommendations
    if system_check['overall'] != 'ok':
        report['recommendations'].append('Address system requirement issues before using FDEF analyzer')
    
    if diagnostics['issues_found'] > 0:
        error_issues = [issue for issue in diagnostics['issues'] if issue['severity'] == 'error']
        if error_issues:
            report['recommendations'].append('Fix critical errors found in diagnostics')
    
    if 'overall' in benchmark and benchmark['overall']['rating'] in ['slow', 'acceptable']:
        report['recommendations'].append('Consider reducing processing parameters for better performance')
    
    if not report['recommendations']:
        report['recommendations'].append('System appears ready for FDEF analysis')
    
    # Save report
    report_file = f"fdef_system_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ System report saved: {report_file}")
    
    return report_file

def main():
    """Main utility function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='FDEF Analyzer Utilities')
    parser.add_argument('--check', action='store_true',
                       help='Check system requirements')
    parser.add_argument('--diagnose', action='store_true',
                       help='Diagnose common issues')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up temporary files')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive system report')
    parser.add_argument('--all', action='store_true',
                       help='Run all utilities')
    
    args = parser.parse_args()
    
    print("🛠️ FDEF Analyzer Utilities")
    print("=" * 50)
    
    if args.all or not any([args.check, args.diagnose, args.benchmark, args.cleanup, args.report]):
        # Run all utilities
        check_system_requirements()
        print()
        diagnose_common_issues()
        print()
        run_performance_benchmark()
        print()
        cleanup_temporary_files()
        print()
        generate_system_report()
        
    else:
        if args.check:
            check_system_requirements()
            
        if args.diagnose:
            diagnose_common_issues()
            
        if args.benchmark:
            run_performance_benchmark()
            
        if args.cleanup:
            cleanup_temporary_files()
            
        if args.report:
            generate_system_report()
    
    print("\n✅ Utilities completed")

if __name__ == "__main__":
    main()