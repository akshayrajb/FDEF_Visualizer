#!/usr/bin/env python3
"""
FDEF Analyzer Examples
Comprehensive examples demonstrating various usage scenarios and configurations.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fdef_analyzer import FdefAnalyzer, setup_logging
import logging

def example_basic_analysis():
    """
    Example 1: Basic PDF analysis with default settings
    """
    
    print("=" * 60)
    print("📖 Example 1: Basic PDF Analysis")
    print("=" * 60)
    
    # Setup logging
    setup_logging('INFO', save_logs=False)
    
    # Create analyzer with default configuration
    analyzer = FdefAnalyzer()
    
    # Example PDF file (you would replace this with your actual PDF)
    pdf_file = "example_fdef_document.pdf"
    
    if not Path(pdf_file).exists():
        print(f"⚠️ Example PDF not found: {pdf_file}")
        print("   Please provide a valid PDF file for this example")
        return
        
    try:
        # Perform analysis
        print(f"🔬 Analyzing: {pdf_file}")
        results = analyzer.analyze_pdf(pdf_file)
        
        # Print summary
        print(f"\n📊 Analysis Results:")
        print(f"   Signals detected: {results.statistics['signals_detected']}")
        print(f"   Dependencies mapped: {results.statistics['dependencies_mapped']}")
        print(f"   Analysis time: {results.metadata['analysis_time']:.2f} seconds")
        print(f"   Average confidence: {results.statistics['average_confidence']:.1%}")
        
        # Export results
        output_dir = "basic_analysis_output"
        exported_files = analyzer.export_results(output_dir)
        
        print(f"\n📤 Results exported to: {output_dir}")
        for file_type, file_path in exported_files.items():
            print(f"   {file_type}: {Path(file_path).name}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        

def example_targeted_analysis():
    """
    Example 2: Targeted analysis focusing on a specific signal
    """
    
    print("=" * 60)
    print("🎯 Example 2: Targeted Signal Analysis")
    print("=" * 60)
    
    setup_logging('INFO', save_logs=False)
    
    # Custom configuration for targeted analysis
    config = {
        'pdf_processing': {
            'dpi': 250,  # Higher DPI for better accuracy
            'enhance_images': True
        },
        'computer_vision': {
            'arrow_detection_threshold': 0.5,  # Lower threshold for more detections
            'enable_template_matching': True,
            'enable_line_detection': True,
            'enable_contour_analysis': True
        },
        'output': {
            'create_html_visualization': True,
            'create_summary_dashboard': True,
            'save_intermediate_results': True  # Save debugging info
        }
    }
    
    analyzer = FdefAnalyzer(config=config)
    
    pdf_file = "vehicle_control_system.pdf"
    target_signal = "Engine_RPM"
    
    if not Path(pdf_file).exists():
        print(f"⚠️ Example PDF not found: {pdf_file}")
        print("   Using mock analysis for demonstration...")
        
        # Create mock results for demonstration
        results = create_mock_analysis_results()
        
    else:
        try:
            print(f"🎯 Analyzing: {pdf_file}")
            print(f"   Target signal: {target_signal}")
            
            results = analyzer.analyze_pdf(pdf_file, target_signal=target_signal)
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return
    
    # Detailed signal analysis
    if results.dependency_graph.has_node(target_signal):
        signal_analysis = analyzer.get_signal_analysis(target_signal)
        
        print(f"\n🔍 Detailed Analysis for '{target_signal}':")
        print(f"   Signal type: {signal_analysis['properties'].get('signal_type', 'unknown')}")
        print(f"   Confidence: {signal_analysis['analysis_confidence']:.2%}")
        print(f"   Direct inputs: {len(signal_analysis['dependencies']['direct_inputs'])}")
        print(f"   Direct outputs: {len(signal_analysis['dependencies']['direct_outputs'])}")
        print(f"   Total dependencies: {signal_analysis['dependencies']['total_dependencies']}")
        
        # Show related equations
        if signal_analysis['related_equations']:
            print(f"\n📐 Related Equations:")
            for eq in signal_analysis['related_equations'][:3]:  # Show first 3
                print(f"   • {eq['equation']} (confidence: {eq['confidence']:.2%})")
                
    # Export with target signal focus
    output_dir = f"targeted_analysis_{target_signal.lower()}"
    exported_files = analyzer.export_results(output_dir, target_signal=target_signal)
    
    print(f"\n📤 Targeted analysis exported to: {output_dir}")
    

def example_batch_processing():
    """
    Example 3: Batch processing multiple PDF files
    """
    
    print("=" * 60)
    print("📚 Example 3: Batch PDF Processing")
    print("=" * 60)
    
    setup_logging('INFO', save_logs=True)
    
    # Configuration optimized for batch processing
    config = {
        'pdf_processing': {
            'dpi': 150,  # Lower DPI for faster processing
            'enhance_images': False,  # Disable enhancement for speed
            'max_pages': 10  # Limit pages for faster processing
        },
        'computer_vision': {
            'arrow_detection_threshold': 0.7,  # Higher threshold for speed
            'enable_contour_analysis': False  # Disable expensive analysis
        },
        'output': {
            'create_html_visualization': True,
            'create_summary_dashboard': False,  # Skip individual dashboards
            'save_intermediate_results': False
        }
    }
    
    analyzer = FdefAnalyzer(config=config)
    
    # List of PDF files to process (replace with your actual files)
    pdf_files = [
        "engine_control_fdef.pdf",
        "transmission_control_fdef.pdf",
        "brake_system_fdef.pdf",
        "body_control_fdef.pdf"
    ]
    
    batch_results = []
    total_start_time = time.time()
    
    print(f"🔄 Processing {len(pdf_files)} PDF files...")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        if not Path(pdf_file).exists():
            print(f"   ⚠️ File {i}/{len(pdf_files)}: {pdf_file} - NOT FOUND")
            continue
            
        try:
            print(f"   📄 File {i}/{len(pdf_files)}: {pdf_file}")
            
            file_start_time = time.time()
            results = analyzer.analyze_pdf(pdf_file)
            file_time = time.time() - file_start_time
            
            # Store results summary
            batch_results.append({
                'file': pdf_file,
                'signals': results.statistics['signals_detected'],
                'dependencies': results.statistics['dependencies_mapped'],
                'confidence': results.statistics['average_confidence'],
                'processing_time': file_time,
                'status': 'success'
            })
            
            # Export results
            output_dir = f"batch_output/{Path(pdf_file).stem}"
            analyzer.export_results(output_dir)
            
            print(f"      ✅ {results.statistics['signals_detected']} signals, "
                  f"{results.statistics['dependencies_mapped']} dependencies "
                  f"({file_time:.1f}s)")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            batch_results.append({
                'file': pdf_file,
                'status': 'failed',
                'error': str(e)
            })
    
    total_time = time.time() - total_start_time
    
    # Generate batch summary
    generate_batch_summary(batch_results, total_time)
    

def example_custom_configuration():
    """
    Example 4: Custom configuration for specific document types
    """
    
    print("=" * 60)
    print("⚙️ Example 4: Custom Configuration")
    print("=" * 60)
    
    # Configuration for high-quality technical documents
    high_quality_config = {
        'pdf_processing': {
            'dpi': 300,  # High DPI for detailed documents
            'enhance_images': True,
            'max_pages': None  # Process all pages
        },
        'computer_vision': {
            'arrow_detection_threshold': 0.4,  # Very sensitive detection
            'block_detection_threshold': 0.5,
            'enable_template_matching': True,
            'enable_line_detection': True,
            'enable_contour_analysis': True
        },
        'equation_parsing': {
            'confidence_threshold': 0.3,  # Accept lower confidence equations
            'enable_automotive_functions': True,
            'parse_lookup_tables': True
        },
        'template_matching': {
            'enable_a2l_patterns': True,
            'enable_autosar_patterns': True,
            'enable_can_patterns': True,
            'enable_diagnostic_patterns': True,
            'fuzzy_matching_threshold': 0.7  # More lenient fuzzy matching
        },
        'graph_building': {
            'merge_similar_signals': True,
            'resolve_cross_references': True,
            'confidence_weight_threshold': 0.3  # Lower threshold
        },
        'output': {
            'create_html_visualization': True,
            'create_summary_dashboard': True,
            'save_intermediate_results': True,
            'output_directory': 'high_quality_analysis'
        },
        'logging': {
            'level': 'DEBUG',  # Detailed logging
            'save_logs': True
        }
    }
    
    # Save configuration to file
    config_file = "high_quality_config.json"
    with open(config_file, 'w') as f:
        json.dump(high_quality_config, f, indent=2)
        
    print(f"📝 Saved configuration to: {config_file}")
    
    # Create analyzer with custom config
    analyzer = FdefAnalyzer(config=high_quality_config)
    
    print(f"⚙️ Analyzer configured with custom settings:")
    print(f"   DPI: {high_quality_config['pdf_processing']['dpi']}")
    print(f"   Arrow threshold: {high_quality_config['computer_vision']['arrow_detection_threshold']}")
    print(f"   Equation threshold: {high_quality_config['equation_parsing']['confidence_threshold']}")
    print(f"   Log level: {high_quality_config['logging']['level']}")
    
    # Example usage (with mock file)
    pdf_file = "technical_specification.pdf"
    
    if Path(pdf_file).exists():
        try:
            results = analyzer.analyze_pdf(pdf_file)
            analyzer.export_results(high_quality_config['output']['output_directory'])
            print(f"✅ High-quality analysis completed")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    else:
        print(f"⚠️ Example file not found: {pdf_file}")
        

def example_programmatic_usage():
    """
    Example 5: Advanced programmatic usage with custom processing
    """
    
    print("=" * 60)
    print("🐍 Example 5: Advanced Programmatic Usage")
    print("=" * 60)
    
    from core_io import PdfLoader
    from cv_arrows import CvArrowDetector
    from sympy_parser import SymPyEquationParser
    from domain_templates import AutomotiveTemplateMatcher
    from graph_builder import DependencyGraphBuilder
    from export_html import HtmlNetworkExporter
    
    # Step-by-step manual processing
    pdf_file = "example_document.pdf"
    
    print("🔧 Manual step-by-step processing:")
    
    if not Path(pdf_file).exists():
        print(f"⚠️ PDF file not found: {pdf_file}")
        print("   Creating synthetic example data...")
        
        # Create synthetic data for demonstration
        create_synthetic_example()
        return
    
    try:
        # Step 1: Load PDF
        print("   📄 Step 1: Loading PDF...")
        loader = PdfLoader(pdf_file, dpi=200, enhance_images=True)
        pages = loader.load_pages(max_pages=5)  # Limit for example
        print(f"      Loaded {len(pages)} pages")
        
        # Step 2: Computer Vision Analysis
        print("   👁️ Step 2: Computer Vision Analysis...")
        cv_detector = CvArrowDetector()
        
        all_arrows = []
        all_blocks = []
        
        for page in pages[:2]:  # Analyze first 2 pages
            arrows = cv_detector.detect_arrow_paths(page.processed_image)
            blocks = cv_detector.detect_blocks(page.processed_image)
            
            all_arrows.extend(arrows)
            all_blocks.extend(blocks)
            
        print(f"      Detected {len(all_arrows)} arrows, {len(all_blocks)} blocks")
        
        # Step 3: Equation Parsing
        print("   🧮 Step 3: Equation Parsing...")
        parser = SymPyEquationParser()
        
        combined_text = '\n'.join([page.text_content for page in pages])
        equations = parser.parse_equations(combined_text)
        lookup_tables = parser.extract_lookup_tables(combined_text)
        
        print(f"      Parsed {len(equations)} equations, {len(lookup_tables)} lookup tables")
        
        # Step 4: Template Matching
        print("   🚗 Step 4: Template Matching...")
        matcher = AutomotiveTemplateMatcher()
        template_matches = matcher.match_automotive_blocks(combined_text)
        
        print(f"      Found {len(template_matches)} template matches")
        
        # Step 5: Graph Construction
        print("   🕸️ Step 5: Graph Construction...")
        builder = DependencyGraphBuilder()
        
        page_texts = [page.text_content for page in pages]
        graph = builder.build_comprehensive_graph(
            cv_arrows=all_arrows,
            cv_blocks=all_blocks,
            equations=equations,
            template_matches=template_matches,
            page_texts=page_texts
        )
        
        print(f"      Built graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Step 6: Custom Analysis
        print("   📊 Step 6: Custom Analysis...")
        
        # Find most connected signals
        node_degrees = [(node, graph.degree(node)) for node in graph.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        print(f"      Most connected signals:")
        for node, degree in node_degrees[:5]:
            print(f"        {node}: {degree} connections")
            
        # Analyze signal types
        signal_types = {}
        for node, data in graph.nodes(data=True):
            signal_type = data.get('signal_type', 'unknown')
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            
        print(f"      Signal type distribution:")
        for signal_type, count in signal_types.items():
            print(f"        {signal_type}: {count}")
            
        # Step 7: Export Results
        print("   📤 Step 7: Export Results...")
        exporter = HtmlNetworkExporter()
        
        output_file = "programmatic_analysis_network.html"
        exporter.export_interactive_network(
            graph, output_file, "Programmatic Analysis Results"
        )
        
        print(f"      Exported visualization: {output_file}")
        
    except Exception as e:
        print(f"❌ Manual processing failed: {e}")
        import traceback
        traceback.print_exc()


def create_mock_analysis_results():
    """Create mock analysis results for demonstration"""
    
    import networkx as nx
    from fdef_analyzer import AnalysisResults
    
    # Create mock graph
    G = nx.DiGraph()
    
    # Add mock nodes
    signals = [
        ("Engine_RPM", "input", 0.9),
        ("Throttle_Position", "input", 0.85),
        ("Vehicle_Speed", "output", 0.8),
        ("Fuel_Rate", "intermediate", 0.75),
        ("Engine_Load", "intermediate", 0.7)
    ]
    
    for signal, signal_type, confidence in signals:
        G.add_node(signal, 
                  signal_type=signal_type, 
                  confidence=confidence,
                  sources=["mock_data"])
                  
    # Add mock edges
    edges = [
        ("Throttle_Position", "Engine_RPM", "direct", 0.8),
        ("Engine_RPM", "Vehicle_Speed", "mathematical", 0.75),
        ("Engine_Load", "Fuel_Rate", "lookup", 0.9),
        ("Fuel_Rate", "Engine_RPM", "feedback", 0.6)
    ]
    
    for source, target, rel_type, confidence in edges:
        G.add_edge(source, target,
                  relationship_type=rel_type,
                  confidence=confidence,
                  evidence=["mock_data"])
    
    # Create mock results
    mock_results = AnalysisResults(
        pages=[],
        equations=[],
        lookup_tables=[],
        cv_arrows=[],
        cv_blocks=[],
        template_matches=[],
        dependency_graph=G,
        statistics={
            'signals_detected': len(G.nodes),
            'dependencies_mapped': len(G.edges),
            'average_confidence': 0.78
        },
        metadata={
            'analysis_time': 2.5,
            'analyzer_version': '1.0.0'
        }
    )
    
    return mock_results


def create_synthetic_example():
    """Create synthetic example for demonstration"""
    
    print("🔧 Creating synthetic analysis example...")
    
    # Create mock signals and relationships
    signals_data = {
        'Engine_RPM': {
            'type': 'input',
            'confidence': 0.92,
            'equation': 'Engine_RPM = Throttle_Position * RPM_Factor + Idle_RPM'
        },
        'Throttle_Position': {
            'type': 'input', 
            'confidence': 0.88,
            'equation': None
        },
        'Vehicle_Speed': {
            'type': 'output',
            'confidence': 0.85,
            'equation': 'Vehicle_Speed = Engine_RPM * Gear_Ratio * Wheel_Factor'
        },
        'Fuel_Injection_Rate': {
            'type': 'intermediate',
            'confidence': 0.79,
            'equation': 'Fuel_Injection_Rate = lookup(Engine_Load, Fuel_Map)'
        }
    }
    
    print(f"   📊 Synthetic signals: {len(signals_data)}")
    
    for signal, data in signals_data.items():
        print(f"      {signal} ({data['type']}) - {data['confidence']:.1%}")
        if data['equation']:
            print(f"        Equation: {data['equation']}")
    
    # Simulate dependency relationships
    dependencies = [
        ('Throttle_Position', 'Engine_RPM', 'mathematical'),
        ('Engine_RPM', 'Vehicle_Speed', 'mathematical'),
        ('Engine_Load', 'Fuel_Injection_Rate', 'lookup'),
        ('Fuel_Injection_Rate', 'Engine_RPM', 'feedback')
    ]
    
    print(f"   🕸️ Dependencies: {len(dependencies)}")
    for source, target, rel_type in dependencies:
        print(f"      {source} → {target} ({rel_type})")


def generate_batch_summary(batch_results, total_time):
    """Generate summary for batch processing"""
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"   Total processing time: {total_time:.1f} seconds")
    
    successful = [r for r in batch_results if r.get('status') == 'success']
    failed = [r for r in batch_results if r.get('status') == 'failed']
    
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    
    if successful:
        total_signals = sum(r['signals'] for r in successful)
        total_deps = sum(r['dependencies'] for r in successful)
        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        
        print(f"\n   Overall Statistics:")
        print(f"     Total signals detected: {total_signals}")
        print(f"     Total dependencies: {total_deps}")
        print(f"     Average confidence: {avg_confidence:.1%}")
        print(f"     Average processing time: {avg_time:.1f}s per file")
        
    # Save detailed summary
    summary_file = "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': {
                'total_files': len(batch_results),
                'successful': len(successful),
                'failed': len(failed),
                'total_time': total_time
            },
            'results': batch_results
        }, f, indent=2)
        
    print(f"\n📄 Detailed summary saved: {summary_file}")


def create_configuration_templates():
    """Create various configuration templates"""
    
    print("=" * 60)
    print("📝 Creating Configuration Templates")
    print("=" * 60)
    
    configs = {
        'fast_processing.json': {
            'description': 'Optimized for speed - good for batch processing',
            'config': {
                'pdf_processing': {
                    'dpi': 150,
                    'enhance_images': False,
                    'max_pages': 10
                },
                'computer_vision': {
                    'arrow_detection_threshold': 0.8,
                    'enable_contour_analysis': False
                },
                'equation_parsing': {
                    'confidence_threshold': 0.6
                },
                'output': {
                    'save_intermediate_results': False
                }
            }
        },
        
        'high_accuracy.json': {
            'description': 'Optimized for accuracy - slower but more thorough',
            'config': {
                'pdf_processing': {
                    'dpi': 300,
                    'enhance_images': True,
                    'max_pages': None
                },
                'computer_vision': {
                    'arrow_detection_threshold': 0.4,
                    'block_detection_threshold': 0.5,
                    'enable_template_matching': True,
                    'enable_line_detection': True,
                    'enable_contour_analysis': True
                },
                'equation_parsing': {
                    'confidence_threshold': 0.3,
                    'enable_automotive_functions': True,
                    'parse_lookup_tables': True
                },
                'template_matching': {
                    'fuzzy_matching_threshold': 0.7
                },
                'output': {
                    'save_intermediate_results': True
                },
                'logging': {
                    'level': 'DEBUG'
                }
            }
        },
        
        'automotive_specific.json': {
            'description': 'Specialized for automotive documents (A2L, AUTOSAR, CAN)',
            'config': {
                'template_matching': {
                    'enable_a2l_patterns': True,
                    'enable_autosar_patterns': True,
                    'enable_can_patterns': True,
                    'enable_diagnostic_patterns': True,
                    'fuzzy_matching_threshold': 0.8
                },
                'equation_parsing': {
                    'enable_automotive_functions': True,
                    'parse_lookup_tables': True
                },
                'graph_building': {
                    'merge_similar_signals': True,
                    'resolve_cross_references': True
                }
            }
        },
        
        'debugging.json': {
            'description': 'For debugging and development - saves all intermediate results',
            'config': {
                'output': {
                    'save_intermediate_results': True,
                    'create_html_visualization': True,
                    'create_summary_dashboard': True
                },
                'logging': {
                    'level': 'DEBUG',
                    'save_logs': True
                }
            }
        }
    }
    
    for filename, template in configs.items():
        with open(filename, 'w') as f:
            json.dump(template['config'], f, indent=2)
            
        print(f"✅ {filename}")
        print(f"   {template['description']}")
        
    print(f"\n📚 Created {len(configs)} configuration templates")
    print("   Use with: python fdef_analyzer.py document.pdf --config <template>.json")


def main():
    """Main function to run examples"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='FDEF Analyzer Examples')
    parser.add_argument('--example', type=int, choices=range(1, 6),
                       help='Run specific example (1-5)')
    parser.add_argument('--all', action='store_true',
                       help='Run all examples')
    parser.add_argument('--configs', action='store_true',
                       help='Create configuration templates only')
    
    args = parser.parse_args()
    
    print("🚀 FDEF Analyzer Examples")
    print("=" * 60)
    
    if args.configs:
        create_configuration_templates()
        return
        
    examples = {
        1: example_basic_analysis,
        2: example_targeted_analysis,
        3: example_batch_processing,
        4: example_custom_configuration,
        5: example_programmatic_usage
    }
    
    if args.example:
        examples[args.example]()
    elif args.all:
        for i in sorted(examples.keys()):
            try:
                examples[i]()
                print()  # Add spacing between examples
            except KeyboardInterrupt:
                print("\n❌ Examples interrupted by user")
                break
            except Exception as e:
                print(f"❌ Example {i} failed: {e}")
                print()
    else:
        print("Available examples:")
        print("  1. Basic PDF Analysis")
        print("  2. Targeted Signal Analysis")
        print("  3. Batch PDF Processing")
        print("  4. Custom Configuration")
        print("  5. Advanced Programmatic Usage")
        print()
        print("Usage:")
        print("  python examples.py --example 1")
        print("  python examples.py --all")
        print("  python examples.py --configs")


if __name__ == "__main__":
    main()