#!/usr/bin/env python3
"""
FDEF Analyzer - Main Analysis Engine
Comprehensive Function Design and Engineering Flow analysis for automotive PDFs.
Orchestrates all analysis modules to extract signal dependencies and create visualizations.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import argparse

# Import our analysis modules
from core_io import PdfLoader, PageData, validate_tesseract_installation
from cv_arrows import CvArrowDetector, ArrowDetection, BlockDetection
from sympy_parser import SymPyEquationParser, ParsedEquation, LookupTable
from domain_templates import AutomotiveTemplateMatcher, AutomotiveMatch
from graph_builder import DependencyGraphBuilder, SignalNode, DependencyEdge
from export_html import HtmlNetworkExporter, create_summary_dashboard

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResults:
    """Container for complete analysis results"""
    pages: List[PageData]
    equations: List[ParsedEquation]
    lookup_tables: List[LookupTable]
    cv_arrows: List[ArrowDetection]
    cv_blocks: List[BlockDetection]
    template_matches: List[AutomotiveMatch]
    dependency_graph: Any  # nx.DiGraph
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]

class FdefAnalyzer:
    """
    Main FDEF Analysis Engine
    Orchestrates all analysis modules for comprehensive signal dependency extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.analysis_start_time = None
        
        # Initialize analysis modules
        self.pdf_loader = None
        self.cv_detector = CvArrowDetector()
        self.equation_parser = SymPyEquationParser()
        self.template_matcher = AutomotiveTemplateMatcher()
        self.graph_builder = DependencyGraphBuilder()
        self.html_exporter = HtmlNetworkExporter()
        
        # Results storage
        self.results = None
        
        logger.info("🚀 FDEF Analyzer initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        """Initialize default configuration parameters"""
        
        return {
            'pdf_processing': {
                'dpi': 200,
                'enhance_images': True,
                'max_pages': None
            },
            'computer_vision': {
                'arrow_detection_threshold': 0.6,
                'block_detection_threshold': 0.7,
                'enable_template_matching': True,
                'enable_line_detection': True,
                'enable_contour_analysis': True
            },
            'equation_parsing': {
                'confidence_threshold': 0.5,
                'enable_automotive_functions': True,
                'parse_lookup_tables': True
            },
            'template_matching': {
                'enable_a2l_patterns': True,
                'enable_autosar_patterns': True,
                'enable_can_patterns': True,
                'enable_diagnostic_patterns': True,
                'fuzzy_matching_threshold': 0.8
            },
            'graph_building': {
                'merge_similar_signals': True,
                'resolve_cross_references': True,
                'confidence_weight_threshold': 0.4
            },
            'output': {
                'create_html_visualization': True,
                'create_summary_dashboard': True,
                'save_intermediate_results': False,
                'output_directory': 'fdef_analysis_output'
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True
            }
        }
        
    def analyze_pdf(self, pdf_path: str, target_signal: Optional[str] = None) -> AnalysisResults:
        """
        Perform complete FDEF analysis on a PDF document
        
        Args:
            pdf_path: Path to the PDF file to analyze
            target_signal: Optional target signal to focus analysis on
            
        Returns:
            AnalysisResults containing all extracted information
        """
        
        self.analysis_start_time = time.time()
        
        logger.info(f"🔬 Starting FDEF analysis of: {Path(pdf_path).name}")
        
        # Validate prerequisites
        if not self._validate_prerequisites():
            raise RuntimeError("Prerequisites not met - cannot proceed with analysis")
            
        # Phase 1: Load and preprocess PDF
        logger.info("📖 Phase 1: PDF Loading and Preprocessing")
        pages = self._load_pdf(pdf_path)
        
        # Phase 2: Computer vision analysis
        logger.info("👁️ Phase 2: Computer Vision Analysis")
        cv_arrows, cv_blocks = self._analyze_computer_vision(pages)
        
        # Phase 3: Mathematical equation parsing
        logger.info("🧮 Phase 3: Mathematical Equation Parsing")
        equations, lookup_tables = self._parse_equations(pages)
        
        # Phase 4: Automotive template matching
        logger.info("🚗 Phase 4: Automotive Template Matching")
        template_matches = self._match_templates(pages)
        
        # Phase 5: Dependency graph construction
        logger.info("🕸️ Phase 5: Dependency Graph Construction")
        dependency_graph = self._build_dependency_graph(
            cv_arrows, cv_blocks, equations, template_matches, pages, target_signal
        )
        
        # Phase 6: Statistics and metadata collection
        logger.info("📊 Phase 6: Statistics Collection")
        statistics, metadata = self._collect_statistics(
            pages, equations, cv_arrows, cv_blocks, template_matches, dependency_graph
        )
        
        # Create results object
        self.results = AnalysisResults(
            pages=pages,
            equations=equations,
            lookup_tables=lookup_tables,
            cv_arrows=cv_arrows,
            cv_blocks=cv_blocks,
            template_matches=template_matches,
            dependency_graph=dependency_graph,
            statistics=statistics,
            metadata=metadata
        )
        
        total_time = time.time() - self.analysis_start_time
        logger.info(f"✅ Analysis complete in {total_time:.2f} seconds")
        
        return self.results
        
    def export_results(self, output_dir: str, target_signal: Optional[str] = None) -> Dict[str, str]:
        """
        Export analysis results to various formats
        
        Args:
            output_dir: Directory to save results
            target_signal: Optional target signal for focused visualization
            
        Returns:
            Dictionary mapping output type to file path
        """
        
        if not self.results:
            raise ValueError("No analysis results available - run analyze_pdf() first")
            
        logger.info(f"📤 Exporting results to: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export interactive HTML network visualization
        if self.config['output']['create_html_visualization']:
            html_file = output_path / "signal_dependency_network.html"
            
            if self.results.dependency_graph.nodes:
                self.html_exporter.export_interactive_network(
                    self.results.dependency_graph,
                    str(html_file),
                    "FDEF Signal Dependency Network",
                    target_signal
                )
                exported_files['network_visualization'] = str(html_file)
                logger.info(f"   📊 Interactive network: {html_file}")
            else:
                logger.warning("   ⚠️ No dependency graph to visualize")
                
        # Export summary dashboard
        if self.config['output']['create_summary_dashboard']:
            dashboard_file = output_path / "analysis_summary.html"
            
            dashboard_data = {
                'statistics': self.results.statistics,
                'metadata': self.results.metadata
            }
            
            create_summary_dashboard(dashboard_data, str(dashboard_file))
            exported_files['summary_dashboard'] = str(dashboard_file)
            logger.info(f"   📋 Summary dashboard: {dashboard_file}")
            
        # Export detailed analysis report
        report_file = output_path / "detailed_analysis_report.json"
        self._export_detailed_report(str(report_file))
        exported_files['detailed_report'] = str(report_file)
        logger.info(f"   📄 Detailed report: {report_file}")
        
        # Export signal list
        signals_file = output_path / "extracted_signals.txt"
        self._export_signal_list(str(signals_file))
        exported_files['signal_list'] = str(signals_file)
        logger.info(f"   📝 Signal list: {signals_file}")
        
        # Save intermediate results if requested
        if self.config['output']['save_intermediate_results']:
            intermediate_dir = output_path / "intermediate"
            self._save_intermediate_results(str(intermediate_dir))
            exported_files['intermediate_results'] = str(intermediate_dir)
            logger.info(f"   🔍 Intermediate results: {intermediate_dir}")
            
        logger.info(f"✅ Export complete - {len(exported_files)} files created")
        return exported_files
        
    def get_signal_analysis(self, signal_name: str) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific signal
        
        Args:
            signal_name: Name of the signal to analyze
            
        Returns:
            Dictionary containing detailed signal analysis
        """
        
        if not self.results:
            raise ValueError("No analysis results available")
            
        if signal_name not in self.results.dependency_graph.nodes:
            return {'error': f'Signal "{signal_name}" not found in dependency graph'}
            
        # Get signal dependencies
        dependencies = self.graph_builder.get_signal_dependencies(signal_name)
        
        # Get node properties
        node_data = self.results.dependency_graph.nodes[signal_name]
        
        # Find equations involving this signal
        related_equations = []
        for eq in self.results.equations:
            if signal_name in eq.variables or signal_name == eq.output_variable:
                related_equations.append({
                    'equation': eq.raw_text,
                    'type': eq.equation_type.value,
                    'confidence': eq.confidence,
                    'role': 'output' if signal_name == eq.output_variable else 'input'
                })
                
        # Find template matches
        related_templates = []
        for match in self.results.template_matches:
            if signal_name in str(match.extracted_data) or signal_name in match.text_content:
                related_templates.append({
                    'type': match.block_type.value,
                    'confidence': match.confidence,
                    'data': match.extracted_data
                })
                
        return {
            'signal_name': signal_name,
            'properties': node_data,
            'dependencies': dependencies,
            'related_equations': related_equations,
            'related_templates': related_templates,
            'analysis_confidence': node_data.get('confidence', 0.0)
        }
        
    def _validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are available"""
        
        # Check Tesseract installation
        if not validate_tesseract_installation():
            logger.error("Tesseract OCR not found - please install it first")
            return False
            
        # Check required Python packages
        required_packages = [
            'cv2', 'numpy', 'sympy', 'networkx', 'PIL', 'fuzzywuzzy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
                
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            return False
            
        return True
        
    def _load_pdf(self, pdf_path: str) -> List[PageData]:
        """Load and preprocess PDF document"""
        
        self.pdf_loader = PdfLoader(
            pdf_path,
            dpi=self.config['pdf_processing']['dpi'],
            enhance_images=self.config['pdf_processing']['enhance_images']
        )
        
        pages = self.pdf_loader.load_pages(
            max_pages=self.config['pdf_processing']['max_pages']
        )
        
        logger.info(f"   Loaded {len(pages)} pages from PDF")
        return pages
        
    def _analyze_computer_vision(self, pages: List[PageData]) -> Tuple[List[ArrowDetection], List[BlockDetection]]:
        """Perform computer vision analysis on all pages"""
        
        all_arrows = []
        all_blocks = []
        
        for page in pages:
            # Detect arrows
            if (self.config['computer_vision']['enable_template_matching'] or
                self.config['computer_vision']['enable_line_detection'] or
                self.config['computer_vision']['enable_contour_analysis']):
                
                arrows = self.cv_detector.detect_arrow_paths(page.processed_image)
                all_arrows.extend(arrows)
                
            # Detect blocks
            blocks = self.cv_detector.detect_blocks(page.processed_image)
            all_blocks.extend(blocks)
            
        logger.info(f"   Detected {len(all_arrows)} arrows and {len(all_blocks)} blocks")
        return all_arrows, all_blocks
        
    def _parse_equations(self, pages: List[PageData]) -> Tuple[List[ParsedEquation], List[LookupTable]]:
        """Parse mathematical equations from all pages"""
        
        all_equations = []
        all_tables = []
        
        # Combine text from all pages
        combined_text = '\n'.join([page.text_content for page in pages])
        
        # Parse equations
        equations = self.equation_parser.parse_equations(combined_text)
        
        # Filter by confidence threshold
        threshold = self.config['equation_parsing']['confidence_threshold']
        filtered_equations = [eq for eq in equations if eq.confidence >= threshold]
        
        all_equations.extend(filtered_equations)
        
        # Parse lookup tables if enabled
        if self.config['equation_parsing']['parse_lookup_tables']:
            tables = self.equation_parser.extract_lookup_tables(combined_text)
            all_tables.extend(tables)
            
        logger.info(f"   Parsed {len(all_equations)} equations and {len(all_tables)} lookup tables")
        return all_equations, all_tables
        
    def _match_templates(self, pages: List[PageData]) -> List[AutomotiveMatch]:
        """Perform automotive template matching on all pages"""
        
        all_matches = []
        
        # Combine text from all pages
        combined_text = '\n'.join([page.text_content for page in pages])
        
        # Perform template matching
        matches = self.template_matcher.match_automotive_blocks(combined_text)
        
        # Filter by confidence threshold
        threshold = self.config['template_matching']['fuzzy_matching_threshold']
        filtered_matches = [match for match in matches if match.confidence >= threshold]
        
        all_matches.extend(filtered_matches)
        
        logger.info(f"   Found {len(all_matches)} template matches")
        return all_matches
        
    def _build_dependency_graph(self, arrows: List[ArrowDetection], blocks: List[BlockDetection],
                               equations: List[ParsedEquation], templates: List[AutomotiveMatch],
                               pages: List[PageData], target_signal: Optional[str]) -> Any:
        """Build comprehensive dependency graph"""
        
        # Extract page texts
        page_texts = [page.text_content for page in pages]
        
        # Build graph
        graph = self.graph_builder.build_comprehensive_graph(
            cv_arrows=arrows,
            cv_blocks=blocks,
            equations=equations,
            template_matches=templates,
            page_texts=page_texts
        )
        
        logger.info(f"   Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
        
    def _collect_statistics(self, pages: List[PageData], equations: List[ParsedEquation],
                           arrows: List[ArrowDetection], blocks: List[BlockDetection],
                           templates: List[AutomotiveMatch], graph: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Collect comprehensive statistics and metadata"""
        
        # Basic statistics
        statistics = {
            'signals_detected': len(graph.nodes) if graph else 0,
            'dependencies_mapped': len(graph.edges) if graph else 0,
            'pages_processed': len(pages),
            'equations_found': len(equations),
            'arrows_detected': len(arrows),
            'blocks_detected': len(blocks),
            'template_matches': len(templates),
            'average_confidence': 0.0
        }
        
        # Calculate average confidence
        if graph and graph.nodes:
            confidences = [data.get('confidence', 0.0) for _, data in graph.nodes(data=True)]
            statistics['average_confidence'] = sum(confidences) / len(confidences)
            
        # Graph analysis
        if graph:
            graph_stats = self.graph_builder.analyze_graph_properties()
            statistics.update(graph_stats)
            
        # Metadata
        metadata = {
            'analysis_time': time.time() - self.analysis_start_time,
            'analyzer_version': '1.0.0',
            'config_used': self.config,
            'pdf_statistics': self.pdf_loader.get_page_statistics() if self.pdf_loader else {},
            'equation_statistics': self.equation_parser.get_parsing_statistics(equations),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return statistics, metadata
        
    def _export_detailed_report(self, output_file: str):
        """Export detailed analysis report as JSON"""
        
        # Convert results to serializable format
        report_data = {
            'summary': {
                'total_signals': len(self.results.dependency_graph.nodes),
                'total_dependencies': len(self.results.dependency_graph.edges),
                'analysis_time': self.results.metadata['analysis_time'],
                'confidence': self.results.statistics['average_confidence']
            },
            'statistics': self.results.statistics,
            'metadata': self.results.metadata,
            'signals': [],
            'equations': [],
            'template_matches': []
        }
        
        # Add signal information
        for node, data in self.results.dependency_graph.nodes(data=True):
            signal_info = {
                'name': node,
                'type': data.get('signal_type', 'unknown'),
                'confidence': data.get('confidence', 0.0),
                'sources': data.get('sources', []),
                'properties': {k: v for k, v in data.items() if k not in ['signal_type', 'confidence', 'sources']}
            }
            report_data['signals'].append(signal_info)
            
        # Add equation information
        for eq in self.results.equations:
            eq_info = {
                'output_variable': eq.output_variable,
                'equation': eq.raw_text,
                'type': eq.equation_type.value,
                'confidence': eq.confidence,
                'variables': list(eq.variables),
                'functions': eq.functions
            }
            report_data['equations'].append(eq_info)
            
        # Add template match information
        for match in self.results.template_matches:
            match_info = {
                'type': match.block_type.value,
                'confidence': match.confidence,
                'content': match.text_content[:200],  # Truncate long content
                'extracted_data': match.extracted_data
            }
            report_data['template_matches'].append(match_info)
            
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
    def _export_signal_list(self, output_file: str):
        """Export simple list of detected signals"""
        
        signals = []
        
        for node, data in self.results.dependency_graph.nodes(data=True):
            signal_type = data.get('signal_type', 'unknown')
            confidence = data.get('confidence', 0.0)
            sources = data.get('sources', [])
            
            signals.append(f"{node}\t{signal_type}\t{confidence:.3f}\t{','.join(sources)}")
            
        # Sort by confidence (highest first)
        signals.sort(key=lambda x: float(x.split('\t')[2]), reverse=True)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Signal_Name\tType\tConfidence\tSources\n")
            f.write("\n".join(signals))
            
    def _save_intermediate_results(self, output_dir: str):
        """Save intermediate analysis results for debugging"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed images
        if self.pdf_loader:
            image_dir = output_path / "processed_images"
            self.pdf_loader.save_processed_images(str(image_dir))
            
        # Save CV detection results
        if self.results.cv_arrows or self.results.cv_blocks:
            cv_file = output_path / "cv_detections.json"
            cv_data = {
                'arrows': [
                    {
                        'start': arrow.start_point,
                        'end': arrow.end_point,
                        'direction': arrow.direction.value,
                        'confidence': arrow.confidence
                    }
                    for arrow in self.results.cv_arrows
                ],
                'blocks': [
                    {
                        'center': block.center_point,
                        'type': block.block_type,
                        'confidence': block.confidence,
                        'text': block.text_content
                    }
                    for block in self.results.cv_blocks
                ]
            }
            
            with open(cv_file, 'w') as f:
                json.dump(cv_data, f, indent=2)

def setup_logging(log_level: str = 'INFO', save_logs: bool = True, log_file: Optional[str] = None):
    """Set up logging configuration"""
    
    # Configure logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
        
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=[console_handler],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler if requested
    if save_logs:
        if not log_file:
            log_file = f"fdef_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")

def main():
    """Main command-line interface"""
    
    parser = argparse.ArgumentParser(
        description='FDEF Analyzer - Automotive Signal Dependency Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s document.pdf --target-signal Engine_RPM
  %(prog)s document.pdf --output results/ --config custom_config.json
        """
    )
    
    parser.add_argument('pdf_file', help='PDF file to analyze')
    parser.add_argument('--target-signal', help='Target signal to focus analysis on')
    parser.add_argument('--output', default='fdef_analysis_output', 
                       help='Output directory (default: fdef_analysis_output)')
    parser.add_argument('--config', help='Configuration file (JSON format)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--no-html', action='store_true', 
                       help='Skip HTML visualization generation')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate analysis results')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, save_logs=True)
    
    try:
        # Load configuration
        config = None
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
                
        # Create analyzer
        analyzer = FdefAnalyzer(config)
        
        # Override config with command line arguments
        if args.no_html:
            analyzer.config['output']['create_html_visualization'] = False
            
        if args.save_intermediate:
            analyzer.config['output']['save_intermediate_results'] = True
            
        # Validate PDF file
        if not Path(args.pdf_file).exists():
            logger.error(f"PDF file not found: {args.pdf_file}")
            sys.exit(1)
            
        # Perform analysis
        logger.info(f"🚀 Starting FDEF analysis...")
        results = analyzer.analyze_pdf(args.pdf_file, args.target_signal)
        
        # Export results
        exported_files = analyzer.export_results(args.output, args.target_signal)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        print(f"📄 Document: {Path(args.pdf_file).name}")
        print(f"🔍 Signals detected: {results.statistics['signals_detected']}")
        print(f"🕸️ Dependencies mapped: {results.statistics['dependencies_mapped']}")
        print(f"⏱️ Analysis time: {results.metadata['analysis_time']:.2f} seconds")
        print(f"📊 Average confidence: {results.statistics['average_confidence']:.1%}")
        
        print(f"\n📤 Results exported to: {args.output}")
        for output_type, file_path in exported_files.items():
            print(f"   {output_type}: {Path(file_path).name}")
            
        if 'network_visualization' in exported_files:
            print(f"\n🌐 Open the interactive visualization:")
            print(f"   {exported_files['network_visualization']}")
            
        logger.info("✅ Analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()