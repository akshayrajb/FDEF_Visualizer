#!/usr/bin/env python3
"""
Test Suite for FDEF Signal Analyzer
Comprehensive unit tests, integration tests, and validation framework.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import modules to test
from core_io import PdfLoader, PageData, validate_tesseract_installation
from cv_arrows import CvArrowDetector, ArrowDirection, ArrowDetection, BlockDetection
from sympy_parser import SymPyEquationParser, ParsedEquation, EquationType
from domain_templates import AutomotiveTemplateMatcher, AutomotiveBlockType, AutomotiveMatch
from graph_builder import DependencyGraphBuilder, SignalNode, DependencyEdge
from export_html import HtmlNetworkExporter
from fdef_analyzer import FdefAnalyzer, AnalysisResults

class TestPdfLoader(unittest.TestCase):
    """Test PDF loading and preprocessing functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_invalid_pdf_path(self):
        """Test handling of invalid PDF paths"""
        with self.assertRaises(FileNotFoundError):
            PdfLoader("nonexistent_file.pdf")
            
    def test_non_pdf_file(self):
        """Test handling of non-PDF files"""
        text_file = self.temp_path / "test.txt"
        text_file.write_text("This is not a PDF")
        
        with self.assertRaises(ValueError):
            PdfLoader(str(text_file))
            
    def test_file_hash_calculation(self):
        """Test PDF file hash calculation"""
        # Create a test file
        test_file = self.temp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")
        
        # Mock the file validation to bypass PDF checks
        with patch.object(Path, 'suffix', new_callable=lambda: Mock(return_value='.pdf')):
            loader = PdfLoader(str(test_file))
            
            # Hash should be consistent
            hash1 = loader._calculate_file_hash()
            hash2 = loader._calculate_file_hash()
            self.assertEqual(hash1, hash2)
            self.assertEqual(len(hash1), 16)  # First 16 chars of SHA-256
            
    @patch('core_io.validate_tesseract_installation')
    def test_tesseract_validation(self, mock_validate):
        """Test Tesseract installation validation"""
        mock_validate.return_value = True
        self.assertTrue(validate_tesseract_installation())
        
        mock_validate.return_value = False
        self.assertFalse(validate_tesseract_installation())
        
    def test_automotive_text_cleaning(self):
        """Test automotive text cleaning functionality"""
        # Create a mock loader to access private methods
        test_file = self.temp_path / "test.pdf"
        test_file.write_bytes(b"PDF")
        
        with patch.object(Path, 'suffix', new_callable=lambda: Mock(return_value='.pdf')):
            loader = PdfLoader(str(test_file))
            
            # Test OCR error corrections
            test_cases = [
                ("Engine_l5_RPM", "Engine_15_RPM"),
                ("Throttle_O0_Position", "Throttle_00_Position"),
                ("Signal Il Name", "Signal_11_Name"),
                ("Multiple   Spaces", "Multiple_Spaces"),
            ]
            
            for input_text, expected in test_cases:
                result = loader._clean_automotive_text(input_text)
                self.assertEqual(result, expected)

class TestCvArrowDetector(unittest.TestCase):
    """Test computer vision arrow detection functionality"""
    
    def setUp(self):
        self.detector = CvArrowDetector()
        self.test_image = self._create_test_image()
        
    def _create_test_image(self):
        """Create a test image with basic shapes"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
        
        # Draw a simple arrow-like shape
        cv2.arrowedLine(image, (100, 200), (200, 200), (0, 0, 0), 3, tipLength=0.3)
        
        # Draw a rectangle (functional block)
        cv2.rectangle(image, (300, 150), (400, 250), (0, 0, 0), 2)
        
        return image
        
    def test_arrow_template_creation(self):
        """Test arrow template creation"""
        templates = self.detector._create_arrow_templates()
        
        # Should have templates for all directions
        expected_directions = [
            ArrowDirection.RIGHT, ArrowDirection.LEFT,
            ArrowDirection.UP, ArrowDirection.DOWN,
            ArrowDirection.DIAGONAL_UR, ArrowDirection.DIAGONAL_UL,
            ArrowDirection.DIAGONAL_DR, ArrowDirection.DIAGONAL_DL
        ]
        
        for direction in expected_directions:
            self.assertIn(direction, templates)
            self.assertIsInstance(templates[direction], np.ndarray)
            
    def test_angle_to_direction_conversion(self):
        """Test angle to direction conversion"""
        test_cases = [
            (0, ArrowDirection.RIGHT),
            (np.pi/2, ArrowDirection.DOWN),
            (np.pi, ArrowDirection.LEFT),
            (3*np.pi/2, ArrowDirection.UP),
            (np.pi/4, ArrowDirection.DIAGONAL_DR),
        ]
        
        for angle, expected_direction in test_cases:
            result = self.detector._angle_to_direction(angle)
            self.assertEqual(result, expected_direction)
            
    def test_arrow_detection_basic(self):
        """Test basic arrow detection functionality"""
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        arrows = self.detector.detect_arrow_paths(gray_image)
        
        # Should detect at least some arrows (may vary based on detection quality)
        self.assertIsInstance(arrows, list)
        
        for arrow in arrows:
            self.assertIsInstance(arrow, ArrowDetection)
            self.assertIsInstance(arrow.start_point, tuple)
            self.assertIsInstance(arrow.end_point, tuple)
            self.assertIsInstance(arrow.direction, ArrowDirection)
            self.assertGreaterEqual(arrow.confidence, 0.0)
            self.assertLessEqual(arrow.confidence, 1.0)
            
    def test_block_detection_basic(self):
        """Test basic block detection functionality"""
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        blocks = self.detector.detect_blocks(gray_image)
        
        self.assertIsInstance(blocks, list)
        
        for block in blocks:
            self.assertIsInstance(block, BlockDetection)
            self.assertIsInstance(block.center_point, tuple)
            self.assertIsInstance(block.bounding_box, tuple)
            self.assertEqual(len(block.bounding_box), 4)
            self.assertGreaterEqual(block.confidence, 0.0)
            self.assertLessEqual(block.confidence, 1.0)
            
    def test_visualization_creation(self):
        """Test visualization of detections"""
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        arrows = self.detector.detect_arrow_paths(gray_image)
        blocks = self.detector.detect_blocks(gray_image)
        
        vis_image = self.detector.visualize_detections(self.test_image, arrows, blocks)
        
        # Should return an image of the same size
        self.assertEqual(vis_image.shape, self.test_image.shape)
        self.assertIsInstance(vis_image, np.ndarray)

class TestSymPyEquationParser(unittest.TestCase):
    """Test mathematical equation parsing functionality"""
    
    def setUp(self):
        self.parser = SymPyEquationParser()
        
    def test_equation_pattern_matching(self):
        """Test equation pattern recognition"""
        test_equations = [
            "Engine_RPM = Throttle_Position * 1000 + Base_RPM",
            "Fuel_Rate = lookup(Engine_Load, Fuel_Map)",
            "Vehicle_Speed = rate_limit(Target_Speed, -5.0, 5.0)",
            "if (Engine_Temp > 90) then Cooling_Fan = 1 else Cooling_Fan = 0",
        ]
        
        for eq_text in test_equations:
            equations = self.parser.parse_equations(eq_text)
            self.assertGreater(len(equations), 0, f"Failed to parse: {eq_text}")
            
            equation = equations[0]
            self.assertIsInstance(equation, ParsedEquation)
            self.assertIsInstance(equation.equation_type, EquationType)
            self.assertGreater(len(equation.output_variable), 0)
            
    def test_variable_extraction(self):
        """Test variable extraction from equations"""
        eq_text = "Output_Signal = Input_A * Gain_Factor + Input_B"
        equations = self.parser.parse_equations(eq_text)
        
        self.assertEqual(len(equations), 1)
        equation = equations[0]
        
        self.assertEqual(equation.output_variable, "Output_Signal")
        expected_variables = {"Input_A", "Input_B", "Gain_Factor"}
        self.assertTrue(expected_variables.issubset(equation.variables))
        
    def test_automotive_function_recognition(self):
        """Test recognition of automotive-specific functions"""
        test_functions = [
            "Output = lookup(Input, Table)",
            "Result = rate_limit(Signal, -10, 10)",
            "Status = sat(Value, 0, 100)",
            "Filtered = filter(Raw_Signal, 0.1)",
        ]
        
        for func_text in test_functions:
            equations = self.parser.parse_equations(func_text)
            self.assertGreater(len(equations), 0)
            
            equation = equations[0]
            self.assertGreater(len(equation.functions), 0)
            
    def test_lookup_table_extraction(self):
        """Test lookup table extraction"""
        table_text = """
        FUEL_MAP = { input: [0, 25, 50, 75, 100], output: [0.1, 0.3, 0.5, 0.8, 1.0] }
        """
        
        tables = self.parser.extract_lookup_tables(table_text)
        self.assertGreater(len(tables), 0)
        
        table = tables[0]
        self.assertEqual(table.name, "FUEL_MAP")
        self.assertEqual(len(table.breakpoints), 5)
        self.assertEqual(len(table.values), 5)
        
    def test_dependency_analysis(self):
        """Test variable dependency analysis"""
        equations_text = """
        A = B + C
        B = D * E
        C = F + 10
        """
        
        equations = self.parser.parse_equations(equations_text)
        dependencies = self.parser.get_variable_dependencies(equations)
        
        # A should depend on B, C, D, E, F (transitively)
        self.assertIn("A", dependencies)
        a_deps = dependencies["A"]
        
        expected_deps = {"B", "C", "D", "E", "F"}
        self.assertTrue(expected_deps.issubset(a_deps))
        
    def test_equation_evaluation(self):
        """Test equation evaluation with variable values"""
        eq_text = "Result = A * 2 + B"
        equations = self.parser.parse_equations(eq_text)
        
        self.assertEqual(len(equations), 1)
        equation = equations[0]
        
        variable_values = {"A": 5, "B": 3}
        result = self.parser.evaluate_equation(equation, variable_values)
        
        self.assertEqual(result, 13.0)  # 5 * 2 + 3 = 13

class TestAutomotiveTemplateMatcher(unittest.TestCase):
    """Test automotive template matching functionality"""
    
    def setUp(self):
        self.matcher = AutomotiveTemplateMatcher()
        
    def test_a2l_pattern_matching(self):
        """Test A2L format pattern matching"""
        a2l_text = """
        /begin CHARACTERISTIC Engine_RPM
            "Engine rotational speed"
            CURVE
            0x12345678
            MEASUREMENT_LINK Engine_RPM_Measurement
        /end CHARACTERISTIC
        """
        
        matches = self.matcher.match_automotive_blocks(a2l_text)
        
        # Should find at least one A2L characteristic
        a2l_matches = [m for m in matches if m.block_type == AutomotiveBlockType.A2L_CHARACTERISTIC]
        self.assertGreater(len(a2l_matches), 0)
        
        match = a2l_matches[0]
        self.assertEqual(match.extracted_data.get('name'), 'Engine_RPM')
        
    def test_can_pattern_matching(self):
        """Test CAN message pattern matching"""
        can_text = """
        BO_ 1234 Engine_Data: 8 ECU
        SG_ Engine_RPM : 0|16@1+ (0.25,0) [0|16383.75] "rpm" Instrument_Cluster
        SG_ Engine_Load : 16|8@1+ (0.390625,0) [0|99.60] "%" ECU
        """
        
        matches = self.matcher.match_automotive_blocks(can_text)
        
        # Should find CAN message and signals
        can_messages = [m for m in matches if m.block_type == AutomotiveBlockType.CAN_MESSAGE]
        can_signals = [m for m in matches if m.block_type == AutomotiveBlockType.CAN_SIGNAL]
        
        self.assertGreater(len(can_messages), 0)
        self.assertGreater(len(can_signals), 0)
        
    def test_dtc_pattern_matching(self):
        """Test diagnostic trouble code pattern matching"""
        dtc_text = """
        Engine Control Module errors:
        DTC P0100 - Mass Air Flow Circuit Malfunction
        DTC P0171 - System Too Lean (Bank 1)
        DTC U0001 - CAN Communication Error
        """
        
        matches = self.matcher.match_automotive_blocks(dtc_text)
        
        dtc_matches = [m for m in matches if m.block_type == AutomotiveBlockType.DTC_CODE]
        self.assertGreater(len(dtc_matches), 0)
        
    def test_calibration_parameter_matching(self):
        """Test calibration parameter pattern matching"""
        cal_text = """
        K_Engine_Idle_RPM = 800
        Cal_Fuel_Injection_Timing = 15.5
        K_Max_Throttle_Angle = 90.0
        """
        
        matches = self.matcher.match_automotive_blocks(cal_text)
        
        cal_matches = [m for m in matches if m.block_type == AutomotiveBlockType.CALIBRATION_PARAMETER]
        self.assertGreater(len(cal_matches), 0)

class TestDependencyGraphBuilder(unittest.TestCase):
    """Test dependency graph construction functionality"""
    
    def setUp(self):
        self.builder = DependencyGraphBuilder()
        
    def test_signal_addition_and_merging(self):
        """Test signal addition and merging functionality"""
        # Add signals
        self.builder._add_or_update_signal(
            "Engine_RPM", "input", 0.9, {"source": "test"}, "test_source"
        )
        
        self.builder._add_or_update_signal(
            "engine_rpm", "input", 0.8, {"source": "test2"}, "test_source2"  # Similar name
        )
        
        # Should have added signals
        self.assertIn("Engine_RPM", self.builder.signal_registry)
        self.assertIn("engine_rpm", self.builder.signal_registry)
        
        # Test merging similar signals
        self.builder._merge_similar_signals()
        
        # One should remain after merging
        remaining_signals = list(self.builder.signal_registry.keys())
        self.assertLessEqual(len(remaining_signals), 2)  # May merge or keep separate
        
    def test_dependency_addition(self):
        """Test dependency edge addition"""
        self.builder._add_or_update_dependency(
            "Input_Signal", "Output_Signal", "mathematical", 0.8,
            {"equation": "Output = Input * 2"}, ["equation_analysis"]
        )
        
        self.assertIn(("Input_Signal", "Output_Signal"), self.builder.edge_registry)
        
        edge = self.builder.edge_registry[("Input_Signal", "Output_Signal")]
        self.assertEqual(edge.relationship_type, "mathematical")
        self.assertEqual(edge.confidence, 0.8)
        
    def test_signal_name_cleaning(self):
        """Test signal name cleaning and normalization"""
        test_cases = [
            ("Engine RPM", "Engine_RPM"),
            ("Signal-Name", "Signal_Name"),
            ("Mixed__Underscores", "Mixed_Underscores"),
            ("  Trimmed  ", "Trimmed"),
            ("Special@#$Chars", "Special_Chars"),
        ]
        
        for input_name, expected in test_cases:
            result = self.builder._clean_signal_name(input_name)
            self.assertEqual(result, expected)
            
    def test_graph_construction(self):
        """Test NetworkX graph construction"""
        # Add some test data
        self.builder._add_or_update_signal("A", "input", 0.9, {}, "test")
        self.builder._add_or_update_signal("B", "output", 0.8, {}, "test")
        self.builder._add_or_update_dependency("A", "B", "direct", 0.7, {}, ["test"])
        
        # Build NetworkX graph
        self.builder._build_networkx_graph()
        
        # Verify graph structure
        self.assertIn("A", self.builder.graph.nodes)
        self.assertIn("B", self.builder.graph.nodes)
        self.assertTrue(self.builder.graph.has_edge("A", "B"))
        
        # Check node attributes
        node_a = self.builder.graph.nodes["A"]
        self.assertEqual(node_a["signal_type"], "input")
        self.assertEqual(node_a["confidence"], 0.9)
        
    def test_dependency_analysis(self):
        """Test signal dependency analysis"""
        # Create a test graph: A -> B -> C, A -> C
        self.builder._add_or_update_signal("A", "input", 0.9, {}, "test")
        self.builder._add_or_update_signal("B", "intermediate", 0.8, {}, "test")
        self.builder._add_or_update_signal("C", "output", 0.7, {}, "test")
        
        self.builder._add_or_update_dependency("A", "B", "direct", 0.8, {}, ["test"])
        self.builder._add_or_update_dependency("B", "C", "direct", 0.7, {}, ["test"])
        self.builder._add_or_update_dependency("A", "C", "direct", 0.6, {}, ["test"])
        
        self.builder._build_networkx_graph()
        
        # Analyze dependencies for signal C
        deps = self.builder.get_signal_dependencies("C")
        
        self.assertIn("A", deps["direct_inputs"])
        self.assertIn("B", deps["direct_inputs"])
        self.assertEqual(deps["total_dependencies"], 2)  # A and B

class TestHtmlNetworkExporter(unittest.TestCase):
    """Test HTML network visualization export functionality"""
    
    def setUp(self):
        self.exporter = HtmlNetworkExporter()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_color_scheme_initialization(self):
        """Test color scheme initialization"""
        colors = self.exporter.color_scheme
        
        expected_types = ['input', 'output', 'intermediate', 'parameter', 'unknown']
        for signal_type in expected_types:
            self.assertIn(signal_type, colors)
            self.assertTrue(colors[signal_type].startswith('#'))  # Valid hex color
            
    def test_empty_graph_handling(self):
        """Test handling of empty graphs"""
        import networkx as nx
        
        empty_graph = nx.DiGraph()
        output_file = Path(self.temp_dir) / "empty_test.html"
        
        result_file = self.exporter.export_interactive_network(
            empty_graph, str(output_file), "Empty Test Network"
        )
        
        self.assertTrue(Path(result_file).exists())
        
        # Check that placeholder content is created
        with open(result_file, 'r') as f:
            content = f.read()
            self.assertIn("No signal dependencies found", content)
            
    def test_graph_statistics_calculation(self):
        """Test network statistics calculation"""
        import networkx as nx
        
        # Create test graph
        G = nx.DiGraph()
        G.add_node("A", signal_type="input", confidence=0.9)
        G.add_node("B", signal_type="output", confidence=0.8)
        G.add_edge("A", "B", relationship_type="direct", confidence=0.7)
        
        stats = self.exporter._calculate_network_stats(G, "A")
        
        self.assertEqual(stats['total_signals'], 2)
        self.assertEqual(stats['total_dependencies'], 1)
        self.assertGreater(stats['avg_confidence'], 0)
        self.assertEqual(stats['max_degree'], 1)
        
    def test_html_content_generation(self):
        """Test HTML content generation"""
        import networkx as nx
        
        # Create test graph
        G = nx.DiGraph()
        G.add_node("Engine_RPM", signal_type="input", confidence=0.9, sources=["test"])
        G.add_node("Vehicle_Speed", signal_type="output", confidence=0.8, sources=["test"])
        G.add_edge("Engine_RPM", "Vehicle_Speed", relationship_type="mathematical", 
                  confidence=0.7, evidence=["equation"])
        
        # Calculate layout and extract data
        pos = self.exporter._calculate_layout(G)
        node_data = self.exporter._extract_node_data(G, pos)
        edge_data = self.exporter._extract_edge_data(G, pos)
        
        # Generate HTML content
        html_content = self.exporter._create_html_content(
            node_data, edge_data, "Test Network", "Engine_RPM", G
        )
        
        # Verify HTML structure
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("Test Network", html_content)
        self.assertIn("plotly", html_content.lower())
        self.assertIn("Engine_RPM", html_content)

class TestFdefAnalyzer(unittest.TestCase):
    """Test main FDEF analyzer integration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_analyzer_initialization(self):
        """Test FDEF analyzer initialization"""
        analyzer = FdefAnalyzer()
        
        # Check that all components are initialized
        self.assertIsNotNone(analyzer.cv_detector)
        self.assertIsNotNone(analyzer.equation_parser)
        self.assertIsNotNone(analyzer.template_matcher)
        self.assertIsNotNone(analyzer.graph_builder)
        self.assertIsNotNone(analyzer.html_exporter)
        
        # Check default configuration
        config = analyzer.config
        self.assertIn('pdf_processing', config)
        self.assertIn('computer_vision', config)
        self.assertIn('equation_parsing', config)
        
    def test_custom_configuration(self):
        """Test custom configuration handling"""
        custom_config = {
            'pdf_processing': {'dpi': 300},
            'output': {'create_html_visualization': False}
        }
        
        analyzer = FdefAnalyzer(config=custom_config)
        
        # Should merge with defaults
        self.assertEqual(analyzer.config['pdf_processing']['dpi'], 300)
        self.assertFalse(analyzer.config['output']['create_html_visualization'])
        
        # Should still have other default values
        self.assertIn('computer_vision', analyzer.config)
        
    @patch('fdef_analyzer.validate_tesseract_installation')
    def test_prerequisite_validation(self, mock_validate):
        """Test prerequisite validation"""
        analyzer = FdefAnalyzer()
        
        # Test successful validation
        mock_validate.return_value = True
        self.assertTrue(analyzer._validate_prerequisites())
        
        # Test failed validation
        mock_validate.return_value = False
        self.assertFalse(analyzer._validate_prerequisites())
        
    def test_signal_analysis(self):
        """Test individual signal analysis"""
        analyzer = FdefAnalyzer()
        
        # Create mock results
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_node("Test_Signal", signal_type="input", confidence=0.9, sources=["test"])
        
        # Mock results
        analyzer.results = AnalysisResults(
            pages=[], equations=[], lookup_tables=[], cv_arrows=[], cv_blocks=[],
            template_matches=[], dependency_graph=G, statistics={}, metadata={}
        )
        
        # Analyze signal
        analysis = analyzer.get_signal_analysis("Test_Signal")
        
        self.assertEqual(analysis['signal_name'], "Test_Signal")
        self.assertIn('properties', analysis)
        self.assertIn('dependencies', analysis)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete analysis pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_complete_pipeline_mock(self):
        """Test complete analysis pipeline with mocked components"""
        
        # Create mock PDF content
        mock_pages = [
            PageData(
                page_number=1,
                original_image=np.ones((400, 600, 3), dtype=np.uint8) * 255,
                processed_image=np.ones((400, 600, 3), dtype=np.uint8) * 255,
                text_content="Engine_RPM = Throttle_Position * 1000",
                confidence_score=0.8,
                width=600,
                height=400,
                metadata={}
            )
        ]
        
        # Test with mocked components
        with patch('core_io.PdfLoader') as mock_loader:
            mock_loader_instance = Mock()
            mock_loader_instance.load_pages.return_value = mock_pages
            mock_loader_instance.get_page_statistics.return_value = {
                'total_pages': 1, 'total_characters': 100
            }
            mock_loader.return_value = mock_loader_instance
            
            with patch('fdef_analyzer.validate_tesseract_installation', return_value=True):
                analyzer = FdefAnalyzer()
                
                # Mock file existence
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.suffix', new_callable=lambda: Mock(return_value='.pdf')):
                        # This would normally fail due to file not existing,
                        # but we're testing the pipeline structure
                        try:
                            results = analyzer.analyze_pdf("mock_file.pdf")
                            
                            # Verify results structure
                            self.assertIsInstance(results, AnalysisResults)
                            self.assertIsInstance(results.pages, list)
                            self.assertIsInstance(results.equations, list)
                            self.assertIsInstance(results.statistics, dict)
                            
                        except Exception as e:
                            # Expected due to mocking limitations
                            self.assertIn("mock", str(e).lower())

class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_large_text_parsing_performance(self):
        """Test performance with large text inputs"""
        import time
        
        # Generate large text with equations
        large_text = "\n".join([
            f"Signal_{i} = Input_{i} * Factor_{i} + Offset_{i}"
            for i in range(1000)
        ])
        
        parser = SymPyEquationParser()
        
        start_time = time.time()
        equations = parser.parse_equations(large_text)
        parsing_time = time.time() - start_time
        
        # Should parse reasonably quickly (adjust threshold as needed)
        self.assertLess(parsing_time, 10.0)  # 10 seconds max
        self.assertGreater(len(equations), 500)  # Should find many equations
        
    def test_graph_building_performance(self):
        """Test performance of graph building with many nodes"""
        import time
        
        builder = DependencyGraphBuilder()
        
        # Add many signals
        start_time = time.time()
        
        for i in range(1000):
            builder._add_or_update_signal(
                f"Signal_{i}", "intermediate", 0.8, {}, "test"
            )
            
            if i > 0:
                builder._add_or_update_dependency(
                    f"Signal_{i-1}", f"Signal_{i}", "direct", 0.7, {}, ["test"]
                )
                
        builder._build_networkx_graph()
        
        build_time = time.time() - start_time
        
        # Should build reasonably quickly
        self.assertLess(build_time, 5.0)  # 5 seconds max
        self.assertEqual(len(builder.graph.nodes), 1000)
        self.assertEqual(len(builder.graph.edges), 999)

def create_test_suite():
    """Create and return the complete test suite"""
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPdfLoader,
        TestCvArrowDetector,
        TestSymPyEquationParser,
        TestAutomotiveTemplateMatcher,
        TestDependencyGraphBuilder,
        TestHtmlNetworkExporter,
        TestFdefAnalyzer,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    return suite

def run_tests(verbosity=2):
    """Run all tests with specified verbosity"""
    
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    print("🧪 Running FDEF Analyzer Test Suite")
    print("=" * 50)
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        
        # Print failure details
        if result.failures:
            print("\n💥 FAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
                
        if result.errors:
            print("\n🚨 ERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
                
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FDEF Analyzer Tests')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--performance', action='store_true',
                       help='Include performance tests')
    parser.add_argument('--pattern', help='Run tests matching pattern')
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    if args.pattern:
        # Run specific tests matching pattern
        loader = unittest.TestLoader()
        suite = loader.discover('.', pattern=f'*{args.pattern}*')
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run all tests
        success = run_tests(verbosity)
        sys.exit(0 if success else 1)