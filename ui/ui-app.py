"""
Main Streamlit Application for FDEF Dependency Visualizer

Web-based interface for:
- File upload (FDEF PDFs and Excel mapping)
- Graph building and caching
- Interactive signal dependency visualization
- Signal search and exploration
"""

import streamlit as st
import logging
from pathlib import Path
import pandas as pd
from pyvis.network import Network
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from parser.mapping_loader import MappingLoader
from parser.pdf_parser import PDFRuleParser
from parser.dependency_graph import DependencyGraph
from ocr_enhanced.diagram_ocr import DiagramOCR

# Configuration
DATA_DIR = Path(__file__).parent.parent / "resources"
CACHE_DIR = DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "dependency_graph.json"

# Streamlit page configuration
st.set_page_config(
    page_title="FDEF Dependency Visualizer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_directories():
    """Initialize required directories."""
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

def save_uploaded_files(uploaded_files, file_type):
    """Save uploaded files to the data directory."""
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
        logger.info(f"Saved {file_type} file: {file_path}")
    
    return saved_paths

def display_graph_statistics(dep_graph):
    """Display graph statistics in a nice format."""
    stats = dep_graph.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", stats.get('nodes', 0))
    
    with col2:
        st.metric("Total Connections", stats.get('edges', 0))
    
    with col3:
        st.metric("Graph Density", f"{stats.get('density', 0):.3f}")
    
    with col4:
        is_dag = stats.get('is_dag', False)
        st.metric("Is DAG", "✅ Yes" if is_dag else "❌ No")

def create_pyvis_network(subgraph, signal_name):
    """Create a PyVis network from a NetworkX subgraph."""
    # Initialize network
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    # Configure physics
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09
            }
        },
        "layout": {
            "hierarchical": {
                "enabled": false
            }
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        }
    }
    """)
    
    # Add nodes with different colors based on their role
    for node, attrs in subgraph.nodes(data=True):
        if node == signal_name:
            color = "#ff6b6b"  # Red for target signal
            size = 30
        elif subgraph.in_degree(node) == 0:
            color = "#4ecdc4"  # Teal for source signals
            size = 20
        elif subgraph.out_degree(node) == 0:
            color = "#45b7d1"  # Blue for leaf signals
            size = 20
        else:
            color = "#96ceb4"  # Green for intermediate signals
            size = 25
        
        # Create title with signal information
        title = f"Signal: {node}\\nIn-degree: {subgraph.in_degree(node)}\\nOut-degree: {subgraph.out_degree(node)}"
        
        net.add_node(
            node,
            label=node,
            title=title,
            color=color,
            size=size
        )
    
    # Add edges with different colors based on operator
    for source, target, edge_data in subgraph.edges(data=True):
        operator = edge_data.get('operator', 'UNKNOWN')
        
        # Color edges based on operator
        if operator == 'AND':
            color = "#ff9999"
        elif operator == 'OR':
            color = "#99ccff"
        elif operator == 'NOT':
            color = "#ffcc99"
        else:
            color = "#cccccc"
        
        title = f"Operator: {operator}"
        
        net.add_edge(
            source,
            target,
            title=title,
            color=color,
            width=2,
            arrows="to"
        )
    
    return net

def main():
    """Main application function."""
    initialize_directories()
    
    # Header
    st.markdown('<div class="main-header">🔗 FDEF Dependency Visualizer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Welcome!</strong> This tool analyzes Mercedes-Benz FDEF (Functional Description) documents 
    to create interactive dependency graphs for signal tracing.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads and configuration
    with st.sidebar:
        st.header("📂 Input Files")
        
        # Excel mapping file upload
        st.subheader("Mapping Sheet")
        mapping_file = st.file_uploader(
            "Upload Excel mapping sheet (columns C, D, F)",
            type=["xlsx", "xls"],
            help="Excel file with Network Line (C), Network Name (D), and Internal Name (F) columns"
        )
        
        # PDF files upload
        st.subheader("FDEF Documents")
        pdf_files = st.file_uploader(
            "Upload FDEF PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="One or more PDF files containing signal definitions and dependencies"
        )
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        
        use_ocr = st.checkbox(
            "Enhanced OCR for scanned PDFs",
            value=True,
            help="Use advanced OCR for better text extraction from scanned documents"
        )
        
        force_rebuild = st.checkbox(
            "Force rebuild graph",
            value=False,
            help="Rebuild the dependency graph even if cache exists"
        )
        
        # Build button
        build_button = st.button("🔨 Build/Rebuild Graph", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Graph building section
        if build_button:
            if not mapping_file or not pdf_files:
                st.error("Please upload both mapping sheet and PDF files before building the graph.")
            else:
                with st.spinner("Building dependency graph... This may take a few minutes."):
                    try:
                        # Save uploaded files
                        mapping_path = DATA_DIR / mapping_file.name
                        with open(mapping_path, "wb") as f:
                            f.write(mapping_file.getbuffer())
                        
                        pdf_paths = save_uploaded_files(pdf_files, "PDF")
                        
                        # Load mapping
                        st.info("📊 Loading mapping sheet...")
                        mapper = MappingLoader(mapping_path)
                        mapping_df = mapper.load()
                        
                        # Validate mapping
                        warnings = mapper.validate_mapping()
                        if warnings:
                            st.warning(f"Mapping validation warnings: {', '.join(warnings)}")
                        
                        # Extract rules from PDFs
                        st.info("📄 Extracting rules from PDF documents...")
                        
                        if use_ocr:
                            # Use enhanced OCR
                            ocr_processor = DiagramOCR()
                            processed_pdfs = []
                            for pdf_path in pdf_paths:
                                processed_path = ocr_processor.process_pdf(pdf_path)
                                processed_pdfs.append(processed_path)
                            parser = PDFRuleParser(processed_pdfs)
                        else:
                            parser = PDFRuleParser(pdf_paths)
                        
                        rules = parser.extract_rules()
                        
                        if not rules:
                            st.warning("No dependency rules found in the PDF documents. Please check the file content.")
                        else:
                            st.success(f"Extracted {len(rules)} dependency rules from {len(pdf_files)} PDF(s)")
                        
                        # Build dependency graph
                        st.info("🕸️ Building dependency graph...")
                        dep_graph = DependencyGraph(cache_file=CACHE_FILE)
                        dep_graph.build_from_rules(rules)
                        dep_graph.save()
                        
                        st.success("✅ Graph built successfully!")
                        
                        # Display statistics
                        display_graph_statistics(dep_graph)
                        
                        # Store in session state
                        st.session_state['graph_built'] = True
                        st.session_state['mapping_loader'] = mapper
                        
                    except Exception as e:
                        st.error(f"Error building graph: {str(e)}")
                        logger.error(f"Graph building error: {e}")
        
        # Signal search and visualization section
        st.markdown('<div class="section-header">🔍 Signal Analysis</div>', unsafe_allow_html=True)
        
        # Check if graph exists
        if not CACHE_FILE.exists() and not st.session_state.get('graph_built', False):
            st.markdown("""
            <div class="warning-box">
            <strong>No dependency graph found.</strong><br>
            Please upload your files and build the graph first using the sidebar.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Load existing graph
            try:
                dep_graph = DependencyGraph(cache_file=CACHE_FILE)
                
                if dep_graph.G.number_of_nodes() == 0:
                    st.warning("The dependency graph is empty. Please rebuild with valid input files.")
                else:
                    # Signal search
                    col_search, col_depth = st.columns([3, 1])
                    
                    with col_search:
                        signal_input = st.text_input(
                            "Enter signal name to analyze:",
                            placeholder="e.g., PT_Rdy, Engine_Start",
                            help="Enter a network name or internal signal name"
                        )
                    
                    with col_depth:
                        depth = st.slider(
                            "Graph depth",
                            min_value=1,
                            max_value=6,
                            value=3,
                            help="Number of dependency levels to show"
                        )
                    
                    # Search suggestions
                    if signal_input:
                        matches = dep_graph.search_signals(signal_input, case_sensitive=False)
                        if matches and len(matches) <= 10:
                            st.info(f"Similar signals found: {', '.join(matches[:5])}")
                    
                    # Analysis button
                    analyze_button = st.button("🔍 Analyze Dependencies", type="secondary")
                    
                    if analyze_button and signal_input:
                        # Try to find the signal (could be network or internal name)
                        resolved_signal = signal_input
                        
                        # Check mapping if available
                        if st.session_state.get('mapping_loader'):
                            mapper = st.session_state['mapping_loader']
                            network_name, internal_name = mapper.resolve_signal_name(signal_input)
                            
                            # Try both names
                            if internal_name in dep_graph.G:
                                resolved_signal = internal_name
                            elif network_name in dep_graph.G:
                                resolved_signal = network_name
                        
                        if resolved_signal not in dep_graph.G:
                            st.error(f"Signal '{signal_input}' not found in the dependency graph.")
                            
                            # Suggest similar signals
                            suggestions = dep_graph.search_signals(signal_input, case_sensitive=False)
                            if suggestions:
                                st.info(f"Did you mean: {', '.join(suggestions[:5])}?")
                        else:
                            # Extract subgraph
                            subgraph = dep_graph.subgraph_upstream(resolved_signal, depth)
                            
                            if subgraph.number_of_nodes() == 0:
                                st.warning(f"No dependencies found for signal '{resolved_signal}'")
                            else:
                                st.success(f"Found {subgraph.number_of_nodes()} related signals with {subgraph.number_of_edges()} connections")
                                
                                # Create and display the visualization
                                pyvis_net = create_pyvis_network(subgraph, resolved_signal)
                                
                                # Save to temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w') as tmp_file:
                                    pyvis_net.show(tmp_file.name)
                                    
                                    # Read and display
                                    with open(tmp_file.name, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    
                                    st.components.v1.html(html_content, height=650, scrolling=True)
                                    
                                    # Clean up
                                    os.unlink(tmp_file.name)
                                
                                # Display signal information
                                signal_info = dep_graph.get_signal_info(resolved_signal)
                                
                                with st.expander("📋 Signal Details"):
                                    col_info1, col_info2 = st.columns(2)
                                    
                                    with col_info1:
                                        st.write(f"**Signal Name:** {signal_info['name']}")
                                        st.write(f"**Type:** {signal_info['type']}")
                                        st.write(f"**Input Connections:** {signal_info['in_degree']}")
                                        st.write(f"**Output Connections:** {signal_info['out_degree']}")
                                    
                                    with col_info2:
                                        if signal_info['predecessors']:
                                            st.write("**Input Signals:**")
                                            for pred in signal_info['predecessors']:
                                                st.write(f"• {pred}")
                                        
                                        if signal_info['successors']:
                                            st.write("**Output Signals:**")
                                            for succ in signal_info['successors']:
                                                st.write(f"• {succ}")
            
            except Exception as e:
                st.error(f"Error loading dependency graph: {str(e)}")
    
    with col2:
        # Statistics and information panel
        st.markdown('<div class="section-header">📊 Graph Information</div>', unsafe_allow_html=True)
        
        if CACHE_FILE.exists():
            try:
                dep_graph = DependencyGraph(cache_file=CACHE_FILE)
                
                # Display current statistics
                display_graph_statistics(dep_graph)
                
                # Critical signals
                critical_signals = dep_graph.get_critical_signals(min_connections=3)
                if critical_signals:
                    st.subheader("🔥 Critical Signals")
                    for signal, connections in critical_signals[:5]:
                        st.write(f"• **{signal}** ({connections} connections)")
                
                # Cycle detection
                cycles = dep_graph.detect_cycles()
                if cycles:
                    st.subheader("⚠️ Dependency Cycles")
                    st.warning(f"Found {len(cycles)} dependency cycles")
                    with st.expander("View cycles"):
                        for i, cycle in enumerate(cycles[:3]):
                            st.write(f"Cycle {i+1}: {' → '.join(cycle)}")
                
                # Export options
                st.subheader("💾 Export Options")
                if st.button("Export Graph Data"):
                    export_dir = DATA_DIR / "exports"
                    dep_graph.export_to_formats(export_dir, ['json', 'gexf'])
                    st.success(f"Graph exported to {export_dir}")
                
            except Exception as e:
                st.error(f"Error loading graph statistics: {str(e)}")
        else:
            st.info("Build a dependency graph to see statistics here.")

if __name__ == "__main__":
    main()