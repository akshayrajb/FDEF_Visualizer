#!/usr/bin/env python3
"""
HTML Network Exporter
Advanced interactive HTML visualization for FDEF signal dependency networks.
Creates professional Plotly-based visualizations with hover details and filtering.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

class HtmlNetworkExporter:
    """
    Export dependency graphs as interactive HTML visualizations
    """
    
    def __init__(self):
        self.color_scheme = self._init_color_scheme()
        self.layout_params = self._init_layout_parameters()
        
        logger.info("🌐 HTML Network Exporter initialized")
        
    def _init_color_scheme(self) -> Dict[str, str]:
        """Initialize color scheme for different signal types"""
        
        return {
            'input': '#2E8B57',        # Sea Green
            'output': '#DC143C',       # Crimson
            'intermediate': '#4682B4', # Steel Blue
            'parameter': '#DAA520',    # Goldenrod
            'unknown': '#808080',      # Gray
            
            # Edge colors
            'direct': '#1f77b4',       # Blue
            'conditional': '#ff7f0e',  # Orange
            'mathematical': '#2ca02c', # Green
            'lookup': '#d62728',       # Red
            'cross_reference': '#9467bd' # Purple
        }
        
    def _init_layout_parameters(self) -> Dict[str, Any]:
        """Initialize layout parameters for network visualization"""
        
        return {
            'k': 3,              # Spring layout optimal distance
            'iterations': 50,    # Layout iterations
            'node_size_range': (10, 30),
            'edge_width_range': (1, 5),
            'figure_size': (1200, 800),
            'margin': 50
        }
        
    def export_interactive_network(self, graph: nx.DiGraph, output_file: str,
                                 title: str = "Signal Dependency Network",
                                 target_signal: Optional[str] = None) -> str:
        """
        Export NetworkX graph as interactive HTML visualization
        """
        
        logger.info(f"🎨 Creating interactive network visualization...")
        
        if not graph.nodes:
            logger.warning("Empty graph - creating placeholder visualization")
            return self._create_empty_visualization(output_file, title)
            
        # Calculate layout
        pos = self._calculate_layout(graph, target_signal)
        
        # Extract node and edge data
        node_data = self._extract_node_data(graph, pos, target_signal)
        edge_data = self._extract_edge_data(graph, pos)
        
        # Create interactive HTML
        html_content = self._create_html_content(
            node_data, edge_data, title, target_signal, graph
        )
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"   Saved interactive visualization: {output_path}")
        return str(output_path)
        
    def _calculate_layout(self, graph: nx.DiGraph, target_signal: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
        """Calculate optimal layout for the network"""
        
        # Use spring layout as base
        try:
            pos = nx.spring_layout(
                graph, 
                k=self.layout_params['k'],
                iterations=self.layout_params['iterations'],
                seed=42  # Reproducible layout
            )
        except:
            # Fallback to random layout if spring layout fails
            pos = nx.random_layout(graph, seed=42)
            
        # If target signal specified, center it
        if target_signal and target_signal in pos:
            target_pos = pos[target_signal]
            
            # Move target to center and adjust others
            center_x, center_y = 0.5, 0.5
            offset_x = center_x - target_pos[0]
            offset_y = center_y - target_pos[1]
            
            for node in pos:
                x, y = pos[node]
                pos[node] = (x + offset_x, y + offset_y)
                
        return pos
        
    def _extract_node_data(self, graph: nx.DiGraph, pos: Dict[str, Tuple[float, float]],
                          target_signal: Optional[str] = None) -> Dict[str, Any]:
        """Extract node data for visualization"""
        
        nodes = {
            'x': [],
            'y': [],
            'text': [],
            'hovertext': [],
            'color': [],
            'size': [],
            'opacity': []
        }
        
        for node, data in graph.nodes(data=True):
            if node not in pos:
                continue
                
            x, y = pos[node]
            nodes['x'].append(x)
            nodes['y'].append(y)
            nodes['text'].append(node)
            
            # Create hover text with detailed information
            hover_lines = [f"<b>{node}</b>"]
            
            signal_type = data.get('signal_type', 'unknown')
            confidence = data.get('confidence', 0.0)
            sources = data.get('sources', [])
            
            hover_lines.append(f"Type: {signal_type}")
            hover_lines.append(f"Confidence: {confidence:.2f}")
            
            if sources:
                hover_lines.append(f"Sources: {', '.join(sources)}")
                
            nodes['hovertext'].append('<br>'.join(hover_lines))
            
            # Color based on signal type
            nodes['color'].append(self.color_scheme.get(signal_type, self.color_scheme['unknown']))
            
            # Size based on connectivity (degree)
            degree = graph.degree(node)
            min_size, max_size = self.layout_params['node_size_range']
            max_degree = max([graph.degree(n) for n in graph.nodes()]) if graph.nodes() else 1
            
            if max_degree > 0:
                size = min_size + (max_size - min_size) * (degree / max_degree)
            else:
                size = min_size
                
            nodes['size'].append(size)
            
            # Highlight target signal
            if node == target_signal:
                nodes['opacity'].append(1.0)
            else:
                nodes['opacity'].append(0.8)
                
        return nodes
        
    def _extract_edge_data(self, graph: nx.DiGraph, pos: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Extract edge data for visualization"""
        
        edges = {
            'x': [],
            'y': [],
            'color': [],
            'width': [],
            'hovertext': []
        }
        
        for source, target, data in graph.edges(data=True):
            if source not in pos or target not in pos:
                continue
                
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Add line coordinates (with None separator for discontinuous lines)
            edges['x'].extend([x0, x1, None])
            edges['y'].extend([y0, y1, None])
            
            # Edge color based on relationship type
            relationship_type = data.get('relationship_type', 'direct')
            edge_color = self.color_scheme.get(relationship_type, self.color_scheme['direct'])
            edges['color'].extend([edge_color, edge_color, edge_color])
            
            # Edge width based on confidence
            confidence = data.get('confidence', 0.5)
            min_width, max_width = self.layout_params['edge_width_range']
            width = min_width + (max_width - min_width) * confidence
            edges['width'].extend([width, width, width])
            
            # Create hover text for edge
            hover_lines = [f"<b>{source} → {target}</b>"]
            hover_lines.append(f"Type: {relationship_type}")
            hover_lines.append(f"Confidence: {confidence:.2f}")
            
            evidence = data.get('evidence', [])
            if evidence:
                hover_lines.append(f"Evidence: {', '.join(evidence)}")
                
            hover_text = '<br>'.join(hover_lines)
            edges['hovertext'].extend([hover_text, hover_text, ''])
            
        return edges
        
    def _create_html_content(self, node_data: Dict[str, Any], edge_data: Dict[str, Any],
                           title: str, target_signal: Optional[str], graph: nx.DiGraph) -> str:
        """Create complete HTML content with Plotly visualization"""
        
        # Calculate statistics
        stats = self._calculate_network_stats(graph, target_signal)
        
        # Create the main HTML structure
        html_parts = []
        
        # HTML head and styles
        html_parts.append(f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #4a6741 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .stats-panel {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .stat-item {{
            text-align: center;
            padding: 10px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .legend {{
            padding: 20px;
            background: #ffffff;
            border-bottom: 1px solid #e9ecef;
        }}
        .legend h3 {{
            margin: 0 0 15px 0;
            color: #2c3e50;
        }}
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .legend-section {{
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .network-container {{
            padding: 20px;
            background: white;
        }}
        #network-plot {{
            border: 1px solid #e9ecef;
            border-radius: 8px;
        }}
    </style>
</head>
<body>''')
        
        # Body content
        html_parts.append(f'''
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Interactive Signal Dependency Network Analysis</p>
            {f'<p style="margin-top: 10px;">🎯 Target Signal: <strong>{target_signal}</strong></p>' if target_signal else ''}
        </div>
        
        <div class="stats-panel">
            <div class="stat-item">
                <div class="stat-value">{stats['total_signals']}</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['total_dependencies']}</div>
                <div class="stat-label">Dependencies</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['avg_confidence']:.1%}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['max_degree']}</div>
                <div class="stat-label">Max Connections</div>
            </div>
        </div>
        
        <div class="legend">
            <h3>🎨 Visualization Legend</h3>
            <div class="legend-grid">
                <div class="legend-section">
                    <h4>Signal Types</h4>
                    <div class="legend-item">
                        <div class="legend-color" style="background: {self.color_scheme['input']};"></div>
                        <span>Input Signals</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: {self.color_scheme['output']};"></div>
                        <span>Output Signals</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: {self.color_scheme['intermediate']};"></div>
                        <span>Intermediate</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: {self.color_scheme['parameter']};"></div>
                        <span>Parameters</span>
                    </div>
                </div>
                
                <div class="legend-section">
                    <h4>Visual Encoding</h4>
                    <div style="font-size: 0.9em; line-height: 1.6;">
                        • <strong>Node Size:</strong> Connection count<br>
                        • <strong>Edge Width:</strong> Confidence level<br>
                        • <strong>Hover:</strong> Detailed information<br>
                        • <strong>Zoom/Pan:</strong> Interactive exploration
                    </div>
                </div>
            </div>
        </div>
        
        <div class="network-container">
            <div id="network-plot"></div>
            
            <div style="margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <h4>📊 Network Analysis</h4>
                <p><strong>Graph Density:</strong> {stats['density']:.3f} - Indicates how interconnected the network is</p>
                <p><strong>Analysis Method:</strong> Combined computer vision, equation parsing, and template matching</p>
                <p><strong>Total Nodes:</strong> {stats['total_signals']} signals detected and analyzed</p>
            </div>
        </div>
    </div>''')
        
        # JavaScript
        html_parts.append(f'''
    <script>
        const nodeData = {json.dumps(node_data)};
        const edgeData = {json.dumps(edge_data)};
        
        const edgeTrace = {{
            x: edgeData.x,
            y: edgeData.y,
            mode: 'lines',
            line: {{
                color: edgeData.color,
                width: edgeData.width
            }},
            hoverinfo: 'text',
            hovertext: edgeData.hovertext,
            type: 'scatter',
            showlegend: false
        }};
        
        const nodeTrace = {{
            x: nodeData.x,
            y: nodeData.y,
            mode: 'markers+text',
            marker: {{
                size: nodeData.size,
                color: nodeData.color,
                opacity: nodeData.opacity,
                line: {{
                    width: 2,
                    color: 'white'
                }}
            }},
            text: nodeData.text,
            textposition: 'middle center',
            textfont: {{
                size: 10,
                color: 'white',
                family: 'Arial Black'
            }},
            hoverinfo: 'text',
            hovertext: nodeData.hovertext,
            type: 'scatter',
            showlegend: false
        }};
        
        const data = [edgeTrace, nodeTrace];
        
        const layout = {{
            title: '',
            showlegend: false,
            hovermode: 'closest',
            margin: {{
                b: 20,
                l: 5,
                r: 5,
                t: 40
            }},
            annotations: [
                {{
                    text: "Hover over nodes and edges for detailed information",
                    showarrow: false,
                    xref: "paper", yref: "paper",
                    x: 0.005, y: -0.002,
                    xanchor: 'left', yanchor: 'bottom',
                    font: {{size: 12, color: '#999'}}
                }}
            ],
            xaxis: {{
                showgrid: false,
                zeroline: false,
                showticklabels: false
            }},
            yaxis: {{
                showgrid: false,
                zeroline: false,
                showticklabels: false
            }},
            plot_bgcolor: 'rgba(248,249,250,0.8)',
            paper_bgcolor: 'white'
        }};
        
        const config = {{
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            displaylogo: false,
            responsive: true
        }};
        
        Plotly.newPlot('network-plot', data, layout, config);
        
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize('network-plot');
        }});
    </script>
</body>
</html>''')
        
        return ''.join(html_parts)
        
    def _calculate_network_stats(self, graph: nx.DiGraph, target_signal: Optional[str]) -> Dict[str, Any]:
        """Calculate network statistics for display"""
        
        if not graph.nodes:
            return {
                'total_signals': 0,
                'total_dependencies': 0,
                'avg_confidence': 0.0,
                'max_degree': 0,
                'density': 0.0
            }
            
        # Basic counts
        total_signals = len(graph.nodes)
        total_dependencies = len(graph.edges)
        
        # Confidence statistics
        node_confidences = [data.get('confidence', 0.0) for _, data in graph.nodes(data=True)]
        edge_confidences = [data.get('confidence', 0.0) for _, _, data in graph.edges(data=True)]
        
        all_confidences = node_confidences + edge_confidences
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        # Degree statistics
        degrees = [graph.degree(node) for node in graph.nodes]
        max_degree = max(degrees) if degrees else 0
        
        # Network density
        density = nx.density(graph)
        
        return {
            'total_signals': total_signals,
            'total_dependencies': total_dependencies,
            'avg_confidence': avg_confidence,
            'max_degree': max_degree,
            'density': density
        }
        
    def _create_empty_visualization(self, output_file: str, title: str) -> str:
        """Create a placeholder visualization for empty graphs"""
        
        simple_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
        .message {{ font-size: 18px; color: #666; margin: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="message">
        <p>⚠️ No signal dependencies found in the document</p>
        <p>This may indicate:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>The document contains primarily textual content</li>
            <li>Signal relationships are not in a recognizable format</li>
            <li>Further analysis with different parameters may be needed</li>
        </ul>
    </div>
</body>
</html>'''
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(simple_html)
            
        return str(output_path)

def create_summary_dashboard(analysis_results: Dict[str, Any], output_file: str) -> str:
    """Create a comprehensive dashboard summarizing all analysis results"""
    
    dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FDEF Analysis Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #4a6741 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #007bff;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        .metric-label {
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🚀 FDEF Analysis Complete</h1>
            <p>Comprehensive Signal Dependency Analysis Results</p>
        </div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{signals_detected}</div>
                <div class="metric-label">Signals Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dependencies_mapped}</div>
                <div class="metric-label">Dependencies Mapped</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{processing_time:.1f}s</div>
                <div class="metric-label">Processing Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-label">Average Confidence</div>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    # Extract metrics from analysis results
    stats = analysis_results.get('statistics', {})
    metadata = analysis_results.get('metadata', {})
    
    formatted_html = dashboard_html.format(
        signals_detected=stats.get('signals_detected', 0),
        dependencies_mapped=stats.get('dependencies_mapped', 0),
        processing_time=metadata.get('analysis_time', 0),
        confidence=stats.get('average_confidence', 0)
    )
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_html)
        
    return str(output_path)

if __name__ == "__main__":
    # Test the HTML exporter
    import networkx as nx
    
    logging.basicConfig(level=logging.INFO)
    
    # Create a test graph
    G = nx.DiGraph()
    
    # Add test nodes
    G.add_node("Engine_RPM", signal_type="input", confidence=0.9, sources=["equation_parsing"])
    G.add_node("Vehicle_Speed", signal_type="output", confidence=0.8, sources=["cv_detection"])
    G.add_node("Throttle_Position", signal_type="input", confidence=0.85, sources=["template_matching"])
    
    # Add test edges
    G.add_edge("Engine_RPM", "Vehicle_Speed", relationship_type="mathematical", confidence=0.7, evidence=["equation"])
    G.add_edge("Throttle_Position", "Engine_RPM", relationship_type="direct", confidence=0.8, evidence=["cv_arrow"])
    
    # Export visualization
    exporter = HtmlNetworkExporter()
    output_file = exporter.export_interactive_network(
        G, 
        "test_network.html", 
        "Test FDEF Network",
        "Vehicle_Speed"
    )
    
    print(f"Created test visualization: {output_file}")