"""
Dependency Graph Builder for FDEF Signal Analysis

Creates and manipulates NetworkX directed graphs representing signal dependencies.
Supports caching, subgraph extraction, and interactive exploration.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import networkx as nx

logger = logging.getLogger(__name__)


class DependencyGraph:
    """
    Manages signal dependency graphs using NetworkX.
    
    Features:
    - Directed graph representation of signal dependencies
    - Caching for fast repeated access
    - Subgraph extraction for visualization
    - Graph analysis and statistics
    """
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize the dependency graph.
        
        Args:
            cache_file: Optional path to cache file for persistence
        """
        self.G = nx.DiGraph()
        self.cache_file = cache_file
        self.metadata = {
            'creation_time': None,
            'last_modified': None,
            'source_files': [],
            'statistics': {}
        }
        
        # Load from cache if available
        if cache_file and Path(cache_file).exists():
            self._load_from_cache()
    
    def add_rule(self, source: str, target: str, operator: str = 'UNKNOWN', **kwargs):
        """
        Add a single dependency rule to the graph.
        
        Args:
            source: Source signal name
            target: Target signal name
            operator: Logical operator (AND, OR, NOT, etc.)
            **kwargs: Additional edge attributes
        """
        # Add nodes with default attributes
        self.G.add_node(source, 
                       label=source, 
                       type='signal',
                       properties=kwargs.get('source_props', {}))
        
        self.G.add_node(target,
                       label=target,
                       type='signal', 
                       properties=kwargs.get('target_props', {}))
        
        # Add edge with operator and additional attributes
        edge_attrs = {
            'operator': operator,
            'weight': kwargs.get('weight', 1.0),
            'confidence': kwargs.get('confidence', 1.0)
        }
        edge_attrs.update(kwargs)
        
        self.G.add_edge(source, target, **edge_attrs)
        
        logger.debug(f"Added rule: {source} --{operator}--> {target}")
    
    def build_from_rules(self, rules: List[Tuple[str, str, str]]):
        """
        Build the graph from a list of dependency rules.
        
        Args:
            rules: List of (source, target, operator) tuples
        """
        logger.info(f"Building graph from {len(rules)} rules")
        
        # Clear existing graph
        self.G.clear()
        
        # Add all rules
        for source, target, operator in rules:
            self.add_rule(source, target, operator)
        
        # Update statistics
        self._update_statistics()
        
        logger.info(f"Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def subgraph_upstream(self, signal: str, depth: int = 3) -> nx.DiGraph:
        """
        Extract subgraph showing upstream dependencies of a signal.
        
        Args:
            signal: Target signal to trace
            depth: Maximum depth to traverse upstream
            
        Returns:
            NetworkX DiGraph containing the subgraph
        """
        if signal not in self.G:
            logger.warning(f"Signal '{signal}' not found in graph")
            return nx.DiGraph()
        
        visited_nodes = {signal}
        current_frontier = {signal}
        
        # Traverse upstream layer by layer
        for level in range(depth):
            next_frontier = set()
            
            for node in current_frontier:
                # Get all predecessors (sources)
                predecessors = set(self.G.predecessors(node))
                next_frontier.update(predecessors)
                visited_nodes.update(predecessors)
            
            current_frontier = next_frontier
            
            # Stop if no more predecessors found
            if not current_frontier:
                break
        
        # Create subgraph with visited nodes
        subgraph = self.G.subgraph(visited_nodes).copy()
        
        logger.debug(f"Extracted upstream subgraph for '{signal}': "
                    f"{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def subgraph_downstream(self, signal: str, depth: int = 3) -> nx.DiGraph:
        """
        Extract subgraph showing downstream dependencies of a signal.
        
        Args:
            signal: Source signal to trace
            depth: Maximum depth to traverse downstream
            
        Returns:
            NetworkX DiGraph containing the subgraph
        """
        if signal not in self.G:
            logger.warning(f"Signal '{signal}' not found in graph")
            return nx.DiGraph()
        
        visited_nodes = {signal}
        current_frontier = {signal}
        
        # Traverse downstream layer by layer
        for level in range(depth):
            next_frontier = set()
            
            for node in current_frontier:
                # Get all successors (targets)
                successors = set(self.G.successors(node))
                next_frontier.update(successors)
                visited_nodes.update(successors)
            
            current_frontier = next_frontier
            
            # Stop if no more successors found
            if not current_frontier:
                break
        
        # Create subgraph with visited nodes
        subgraph = self.G.subgraph(visited_nodes).copy()
        
        logger.debug(f"Extracted downstream subgraph for '{signal}': "
                    f"{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def find_paths(self, source: str, target: str, max_paths: int = 10) -> List[List[str]]:
        """
        Find all paths between two signals.
        
        Args:
            source: Source signal
            target: Target signal
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths (each path is a list of signal names)
        """
        if source not in self.G or target not in self.G:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.G, source, target))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def get_signal_info(self, signal: str) -> Dict:
        """
        Get detailed information about a signal.
        
        Args:
            signal: Signal name
            
        Returns:
            Dictionary with signal information
        """
        if signal not in self.G:
            return {'exists': False}
        
        node_data = self.G.nodes[signal]
        
        # Get connection information
        predecessors = list(self.G.predecessors(signal))
        successors = list(self.G.successors(signal))
        
        # Get edge information
        incoming_edges = [(pred, signal, self.G[pred][signal]) 
                         for pred in predecessors]
        outgoing_edges = [(signal, succ, self.G[signal][succ]) 
                         for succ in successors]
        
        return {
            'exists': True,
            'name': signal,
            'properties': node_data.get('properties', {}),
            'type': node_data.get('type', 'signal'),
            'label': node_data.get('label', signal),
            'predecessors': predecessors,
            'successors': successors,
            'incoming_edges': incoming_edges,
            'outgoing_edges': outgoing_edges,
            'in_degree': self.G.in_degree(signal),
            'out_degree': self.G.out_degree(signal)
        }
    
    def search_signals(self, pattern: str, case_sensitive: bool = False) -> List[str]:
        """
        Search for signals matching a pattern.
        
        Args:
            pattern: Search pattern
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching signal names
        """
        search_pattern = pattern if case_sensitive else pattern.lower()
        matches = []
        
        for node in self.G.nodes():
            search_node = node if case_sensitive else node.lower()
            if search_pattern in search_node:
                matches.append(node)
        
        return sorted(matches)
    
    def get_critical_signals(self, min_connections: int = 3) -> List[Tuple[str, int]]:
        """
        Identify critical signals with many connections.
        
        Args:
            min_connections: Minimum number of connections to consider critical
            
        Returns:
            List of (signal_name, total_connections) tuples
        """
        critical = []
        
        for node in self.G.nodes():
            total_connections = self.G.degree(node)
            if total_connections >= min_connections:
                critical.append((node, total_connections))
        
        # Sort by number of connections (descending)
        critical.sort(key=lambda x: x[1], reverse=True)
        
        return critical
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the dependency graph.
        
        Returns:
            List of cycles (each cycle is a list of signal names)
        """
        try:
            cycles = list(nx.simple_cycles(self.G))
            return cycles
        except:
            return []
    
    def _update_statistics(self):
        """Update graph statistics in metadata."""
        self.metadata['statistics'] = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'is_dag': nx.is_directed_acyclic_graph(self.G),
            'connected_components': nx.number_weakly_connected_components(self.G),
            'average_degree': sum(dict(self.G.degree()).values()) / max(1, self.G.number_of_nodes())
        }
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        self._update_statistics()
        return self.metadata['statistics'].copy()
    
    def save(self):
        """Save the graph to cache file."""
        if not self.cache_file:
            logger.warning("No cache file specified, cannot save")
            return
        
        try:
            # Create cache directory
            cache_path = Path(self.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update metadata
            import datetime
            self.metadata['last_modified'] = datetime.datetime.now().isoformat()
            self._update_statistics()
            
            # Prepare data for JSON serialization
            graph_data = nx.node_link_data(self.G)
            
            cache_data = {
                'graph': graph_data,
                'metadata': self.metadata
            }
            
            # Write to cache file
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Graph saved to cache: {cache_path}")
            
        except Exception as e:
            logger.error(f"Error saving graph cache: {e}")
    
    def _load_from_cache(self):
        """Load the graph from cache file."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Load graph
            graph_data = cache_data.get('graph', {})
            self.G = nx.node_link_graph(graph_data)
            
            # Load metadata
            self.metadata.update(cache_data.get('metadata', {}))
            
            logger.info(f"Graph loaded from cache: {self.cache_file}")
            logger.info(f"Loaded {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error loading graph cache: {e}")
            # Initialize empty graph
            self.G = nx.DiGraph()
    
    def export_to_formats(self, output_dir: Path, formats: List[str] = None):
        """
        Export graph to various formats.
        
        Args:
            output_dir: Directory to save exported files
            formats: List of formats ('gexf', 'graphml', 'edgelist', 'json')
        """
        if formats is None:
            formats = ['gexf', 'json']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            try:
                if fmt == 'gexf':
                    nx.write_gexf(self.G, output_dir / 'dependency_graph.gexf')
                elif fmt == 'graphml':
                    nx.write_graphml(self.G, output_dir / 'dependency_graph.graphml')
                elif fmt == 'edgelist':
                    nx.write_edgelist(self.G, output_dir / 'dependency_graph.edgelist')
                elif fmt == 'json':
                    with open(output_dir / 'dependency_graph.json', 'w') as f:
                        json.dump(nx.node_link_data(self.G), f, indent=2)
                
                logger.info(f"Exported graph to {fmt} format")
                
            except Exception as e:
                logger.error(f"Error exporting to {fmt}: {e}")
    
    def clear(self):
        """Clear the graph and reset metadata."""
        self.G.clear()
        self.metadata = {
            'creation_time': None,
            'last_modified': None,
            'source_files': [],
            'statistics': {}
        }