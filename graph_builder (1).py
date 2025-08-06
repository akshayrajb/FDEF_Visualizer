#!/usr/bin/env python3
"""
Dependency Graph Builder
Advanced graph construction and analysis for signal dependencies in FDEF documents.
Combines computer vision, mathematical analysis, and template matching results.
"""

import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import networkx as nx
import numpy as np

# Import our analysis modules
from cv_arrows import ArrowDetection, BlockDetection, ArrowDirection
from sympy_parser import ParsedEquation, LookupTable
from domain_templates import AutomotiveMatch, AutomotiveBlockType

logger = logging.getLogger(__name__)

@dataclass
class SignalNode:
    """Represents a signal node in the dependency graph"""
    name: str
    signal_type: str  # 'input', 'output', 'intermediate', 'parameter'
    confidence: float
    properties: Dict[str, Any]
    sources: List[str]  # Which analysis method detected this signal

@dataclass
class DependencyEdge:
    """Represents a dependency relationship between signals"""
    source: str
    target: str
    relationship_type: str  # 'direct', 'conditional', 'lookup', 'mathematical'
    confidence: float
    properties: Dict[str, Any]
    evidence: List[str]  # Supporting evidence for this relationship

class DependencyGraphBuilder:
    """
    Advanced dependency graph builder that combines multiple analysis results
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.signal_registry = {}  # name -> SignalNode
        self.edge_registry = {}    # (source, target) -> DependencyEdge
        self.confidence_weights = self._init_confidence_weights()
        
        logger.info("🕸️ Dependency Graph Builder initialized")
        
    def _init_confidence_weights(self) -> Dict[str, float]:
        """Initialize confidence weights for different evidence types"""
        
        return {
            'template_matching': 0.9,
            'equation_parsing': 0.95,
            'cv_arrow_detection': 0.7,
            'fuzzy_matching': 0.6,
            'name_similarity': 0.5,
            'spatial_proximity': 0.4
        }
        
    def build_comprehensive_graph(self, 
                                cv_arrows: List[ArrowDetection],
                                cv_blocks: List[BlockDetection],
                                equations: List[ParsedEquation],
                                template_matches: List[AutomotiveMatch],
                                page_texts: List[str]) -> nx.DiGraph:
        """
        Build comprehensive dependency graph from all analysis results
        """
        
        logger.info("🔧 Building comprehensive dependency graph...")
        
        # Clear previous graph
        self.graph.clear()
        self.signal_registry.clear()
        self.edge_registry.clear()
        
        # Phase 1: Extract signals from all sources
        self._extract_signals_from_equations(equations)
        self._extract_signals_from_templates(template_matches)
        self._extract_signals_from_cv_blocks(cv_blocks)
        self._extract_signals_from_text(page_texts)
        
        # Phase 2: Build dependencies from all sources
        self._build_dependencies_from_equations(equations)
        self._build_dependencies_from_cv_arrows(cv_arrows, cv_blocks)
        self._build_dependencies_from_templates(template_matches)
        
        # Phase 3: Resolve cross-references and merge similar signals
        self._resolve_cross_references(page_texts)
        self._merge_similar_signals()
        
        # Phase 4: Add consolidated nodes and edges to NetworkX graph
        self._build_networkx_graph()
        
        # Phase 5: Calculate final confidence scores
        self._calculate_final_confidences()
        
        logger.info(f"   Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
        
    def _extract_signals_from_equations(self, equations: List[ParsedEquation]):
        """Extract signal nodes from parsed equations"""
        
        for equation in equations:
            # Add output signal
            self._add_or_update_signal(
                name=equation.output_variable,
                signal_type='output',
                confidence=equation.confidence,
                properties={
                    'equation_type': equation.equation_type.value,
                    'raw_equation': equation.raw_text,
                    'functions_used': equation.functions
                },
                source='equation_parsing'
            )
            
            # Add input signals (variables)
            for variable in equation.variables:
                if variable != equation.output_variable:  # Don't count output as input
                    self._add_or_update_signal(
                        name=variable,
                        signal_type='input',
                        confidence=equation.confidence * 0.8,  # Slightly lower confidence for inputs
                        properties={
                            'referenced_in_equation': equation.output_variable,
                            'equation_type': equation.equation_type.value
                        },
                        source='equation_parsing'
                    )
                    
            # Add constants as parameters
            for constant in equation.constants:
                self._add_or_update_signal(
                    name=constant,
                    signal_type='parameter',
                    confidence=equation.confidence * 0.9,
                    properties={
                        'constant_value': True,
                        'used_in_equation': equation.output_variable
                    },
                    source='equation_parsing'
                )
                
    def _extract_signals_from_templates(self, template_matches: List[AutomotiveMatch]):
        """Extract signal nodes from automotive template matches"""
        
        for match in template_matches:
            signal_name = self._extract_signal_name_from_match(match)
            
            if signal_name:
                signal_type = self._determine_signal_type_from_template(match.block_type)
                
                self._add_or_update_signal(
                    name=signal_name,
                    signal_type=signal_type,
                    confidence=match.confidence,
                    properties={
                        'automotive_type': match.block_type.value,
                        'template_data': match.extracted_data,
                        'matched_patterns': match.matched_patterns
                    },
                    source='template_matching'
                )
                
    def _extract_signals_from_cv_blocks(self, cv_blocks: List[BlockDetection]):
        """Extract signal nodes from computer vision block detections"""
        
        for block in cv_blocks:
            # Use block center coordinates as signal name if no text content
            if block.text_content:
                signal_name = self._clean_signal_name(block.text_content)
            else:
                signal_name = f"Block_{block.center_point[0]}_{block.center_point[1]}"
                
            self._add_or_update_signal(
                name=signal_name,
                signal_type='intermediate',  # Blocks are typically intermediate processing
                confidence=block.confidence,
                properties={
                    'block_type': block.block_type,
                    'center_point': block.center_point,
                    'bounding_box': block.bounding_box,
                    'visual_block': True
                },
                source='cv_block_detection'
            )
            
    def _extract_signals_from_text(self, page_texts: List[str]):
        """Extract potential signals from raw text using pattern matching"""
        
        import re
        
        # Common signal naming patterns
        patterns = [
            r'\b([A-Za-z_][A-Za-z0-9_]*(?:_Signal|_Input|_Output|_Status|_Value|_Rate|_Temp|_Press|_Speed|_Torque|_RPM))\b',
            r'\b(Engine_[A-Za-z0-9_]*|Brake_[A-Za-z0-9_]*|Trans_[A-Za-z0-9_]*|Body_[A-Za-z0-9_]*)\b',
            r'\b([A-Z]{2,}[_][A-Za-z0-9_]*)\b',  # ALL_CAPS_SIGNALS
        ]
        
        for page_idx, text in enumerate(page_texts):
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    signal_name = self._clean_signal_name(match)
                    
                    if len(signal_name) > 2:  # Skip very short matches
                        self._add_or_update_signal(
                            name=signal_name,
                            signal_type='intermediate',
                            confidence=0.6,  # Lower confidence for text extraction
                            properties={
                                'page_number': page_idx + 1,
                                'pattern_matched': pattern,
                                'text_extraction': True
                            },
                            source='text_pattern_matching'
                        )
                        
    def _build_dependencies_from_equations(self, equations: List[ParsedEquation]):
        """Build dependency edges from equation analysis"""
        
        for equation in equations:
            output_signal = equation.output_variable
            
            # Create dependencies from input variables to output
            for input_var in equation.variables:
                if input_var != output_signal:
                    relationship_type = self._determine_relationship_type_from_equation(equation)
                    
                    self._add_or_update_dependency(
                        source=input_var,
                        target=output_signal,
                        relationship_type=relationship_type,
                        confidence=equation.confidence,
                        properties={
                            'equation': equation.raw_text,
                            'equation_type': equation.equation_type.value,
                            'functions': equation.functions
                        },
                        evidence=['equation_analysis']
                    )
                    
    def _build_dependencies_from_cv_arrows(self, cv_arrows: List[ArrowDetection], 
                                         cv_blocks: List[BlockDetection]):
        """Build dependency edges from computer vision arrow detection"""
        
        # Create a spatial lookup for blocks
        block_lookup = self._create_spatial_block_lookup(cv_blocks)
        
        for arrow in cv_arrows:
            # Find blocks near arrow start and end points
            start_block = self._find_nearest_block(arrow.start_point, block_lookup)
            end_block = self._find_nearest_block(arrow.end_point, block_lookup)
            
            if start_block and end_block:
                # Extract signal names from blocks
                source_signal = self._get_signal_name_from_block(start_block)
                target_signal = self._get_signal_name_from_block(end_block)
                
                if source_signal and target_signal and source_signal != target_signal:
                    self._add_or_update_dependency(
                        source=source_signal,
                        target=target_signal,
                        relationship_type='direct',
                        confidence=arrow.confidence,
                        properties={
                            'arrow_direction': arrow.direction.value,
                            'arrow_type': arrow.arrow_type,
                            'spatial_connection': True,
                            'start_point': arrow.start_point,
                            'end_point': arrow.end_point
                        },
                        evidence=['cv_arrow_detection']
                    )
                    
    def _build_dependencies_from_templates(self, template_matches: List[AutomotiveMatch]):
        """Build dependency edges from automotive template analysis"""
        
        # Group matches by type to identify relationships
        can_messages = [m for m in template_matches if m.block_type == AutomotiveBlockType.CAN_MESSAGE]
        can_signals = [m for m in template_matches if m.block_type == AutomotiveBlockType.CAN_SIGNAL]
        
        # Link CAN signals to their messages
        for signal_match in can_signals:
            signal_name = self._extract_signal_name_from_match(signal_match)
            
            # Find corresponding message (simplified - would need more sophisticated matching)
            for message_match in can_messages:
                message_name = self._extract_signal_name_from_match(message_match)
                
                if signal_name and message_name:
                    # Assume signal belongs to message if they're textually close
                    self._add_or_update_dependency(
                        source=signal_name,
                        target=message_name,
                        relationship_type='packaging',
                        confidence=min(signal_match.confidence, message_match.confidence),
                        properties={
                            'can_relationship': True,
                            'signal_type': 'can_signal',
                            'message_type': 'can_message'
                        },
                        evidence=['template_matching']
                    )
                    
    def _resolve_cross_references(self, page_texts: List[str]):
        """Resolve cross-page references and add dependencies"""
        
        import re
        
        # Look for explicit references like "see page X", "ref: Signal_Name", etc.
        reference_patterns = [
            r'see\s+page\s+(\d+)',
            r'ref(?:erence)?\s*:\s*([A-Za-z_][A-Za-z0-9_]*)',
            r'input\s+from\s+([A-Za-z_][A-Za-z0-9_]*)',
            r'output\s+to\s+([A-Za-z_][A-Za-z0-9_]*)'
        ]
        
        for page_idx, text in enumerate(page_texts):
            for pattern in reference_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if pattern.startswith('ref') or pattern.startswith('input') or pattern.startswith('output'):
                        # This is a signal reference
                        referenced_signal = self._clean_signal_name(match)
                        
                        if referenced_signal in self.signal_registry:
                            # Create cross-reference relationship
                            page_signals = self._get_signals_from_page(page_idx, page_texts)
                            
                            for page_signal in page_signals:
                                if page_signal != referenced_signal:
                                    relationship_type = 'cross_reference'
                                    if 'input' in pattern:
                                        # Referenced signal is input to page signal
                                        self._add_or_update_dependency(
                                            source=referenced_signal,
                                            target=page_signal,
                                            relationship_type=relationship_type,
                                            confidence=0.7,
                                            properties={'cross_page_reference': True},
                                            evidence=['cross_reference_analysis']
                                        )
                                    elif 'output' in pattern:
                                        # Page signal is output to referenced signal
                                        self._add_or_update_dependency(
                                            source=page_signal,
                                            target=referenced_signal,
                                            relationship_type=relationship_type,
                                            confidence=0.7,
                                            properties={'cross_page_reference': True},
                                            evidence=['cross_reference_analysis']
                                        )
                                        
    def _merge_similar_signals(self):
        """Merge signals that likely refer to the same entity"""
        
        from fuzzywuzzy import fuzz
        
        # Find groups of similar signal names
        signal_names = list(self.signal_registry.keys())
        merge_groups = []
        processed = set()
        
        for i, name1 in enumerate(signal_names):
            if name1 in processed:
                continue
                
            group = [name1]
            processed.add(name1)
            
            for j, name2 in enumerate(signal_names[i+1:], i+1):
                if name2 in processed:
                    continue
                    
                # Check similarity
                similarity = fuzz.ratio(name1.lower(), name2.lower())
                
                if similarity > 85:  # High similarity threshold
                    group.append(name2)
                    processed.add(name2)
                    
            if len(group) > 1:
                merge_groups.append(group)
                
        # Merge similar signals
        for group in merge_groups:
            primary_name = max(group, key=lambda name: self.signal_registry[name].confidence)
            
            # Merge other signals into the primary one
            for name in group:
                if name != primary_name:
                    self._merge_signals(name, primary_name)
                    
    def _merge_signals(self, source_name: str, target_name: str):
        """Merge one signal into another"""
        
        source_signal = self.signal_registry[source_name]
        target_signal = self.signal_registry[target_name]
        
        # Combine properties
        combined_properties = {**source_signal.properties, **target_signal.properties}
        combined_sources = list(set(source_signal.sources + target_signal.sources))
        
        # Use higher confidence
        final_confidence = max(source_signal.confidence, target_signal.confidence)
        
        # Update target signal
        self.signal_registry[target_name] = SignalNode(
            name=target_name,
            signal_type=target_signal.signal_type,  # Keep target's type
            confidence=final_confidence,
            properties=combined_properties,
            sources=combined_sources
        )
        
        # Update edge registry to redirect edges from source to target
        edges_to_update = []
        for (edge_source, edge_target), edge in self.edge_registry.items():
            if edge_source == source_name:
                edges_to_update.append(((edge_source, edge_target), (target_name, edge_target)))
            elif edge_target == source_name:
                edges_to_update.append(((edge_source, edge_target), (edge_source, target_name)))
                
        for old_key, new_key in edges_to_update:
            if new_key not in self.edge_registry:  # Avoid duplicates
                edge = self.edge_registry.pop(old_key)
                edge.source = new_key[0]
                edge.target = new_key[1]
                self.edge_registry[new_key] = edge
            else:
                # Merge with existing edge
                old_edge = self.edge_registry.pop(old_key)
                existing_edge = self.edge_registry[new_key]
                existing_edge.confidence = max(old_edge.confidence, existing_edge.confidence)
                existing_edge.evidence.extend(old_edge.evidence)
                
        # Remove source signal
        del self.signal_registry[source_name]
        
    def _build_networkx_graph(self):
        """Build the final NetworkX graph from signal and edge registries"""
        
        # Add nodes
        for signal_name, signal_node in self.signal_registry.items():
            self.graph.add_node(
                signal_name,
                signal_type=signal_node.signal_type,
                confidence=signal_node.confidence,
                sources=signal_node.sources,
                **signal_node.properties
            )
            
        # Add edges
        for (source, target), edge in self.edge_registry.items():
            if source in self.graph.nodes and target in self.graph.nodes:
                self.graph.add_edge(
                    source,
                    target,
                    relationship_type=edge.relationship_type,
                    confidence=edge.confidence,
                    evidence=edge.evidence,
                    **edge.properties
                )
                
    def _calculate_final_confidences(self):
        """Calculate final confidence scores for nodes and edges"""
        
        # Update node confidences based on multiple sources
        for node_name in self.graph.nodes():
            node_data = self.graph.nodes[node_name]
            sources = node_data.get('sources', [])
            
            # Combine confidences from multiple sources
            if len(sources) > 1:
                base_confidence = node_data.get('confidence', 0.5)
                source_bonus = min(0.2, 0.05 * len(sources))  # Bonus for multiple sources
                final_confidence = min(1.0, base_confidence + source_bonus)
                
                self.graph.nodes[node_name]['confidence'] = final_confidence
                
        # Update edge confidences based on evidence
        for source, target in self.graph.edges():
            edge_data = self.graph.edges[source, target]
            evidence = edge_data.get('evidence', [])
            
            if len(evidence) > 1:
                base_confidence = edge_data.get('confidence', 0.5)
                evidence_bonus = min(0.3, 0.1 * len(evidence))  # Bonus for multiple evidence
                final_confidence = min(1.0, base_confidence + evidence_bonus)
                
                self.graph.edges[source, target]['confidence'] = final_confidence
                
    # Helper methods
    
    def _add_or_update_signal(self, name: str, signal_type: str, confidence: float,
                            properties: Dict[str, Any], source: str):
        """Add a new signal or update existing one"""
        
        cleaned_name = self._clean_signal_name(name)
        
        if cleaned_name in self.signal_registry:
            # Update existing signal
            existing = self.signal_registry[cleaned_name]
            
            # Use higher confidence
            if confidence > existing.confidence:
                existing.confidence = confidence
                existing.signal_type = signal_type  # Update type if higher confidence
                
            # Merge properties
            existing.properties.update(properties)
            
            # Add source
            if source not in existing.sources:
                existing.sources.append(source)
        else:
            # Add new signal
            self.signal_registry[cleaned_name] = SignalNode(
                name=cleaned_name,
                signal_type=signal_type,
                confidence=confidence,
                properties=properties,
                sources=[source]
            )
            
    def _add_or_update_dependency(self, source: str, target: str, relationship_type: str,
                                confidence: float, properties: Dict[str, Any], evidence: List[str]):
        """Add a new dependency or update existing one"""
        
        clean_source = self._clean_signal_name(source)
        clean_target = self._clean_signal_name(target)
        
        edge_key = (clean_source, clean_target)
        
        if edge_key in self.edge_registry:
            # Update existing edge
            existing = self.edge_registry[edge_key]
            
            # Use higher confidence
            if confidence > existing.confidence:
                existing.confidence = confidence
                existing.relationship_type = relationship_type
                
            # Merge properties
            existing.properties.update(properties)
            
            # Add evidence
            for ev in evidence:
                if ev not in existing.evidence:
                    existing.evidence.append(ev)
        else:
            # Add new edge
            self.edge_registry[edge_key] = DependencyEdge(
                source=clean_source,
                target=clean_target,
                relationship_type=relationship_type,
                confidence=confidence,
                properties=properties,
                evidence=evidence.copy()
            )
            
    def _clean_signal_name(self, name: str) -> str:
        """Clean and normalize signal names"""
        
        if not name:
            return ""
            
        # Remove whitespace and special characters
        import re
        cleaned = re.sub(r'[^A-Za-z0-9_]', '_', name.strip())
        
        # Remove multiple underscores
        cleaned = re.sub(r'_{2,}', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
        
    def _extract_signal_name_from_match(self, match: AutomotiveMatch) -> Optional[str]:
        """Extract signal name from automotive template match"""
        
        # Try common field names
        common_fields = ['name', 'signal_name', 'short_name', 'message_name', 'parameter_name']
        
        for field in common_fields:
            if field in match.extracted_data:
                return self._clean_signal_name(match.extracted_data[field])
                
        # Fall back to text content
        if match.text_content:
            return self._clean_signal_name(match.text_content)
            
        return None
        
    def _determine_signal_type_from_template(self, block_type: AutomotiveBlockType) -> str:
        """Determine signal type from automotive block type"""
        
        type_mapping = {
            AutomotiveBlockType.SENSOR_INPUT: 'input',
            AutomotiveBlockType.ACTUATOR_OUTPUT: 'output',
            AutomotiveBlockType.CALIBRATION_PARAMETER: 'parameter',
            AutomotiveBlockType.CAN_SIGNAL: 'intermediate',
            AutomotiveBlockType.CAN_MESSAGE: 'intermediate',
            AutomotiveBlockType.A2L_MEASUREMENT: 'input',
            AutomotiveBlockType.A2L_CHARACTERISTIC: 'parameter',
            AutomotiveBlockType.ECU_FUNCTION: 'intermediate'
        }
        
        return type_mapping.get(block_type, 'intermediate')
        
    def _determine_relationship_type_from_equation(self, equation: ParsedEquation) -> str:
        """Determine relationship type from equation analysis"""
        
        from sympy_parser import EquationType
        
        type_mapping = {
            EquationType.ASSIGNMENT: 'mathematical',
            EquationType.CONDITIONAL: 'conditional',
            EquationType.LOOKUP_TABLE: 'lookup',
            EquationType.RATE_LIMIT: 'rate_limited',
            EquationType.COMPARISON: 'conditional'
        }
        
        return type_mapping.get(equation.equation_type, 'direct')
        
    def _create_spatial_block_lookup(self, cv_blocks: List[BlockDetection]) -> Dict[Tuple[int, int], BlockDetection]:
        """Create spatial lookup for blocks"""
        
        lookup = {}
        for block in cv_blocks:
            lookup[block.center_point] = block
            
        return lookup
        
    def _find_nearest_block(self, point: Tuple[int, int], 
                          block_lookup: Dict[Tuple[int, int], BlockDetection]) -> Optional[BlockDetection]:
        """Find the nearest block to a given point"""
        
        if not block_lookup:
            return None
            
        min_distance = float('inf')
        nearest_block = None
        
        for block_center, block in block_lookup.items():
            distance = np.sqrt((point[0] - block_center[0])**2 + (point[1] - block_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_block = block
                
        # Only return if reasonably close (within 100 pixels)
        return nearest_block if min_distance < 100 else None
        
    def _get_signal_name_from_block(self, block: BlockDetection) -> Optional[str]:
        """Get signal name from block detection"""
        
        if block.text_content:
            return self._clean_signal_name(block.text_content)
        else:
            return None
            
    def _get_signals_from_page(self, page_idx: int, page_texts: List[str]) -> List[str]:
        """Get signals that were extracted from a specific page"""
        
        page_signals = []
        
        for signal_name, signal_node in self.signal_registry.items():
            if signal_node.properties.get('page_number') == page_idx + 1:
                page_signals.append(signal_name)
                
        return page_signals
        
    # Analysis methods
    
    def get_signal_dependencies(self, signal_name: str) -> Dict[str, List[str]]:
        """Get comprehensive dependency information for a specific signal"""
        
        if signal_name not in self.graph.nodes:
            return {}
            
        # Direct inputs (signals that directly affect this signal)
        direct_inputs = list(self.graph.predecessors(signal_name))
        
        # Direct outputs (signals directly affected by this signal)
        direct_outputs = list(self.graph.successors(signal_name))
        
        # Indirect inputs (transitive dependencies)
        indirect_inputs = []
        visited = set()
        queue = deque(direct_inputs)
        
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                predecessors = list(self.graph.predecessors(current))
                indirect_inputs.extend(predecessors)
                queue.extend(predecessors)
                
        # Remove direct inputs from indirect list
        indirect_inputs = [sig for sig in set(indirect_inputs) if sig not in direct_inputs and sig != signal_name]
        
        # Indirect outputs (signals transitively affected)
        indirect_outputs = []
        visited = set()
        queue = deque(direct_outputs)
        
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                successors = list(self.graph.successors(current))
                indirect_outputs.extend(successors)
                queue.extend(successors)
                
        # Remove direct outputs from indirect list
        indirect_outputs = [sig for sig in set(indirect_outputs) if sig not in direct_outputs and sig != signal_name]
        
        return {
            'direct_inputs': direct_inputs,
            'direct_outputs': direct_outputs,
            'indirect_inputs': indirect_outputs,
            'indirect_outputs': indirect_outputs,
            'total_dependencies': len(direct_inputs) + len(indirect_inputs),
            'total_dependents': len(direct_outputs) + len(indirect_outputs)
        }
        
    def analyze_graph_properties(self) -> Dict[str, Any]:
        """Analyze various properties of the dependency graph"""
        
        if not self.graph.nodes:
            return {}
            
        # Basic graph metrics
        num_nodes = len(self.graph.nodes)
        num_edges = len(self.graph.edges)
        
        # Connectivity metrics
        if num_nodes > 0:
            density = nx.density(self.graph)
            
            # Convert to undirected for some metrics
            undirected = self.graph.to_undirected()
            connected_components = list(nx.connected_components(undirected))
            largest_component_size = max(len(comp) for comp in connected_components) if connected_components else 0
        else:
            density = 0
            connected_components = []
            largest_component_size = 0
            
        # Node degree statistics
        in_degrees = [self.graph.in_degree(node) for node in self.graph.nodes]
        out_degrees = [self.graph.out_degree(node) for node in self.graph.nodes]
        
        # Signal type distribution
        signal_types = {}
        confidence_scores = []
        
        for node in self.graph.nodes(data=True):
            signal_type = node[1].get('signal_type', 'unknown')
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            confidence_scores.append(node[1].get('confidence', 0.0))
            
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'num_connected_components': len(connected_components),
            'largest_component_size': largest_component_size,
            'avg_in_degree': np.mean(in_degrees) if in_degrees else 0,
            'avg_out_degree': np.mean(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'signal_type_distribution': signal_types,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0
        }

if __name__ == "__main__":
    # Test the graph builder
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple test
    builder = DependencyGraphBuilder()
    
    # This would normally be called with real analysis results
    # For testing, we'll create a minimal graph
    builder._add_or_update_signal("Engine_RPM", "input", 0.9, {}, "test")
    builder._add_or_update_signal("Vehicle_Speed", "output", 0.8, {}, "test")
    builder._add_or_update_dependency("Engine_RPM", "Vehicle_Speed", "direct", 0.7, {}, ["test"])
    
    builder._build_networkx_graph()
    
    print(f"Test graph: {len(builder.graph.nodes)} nodes, {len(builder.graph.edges)} edges")
    
    # Analyze properties
    properties = builder.analyze_graph_properties()
    print("Graph properties:")
    for key, value in properties.items():
        print(f"  {key}: {value}")