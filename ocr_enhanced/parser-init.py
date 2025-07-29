"""
FDEF Parser Package

Core parsing modules for FDEF dependency analysis:
- mapping_loader: Excel mapping sheet reader
- pdf_parser: PDF text extraction and rule parsing
- dependency_graph: NetworkX graph builder and manipulation
"""

__version__ = "1.0.0"
__author__ = "FDEF Dependency Visualizer"

from .mapping_loader import MappingLoader
from .pdf_parser import PDFRuleParser
from .dependency_graph import DependencyGraph

__all__ = [
    'MappingLoader',
    'PDFRuleParser', 
    'DependencyGraph'
]