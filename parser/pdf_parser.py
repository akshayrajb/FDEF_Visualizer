"""
PDF Parser for FDEF Documents

Extracts signal dependencies and logical rules from FDEF PDF documents.
Handles both text-searchable and scanned (OCR-processed) PDFs.

Supports pattern recognition for:
- Logic gates (AND, OR, NOT)
- Signal connections and dependencies
- Switch conditions and state transitions
- Complex Simulink/Stateflow diagrams
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFRuleParser:
    """
    Parses FDEF PDF documents to extract signal dependency rules.
    
    Recognizes patterns like:
    - Signal_A OR Signal_B -> Signal_C
    - X AND (Y OR Z) -> PT_Rdy
    - Switch(condition): True -> Signal_A, False -> Signal_B
    """
    
    # Regex patterns for different rule formats
    RULE_PATTERNS = [
        # Standard arrow notation: Signal_A -> Signal_B
        re.compile(r'(?P<sources>[\w\s\[\],()]+?)\s*(?:->|→|=)\s*(?P<target>\w+)', re.I),
        
        # Logic notation: Signal_A AND Signal_B = Signal_C
        re.compile(r'(?P<sources>[\w\s\[\],()]+?)\s*(?:AND|OR)\s*[\w\s\[\],()]+?\s*=\s*(?P<target>\w+)', re.I),
        
        # Switch notation: Switch(condition) -> True: Signal_A, False: Signal_B
        re.compile(r'Switch\s*\(\s*(?P<condition>\w+)\s*\)\s*(?:->|:)\s*(?P<branches>.+)', re.I),
        
        # Assignment notation: Signal_C := Signal_A OR Signal_B
        re.compile(r'(?P<target>\w+)\s*:=\s*(?P<sources>[\w\s\[\],()]+)', re.I),
        
        # Function notation: f(Signal_A, Signal_B) -> Signal_C
        re.compile(r'(?P<function>\w+)\s*\(\s*(?P<sources>[\w\s,]+)\s*\)\s*(?:->|=)\s*(?P<target>\w+)', re.I)
    ]
    
    # Logical operators
    OPERATORS = {
        'AND': 'AND',
        'OR': 'OR', 
        'NOT': 'NOT',
        '&': 'AND',
        '|': 'OR',
        '!': 'NOT',
        '+': 'OR',
        '*': 'AND'
    }
    
    def __init__(self, pdf_paths: List[Path]):
        """
        Initialize the PDF parser.
        
        Args:
            pdf_paths: List of paths to FDEF PDF files
        """
        self.pdf_paths = [Path(p) for p in pdf_paths]
        self.extracted_rules = []
        self.extracted_text = {}  # Store extracted text per PDF
        
    def extract_rules(self) -> List[Tuple[str, str, str]]:
        """
        Extract all dependency rules from the PDF files.
        
        Returns:
            List of tuples (source_signal, target_signal, operator)
        """
        self.extracted_rules.clear()
        
        for pdf_path in self.pdf_paths:
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
                
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                rules = self._extract_from_pdf(pdf_path)
                self.extracted_rules.extend(rules)
                
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        # Remove duplicates while preserving order
        unique_rules = []
        seen = set()
        for rule in self.extracted_rules:
            if rule not in seen:
                unique_rules.append(rule)
                seen.add(rule)
        
        logger.info(f"Extracted {len(unique_rules)} unique rules from {len(self.pdf_paths)} PDFs")
        return unique_rules
    
    def _extract_from_pdf(self, pdf_path: Path) -> List[Tuple[str, str, str]]:
        """
        Extract rules from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted rules
        """
        rules = []
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text("text")
                full_text += text + "\n"
                
                # Extract rules from this page
                page_rules = self._parse_text_for_rules(text, page_num)
                rules.extend(page_rules)
            
            # Store extracted text for debugging
            self.extracted_text[str(pdf_path)] = full_text
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
        
        return rules
    
    def _parse_text_for_rules(self, text: str, page_num: int = 0) -> List[Tuple[str, str, str]]:
        """
        Parse text content to extract dependency rules.
        
        Args:
            text: Text content to parse
            page_num: Page number for logging
            
        Returns:
            List of extracted rules
        """
        rules = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Try each pattern
            for pattern in self.RULE_PATTERNS:
                matches = pattern.finditer(line)
                for match in matches:
                    extracted_rules = self._process_match(match, line)
                    rules.extend(extracted_rules)
        
        return rules
    
    def _process_match(self, match: re.Match, original_line: str) -> List[Tuple[str, str, str]]:
        """
        Process a regex match to extract rules.
        
        Args:
            match: Regex match object
            original_line: Original text line
            
        Returns:
            List of extracted rules
        """
        rules = []
        groups = match.groupdict()
        
        if 'sources' in groups and 'target' in groups:
            # Standard source -> target pattern
            sources_text = groups['sources'].strip()
            target = groups['target'].strip()
            
            # Parse sources and operator
            sources, operator = self._parse_sources(sources_text)
            
            for source in sources:
                if source and target:
                    rules.append((source, target, operator))
        
        elif 'target' in groups and 'sources' in groups:
            # Assignment pattern: target := sources
            target = groups['target'].strip()
            sources_text = groups['sources'].strip()
            
            sources, operator = self._parse_sources(sources_text)
            
            for source in sources:
                if source and target:
                    rules.append((source, target, operator))
        
        elif 'condition' in groups and 'branches' in groups:
            # Switch pattern
            condition = groups['condition'].strip()
            branches_text = groups['branches'].strip()
            
            # Parse switch branches
            switch_rules = self._parse_switch_branches(condition, branches_text)
            rules.extend(switch_rules)
        
        elif 'function' in groups and 'sources' in groups and 'target' in groups:
            # Function pattern
            function = groups['function'].strip()
            sources_text = groups['sources'].strip()
            target = groups['target'].strip()
            
            sources, operator = self._parse_sources(sources_text)
            
            # Add function as intermediate node
            for source in sources:
                if source:
                    rules.append((source, function, operator))
            
            if target:
                rules.append((function, target, 'FUNC'))
        
        return rules
    
    def _parse_sources(self, sources_text: str) -> Tuple[List[str], str]:
        """
        Parse source signals and determine the logical operator.
        
        Args:
            sources_text: Text containing source signals
            
        Returns:
            Tuple of (source_list, operator)
        """
        # Determine operator by checking for keywords
        operator = 'AND'  # Default
        
        for op_text, op_type in self.OPERATORS.items():
            if op_text.upper() in sources_text.upper():
                operator = op_type
                break
        
        # Split sources by common delimiters
        delimiters = r'[,\s]+(?:AND|OR|&|\||\+|\*)?[,\s]*'
        sources = re.split(delimiters, sources_text, flags=re.I)
        
        # Clean up sources
        cleaned_sources = []
        for source in sources:
            source = source.strip()
            # Remove operator keywords
            for op_text in self.OPERATORS.keys():
                source = re.sub(rf'\b{re.escape(op_text)}\b', '', source, flags=re.I)
            source = source.strip()
            
            # Remove parentheses and other noise
            source = re.sub(r'[^\w]', '', source)
            
            if source and len(source) > 1:  # Ignore single characters
                cleaned_sources.append(source)
        
        return cleaned_sources, operator
    
    def _parse_switch_branches(self, condition: str, branches_text: str) -> List[Tuple[str, str, str]]:
        """
        Parse switch statement branches.
        
        Args:
            condition: Switch condition signal
            branches_text: Text describing the branches
            
        Returns:
            List of rules for switch branches
        """
        rules = []
        
        # Common switch patterns
        true_pattern = re.compile(r'(?:True|1|High)[\s:]*(?:->|:)\s*(\w+)', re.I)
        false_pattern = re.compile(r'(?:False|0|Low)[\s:]*(?:->|:)\s*(\w+)', re.I)
        
        true_match = true_pattern.search(branches_text)
        false_match = false_pattern.search(branches_text)
        
        if true_match:
            true_target = true_match.group(1).strip()
            rules.append((condition, true_target, 'SWITCH_TRUE'))
        
        if false_match:
            false_target = false_match.group(1).strip()
            rules.append((condition, false_target, 'SWITCH_FALSE'))
        
        return rules
    
    def get_extracted_text(self, pdf_path: Optional[Path] = None) -> str:
        """
        Get extracted text from PDFs for debugging.
        
        Args:
            pdf_path: Specific PDF path, or None for all text
            
        Returns:
            Extracted text content
        """
        if pdf_path:
            return self.extracted_text.get(str(pdf_path), "")
        else:
            return "\n\n".join(self.extracted_text.values())
    
    def validate_rules(self, rules: List[Tuple[str, str, str]]) -> Dict[str, int]:
        """
        Validate extracted rules and provide statistics.
        
        Args:
            rules: List of extracted rules
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'total_rules': len(rules),
            'unique_sources': len(set(rule[0] for rule in rules)),
            'unique_targets': len(set(rule[1] for rule in rules)),
            'operators': {}
        }
        
        # Count operators
        for _, _, operator in rules:
            stats['operators'][operator] = stats['operators'].get(operator, 0) + 1
        
        return stats
    
    def search_signal_references(self, signal_name: str) -> List[Dict]:
        """
        Search for all references to a specific signal in the extracted text.
        
        Args:
            signal_name: Signal name to search for
            
        Returns:
            List of reference locations and contexts
        """
        references = []
        
        for pdf_path, text in self.extracted_text.items():
            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                if signal_name.lower() in line.lower():
                    references.append({
                        'pdf_path': pdf_path,
                        'line_number': line_num + 1,
                        'context': line.strip(),
                        'line_text': line
                    })
        
        return references