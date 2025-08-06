#!/usr/bin/env python3
"""
SymPy Mathematical Expression Parser
Advanced equation parsing and analysis for automotive FDEF documents.
Supports symbolic computation, variable dependency analysis, and automotive-specific functions.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import sympy as sp
from sympy import symbols, sympify, latex
from sympy.parsing.sympy_parser import parse_expr
import numpy as np

logger = logging.getLogger(__name__)

class EquationType(Enum):
    """Types of equations commonly found in automotive documents"""
    ASSIGNMENT = "assignment"
    COMPARISON = "comparison"
    CONDITIONAL = "conditional"
    LOOKUP_TABLE = "lookup_table"
    RATE_LIMIT = "rate_limit"
    TRANSFER_FUNCTION = "transfer_function"
    UNKNOWN = "unknown"

@dataclass
class ParsedEquation:
    """Container for parsed equation information"""
    output_variable: str
    raw_text: str
    sympy_expr: Optional[sp.Expr]
    variables: Set[str]
    constants: Set[str]
    operators: List[str]
    functions: List[str]
    equation_type: EquationType
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class LookupTable:
    """Container for lookup table information"""
    name: str
    input_variable: str
    output_variable: str
    breakpoints: List[float]
    values: List[float]
    interpolation_method: str
    metadata: Dict[str, Any]

class SymPyEquationParser:
    """
    Advanced equation parser using SymPy for symbolic computation
    """
    
    def __init__(self):
        self.automotive_functions = self._init_automotive_functions()
        self.equation_patterns = self._init_equation_patterns()
        self.variable_cache = {}
        
        logger.info("🧮 SymPy Equation Parser initialized")
        
    def _init_automotive_functions(self) -> Dict[str, Any]:
        """Initialize automotive-specific function definitions"""
        
        # Define custom functions for automotive applications
        functions = {
            'lookup': sp.Function('lookup'),
            'interp': sp.Function('interp'),
            'rate_limit': sp.Function('rate_limit'),
            'sat': sp.Function('sat'),           # Saturation
            'deadband': sp.Function('deadband'),
            'hysteresis': sp.Function('hysteresis'),
            'filter': sp.Function('filter'),
            'pid': sp.Function('pid'),
            'integrator': sp.Function('integrator'),
            'differentiator': sp.Function('differentiator'),
            'delay': sp.Function('delay'),
            'switch': sp.Function('switch'),
            'ramp': sp.Function('ramp'),
            'step': sp.Function('step'),
            'pwm': sp.Function('pwm'),
            'can_send': sp.Function('can_send'),
            'can_receive': sp.Function('can_receive')
        }
        
        return functions
        
    def _init_equation_patterns(self) -> List[Dict[str, Any]]:
        """Initialize regular expressions for equation pattern matching"""
        
        patterns = [
            # Assignment equations: y = f(x)
            {
                'name': 'assignment',
                'pattern': r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)',
                'type': EquationType.ASSIGNMENT,
                'confidence': 0.9
            },
            
            # Comparison equations: x > y, x == y, etc.
            {
                'name': 'comparison',
                'pattern': r'(.+)\s*(==|!=|>=|<=|>|<)\s*(.+)',
                'type': EquationType.COMPARISON,
                'confidence': 0.8
            },
            
            # Conditional equations: if-then-else
            {
                'name': 'conditional',
                'pattern': r'if\s*\((.+)\)\s*then\s*(.+)\s*else\s*(.+)',
                'type': EquationType.CONDITIONAL,
                'confidence': 0.85
            },
            
            # Lookup tables: y = lookup(x, table)
            {
                'name': 'lookup',
                'pattern': r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*lookup\s*\(([^,]+),\s*([^)]+)\)',
                'type': EquationType.LOOKUP_TABLE,
                'confidence': 0.95
            },
            
            # Rate limiting: y = rate_limit(x, min_rate, max_rate)
            {
                'name': 'rate_limit',
                'pattern': r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*rate_limit\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
                'type': EquationType.RATE_LIMIT,
                'confidence': 0.9
            }
        ]
        
        return patterns
        
    def parse_equations(self, text: str) -> List[ParsedEquation]:
        """
        Parse mathematical equations from text content
        """
        
        logger.info("🔍 Parsing mathematical equations...")
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Split into potential equation lines
        lines = cleaned_text.split('\n')
        
        equations = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue
                
            # Try to parse the line as an equation
            parsed_eq = self._parse_single_equation(line, line_num)
            
            if parsed_eq:
                equations.append(parsed_eq)
                
        logger.info(f"   Parsed {len(equations)} equations")
        return equations
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for equation parsing"""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize mathematical operators
        replacements = {
            '×': '*',
            '÷': '/',
            '≥': '>=',
            '≤': '<=',
            '≠': '!=',
            '²': '**2',
            '³': '**3',
            '√': 'sqrt',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Fix common OCR errors in mathematical expressions
        text = re.sub(r'\bl\b', '1', text)  # l -> 1
        text = re.sub(r'\bO\b', '0', text)  # O -> 0
        text = re.sub(r'\|(?=[a-zA-Z])', 'l', text)  # | -> l (in variable names)
        
        return text
        
    def _parse_single_equation(self, line: str, line_num: int) -> Optional[ParsedEquation]:
        """Parse a single equation line"""
        
        # Try each equation pattern
        for pattern_info in self.equation_patterns:
            pattern = pattern_info['pattern']
            eq_type = pattern_info['type']
            base_confidence = pattern_info['confidence']
            
            match = re.match(pattern, line, re.IGNORECASE)
            
            if match:
                return self._create_parsed_equation(
                    line, match, eq_type, base_confidence, line_num
                )
                
        # If no pattern matched, try general symbolic parsing
        return self._try_general_parsing(line, line_num)
        
    def _create_parsed_equation(self, line: str, match: re.Match, 
                               eq_type: EquationType, confidence: float,
                               line_num: int) -> Optional[ParsedEquation]:
        """Create a ParsedEquation object from a regex match"""
        
        try:
            if eq_type == EquationType.ASSIGNMENT:
                output_var = match.group(1).strip()
                expression_text = match.group(2).strip()
                
                # Parse the right-hand side with SymPy
                sympy_expr = self._safe_sympify(expression_text)
                
                if sympy_expr is not None:
                    variables, constants = self._extract_symbols(sympy_expr)
                    operators = self._extract_operators(expression_text)
                    functions = self._extract_functions(expression_text)
                    
                    return ParsedEquation(
                        output_variable=output_var,
                        raw_text=line,
                        sympy_expr=sympy_expr,
                        variables=variables,
                        constants=constants,
                        operators=operators,
                        functions=functions,
                        equation_type=eq_type,
                        confidence=confidence,
                        metadata={
                            'line_number': line_num,
                            'expression_text': expression_text,
                            'parsing_method': 'pattern_matching'
                        }
                    )
                    
            elif eq_type == EquationType.LOOKUP_TABLE:
                output_var = match.group(1).strip()
                input_var = match.group(2).strip()
                table_name = match.group(3).strip()
                
                # Create a symbolic representation
                lookup_expr = self.automotive_functions['lookup'](
                    symbols(input_var), symbols(table_name)
                )
                
                return ParsedEquation(
                    output_variable=output_var,
                    raw_text=line,
                    sympy_expr=lookup_expr,
                    variables={input_var, table_name},
                    constants=set(),
                    operators=[],
                    functions=['lookup'],
                    equation_type=eq_type,
                    confidence=confidence,
                    metadata={
                        'line_number': line_num,
                        'input_variable': input_var,
                        'table_name': table_name,
                        'parsing_method': 'lookup_pattern'
                    }
                )
                
            elif eq_type == EquationType.RATE_LIMIT:
                output_var = match.group(1).strip()
                input_var = match.group(2).strip()
                min_rate = match.group(3).strip()
                max_rate = match.group(4).strip()
                
                # Create symbolic representation
                rate_limit_expr = self.automotive_functions['rate_limit'](
                    symbols(input_var), symbols(min_rate), symbols(max_rate)
                )
                
                return ParsedEquation(
                    output_variable=output_var,
                    raw_text=line,
                    sympy_expr=rate_limit_expr,
                    variables={input_var, min_rate, max_rate},
                    constants=set(),
                    operators=[],
                    functions=['rate_limit'],
                    equation_type=eq_type,
                    confidence=confidence,
                    metadata={
                        'line_number': line_num,
                        'input_variable': input_var,
                        'min_rate': min_rate,
                        'max_rate': max_rate,
                        'parsing_method': 'rate_limit_pattern'
                    }
                )
                
        except Exception as e:
            logger.debug(f"Failed to parse equation at line {line_num}: {e}")
            
        return None
        
    def _try_general_parsing(self, line: str, line_num: int) -> Optional[ParsedEquation]:
        """Try to parse equation using general SymPy parsing"""
        
        # Look for assignment-like patterns
        if '=' in line and not any(op in line for op in ['==', '!=', '>=', '<=']):
            parts = line.split('=', 1)
            if len(parts) == 2:
                output_var = parts[0].strip()
                expression_text = parts[1].strip()
                
                # Try to parse with SymPy
                sympy_expr = self._safe_sympify(expression_text)
                
                if sympy_expr is not None:
                    variables, constants = self._extract_symbols(sympy_expr)
                    operators = self._extract_operators(expression_text)
                    functions = self._extract_functions(expression_text)
                    
                    return ParsedEquation(
                        output_variable=output_var,
                        raw_text=line,
                        sympy_expr=sympy_expr,
                        variables=variables,
                        constants=constants,
                        operators=operators,
                        functions=functions,
                        equation_type=EquationType.ASSIGNMENT,
                        confidence=0.6,  # Lower confidence for general parsing
                        metadata={
                            'line_number': line_num,
                            'expression_text': expression_text,
                            'parsing_method': 'general_parsing'
                        }
                    )
                    
        return None
        
    def _safe_sympify(self, expression: str) -> Optional[sp.Expr]:
        """Safely convert string to SymPy expression"""
        
        try:
            # Replace automotive functions with symbolic functions
            expr_text = expression
            
            for func_name, func_symbol in self.automotive_functions.items():
                # Replace function calls with symbolic versions
                pattern = rf'\b{func_name}\s*\('
                if re.search(pattern, expr_text):
                    # This is a complex replacement - for now, create a symbolic function
                    expr_text = re.sub(pattern, f'{func_name}(', expr_text)
                    
            # Try to parse with SymPy
            expr = sympify(expr_text, evaluate=False)
            return expr
            
        except Exception as e:
            logger.debug(f"Failed to sympify expression '{expression}': {e}")
            return None
            
    def _extract_symbols(self, expr: sp.Expr) -> Tuple[Set[str], Set[str]]:
        """Extract variables and constants from SymPy expression"""
        
        all_symbols = expr.free_symbols
        
        variables = set()
        constants = set()
        
        for symbol in all_symbols:
            symbol_name = str(symbol)
            
            # Check if it looks like a constant (all uppercase, starts with K_, etc.)
            if (symbol_name.isupper() or 
                symbol_name.startswith('K_') or 
                symbol_name.startswith('C_') or
                symbol_name.replace('.', '').replace('_', '').isdigit()):
                constants.add(symbol_name)
            else:
                variables.add(symbol_name)
                
        return variables, constants
        
    def _extract_operators(self, expression: str) -> List[str]:
        """Extract mathematical operators from expression string"""
        
        operator_pattern = r'[+\-*/^%=<>!&|]+'
        operators = re.findall(operator_pattern, expression)
        
        return list(set(operators))  # Remove duplicates
        
    def _extract_functions(self, expression: str) -> List[str]:
        """Extract function names from expression string"""
        
        # Look for function calls: word followed by opening parenthesis
        function_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        functions = re.findall(function_pattern, expression)
        
        return list(set(functions))  # Remove duplicates
        
    def extract_lookup_tables(self, text: str) -> List[LookupTable]:
        """
        Extract lookup table definitions from text
        """
        
        logger.info("🔍 Extracting lookup tables...")
        
        tables = []
        
        # Pattern for table definitions
        # Format: TABLE_NAME = { input: [x1, x2, x3], output: [y1, y2, y3] }
        table_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{[^}]*input\s*:\s*\[([^\]]+)\][^}]*output\s*:\s*\[([^\]]+)\][^}]*\}'
        
        matches = re.findall(table_pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            table_name = match[0].strip()
            input_values_str = match[1].strip()
            output_values_str = match[2].strip()
            
            try:
                # Parse input values
                input_values = self._parse_numeric_list(input_values_str)
                output_values = self._parse_numeric_list(output_values_str)
                
                if len(input_values) == len(output_values) and len(input_values) > 0:
                    table = LookupTable(
                        name=table_name,
                        input_variable="x",  # Generic input variable
                        output_variable="y", # Generic output variable
                        breakpoints=input_values,
                        values=output_values,
                        interpolation_method="linear",
                        metadata={
                            'source_text': match[0],
                            'num_points': len(input_values)
                        }
                    )
                    
                    tables.append(table)
                    
            except Exception as e:
                logger.debug(f"Failed to parse lookup table {table_name}: {e}")
                
        # Also look for simpler table formats
        simple_tables = self._extract_simple_tables(text)
        tables.extend(simple_tables)
        
        logger.info(f"   Found {len(tables)} lookup tables")
        return tables
        
    def _parse_numeric_list(self, values_str: str) -> List[float]:
        """Parse a string containing numeric values into a list of floats"""
        
        # Remove brackets and split by comma
        values_str = values_str.replace('[', '').replace(']', '')
        value_strings = values_str.split(',')
        
        values = []
        for value_str in value_strings:
            value_str = value_str.strip()
            if value_str:
                try:
                    # Try to convert to float
                    value = float(value_str)
                    values.append(value)
                except ValueError:
                    # If not a number, skip it
                    continue
                    
        return values
        
    def _extract_simple_tables(self, text: str) -> List[LookupTable]:
        """Extract lookup tables from simple tabular data"""
        
        tables = []
        
        # Look for patterns like:
        # X: 0, 10, 20, 30
        # Y: 0, 5,  15, 25
        
        lines = text.split('\n')
        
        for i in range(len(lines) - 1):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()
            
            # Check if both lines contain numeric data with labels
            if ':' in line1 and ':' in line2:
                try:
                    # Parse first line
                    label1, values1_str = line1.split(':', 1)
                    values1 = self._parse_numeric_list(values1_str)
                    
                    # Parse second line
                    label2, values2_str = line2.split(':', 1)
                    values2 = self._parse_numeric_list(values2_str)
                    
                    if (len(values1) == len(values2) and 
                        len(values1) >= 2 and
                        len(values2) >= 2):
                        
                        table_name = f"{label1.strip()}_{label2.strip()}_table"
                        
                        table = LookupTable(
                            name=table_name,
                            input_variable=label1.strip(),
                            output_variable=label2.strip(),
                            breakpoints=values1,
                            values=values2,
                            interpolation_method="linear",
                            metadata={
                                'source_format': 'simple_table',
                                'line_numbers': [i, i + 1]
                            }
                        )
                        
                        tables.append(table)
                        
                except Exception as e:
                    logger.debug(f"Failed to parse simple table at lines {i}-{i+1}: {e}")
                    
        return tables
        
    def get_variable_dependencies(self, equations: List[ParsedEquation]) -> Dict[str, Set[str]]:
        """
        Analyze variable dependencies across all equations
        """
        
        logger.info("🔍 Analyzing variable dependencies...")
        
        dependencies = {}
        
        for equation in equations:
            output_var = equation.output_variable
            input_vars = equation.variables.copy()
            
            # Remove the output variable from inputs (it doesn't depend on itself)
            input_vars.discard(output_var)
            
            if output_var not in dependencies:
                dependencies[output_var] = set()
                
            dependencies[output_var].update(input_vars)
            
        # Compute transitive dependencies
        transitive_deps = self._compute_transitive_dependencies(dependencies)
        
        logger.info(f"   Analyzed dependencies for {len(dependencies)} variables")
        return transitive_deps
        
    def _compute_transitive_dependencies(self, direct_deps: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Compute transitive closure of dependencies"""
        
        transitive = {}
        
        for var in direct_deps:
            transitive[var] = set()
            visited = set()
            self._dfs_dependencies(var, direct_deps, transitive[var], visited)
            
        return transitive
        
    def _dfs_dependencies(self, var: str, deps: Dict[str, Set[str]], 
                         result: Set[str], visited: Set[str]):
        """Depth-first search for dependencies"""
        
        if var in visited:
            return  # Avoid cycles
            
        visited.add(var)
        
        if var in deps:
            for dep_var in deps[var]:
                result.add(dep_var)
                self._dfs_dependencies(dep_var, deps, result, visited)
                
    def evaluate_equation(self, equation: ParsedEquation, 
                         variable_values: Dict[str, float]) -> Optional[float]:
        """
        Evaluate an equation given variable values
        """
        
        if equation.sympy_expr is None:
            return None
            
        try:
            # Substitute variable values
            substituted = equation.sympy_expr
            
            for var_name, value in variable_values.items():
                if var_name in equation.variables:
                    var_symbol = symbols(var_name)
                    substituted = substituted.subs(var_symbol, value)
                    
            # Evaluate numerically
            result = float(substituted.evalf())
            return result
            
        except Exception as e:
            logger.debug(f"Failed to evaluate equation {equation.output_variable}: {e}")
            return None
            
    def to_latex(self, equation: ParsedEquation) -> str:
        """Convert equation to LaTeX format"""
        
        if equation.sympy_expr is None:
            return equation.raw_text
            
        try:
            latex_str = latex(equation.sympy_expr)
            return f"{equation.output_variable} = {latex_str}"
        except Exception:
            return equation.raw_text
            
    def get_parsing_statistics(self, equations: List[ParsedEquation]) -> Dict[str, Any]:
        """Get statistics about parsed equations"""
        
        if not equations:
            return {}
            
        # Count by equation type
        type_counts = {}
        for eq in equations:
            eq_type = eq.equation_type.value
            type_counts[eq_type] = type_counts.get(eq_type, 0) + 1
            
        # Calculate average confidence
        avg_confidence = np.mean([eq.confidence for eq in equations])
        
        # Count unique variables
        all_variables = set()
        for eq in equations:
            all_variables.update(eq.variables)
            
        return {
            'total_equations': len(equations),
            'equation_types': type_counts,
            'average_confidence': avg_confidence,
            'unique_variables': len(all_variables),
            'variable_list': sorted(list(all_variables))
        }

if __name__ == "__main__":
    # Test the equation parser
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SymPy Equation Parser')
    parser.add_argument('--text', help='Text containing equations to parse')
    parser.add_argument('--file', help='File containing equations to parse')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get text input
    if args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = """
        Engine_Torque = Engine_RPM * Torque_Factor + Base_Torque
        Fuel_Rate = lookup(Engine_Load, Fuel_Map)
        Vehicle_Speed = rate_limit(Target_Speed, -5.0, 5.0)
        if (Engine_Temp > 90) then Cooling_Fan = 1 else Cooling_Fan = 0
        """
        
    # Parse equations
    equation_parser = SymPyEquationParser()
    equations = equation_parser.parse_equations(text)
    
    print(f"Parsed {len(equations)} equations:")
    
    for eq in equations:
        print(f"\n{eq.output_variable} = {eq.raw_text}")
        print(f"  Type: {eq.equation_type.value}")
        print(f"  Variables: {eq.variables}")
        print(f"  Functions: {eq.functions}")
        print(f"  Confidence: {eq.confidence:.2f}")
        
    # Extract lookup tables
    tables = equation_parser.extract_lookup_tables(text)
    print(f"\nFound {len(tables)} lookup tables:")
    
    for table in tables:
        print(f"  {table.name}: {len(table.breakpoints)} points")
        
    # Get statistics
    stats = equation_parser.get_parsing_statistics(equations)
    print(f"\nParsing Statistics:")
    for key, value in stats.items():
        if key != 'variable_list':  # Skip the long variable list
            print(f"  {key}: {value}")