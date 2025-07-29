"""
Symbol Recognition for Technical Diagrams

Detects and classifies technical symbols in FDEF documents:
- Logic gates (AND, OR, NOT, XOR)
- Electrical components (switches, relays, sensors)
- Connection points and junctions
- Flow direction indicators
- State indicators and labels
"""

import logging
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectedSymbol:
    """Represents a detected symbol in the diagram."""
    symbol_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    properties: Dict[str, any]


class SymbolRecognition:
    """
    Recognizes technical symbols in FDEF diagrams using template matching
    and shape analysis.
    
    Supports detection of:
    - Logical operators (AND, OR, NOT gates)
    - Electrical symbols (switches, connectors)
    - Signal flow indicators
    - Junction points
    """
    
    def __init__(self):
        """Initialize the symbol recognizer."""
        # Template matching thresholds
        self.match_threshold = 0.6
        self.nms_threshold = 0.3
        
        # Symbol templates (in practice, these would be loaded from files)
        self.templates = self._initialize_templates()
        
        # Shape analysis parameters
        self.min_contour_area = 50
        self.max_contour_area = 10000
        
    def detect_symbols(self, img: np.ndarray) -> List[str]:
        """
        Detect symbols in the image and return symbol descriptions.
        
        Args:
            img: Input image region containing diagrams
            
        Returns:
            List of detected symbol descriptions
        """
        try:
            symbols = []
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Preprocess the image
            processed = self._preprocess_for_symbols(gray)
            
            # Template-based detection
            template_symbols = self._detect_with_templates(processed)
            symbols.extend(template_symbols)
            
            # Shape-based detection
            shape_symbols = self._detect_by_shape_analysis(processed)
            symbols.extend(shape_symbols)
            
            # Connection detection
            connections = self._detect_connections(processed)
            symbols.extend(connections)
            
            # Remove duplicates and filter
            unique_symbols = self._filter_detections(symbols)
            
            # Convert to descriptions
            descriptions = [self._symbol_to_description(symbol) for symbol in unique_symbols]
            
            logger.debug(f"Detected {len(descriptions)} symbols")
            return descriptions
            
        except Exception as e:
            logger.error(f"Error in symbol detection: {e}")
            return []
    
    def _initialize_templates(self) -> Dict[str, np.ndarray]:
        """Initialize symbol templates for matching."""
        # In a full implementation, these would be loaded from template files
        # For now, we'll create simple geometric templates
        templates = {}
        
        # AND gate template (simplified)
        and_template = np.zeros((30, 40), dtype=np.uint8)
        cv2.rectangle(and_template, (5, 5), (25, 25), 255, -1)
        cv2.ellipse(and_template, (25, 15), (10, 10), 0, -90, 90, 255, -1)
        templates['AND_GATE'] = and_template
        
        # OR gate template (simplified)
        or_template = np.zeros((30, 40), dtype=np.uint8)
        cv2.ellipse(or_template, (20, 15), (15, 15), 0, 0, 180, 255, 2)
        cv2.ellipse(or_template, (10, 15), (8, 15), 0, 0, 180, 255, 2)
        templates['OR_GATE'] = or_template
        
        # Switch template
        switch_template = np.zeros((20, 30), dtype=np.uint8)
        cv2.line(switch_template, (5, 10), (15, 10), 255, 2)
        cv2.line(switch_template, (15, 10), (20, 5), 255, 2)
        cv2.circle(switch_template, (5, 10), 3, 255, -1)
        cv2.circle(switch_template, (25, 10), 3, 255, -1)
        templates['SWITCH'] = switch_template
        
        return templates
    
    def _preprocess_for_symbols(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for symbol detection."""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _detect_with_templates(self, img: np.ndarray) -> List[DetectedSymbol]:
        """Detect symbols using template matching."""
        detected = []
        
        for symbol_type, template in self.templates.items():
            # Multi-scale template matching
            for scale in [0.8, 1.0, 1.2, 1.5]:
                # Resize template
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_template = cv2.resize(template, (new_w, new_h))
                
                # Template matching
                result = cv2.matchTemplate(img, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= self.match_threshold)
                
                for y, x in zip(*locations):
                    confidence = result[y, x]
                    bbox = (x, y, new_w, new_h)
                    
                    symbol = DetectedSymbol(
                        symbol_type=symbol_type,
                        confidence=confidence,
                        bbox=bbox,
                        properties={'scale': scale, 'method': 'template'}
                    )
                    detected.append(symbol)
        
        return detected
    
    def _detect_by_shape_analysis(self, img: np.ndarray) -> List[DetectedSymbol]:
        """Detect symbols using shape analysis."""
        detected = []
        
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Analyze shape properties
            symbol_type = self._analyze_contour_shape(contour)
            
            if symbol_type:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on shape matching quality
                confidence = self._calculate_shape_confidence(contour, symbol_type)
                
                symbol = DetectedSymbol(
                    symbol_type=symbol_type,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    properties={'area': area, 'method': 'shape'}
                )
                detected.append(symbol)
        
        return detected
    
    def _analyze_contour_shape(self, contour: np.ndarray) -> Optional[str]:
        """Analyze contour shape to determine symbol type."""
        # Calculate shape descriptors
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return None
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Classify based on shape features
        if 0.7 < circularity < 1.3:
            return 'JUNCTION_POINT'
        elif 0.3 < aspect_ratio < 0.7 and solidity > 0.8:
            return 'LOGIC_BLOCK'
        elif aspect_ratio > 2.0 and solidity > 0.9:
            return 'CONNECTOR_LINE'
        elif 1.2 < aspect_ratio < 2.0 and 0.6 < solidity < 0.9:
            return 'SWITCH_SYMBOL'
        
        return None
    
    def _calculate_shape_confidence(self, contour: np.ndarray, symbol_type: str) -> float:
        """Calculate confidence score for shape-based detection."""
        # Basic confidence calculation based on contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Normalize by expected ranges for each symbol type
        if symbol_type == 'JUNCTION_POINT':
            # For circular objects, higher circularity = higher confidence
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            return min(circularity, 1.0)
        elif symbol_type == 'LOGIC_BLOCK':
            # For rectangular objects, check how close to rectangle
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            return min(rectangularity * 1.2, 1.0)
        else:
            # Default confidence
            return 0.6
    
    def _detect_connections(self, img: np.ndarray) -> List[DetectedSymbol]:
        """Detect connection lines and arrows."""
        detected = []
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vertical_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line contours
        h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process horizontal lines
        for contour in h_contours:
            if cv2.contourArea(contour) > 100:  # Minimum line length
                x, y, w, h = cv2.boundingRect(contour)
                if w > h * 3:  # Ensure it's horizontal
                    symbol = DetectedSymbol(
                        symbol_type='HORIZONTAL_CONNECTION',
                        confidence=0.8,
                        bbox=(x, y, w, h),
                        properties={'orientation': 'horizontal', 'method': 'line_detection'}
                    )
                    detected.append(symbol)
        
        # Process vertical lines
        for contour in v_contours:
            if cv2.contourArea(contour) > 100:  # Minimum line length
                x, y, w, h = cv2.boundingRect(contour)
                if h > w * 3:  # Ensure it's vertical
                    symbol = DetectedSymbol(
                        symbol_type='VERTICAL_CONNECTION',
                        confidence=0.8,
                        bbox=(x, y, w, h),
                        properties={'orientation': 'vertical', 'method': 'line_detection'}
                    )
                    detected.append(symbol)
        
        return detected
    
    def _filter_detections(self, symbols: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """Filter and remove duplicate/overlapping detections."""
        if not symbols:
            return symbols
        
        # Sort by confidence (highest first)
        symbols.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for symbol in symbols:
            # Check for significant overlap with already accepted symbols
            overlaps = False
            sx, sy, sw, sh = symbol.bbox
            
            for accepted in filtered:
                ax, ay, aw, ah = accepted.bbox
                
                # Calculate intersection
                ix1, iy1 = max(sx, ax), max(sy, ay)
                ix2, iy2 = min(sx + sw, ax + aw), min(sy + sh, ay + ah)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    symbol_area = sw * sh
                    overlap_ratio = intersection_area / symbol_area if symbol_area > 0 else 0
                    
                    # If significant overlap, skip this symbol
                    if overlap_ratio > self.nms_threshold:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(symbol)
        
        return filtered
    
    def _symbol_to_description(self, symbol: DetectedSymbol) -> str:
        """Convert detected symbol to text description."""
        symbol_descriptions = {
            'AND_GATE': 'AND gate logic block',
            'OR_GATE': 'OR gate logic block',
            'SWITCH': 'Switch element',
            'JUNCTION_POINT': 'Signal junction point',
            'LOGIC_BLOCK': 'Generic logic block',
            'CONNECTOR_LINE': 'Signal connection line',
            'SWITCH_SYMBOL': 'Switch or control element',
            'HORIZONTAL_CONNECTION': 'Horizontal signal line',
            'VERTICAL_CONNECTION': 'Vertical signal line'
        }
        
        description = symbol_descriptions.get(symbol.symbol_type, f"Unknown symbol ({symbol.symbol_type})")
        confidence_text = f" (confidence: {symbol.confidence:.2f})"
        
        return description + confidence_text