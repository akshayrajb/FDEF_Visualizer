"""
Region Detection for Technical Document Analysis

Automatically detects and classifies different regions in FDEF documents:
- Text regions (paragraphs, labels)
- Diagram regions (wiring diagrams, logic circuits)
- Table regions (data tables, lists)
- Mixed regions (annotated diagrams)
"""

import logging
from typing import List, Tuple
import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class RegionDetector:
    """
    Detects and classifies different content regions in technical documents.
    
    Uses computer vision techniques to identify:
    - Pure text areas
    - Diagrammatic content
    - Tabular data
    - Mixed content areas
    """
    
    def __init__(self):
        """Initialize the region detector."""
        # Morphological kernels for different detection tasks
        self.text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        self.line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        self.line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Thresholds for region classification
        self.min_text_area = 500
        self.min_diagram_area = 1000
        self.line_density_threshold = 0.1
        
    def detect_regions(self, img: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Detect and classify regions in the image.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            List of (region_type, bbox) tuples where bbox is (x, y, w, h)
        """
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Apply preprocessing
            processed = self._preprocess_image(gray)
            
            # Detect different types of regions
            text_regions = self._detect_text_regions(processed)
            diagram_regions = self._detect_diagram_regions(processed)
            table_regions = self._detect_table_regions(processed)
            
            # Combine and resolve overlaps
            all_regions = []
            all_regions.extend([('text', bbox) for bbox in text_regions])
            all_regions.extend([('diagram', bbox) for bbox in diagram_regions])
            all_regions.extend([('table', bbox) for bbox in table_regions])
            
            # Remove overlapping regions (keep the largest)
            filtered_regions = self._resolve_overlaps(all_regions)
            
            logger.debug(f"Detected {len(filtered_regions)} regions")
            return filtered_regions
            
        except Exception as e:
            logger.error(f"Error in region detection: {e}")
            # Return whole image as mixed region if detection fails
            h, w = img.shape[:2]
            return [('mixed', (0, 0, w, h))]
    
    def _preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for region detection."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return thresh
    
    def _detect_text_regions(self, thresh: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions likely to contain text."""
        # Apply morphological operations to connect text
        text_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.text_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if (w * h > self.min_text_area and 
                w > h and  # Text regions are typically wider than tall
                h > 10):   # Minimum height for text
                
                # Check if it looks like text (horizontal orientation)
                aspect_ratio = w / h
                if 2 < aspect_ratio < 20:  # Reasonable text aspect ratios
                    text_regions.append((x, y, w, h))
        
        return text_regions
    
    def _detect_diagram_regions(self, thresh: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions likely to contain diagrams."""
        # Detect horizontal and vertical lines
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.line_kernel_h)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.line_kernel_v)
        
        # Combine lines to find diagram-like structures
        combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Apply dilation to connect nearby diagram elements
        diagram_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        diagram_mask = cv2.dilate(combined_lines, diagram_kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(diagram_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diagram_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and characteristics
            if (w * h > self.min_diagram_area and
                min(w, h) > 50):  # Minimum size for diagrams
                
                # Check line density in the region
                region_mask = thresh[y:y+h, x:x+w]
                line_density = np.sum(region_mask > 0) / (w * h)
                
                if line_density > self.line_density_threshold:
                    diagram_regions.append((x, y, w, h))
        
        return diagram_regions
    
    def _detect_table_regions(self, thresh: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions likely to contain tables."""
        # Detect strong horizontal lines (table borders)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect strong vertical lines (table borders)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        table_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Apply morphological operations to connect table structure
        table_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_mask = cv2.morphologyEx(table_lines, cv2.MORPH_CLOSE, table_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and characteristics
            if (w * h > self.min_text_area and
                w > 100 and h > 50):  # Minimum table dimensions
                
                # Check for rectangular shape
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 10:  # Reasonable table proportions
                    
                    # Verify it has grid-like structure
                    region_h_lines = horizontal_lines[y:y+h, x:x+w]
                    region_v_lines = vertical_lines[y:y+h, x:x+w]
                    
                    h_line_count = np.sum(region_h_lines > 0)
                    v_line_count = np.sum(region_v_lines > 0)
                    
                    if h_line_count > 0 and v_line_count > 0:
                        table_regions.append((x, y, w, h))
        
        return table_regions
    
    def _resolve_overlaps(self, regions: List[Tuple[str, Tuple[int, int, int, int]]]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Resolve overlapping regions by keeping the most specific/largest."""
        if not regions:
            return regions
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r[1][2] * r[1][3], reverse=True)
        
        filtered = []
        for region_type, (x, y, w, h) in regions:
            # Check if this region significantly overlaps with any already kept region
            overlaps = False
            
            for kept_type, (kx, ky, kw, kh) in filtered:
                # Calculate intersection area
                ix1, iy1 = max(x, kx), max(y, ky)
                ix2, iy2 = min(x + w, kx + kw), min(y + h, ky + kh)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    current_area = w * h
                    overlap_ratio = intersection_area / current_area
                    
                    # If significant overlap (>50%), skip this region
                    if overlap_ratio > 0.5:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append((region_type, (x, y, w, h)))
        
        return filtered
    
    def visualize_regions(self, img: np.ndarray, regions: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Create a visualization of detected regions.
        
        Args:
            img: Original image
            regions: Detected regions
            
        Returns:
            Image with regions highlighted
        """
        # Create a copy of the image
        if len(img.shape) == 3:
            result = img.copy()
        else:
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Color map for different region types
        color_map = {
            'text': (0, 255, 0),      # Green
            'diagram': (255, 0, 0),   # Blue  
            'table': (0, 0, 255),     # Red
            'mixed': (255, 255, 0)    # Cyan
        }
        
        # Draw rectangles for each region
        for region_type, (x, y, w, h) in regions:
            color = color_map.get(region_type, (128, 128, 128))
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = region_type.upper()
            cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result