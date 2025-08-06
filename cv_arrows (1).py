#!/usr/bin/env python3
"""
Computer Vision Arrow Detection Module
Advanced arrow path detection using template matching, Hough transforms,
and contour analysis for FDEF signal flow diagrams.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class ArrowDirection(Enum):
    """Arrow direction enumeration"""
    RIGHT = "right"
    LEFT = "left" 
    UP = "up"
    DOWN = "down"
    DIAGONAL_UR = "diagonal_up_right"
    DIAGONAL_UL = "diagonal_up_left"
    DIAGONAL_DR = "diagonal_down_right"
    DIAGONAL_DL = "diagonal_down_left"

@dataclass
class ArrowDetection:
    """Container for arrow detection results"""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    direction: ArrowDirection
    confidence: float
    arrow_type: str  # 'straight', 'curved', 'dashed'
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    metadata: Dict[str, Any]

@dataclass
class BlockDetection:
    """Container for functional block detection results"""
    center_point: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    block_type: str  # 'rectangle', 'circle', 'diamond', 'unknown'
    confidence: float
    text_content: str
    metadata: Dict[str, Any]

class CvArrowDetector:
    """
    Advanced computer vision arrow detector for FDEF diagrams
    """
    
    def __init__(self):
        self.arrow_templates = self._create_arrow_templates()
        self.detection_params = self._init_detection_parameters()
        
        logger.info("🏹 Computer Vision Arrow Detector initialized")
        
    def _create_arrow_templates(self) -> Dict[ArrowDirection, np.ndarray]:
        """Create arrow templates for template matching"""
        
        templates = {}
        
        # Base arrow size
        size = 20
        
        # Right arrow
        right_arrow = np.zeros((size, size), dtype=np.uint8)
        points = np.array([[2, size//2], [size-5, size//2-3], [size-5, size//2-1], 
                          [size-2, size//2], [size-5, size//2+1], [size-5, size//2+3]], np.int32)
        cv2.fillPoly(right_arrow, [points], 255)
        templates[ArrowDirection.RIGHT] = right_arrow
        
        # Left arrow (flip horizontally)
        templates[ArrowDirection.LEFT] = cv2.flip(right_arrow, 1)
        
        # Up arrow (rotate 90 degrees counterclockwise)
        templates[ArrowDirection.UP] = cv2.rotate(right_arrow, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Down arrow (rotate 90 degrees clockwise)
        templates[ArrowDirection.DOWN] = cv2.rotate(right_arrow, cv2.ROTATE_90_CLOCKWISE)
        
        # Diagonal arrows (rotate 45 degrees)
        M_45 = cv2.getRotationMatrix2D((size//2, size//2), 45, 1)
        templates[ArrowDirection.DIAGONAL_UR] = cv2.warpAffine(right_arrow, M_45, (size, size))
        
        M_135 = cv2.getRotationMatrix2D((size//2, size//2), 135, 1)
        templates[ArrowDirection.DIAGONAL_UL] = cv2.warpAffine(right_arrow, M_135, (size, size))
        
        M_315 = cv2.getRotationMatrix2D((size//2, size//2), -45, 1)
        templates[ArrowDirection.DIAGONAL_DR] = cv2.warpAffine(right_arrow, M_315, (size, size))
        
        M_225 = cv2.getRotationMatrix2D((size//2, size//2), -135, 1)
        templates[ArrowDirection.DIAGONAL_DL] = cv2.warpAffine(right_arrow, M_225, (size, size))
        
        logger.debug(f"   Created {len(templates)} arrow templates")
        return templates
        
    def _init_detection_parameters(self) -> Dict[str, Any]:
        """Initialize detection parameters"""
        
        return {
            'template_threshold': 0.6,
            'nms_threshold': 0.3,
            'line_threshold': 100,
            'min_line_length': 30,
            'max_line_gap': 10,
            'contour_min_area': 50,
            'contour_max_area': 5000,
            'block_min_area': 500,
            'block_max_area': 50000
        }
        
    def detect_arrow_paths(self, image: np.ndarray) -> List[ArrowDetection]:
        """
        Detect arrow paths in the image using multiple techniques
        """
        
        logger.info("🔍 Detecting arrow paths...")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Multiple detection approaches
        template_arrows = self._detect_template_arrows(gray)
        line_arrows = self._detect_line_based_arrows(gray)
        contour_arrows = self._detect_contour_arrows(gray)
        
        # Combine and filter results
        all_arrows = template_arrows + line_arrows + contour_arrows
        filtered_arrows = self._filter_and_merge_detections(all_arrows)
        
        logger.info(f"   Found {len(filtered_arrows)} arrow paths")
        return filtered_arrows
        
    def _detect_template_arrows(self, gray_image: np.ndarray) -> List[ArrowDetection]:
        """Detect arrows using template matching"""
        
        detections = []
        
        for direction, template in self.arrow_templates.items():
            # Multi-scale template matching
            for scale in [0.8, 1.0, 1.2, 1.5]:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                
                # Template matching
                result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= self.detection_params['template_threshold'])
                
                for pt in zip(*locations[::-1]):  # Switch x and y
                    confidence = result[pt[1], pt[0]]
                    
                    # Calculate arrow endpoints based on direction
                    h, w = scaled_template.shape
                    start_pt, end_pt = self._calculate_arrow_endpoints(pt, (w, h), direction)
                    
                    detection = ArrowDetection(
                        start_point=start_pt,
                        end_point=end_pt,
                        direction=direction,
                        confidence=float(confidence),
                        arrow_type='straight',
                        bounding_box=(pt[0], pt[1], w, h),
                        metadata={
                            'detection_method': 'template_matching',
                            'template_scale': scale,
                            'template_size': (w, h)
                        }
                    )
                    
                    detections.append(detection)
                    
        logger.debug(f"   Template matching found {len(detections)} candidates")
        return detections
        
    def _detect_line_based_arrows(self, gray_image: np.ndarray) -> List[ArrowDetection]:
        """Detect arrows by analyzing line segments and their endpoints"""
        
        detections = []
        
        # Edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.detection_params['line_threshold'],
            minLineLength=self.detection_params['min_line_length'],
            maxLineGap=self.detection_params['max_line_gap']
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check for arrowheads at line endpoints
                arrow_detection = self._analyze_line_for_arrow(gray_image, (x1, y1), (x2, y2))
                
                if arrow_detection:
                    detections.append(arrow_detection)
                    
        logger.debug(f"   Line-based detection found {len(detections)} candidates")
        return detections
        
    def _detect_contour_arrows(self, gray_image: np.ndarray) -> List[ArrowDetection]:
        """Detect arrows using contour analysis"""
        
        detections = []
        
        # Threshold image
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if (self.detection_params['contour_min_area'] < area < 
                self.detection_params['contour_max_area']):
                
                # Analyze contour shape for arrow-like properties
                arrow_detection = self._analyze_contour_for_arrow(contour)
                
                if arrow_detection:
                    detections.append(arrow_detection)
                    
        logger.debug(f"   Contour-based detection found {len(detections)} candidates")
        return detections
        
    def _analyze_line_for_arrow(self, image: np.ndarray, start: Tuple[int, int], 
                               end: Tuple[int, int]) -> Optional[ArrowDetection]:
        """Analyze a line segment to determine if it's part of an arrow"""
        
        x1, y1 = start
        x2, y2 = end
        
        # Calculate line properties
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = math.atan2(y2-y1, x2-x1)
        
        # Check for arrowhead at endpoint
        arrowhead_size = min(20, length * 0.2)
        
        # Sample region around endpoint for arrowhead detection
        region_size = int(arrowhead_size * 1.5)
        
        # Ensure we don't go out of bounds
        y_start = max(0, y2 - region_size//2)
        y_end = min(image.shape[0], y2 + region_size//2)
        x_start = max(0, x2 - region_size//2)
        x_end = min(image.shape[1], x2 + region_size//2)
        
        if y_end > y_start and x_end > x_start:
            endpoint_region = image[y_start:y_end, x_start:x_end]
            
            # Simple arrowhead detection using pixel density
            if endpoint_region.size > 0:
                pixel_density = np.mean(endpoint_region < 128)  # Assuming dark arrows on light background
                
                if pixel_density > 0.3:  # Threshold for arrowhead presence
                    # Determine direction based on line angle
                    direction = self._angle_to_direction(angle)
                    
                    return ArrowDetection(
                        start_point=start,
                        end_point=end,
                        direction=direction,
                        confidence=pixel_density,
                        arrow_type='straight',
                        bounding_box=(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)),
                        metadata={
                            'detection_method': 'line_analysis',
                            'line_length': length,
                            'line_angle': angle,
                            'pixel_density': pixel_density
                        }
                    )
        
        return None
        
    def _analyze_contour_for_arrow(self, contour: np.ndarray) -> Optional[ArrowDetection]:
        """Analyze a contour to determine if it represents an arrow"""
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Arrows typically have 5-7 vertices when approximated
        if 5 <= len(approx) <= 8:
            # Calculate contour properties
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
                
                # Find the most distant point (likely arrow tip)
                distances = [cv2.pointPolygonTest(contour, (cx, cy), True) for point in approx]
                tip_idx = np.argmax([np.linalg.norm(point[0] - [cx, cy]) for point in approx])
                tip_point = tuple(approx[tip_idx][0])
                
                # Estimate arrow direction
                direction_vector = (tip_point[0] - cx, tip_point[1] - cy)
                angle = math.atan2(direction_vector[1], direction_vector[0])
                direction = self._angle_to_direction(angle)
                
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Estimate start point (opposite to tip)
                start_point = (2*cx - tip_point[0], 2*cy - tip_point[1])
                
                return ArrowDetection(
                    start_point=start_point,
                    end_point=tip_point,
                    direction=direction,
                    confidence=0.7,  # Medium confidence for contour-based detection
                    arrow_type='straight',
                    bounding_box=(x, y, w, h),
                    metadata={
                        'detection_method': 'contour_analysis',
                        'vertices': len(approx),
                        'contour_area': cv2.contourArea(contour),
                        'centroid': (cx, cy)
                    }
                )
        
        return None
        
    def _angle_to_direction(self, angle: float) -> ArrowDirection:
        """Convert angle in radians to ArrowDirection enum"""
        
        # Normalize angle to [0, 2π]
        angle = angle % (2 * math.pi)
        
        # Convert to degrees for easier thresholding
        degrees = math.degrees(angle)
        
        if -22.5 <= degrees <= 22.5 or degrees >= 337.5:
            return ArrowDirection.RIGHT
        elif 22.5 < degrees <= 67.5:
            return ArrowDirection.DIAGONAL_DR
        elif 67.5 < degrees <= 112.5:
            return ArrowDirection.DOWN
        elif 112.5 < degrees <= 157.5:
            return ArrowDirection.DIAGONAL_DL
        elif 157.5 < degrees <= 202.5:
            return ArrowDirection.LEFT
        elif 202.5 < degrees <= 247.5:
            return ArrowDirection.DIAGONAL_UL
        elif 247.5 < degrees <= 292.5:
            return ArrowDirection.UP
        else:  # 292.5 < degrees < 337.5
            return ArrowDirection.DIAGONAL_UR
            
    def _calculate_arrow_endpoints(self, top_left: Tuple[int, int], size: Tuple[int, int], 
                                  direction: ArrowDirection) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate start and end points of arrow based on direction"""
        
        x, y = top_left
        w, h = size
        
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Define start and end points based on direction
        if direction == ArrowDirection.RIGHT:
            start_pt = (x, center_y)
            end_pt = (x + w, center_y)
        elif direction == ArrowDirection.LEFT:
            start_pt = (x + w, center_y)
            end_pt = (x, center_y)
        elif direction == ArrowDirection.UP:
            start_pt = (center_x, y + h)
            end_pt = (center_x, y)
        elif direction == ArrowDirection.DOWN:
            start_pt = (center_x, y)
            end_pt = (center_x, y + h)
        elif direction == ArrowDirection.DIAGONAL_UR:
            start_pt = (x, y + h)
            end_pt = (x + w, y)
        elif direction == ArrowDirection.DIAGONAL_UL:
            start_pt = (x + w, y + h)
            end_pt = (x, y)
        elif direction == ArrowDirection.DIAGONAL_DR:
            start_pt = (x, y)
            end_pt = (x + w, y + h)
        else:  # DIAGONAL_DL
            start_pt = (x + w, y)
            end_pt = (x, y + h)
            
        return start_pt, end_pt
        
    def _filter_and_merge_detections(self, detections: List[ArrowDetection]) -> List[ArrowDetection]:
        """Filter overlapping detections and merge similar ones"""
        
        if not detections:
            return []
            
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Non-maximum suppression
        filtered = []
        
        for detection in detections:
            # Check if this detection overlaps significantly with any already accepted detection
            overlap_found = False
            
            for accepted in filtered:
                if self._detections_overlap(detection, accepted):
                    overlap_found = True
                    break
                    
            if not overlap_found:
                filtered.append(detection)
                
        return filtered
        
    def _detections_overlap(self, det1: ArrowDetection, det2: ArrowDetection) -> bool:
        """Check if two arrow detections overlap significantly"""
        
        # Calculate intersection over union (IoU) of bounding boxes
        x1, y1, w1, h1 = det1.bounding_box
        x2, y2, w2, h2 = det2.bounding_box
        
        # Calculate intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = max(0, min(x1 + w1, x2 + w2) - xi)
        hi = max(0, min(y1 + h1, y2 + h2) - yi)
        
        intersection = wi * hi
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # IoU threshold for overlap
        iou = intersection / union if union > 0 else 0
        
        return iou > self.detection_params['nms_threshold']
        
    def detect_blocks(self, image: np.ndarray) -> List[BlockDetection]:
        """
        Detect functional blocks (rectangles, circles, diamonds) in the image
        """
        
        logger.info("🔍 Detecting functional blocks...")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        detections = []
        
        # Detect rectangles
        rect_detections = self._detect_rectangles(gray)
        detections.extend(rect_detections)
        
        # Detect circles
        circle_detections = self._detect_circles(gray)
        detections.extend(circle_detections)
        
        # Filter overlapping detections
        filtered_detections = self._filter_block_detections(detections)
        
        logger.info(f"   Found {len(filtered_detections)} functional blocks")
        return filtered_detections
        
    def _detect_rectangles(self, gray_image: np.ndarray) -> List[BlockDetection]:
        """Detect rectangular blocks"""
        
        detections = []
        
        # Edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if (self.detection_params['block_min_area'] < area < 
                self.detection_params['block_max_area']):
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Rectangles should have 4 vertices
                if len(approx) == 4:
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's roughly rectangular
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.2 <= aspect_ratio <= 5.0:  # Reasonable aspect ratio range
                        detection = BlockDetection(
                            center_point=(x + w//2, y + h//2),
                            bounding_box=(x, y, w, h),
                            block_type='rectangle',
                            confidence=0.8,
                            text_content="",  # Will be filled by OCR if needed
                            metadata={
                                'contour_area': area,
                                'aspect_ratio': aspect_ratio,
                                'vertices': len(approx)
                            }
                        )
                        
                        detections.append(detection)
                        
        return detections
        
    def _detect_circles(self, gray_image: np.ndarray) -> List[BlockDetection]:
        """Detect circular blocks"""
        
        detections = []
        
        # Hough circle detection
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Check if circle area is within acceptable range
                area = math.pi * r * r
                
                if (self.detection_params['block_min_area'] < area < 
                    self.detection_params['block_max_area']):
                    
                    detection = BlockDetection(
                        center_point=(x, y),
                        bounding_box=(x-r, y-r, 2*r, 2*r),
                        block_type='circle',
                        confidence=0.7,
                        text_content="",
                        metadata={
                            'radius': r,
                            'area': area
                        }
                    )
                    
                    detections.append(detection)
                    
        return detections
        
    def _filter_block_detections(self, detections: List[BlockDetection]) -> List[BlockDetection]:
        """Filter overlapping block detections"""
        
        if not detections:
            return []
            
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        
        for detection in detections:
            # Check for overlap with already accepted detections
            overlap_found = False
            
            for accepted in filtered:
                if self._blocks_overlap(detection, accepted):
                    overlap_found = True
                    break
                    
            if not overlap_found:
                filtered.append(detection)
                
        return filtered
        
    def _blocks_overlap(self, block1: BlockDetection, block2: BlockDetection) -> bool:
        """Check if two block detections overlap significantly"""
        
        x1, y1, w1, h1 = block1.bounding_box
        x2, y2, w2, h2 = block2.bounding_box
        
        # Calculate intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = max(0, min(x1 + w1, x2 + w2) - xi)
        hi = max(0, min(y1 + h1, y2 + h2) - yi)
        
        intersection = wi * hi
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # IoU threshold
        iou = intersection / union if union > 0 else 0
        
        return iou > 0.3  # Lower threshold for blocks
        
    def visualize_detections(self, image: np.ndarray, arrows: List[ArrowDetection], 
                           blocks: List[BlockDetection]) -> np.ndarray:
        """Visualize detected arrows and blocks on the image"""
        
        vis_image = image.copy()
        
        # Draw arrows
        for arrow in arrows:
            # Draw arrow line
            cv2.arrowedLine(
                vis_image,
                arrow.start_point,
                arrow.end_point,
                (0, 255, 0),  # Green
                2,
                tipLength=0.3
            )
            
            # Draw bounding box
            x, y, w, h = arrow.bounding_box
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            # Add confidence text
            cv2.putText(
                vis_image,
                f"{arrow.confidence:.2f}",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )
            
        # Draw blocks
        for block in blocks:
            x, y, w, h = block.bounding_box
            
            if block.block_type == 'rectangle':
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue
            elif block.block_type == 'circle':
                center = block.center_point
                radius = w // 2
                cv2.circle(vis_image, center, radius, (255, 0, 0), 2)  # Blue
                
            # Add confidence text
            cv2.putText(
                vis_image,
                f"{block.block_type}: {block.confidence:.2f}",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )
            
        return vis_image

if __name__ == "__main__":
    # Test the arrow detector
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CV Arrow Detector')
    parser.add_argument('image_file', help='Image file to process')
    parser.add_argument('--output', help='Output image file')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load image
    image = cv2.imread(args.image_file)
    if image is None:
        print(f"Could not load image: {args.image_file}")
        sys.exit(1)
        
    # Detect arrows and blocks
    detector = CvArrowDetector()
    
    arrows = detector.detect_arrow_paths(image)
    blocks = detector.detect_blocks(image)
    
    print(f"Detected {len(arrows)} arrows and {len(blocks)} blocks")
    
    # Visualize results
    vis_image = detector.visualize_detections(image, arrows, blocks)
    
    if args.output:
        cv2.imwrite(args.output, vis_image)
        print(f"Saved visualization to {args.output}")
    else:
        cv2.imshow('Detections', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()