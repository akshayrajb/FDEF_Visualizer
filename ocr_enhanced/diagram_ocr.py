"""
Main OCR Pipeline for FDEF Technical Diagrams

Processes complex technical documents using multi-stage OCR:
1. Document preprocessing and enhancement
2. Region detection (text vs diagrams)
3. Symbol recognition for technical elements
4. Text extraction with context awareness
5. Post-processing for technical terminology
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract
import ocrmypdf
import fitz  # PyMuPDF

from .region_detector import RegionDetector
from .symbol_recognition import SymbolRecognition

logger = logging.getLogger(__name__)


class DiagramOCR:
    """
    Advanced OCR processor for FDEF technical documents.
    
    Handles complex layouts with mixed content including:
    - Wiring diagrams
    - Logic block diagrams  
    - Technical tables
    - Annotated text
    """
    
    # OCR configuration optimized for technical documents
    OCR_CONFIG = {
        'dpi': 300,
        'optimize': True,
        'pdfa': False,
        'force_ocr': True,
        'redo_ocr': True,
        'clean': True,
        'deskew': True,
        'remove_background': False,
        'unpaper_args': ['--layout', 'double', '--no-noisefilter']
    }
    
    # Tesseract configuration for technical text
    TESSERACT_CONFIG = {
        'config': '--psm 6 -c preserve_interword_spaces=1',
        'lang': 'eng',
        'nice': 0
    }
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize the OCR processor.
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "fdef_ocr"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize sub-components
        self.region_detector = RegionDetector()
        self.symbol_recognizer = SymbolRecognition()
        
        # Cache for processed documents
        self._processed_cache = {}
        
        logger.info(f"DiagramOCR initialized with temp dir: {self.temp_dir}")
    
    def process_pdf(self, pdf_path: Path) -> Path:
        """
        Process a PDF document with enhanced OCR.
        
        Args:
            pdf_path: Path to the input PDF
            
        Returns:
            Path to the processed PDF with enhanced text layer
        """
        pdf_path = Path(pdf_path)
        
        # Check cache first
        cache_key = f"{pdf_path.name}_{pdf_path.stat().st_mtime}"
        if cache_key in self._processed_cache:
            logger.info(f"Using cached result for {pdf_path.name}")
            return self._processed_cache[cache_key]
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Create output path
            output_path = self.temp_dir / f"processed_{pdf_path.name}"
            
            # Step 1: Initial OCR processing with OCRmyPDF
            intermediate_path = self._preprocess_with_ocrmypdf(pdf_path)
            
            # Step 2: Enhanced processing for technical content
            final_path = self._enhance_technical_content(intermediate_path, output_path)
            
            # Cache the result
            self._processed_cache[cache_key] = final_path
            
            logger.info(f"PDF processing completed: {output_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            # Return original if processing fails
            return pdf_path
    
    def _preprocess_with_ocrmypdf(self, pdf_path: Path) -> Path:
        """
        Initial preprocessing using OCRmyPDF.
        
        Args:
            pdf_path: Input PDF path
            
        Returns:
            Path to preprocessed PDF
        """
        output_path = self.temp_dir / f"ocrmypdf_{pdf_path.name}"
        
        try:
            # Configure OCRmyPDF for technical documents
            ocrmypdf.ocr(
                input_file=pdf_path,
                output_file=output_path,
                dpi=self.OCR_CONFIG['dpi'],
                optimize=self.OCR_CONFIG['optimize'],
                pdfa_image_compression=ocrmypdf.Compression.lossless,
                force_ocr=self.OCR_CONFIG['force_ocr'],
                redo_ocr=self.OCR_CONFIG['redo_ocr'],
                clean=self.OCR_CONFIG['clean'],
                deskew=self.OCR_CONFIG['deskew'],
                remove_background=self.OCR_CONFIG['remove_background'],
                progress_bar=False,
                skip_text=False
            )
            
            logger.info(f"OCRmyPDF preprocessing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"OCRmyPDF failed, using original: {e}")
            return pdf_path
    
    def _enhance_technical_content(self, pdf_path: Path, output_path: Path) -> Path:
        """
        Enhanced processing for technical diagrams and text.
        
        Args:
            pdf_path: Input PDF path
            output_path: Output PDF path
            
        Returns:
            Path to enhanced PDF
        """
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract page as image for analysis
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling
                img_data = pix.tobytes("png")
                
                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process the image
                enhanced_text = self._process_page_image(img, page_num)
                
                # Add enhanced text to the page (this is a simplified approach)
                # In a full implementation, we would overlay the text at correct positions
                if enhanced_text:
                    # Add text annotation with enhanced content
                    text_rect = fitz.Rect(10, 10, 200, 100)
                    page.add_freetext_annot(
                        text_rect,
                        enhanced_text[:500],  # Limit text length
                        fontsize=8,
                        fontname="helv",
                        text_color=(0, 0, 1),
                        fill_color=(1, 1, 0.8)
                    )
            
            # Save the enhanced PDF
            doc.save(output_path)
            doc.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error enhancing technical content: {e}")
            # Copy original to output if enhancement fails
            import shutil
            shutil.copy2(pdf_path, output_path)
            return output_path
    
    def _process_page_image(self, img: np.ndarray, page_num: int) -> str:
        """
        Process a single page image for enhanced text extraction.
        
        Args:
            img: OpenCV image array
            page_num: Page number for logging
            
        Returns:
            Extracted and enhanced text
        """
        try:
            # Step 1: Detect different regions
            regions = self.region_detector.detect_regions(img)
            
            # Step 2: Process each region separately
            all_text = []
            
            for region_type, bbox in regions:
                x, y, w, h = bbox
                region_img = img[y:y+h, x:x+w]
                
                if region_type == 'text':
                    # Enhanced text extraction
                    text = self._extract_text_from_region(region_img)
                    if text.strip():
                        all_text.append(f"[TEXT] {text}")
                
                elif region_type == 'diagram':
                    # Symbol recognition and text extraction
                    symbols = self.symbol_recognizer.detect_symbols(region_img)
                    for symbol in symbols:
                        all_text.append(f"[SYMBOL] {symbol}")
                    
                    # Extract any text labels in diagram
                    diagram_text = self._extract_diagram_labels(region_img)
                    if diagram_text.strip():
                        all_text.append(f"[DIAGRAM_TEXT] {diagram_text}")
                
                elif region_type == 'table':
                    # Table-specific processing
                    table_text = self._extract_table_content(region_img)
                    if table_text.strip():
                        all_text.append(f"[TABLE] {table_text}")
            
            # Step 3: Post-process and combine
            combined_text = self._post_process_text("\n".join(all_text))
            
            logger.debug(f"Page {page_num}: extracted {len(combined_text)} characters")
            return combined_text
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return ""
    
    def _extract_text_from_region(self, img: np.ndarray) -> str:
        """Extract text from a text region with preprocessing."""
        try:
            # Preprocess image for better OCR
            processed_img = self._preprocess_text_image(img)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                processed_img,
                config=self.TESSERACT_CONFIG['config'],
                lang=self.TESSERACT_CONFIG['lang']
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from region: {e}")
            return ""
    
    def _extract_diagram_labels(self, img: np.ndarray) -> str:
        """Extract text labels from diagram regions."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to isolate text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Extract text with specialized config for diagrams
            text = pytesseract.image_to_string(
                processed,
                config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_',
                lang='eng'
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting diagram labels: {e}")
            return ""
    
    def _extract_table_content(self, img: np.ndarray) -> str:
        """Extract content from table regions."""
        try:
            # Preprocess for table structure
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect table lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Extract text with table-specific configuration
            text = pytesseract.image_to_string(
                gray,
                config='--psm 6',
                lang='eng'
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting table content: {e}")
            return ""
    
    def _preprocess_text_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better text recognition."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text for technical terminology."""
        # Technical term corrections
        corrections = {
            'PT_': 'PT_',  # Ensure underscore preservation
            'Eng_': 'Eng_',
            'Brk_': 'Brk_',
            'Spd_': 'Spd_',
            'AND': 'AND',
            'OR': 'OR',
            'NOT': 'NOT',
            '->': '->',
            '=>': '=>',
            ':=': ':='
        }
        
        processed_text = text
        for wrong, correct in corrections.items():
            processed_text = processed_text.replace(wrong, correct)
        
        # Remove excessive whitespace
        lines = processed_text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        return '\n'.join(cleaned_lines)
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about processed documents."""
        return {
            'processed_count': len(self._processed_cache),
            'temp_dir': str(self.temp_dir),
            'cache_size': len(self._processed_cache)
        }
    
    def clear_cache(self):
        """Clear the processing cache."""
        self._processed_cache.clear()
        logger.info("OCR processing cache cleared")