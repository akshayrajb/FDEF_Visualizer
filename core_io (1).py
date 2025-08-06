#!/usr/bin/env python3
"""
Core I/O Module for FDEF Analysis
Advanced PDF loading, image preprocessing, and OCR with automotive optimizations.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
import hashlib
import time

# PDF and Image Processing
try:
    import PyPDF2
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install PyPDF2 pdf2image Pillow pytesseract opencv-python numpy")
    sys.exit(1)

logger = logging.getLogger(__name__)

@dataclass
class PageData:
    """Container for processed page data"""
    page_number: int
    original_image: np.ndarray
    processed_image: np.ndarray
    text_content: str
    confidence_score: float
    width: int
    height: int
    metadata: Dict[str, Any]

def validate_tesseract_installation() -> bool:
    """Validate that Tesseract OCR is properly installed"""
    try:
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        logger.info(f"✅ Tesseract OCR found: {version}")
        return True
    except Exception as e:
        logger.error(f"❌ Tesseract OCR not found: {e}")
        return False

class PdfLoader:
    """
    Advanced PDF loader with automotive FDEF optimizations
    """
    
    def __init__(self, pdf_path: str, dpi: int = 200, enhance_images: bool = True):
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self.enhance_images = enhance_images
        self.pages: List[PageData] = []
        
        # Validate inputs
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not self.pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {pdf_path}")
            
        # Calculate file hash for caching
        self.file_hash = self._calculate_file_hash()
        
        logger.info(f"📄 Initializing PDF loader: {self.pdf_path.name}")
        logger.info(f"   DPI: {self.dpi}, Enhancement: {self.enhance_images}")
        
    def _calculate_file_hash(self) -> str:
        """Calculate SHA-256 hash of PDF file for caching"""
        sha256_hash = hashlib.sha256()
        with open(self.pdf_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()[:16]  # First 16 chars
        
    def load_pages(self, max_pages: Optional[int] = None) -> List[PageData]:
        """Load and process all pages from PDF"""
        
        start_time = time.time()
        logger.info("🔄 Loading PDF pages...")
        
        try:
            # Convert PDF to images
            images = self._pdf_to_images(max_pages)
            
            # Process each page
            for i, image in enumerate(images):
                logger.info(f"   Processing page {i+1}/{len(images)}...")
                
                page_data = self._process_page(i + 1, image)
                self.pages.append(page_data)
                
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded {len(self.pages)} pages in {load_time:.2f} seconds")
            
            return self.pages
            
        except Exception as e:
            logger.error(f"❌ Failed to load PDF: {e}")
            raise
            
    def _pdf_to_images(self, max_pages: Optional[int] = None) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        
        try:
            # Adjust DPI based on file size to manage memory
            file_size_mb = self.pdf_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 50:  # Large file
                adjusted_dpi = min(self.dpi, 150)
                logger.info(f"   Large PDF detected ({file_size_mb:.1f}MB), reducing DPI to {adjusted_dpi}")
            else:
                adjusted_dpi = self.dpi
                
            # Convert with timeout and error handling
            images = convert_from_path(
                self.pdf_path,
                dpi=adjusted_dpi,
                first_page=1,
                last_page=max_pages,
                fmt='RGB',
                thread_count=2,
                poppler_path=None  # Use system poppler
            )
            
            logger.info(f"   Converted {len(images)} pages at {adjusted_dpi} DPI")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
            
    def _process_page(self, page_num: int, pil_image: Image.Image) -> PageData:
        """Process a single page: enhance image and extract text"""
        
        # Convert PIL to OpenCV format (RGB -> BGR)
        original_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Create processed version
        if self.enhance_images:
            processed_cv = self._enhance_image(original_cv.copy())
        else:
            processed_cv = original_cv.copy()
            
        # Extract text with OCR
        text_content, confidence = self._extract_text(processed_cv)
        
        # Create page data
        page_data = PageData(
            page_number=page_num,
            original_image=original_cv,
            processed_image=processed_cv,
            text_content=text_content,
            confidence_score=confidence,
            width=original_cv.shape[1],
            height=original_cv.shape[0],
            metadata={
                'file_hash': self.file_hash,
                'dpi': self.dpi,
                'enhanced': self.enhance_images,
                'processing_time': time.time()
            }
        )
        
        logger.debug(f"   Page {page_num}: {len(text_content)} chars, confidence: {confidence:.2f}%")
        return page_data
        
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR accuracy"""
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 1. Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 2. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(denoised)
        
        # 3. Sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        
        # 4. Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morphed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        
        # 5. Adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to BGR for consistency
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = binary
            
        return enhanced
        
    def _extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using Tesseract OCR with automotive optimizations"""
        
        try:
            # Configure Tesseract for technical documents
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-+=/*()[]{}|<>:;,?!@#$%^&~`"\' '
            
            # Extract text with confidence data
            data = pytesseract.image_to_data(
                image, 
                config=custom_config, 
                output_type=pytesseract.Output.DICT
            )
            
            # Filter out low-confidence detections and combine text
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                
                if text and conf > 30:  # Only include reasonably confident text
                    text_parts.append(text)
                    confidences.append(conf)
                    
            # Combine text with appropriate spacing
            full_text = ' '.join(text_parts)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Clean up text for automotive documents
            cleaned_text = self._clean_automotive_text(full_text)
            
            return cleaned_text, avg_confidence
            
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return "", 0.0
            
    def _clean_automotive_text(self, text: str) -> str:
        """Clean and normalize text for automotive FDEF documents"""
        
        # Remove excessive whitespace
        import re
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in automotive documents
        replacements = {
            'l5': '15',    # Common OCR error
            'S5': '55',    # Common OCR error  
            'O0': '00',    # O vs 0 confusion
            'Il': '11',    # I vs 1 confusion
            '|<': 'K',     # Pipe vs K
            '|}': 'H',     # Pipe vs H
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Normalize signal naming patterns
        # Convert common separators to underscores for consistency
        text = re.sub(r'[-\.\s]+', '_', text)
        
        # Remove duplicate underscores
        text = re.sub(r'_{2,}', '_', text)
        
        return text.strip()
        
    def get_page_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded pages"""
        
        if not self.pages:
            return {}
            
        total_chars = sum(len(page.text_content) for page in self.pages)
        avg_confidence = np.mean([page.confidence_score for page in self.pages])
        
        return {
            'total_pages': len(self.pages),
            'total_characters': total_chars,
            'average_confidence': avg_confidence,
            'file_size_mb': self.pdf_path.stat().st_size / (1024 * 1024),
            'file_hash': self.file_hash,
            'processing_dpi': self.dpi
        }
        
    def save_processed_images(self, output_dir: str) -> List[str]:
        """Save processed images for debugging/inspection"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        for page in self.pages:
            # Save original
            orig_file = output_path / f"page_{page.page_number:03d}_original.png"
            cv2.imwrite(str(orig_file), page.original_image)
            saved_files.append(str(orig_file))
            
            # Save processed
            proc_file = output_path / f"page_{page.page_number:03d}_processed.png"
            cv2.imwrite(str(proc_file), page.processed_image)
            saved_files.append(str(proc_file))
            
        logger.info(f"💾 Saved {len(saved_files)} processed images to {output_path}")
        return saved_files

# Utility functions
def detect_automotive_document_type(text_content: str) -> str:
    """Detect the type of automotive document based on content patterns"""
    
    text_lower = text_content.lower()
    
    # FDEF patterns
    if any(pattern in text_lower for pattern in ['function design', 'fdef', 'signal flow', 'dependency']):
        return 'FDEF'
    
    # A2L patterns  
    elif any(pattern in text_lower for pattern in ['a2l', 'characteristic', 'measurement', 'calibration']):
        return 'A2L'
        
    # AUTOSAR patterns
    elif any(pattern in text_lower for pattern in ['autosar', 'swc', 'rte', 'component']):
        return 'AUTOSAR'
        
    # DBC patterns
    elif any(pattern in text_lower for pattern in ['dbc', 'can message', 'signal group']):
        return 'DBC'
        
    else:
        return 'Unknown'

if __name__ == "__main__":
    # Test the PDF loader
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PDF Loader')
    parser.add_argument('pdf_file', help='PDF file to process')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for conversion')
    parser.add_argument('--output', help='Output directory for processed images')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test Tesseract
    if not validate_tesseract_installation():
        print("Please install Tesseract OCR first")
        sys.exit(1)
        
    # Load PDF
    loader = PdfLoader(args.pdf_file, dpi=args.dpi)
    pages = loader.load_pages()
    
    # Print statistics
    stats = loader.get_page_statistics()
    print(f"\nPDF Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # Detect document type
    if pages:
        doc_type = detect_automotive_document_type(pages[0].text_content)
        print(f"\nDetected document type: {doc_type}")
        
    # Save processed images if requested
    if args.output:
        saved_files = loader.save_processed_images(args.output)
        print(f"\nSaved {len(saved_files)} images to {args.output}")