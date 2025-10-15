"""
File processing service for extracting text from different file formats.
Supports PDF, Markdown, and plain text files.
"""

# Import necessary libraries
import os  # For file operations
import uuid  # For generating unique IDs
from pathlib import Path  # For cross-platform file paths
from typing import Tuple, Optional  # For type hints
import logging  # For logging

# PDF processing libraries
import PyPDF2  # Simple PDF extraction
import fitz  # PyMuPDF for better PDF handling

# Markdown processing
import markdown  # For parsing markdown
from bs4 import BeautifulSoup  # For extracting text from HTML

# Internal imports
from app.config.settings import settings  # Application settings
from app.models.schemas import DocumentMetadata, FileType  # Data models

# ========== Setup Logging ==========
# Create a logger for this module
logger = logging.getLogger(__name__)
# Set logging level from settings
logger.setLevel(settings.LOG_LEVEL)


# ========== File Processor Class ==========
class FileProcessor:
    """
    Processes uploaded files and extracts text content.
    
    This class handles different file formats and extracts
    clean text that can be chunked and embedded.
    """
    
    def __init__(self):
        """Initialize the file processor."""
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        logger.info("FileProcessor initialized")
    
    
    def process_file(
        self,
        file_path: str,
        filename: str,
        file_type: FileType
    ) -> Tuple[str, DocumentMetadata]:
        """
        Process a file and extract text content.
        
        This is the main entry point for file processing.
        It routes to the appropriate handler based on file type.
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            file_type: Type of file (pdf, md, txt)
            
        Returns:
            Tuple of (extracted_text, document_metadata)
            
        Raises:
            ValueError: If file type is not supported
            IOError: If file cannot be read
        """
        logger.info(f"Processing file: {filename} (type: {file_type})")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise IOError(f"File not found: {file_path}")
        
        # Get file size in bytes
        file_size = os.path.getsize(file_path)
        
        # Check file size limit
        max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({settings.MAX_FILE_SIZE_MB} MB)"
            )
        
        # Generate unique document ID
        # UUID ensures uniqueness across all documents
        document_id = str(uuid.uuid4())
        
        # Route to appropriate processor based on file type
        if file_type == FileType.PDF:
            # Process PDF file
            text, page_count = self._process_pdf(file_path)
        elif file_type == FileType.MARKDOWN:
            # Process Markdown file
            text, page_count = self._process_markdown(file_path)
        elif file_type == FileType.TEXT:
            # Process plain text file
            text, page_count = self._process_text(file_path)
        else:
            # Should never happen due to enum validation
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Clean extracted text
        # Remove extra whitespace and normalize line endings
        text = self._clean_text(text)
        
        # Create metadata object
        metadata = DocumentMetadata(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            document_id=document_id,
            page_count=page_count
        )
        
        logger.info(
            f"Successfully processed {filename}: "
            f"{len(text)} characters, {page_count} pages/sections"
        )
        
        return text, metadata
    
    
    def _process_pdf(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF file.
        
        Uses PyMuPDF (fitz) as primary method because it handles
        text extraction better than PyPDF2, especially for complex PDFs.
        Falls back to PyPDF2 if PyMuPDF fails.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, page_count)
            
        Raises:
            IOError: If PDF cannot be read
        """
        logger.debug(f"Extracting text from PDF: {file_path}")
        
        try:
            # Try PyMuPDF first (better text extraction)
            text, page_count = self._extract_with_pymupdf(file_path)
            
            # If extraction failed or returned empty text, try PyPDF2
            if not text or len(text.strip()) < 10:
                logger.warning("PyMuPDF returned little/no text, trying PyPDF2")
                text, page_count = self._extract_with_pypdf2(file_path)
            
            return text, page_count
            
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Failed to process PDF {file_path}: {str(e)}")
            raise IOError(f"Failed to read PDF: {str(e)}")
    
    
    def _extract_with_pymupdf(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text using PyMuPDF (fitz).
        
        PyMuPDF provides better text extraction, especially for:
        - Complex layouts
        - Multi-column documents
        - Tables and structured data
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        # Open PDF document
        # fitz.open() creates a Document object
        doc = fitz.open(file_path)
    
        text_list = []
    
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text_list.append(f"\n--- Page {page_num} ---\n")
            text_list.append(page_text)
    
            # Get page count BEFORE closing
            page_count = len(doc)
            
            # Now close the document to free resources
            doc.close()
            
            # Join all page texts into single string
            full_text = "".join(text_list)
            
            return full_text, page_count
            
    
    def _extract_with_pypdf2(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text using PyPDF2.
        
        Fallback method when PyMuPDF fails or returns poor results.
        PyPDF2 is simpler but less robust for complex PDFs.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        # Open PDF file in binary read mode
        # 'rb' = read binary
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get page count FIRST (inside the 'with' block)
            page_count = len(pdf_reader.pages)
            
            text_list = []
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()
                text_list.append(f"\n--- Page {page_num} ---\n")
                text_list.append(page_text)
            
            full_text = "".join(text_list)
            
            return full_text, page_count
    
    
    def _process_markdown(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from Markdown file.
        
        Converts Markdown to HTML, then extracts plain text.
        This preserves structure while removing formatting.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            Tuple of (extracted_text, section_count)
        """
        logger.debug(f"Processing Markdown file: {file_path}")
        
        # Read markdown file
        # encoding='utf-8' handles international characters
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read entire file content
            markdown_text = file.read()
        
        # Convert Markdown to HTML
        # markdown.markdown() parses MD syntax and converts to HTML
        # extensions=['extra'] enables GitHub-flavored markdown features
        html_content = markdown.markdown(
            markdown_text,
            extensions=['extra', 'nl2br']  # nl2br = newline to <br>
        )
        
        # Parse HTML to extract plain text
        # BeautifulSoup parses HTML structure
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text, preserving some structure
        # get_text() extracts all text, stripping HTML tags
        # separator='\n' adds newline between elements
        text = soup.get_text(separator='\n')
        
        # Count sections (rough estimate based on headers)
        # Count lines starting with # (markdown headers)
        section_count = markdown_text.count('\n#')
        # If no sections found, count as 1 section
        if section_count == 0:
            section_count = 1
        
        return text, section_count
    
    
    def _process_text(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from plain text file.
        
        Simple read operation with encoding detection.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Tuple of (extracted_text, line_count)
        """
        logger.debug(f"Processing text file: {file_path}")
        
        # Try different encodings to handle various text files
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        text = None
        # Try each encoding until one works
        for encoding in encodings:
            try:
                # Try to read with current encoding
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                # If successful, break loop
                logger.debug(f"Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                # This encoding didn't work, try next one
                continue
        
        # If no encoding worked
        if text is None:
            raise IOError("Failed to read text file with any encoding")
        
        # Count lines for metadata
        # split('\n') creates list of lines
        line_count = len(text.split('\n'))
        
        return text, line_count
    
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Performs several cleanup operations:
        1. Normalize whitespace
        2. Remove excessive line breaks
        3. Fix encoding issues
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with single space
        # This removes excessive whitespace from poor PDF extraction
        import re  # Regular expressions for pattern matching
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with maximum of 2
        # Keeps paragraph breaks but removes excessive spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        # Split into lines, strip each, rejoin
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove null bytes if any
        # Some PDFs have null characters that cause issues
        text = text.replace('\x00', '')
        
        # Remove other problematic characters
        # Replace form feeds and other control characters
        text = text.replace('\f', '\n')  # Form feed to newline
        text = text.replace('\r', '')     # Remove carriage returns
        
        # Final strip to remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    
    def validate_file(self, file_path: str, filename: str) -> FileType:
        """
        Validate file and determine its type.
        
        Checks:
        1. File exists
        2. Extension is allowed
        3. File is readable
        
        Args:
            file_path: Path to file
            filename: Original filename
            
        Returns:
            FileType enum
            
        Raises:
            ValueError: If file is invalid
            IOError: If file cannot be accessed
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise IOError(f"File not found: {file_path}")
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            raise IOError(f"File is not readable: {file_path}")
        
        # Extract file extension
        # Path(filename).suffix gets extension with dot (e.g., '.pdf')
        # [1:] removes the dot
        # lower() normalizes to lowercase
        extension = Path(filename).suffix[1:].lower()
        
        # Check if extension is allowed
        if extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type '.{extension}' not allowed. "
                f"Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Map extension to FileType enum
        extension_map = {
            'pdf': FileType.PDF,
            'md': FileType.MARKDOWN,
            'txt': FileType.TEXT
        }
        
        return extension_map[extension]
    
    
    def save_uploaded_file(
        self,
        file_content: bytes,
        filename: str
    ) -> str:
        """
        Save uploaded file to disk.
        
        Generates unique filename to avoid collisions.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Path to saved file
            
        Raises:
            IOError: If file cannot be saved
        """
        # Generate unique filename using UUID
        # This prevents conflicts if multiple users upload same filename
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Create full path
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        
        try:
            # Write file to disk in binary mode
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Saved file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise IOError(f"Failed to save file: {str(e)}")
    
    
    def cleanup_file(self, file_path: str) -> None:
        """
        Delete a file from disk.
        
        Used to cleanup temporary files after processing.
        
        Args:
            file_path: Path to file to delete
        """
        try:
            # Check if file exists before trying to delete
            if os.path.exists(file_path):
                # Delete the file
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            # Log error but don't raise - cleanup failures shouldn't break flow
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")


# ========== Utility Functions ==========
def get_file_processor() -> FileProcessor:
    """
    Get a FileProcessor instance.
    
    This is a factory function that can be used for dependency injection
    in FastAPI.
    
    Returns:
        FileProcessor instance
    """
    return FileProcessor()