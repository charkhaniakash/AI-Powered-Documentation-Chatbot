"""
File processing service for extracting text from different file formats.
Supports PDF, Markdown, plain text, Excel, CSV, and Word documents.
"""

# Import necessary libraries
import os  # For file operations
import uuid  # For generating unique IDs
from pathlib import Path  # For cross-platform file paths
from typing import Tuple, Optional  # For type hints
import logging  # For logging
import io  # For in-memory file handling

# PDF processing libraries
import PyPDF2  # Simple PDF extraction
import fitz  # PyMuPDF for better PDF handling

# Markdown processing
import markdown  # For parsing markdown
from bs4 import BeautifulSoup  # For extracting text from HTML

# Excel and CSV processing
import pandas as pd  # For Excel and CSV processing
import openpyxl  # For .xlsx files
import xlrd  # For .xls files

# Word document processing
import docx  # python-docx for .docx files
import mammoth  # For better .doc/.docx text extraction

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
            file_type: Type of file (pdf, md, txt, xlsx, xls, csv, docx, doc)
            
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
        elif file_type == FileType.XLSX:
            # Process Excel .xlsx file
            text, page_count = self._process_excel(file_path, is_xlsx=True)
        elif file_type == FileType.XLS:
            # Process Excel .xls file (older format)
            text, page_count = self._process_excel(file_path, is_xlsx=False)
        elif file_type == FileType.CSV:
            # Process CSV file
            text, page_count = self._process_csv(file_path)
        elif file_type == FileType.DOCX:
            # Process Word .docx file (newer format)
            text, page_count = self._process_word(file_path, is_docx=True)
        elif file_type == FileType.DOC:
            # Process Word .doc file (older format)
            text, page_count = self._process_word(file_path, is_docx=False)
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
        
        # Get page count FIRST (before closing document)
        page_count = len(doc)
        
        text_list = []
        
        # Iterate through all pages
        for page_num, page in enumerate(doc, start=1):
            # Extract text from current page
            page_text = page.get_text()
            # Add page separator for clarity
            text_list.append(f"\n--- Page {page_num} ---\n")
            text_list.append(page_text)
        
        # Close the document to free resources
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
            
            # Iterate through all pages
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                # Extract text from current page
                page_text = page.extract_text()
                # Add page separator
                text_list.append(f"\n--- Page {page_num} ---\n")
                text_list.append(page_text)
            
            # Combine all text
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
    
    
    def _process_excel(self, file_path: str, is_xlsx: bool = True) -> Tuple[str, int]:
        """
        Extract text from Excel file (.xlsx or .xls).
        
        Reads all sheets and converts data to text format.
        Each sheet is treated as a section.
        
        Args:
            file_path: Path to Excel file
            is_xlsx: True for .xlsx (newer format), False for .xls (older format)
            
        Returns:
            Tuple of (extracted_text, sheet_count)
            
        Raises:
            IOError: If Excel file cannot be read
        """
        logger.debug(f"Processing Excel file: {file_path}")
        
        try:
            # Read Excel file with pandas
            # sheet_name=None reads all sheets into a dictionary
            # engine='openpyxl' for .xlsx, 'xlrd' for .xls
            if is_xlsx:
                # For newer Excel format (.xlsx)
                excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            else:
                # For older Excel format (.xls)
                excel_data = pd.read_excel(file_path, sheet_name=None, engine='xlrd')
            
            text_list = []
            # Count number of sheets
            sheet_count = len(excel_data)
            
            # Process each sheet
            # excel_data is a dict: {sheet_name: dataframe}
            for sheet_name, df in excel_data.items():
                # Add sheet header
                text_list.append(f"\n=== Sheet: {sheet_name} ===\n")
                
                # Convert dataframe to string
                # to_string() creates a formatted text representation
                # index=False removes row numbers
                # na_rep='' replaces NaN values with empty string
                sheet_text = df.to_string(index=False, na_rep='')
                text_list.append(sheet_text)
                text_list.append("\n")
            
            # Combine all sheets into one text
            full_text = "".join(text_list)
            
            return full_text, sheet_count
            
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Failed to process Excel file {file_path}: {str(e)}")
            raise IOError(f"Failed to read Excel file: {str(e)}")
    
    
    def _process_csv(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from CSV file.
        
        Reads CSV and converts to formatted text.
        Handles different encodings automatically.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (extracted_text, row_count)
            
        Raises:
            IOError: If CSV file cannot be read
        """
        logger.debug(f"Processing CSV file: {file_path}")
        
        try:
            # Try different encodings for CSV
            # CSV files can have various encodings depending on source
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            # Try each encoding until one works
            for encoding in encodings:
                try:
                    # Read CSV with pandas
                    # pandas.read_csv() automatically handles delimiters
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.debug(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    # This encoding didn't work, try next one
                    continue
            
            # If no encoding worked
            if df is None:
                raise IOError("Failed to read CSV with any encoding")
            
            # Convert to text
            # to_string() creates formatted text output
            # index=False removes row numbers
            # na_rep='' replaces NaN with empty string
            text = df.to_string(index=False, na_rep='')
            
            # Row count (excluding header)
            # len(df) gives number of data rows
            row_count = len(df)
            
            return text, row_count
            
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Failed to process CSV {file_path}: {str(e)}")
            raise IOError(f"Failed to read CSV: {str(e)}")
    
    
    def _process_word(self, file_path: str, is_docx: bool = True) -> Tuple[str, int]:
        """
        Extract text from Word document (.docx or .doc).
        
        Uses python-docx for .docx files and mammoth for .doc files.
        Extracts both paragraphs and tables.
        
        Args:
            file_path: Path to Word document
            is_docx: True for .docx (newer format), False for .doc (older format)
            
        Returns:
            Tuple of (extracted_text, paragraph_count)
            
        Raises:
            IOError: If Word document cannot be read
        """
        logger.debug(f"Processing Word document: {file_path}")
        
        try:
            if is_docx:
                # Process .docx with python-docx
                # docx.Document() opens the Word file
                doc = docx.Document(file_path)
                
                text_list = []
                paragraph_count = 0
                
                # Extract paragraphs
                # doc.paragraphs gives list of all paragraph objects
                for para in doc.paragraphs:
                    if para.text.strip():  # Only add non-empty paragraphs
                        text_list.append(para.text)
                        paragraph_count += 1
                
                # Extract tables
                # Word documents can contain tables with structured data
                for table in doc.tables:
                    text_list.append("\n--- Table ---")
                    # Iterate through rows
                    for row in table.rows:
                        # Join cells with pipe separator
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        if row_text.strip():
                            text_list.append(row_text)
                    text_list.append("--- End Table ---\n")
                
                # Join all text with newlines
                full_text = "\n".join(text_list)
                
            else:
                # Process .doc with mammoth
                # mammoth converts .doc to HTML, then extracts text
                # Handles older Word format better
                with open(file_path, 'rb') as docx_file:
                    # extract_raw_text() gets plain text without HTML
                    result = mammoth.extract_raw_text(docx_file)
                    full_text = result.value
                
                # Count paragraphs (rough estimate)
                # Split by newlines and count non-empty lines
                paragraph_count = len([p for p in full_text.split('\n') if p.strip()])
            
            # Ensure we have at least 1 paragraph
            if paragraph_count == 0:
                paragraph_count = 1
            
            return full_text, paragraph_count
            
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Failed to process Word document {file_path}: {str(e)}")
            raise IOError(f"Failed to read Word document: {str(e)}")
    
    
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
            'txt': FileType.TEXT,
            'xlsx': FileType.XLSX,
            'xls': FileType.XLS,
            'csv': FileType.CSV,
            'docx': FileType.DOCX,
            'doc': FileType.DOC
        }
        
        return extension_map[extension]
    
    
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