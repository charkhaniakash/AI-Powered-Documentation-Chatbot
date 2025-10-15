"""
Utility functions and helpers for the Documentation Chatbot.
"""

# Import necessary libraries
import os  # For file operations
import hashlib  # For generating hashes
import time  # For timing operations
from datetime import datetime  # For timestamps
from typing import Any, Dict, List, Optional  # For type hints
import logging  # For logging
from functools import wraps  # For decorators

# ========== Setup Logging ==========
logger = logging.getLogger(__name__)


# ========== Timing Decorator ==========
def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Useful for performance monitoring and optimization.
    
    Usage:
        @timing_decorator
        def my_function():
            ...
    """
    @wraps(func)  # Preserves original function metadata
    def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Log execution time
        logger.info(
            f"{func.__name__} completed in {elapsed_time:.2f} seconds"
        )
        
        return result
    
    return wrapper


# ========== File Utilities ==========
def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file for integrity checking.
    
    Useful for:
    - Detecting duplicate documents
    - Verifying file integrity
    - Caching decisions
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex string of file hash
        
    Raises:
        IOError: If file cannot be read
    """
    # Select hash algorithm
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    else:
        hasher = hashlib.sha256()
    
    # Read file in chunks for memory efficiency
    # Important for large files
    chunk_size = 8192  # 8KB chunks
    
    with open(file_path, 'rb') as f:
        # Read and update hash in chunks
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break  # End of file
            hasher.update(chunk)
    
    # Return hex digest
    return hasher.hexdigest()


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Name of file
        
    Returns:
        Extension without dot (lowercase)
    """
    # Split on last dot
    # example.tar.gz -> gz
    _, ext = os.path.splitext(filename)
    # Remove dot and convert to lowercase
    return ext.lstrip('.').lower()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    # Define units
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    # Convert to appropriate unit
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # Format with 2 decimal places
    return f"{size:.2f} {units[unit_index]}"


# ========== Text Utilities ==========
def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Truncate and add suffix
    return text[:max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """
    Clean excessive whitespace from text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    import re
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Simple sentence splitter based on punctuation.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    import re
    
    # Split on sentence-ending punctuation
    # Handles periods, exclamation marks, question marks
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


# ========== Validation Utilities ==========
def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    
    # Simple email regex
    # For production, use a more robust validator
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    
    # Simple URL regex
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


# ========== Data Structure Utilities ==========
def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Example:
        {"a": {"b": 1}} -> {"a_b": 1}
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        # Create new key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # Recursively flatten nested dicts
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Later dictionaries override earlier ones.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        result.update(d)
    
    return result


# ========== Retry Utilities ==========
def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions on exception.
    
    Implements exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    
    Usage:
        @retry_on_exception(max_retries=3, delay=1.0)
        def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # Last attempt, raise exception
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                        f"Retrying in {current_delay}s..."
                    )
                    
                    # Wait before retry
                    time.sleep(current_delay)
                    
                    # Increase delay for next retry
                    current_delay *= backoff
            
            # Should never reach here
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ========== Date/Time Utilities ==========
def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime as string.
    
    Args:
        dt: Datetime object (uses current time if None)
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    
    return dt.strftime(format_str)


def parse_timestamp(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse timestamp string to datetime.
    
    Args:
        timestamp_str: Timestamp string
        format_str: Format string
        
    Returns:
        Datetime object
    """
    return datetime.strptime(timestamp_str, format_str)


def get_time_ago(dt: datetime) -> str:
    """
    Get human-readable "time ago" string.
    
    Args:
        dt: Datetime to compare
        
    Returns:
        Time ago string (e.g., "2 hours ago")
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    # Calculate time units
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)} hours ago"
    elif seconds < 604800:
        return f"{int(seconds / 86400)} days ago"
    else:
        return f"{int(seconds / 604800)} weeks ago"


# ========== Progress Tracking ==========
class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    
    Usage:
        tracker = ProgressTracker(total=100)
        for i in range(100):
            # Do work
            tracker.update(1)
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of task
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, amount: int = 1) -> None:
        """
        Update progress.
        
        Args:
            amount: Amount to increment
        """
        self.current += amount
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print progress bar."""
        # Calculate percentage
        percentage = (self.current / self.total) * 100
        
        # Calculate elapsed and estimated time
        elapsed = time.time() - self.start_time
        if self.current > 0:
            est_total = (elapsed / self.current) * self.total
            est_remaining = est_total - elapsed
        else:
            est_remaining = 0
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * self.current / self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Print (overwrite previous line)
        print(
            f'\r{self.description}: |{bar}| '
            f'{self.current}/{self.total} ({percentage:.1f}%) '
            f'[{elapsed:.1f}s elapsed, {est_remaining:.1f}s remaining]',
            end='',
            flush=True
        )
        
        # Print newline when complete
        if self.current >= self.total:
            print()  # New line
    
    def finish(self) -> None:
        """Mark progress as complete."""
        self.current = self.total
        self._print_progress()


# ========== Logging Utilities ==========
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_exception(e: Exception, context: str = "") -> None:
    """
    Log exception with context.
    
    Args:
        e: Exception to log
        context: Additional context
    """
    import traceback
    
    logger.error(f"Exception in {context}: {str(e)}")
    logger.debug(traceback.format_exc())