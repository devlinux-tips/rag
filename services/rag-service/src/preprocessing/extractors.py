"""
Document text extraction system supporting multiple file formats.
Provides reliable text extraction from PDF, DOCX, and plain text files
with proper encoding handling and error recovery.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# Optional dependencies - handle gracefully if not available
try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    PYPDF_AVAILABLE = False

try:
    from docx import Document

    DOCX_AVAILABLE = True
except ImportError:
    Document = None
    DOCX_AVAILABLE = False


@dataclass
class ExtractionResult:
    """Result of document text extraction."""

    text: str
    character_count: int
    extraction_method: str
    encoding_used: Optional[str] = None
    pages_processed: Optional[int] = None
    error_details: Optional[str] = None


@dataclass
class ExtractionConfig:
    """Configuration for document extraction."""

    supported_formats: List[str]
    text_encodings: List[str]
    max_file_size_mb: int = 50
    enable_logging: bool = True


@runtime_checkable
class FileSystemProvider(Protocol):
    """Protocol for file system operations to enable testing."""

    def file_exists(self, file_path: Path) -> bool:
        """Check if file exists."""
        ...

    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        ...

    def open_binary(self, file_path: Path) -> bytes:
        """Open file in binary mode."""
        ...

    def open_text(self, file_path: Path, encoding: str) -> str:
        """Open file in text mode with specified encoding."""
        ...


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration access."""

    def get_extraction_config(self) -> Dict[str, Any]:
        """Get extraction configuration."""
        ...


@runtime_checkable
class LoggerProvider(Protocol):
    """Protocol for logging operations."""

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def error(self, message: str) -> None:
        """Log error message."""
        ...


# ================================
# PURE BUSINESS LOGIC FUNCTIONS
# ================================


def validate_file_format(
    file_path: Path, supported_formats: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate if file format is supported.

    Args:
        file_path: Path to document file
        supported_formats: List of supported file extensions

    Returns:
        Tuple of (is_valid, error_message)
    """
    suffix = file_path.suffix.lower()
    if suffix not in supported_formats:
        error_msg = f"Unsupported format: {suffix}. Supported: {supported_formats}"
        return False, error_msg
    return True, None


def extract_text_from_pdf_binary(pdf_binary: bytes) -> Tuple[str, int]:
    """
    Extract text from PDF binary data.
    Pure function with no side effects.

    Args:
        pdf_binary: PDF file content as bytes

    Returns:
        Tuple of (extracted_text, pages_processed)

    Raises:
        Exception: If PDF processing fails
        ImportError: If pypdf is not available
    """
    if not PYPDF_AVAILABLE:
        raise ImportError(
            "pypdf is required for PDF extraction. Install with: pip install pypdf"
        )

    from io import BytesIO

    text_content = []

    with BytesIO(pdf_binary) as pdf_stream:
        pdf_reader = PdfReader(pdf_stream)
        pages_processed = len(pdf_reader.pages)

        for page_num in range(pages_processed):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                text_content.append(text.strip())

    full_text = "\n\n".join(text_content)
    return full_text, pages_processed


def extract_text_from_docx_binary(docx_binary: bytes) -> str:
    """
    Extract text from DOCX binary data.
    Pure function with no side effects.

    Args:
        docx_binary: DOCX file content as bytes

    Returns:
        Extracted text content

    Raises:
        Exception: If DOCX processing fails
        ImportError: If python-docx is not available
    """
    if not DOCX_AVAILABLE:
        raise ImportError(
            "python-docx is required for DOCX extraction. Install with: pip install python-docx"
        )

    from io import BytesIO

    text_content = []

    with BytesIO(docx_binary) as docx_stream:
        doc = Document(docx_stream)

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text.strip())

    full_text = "\n\n".join(text_content)
    return full_text


def extract_text_with_encoding_fallback(
    text_binary: bytes, encodings: List[str]
) -> Tuple[str, str]:
    """
    Extract text with multiple encoding fallback.
    Pure function with no side effects.

    Args:
        text_binary: Text file content as bytes
        encodings: List of encodings to try in order

    Returns:
        Tuple of (extracted_text, successful_encoding)

    Raises:
        ValueError: If no encoding succeeds
    """
    for encoding in encodings:
        try:
            text = text_binary.decode(encoding).strip()
            return text, encoding
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode text with any supported encoding: {encodings}")


def post_process_extracted_text(text: str) -> str:
    """
    Post-process extracted text for consistency.
    Pure function with no side effects.

    Args:
        text: Raw extracted text

    Returns:
        Processed text content
    """
    if not text:
        return ""

    # Basic text cleaning
    processed_text = text.strip()

    # Normalize whitespace while preserving paragraphs
    lines = processed_text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    return "\n\n".join(cleaned_lines)


# ================================
# DEPENDENCY INJECTION ORCHESTRATION
# ================================


class DocumentExtractor:
    """Document extractor with dependency injection for testability."""

    def __init__(
        self,
        config_provider: ConfigProvider,
        file_system_provider: FileSystemProvider,
        logger_provider: Optional[LoggerProvider] = None,
    ):
        """Initialize extractor with injected dependencies."""
        self._config_provider = config_provider
        self._file_system = file_system_provider
        self._logger = logger_provider

        # Load configuration once during initialization
        config_data = config_provider.get_extraction_config()

        # Validate required configuration keys
        if "max_file_size_mb" not in config_data:
            raise ValueError("Missing 'max_file_size_mb' in extraction configuration")
        if "enable_logging" not in config_data:
            raise ValueError("Missing 'enable_logging' in extraction configuration")

        self._config = ExtractionConfig(
            supported_formats=config_data["supported_formats"],
            text_encodings=config_data["text_encodings"],
            max_file_size_mb=config_data["max_file_size_mb"],
            enable_logging=config_data["enable_logging"],
        )

    def extract_text(self, file_path: Path) -> ExtractionResult:
        """
        Extract text from document using dependency injection.

        Args:
            file_path: Path to document file

        Returns:
            ExtractionResult with text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or file too large
        """
        # File existence validation
        if not self._file_system.file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # File size validation
        file_size_mb = self._file_system.get_file_size_mb(file_path)
        if file_size_mb > self._config.max_file_size_mb:
            raise ValueError(
                f"File too large: {file_size_mb:.1f}MB > {self._config.max_file_size_mb}MB"
            )

        # Format validation
        is_valid, error_msg = validate_file_format(
            file_path, self._config.supported_formats
        )
        if not is_valid:
            raise ValueError(error_msg)

        self._log_info(f"Extracting text from {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix == ".docx":
            return self._extract_docx(file_path)
        elif suffix == ".txt":
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _extract_pdf(self, file_path: Path) -> ExtractionResult:
        """Extract text from PDF using pure function."""
        pdf_binary = self._file_system.open_binary(file_path)
        text, pages_processed = extract_text_from_pdf_binary(pdf_binary)
        processed_text = post_process_extracted_text(text)

        self._log_info(
            f"Extracted {len(processed_text)} characters from {pages_processed} PDF pages"
        )

        return ExtractionResult(
            text=processed_text,
            character_count=len(processed_text),
            extraction_method="PDF",
            pages_processed=pages_processed,
        )

    def _extract_docx(self, file_path: Path) -> ExtractionResult:
        """Extract text from DOCX using pure function."""
        docx_binary = self._file_system.open_binary(file_path)
        text = extract_text_from_docx_binary(docx_binary)
        processed_text = post_process_extracted_text(text)

        self._log_info(f"Extracted {len(processed_text)} characters from DOCX")

        return ExtractionResult(
            text=processed_text,
            character_count=len(processed_text),
            extraction_method="DOCX",
        )

    def _extract_txt(self, file_path: Path) -> ExtractionResult:
        """Extract text from TXT using pure function with encoding fallback."""
        text_binary = self._file_system.open_binary(file_path)
        text, encoding_used = extract_text_with_encoding_fallback(
            text_binary, self._config.text_encodings
        )
        processed_text = post_process_extracted_text(text)

        self._log_info(f"Successfully read TXT file with {encoding_used} encoding")
        self._log_info(f"Extracted {len(processed_text)} characters from TXT")

        return ExtractionResult(
            text=processed_text,
            character_count=len(processed_text),
            extraction_method="TXT",
            encoding_used=encoding_used,
        )

    def _log_info(self, message: str) -> None:
        """Log info message if logger available."""
        if self._logger and self._config.enable_logging:
            self._logger.info(message)

    def _log_error(self, message: str) -> None:
        """Log error message if logger available."""
        if self._logger and self._config.enable_logging:
            self._logger.error(message)


# ================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# ================================


def extract_document_text(
    file_path: str | Path,
    config_provider: Optional[ConfigProvider] = None,
    file_system_provider: Optional[FileSystemProvider] = None,
) -> str:
    """
    Convenience function for backward compatibility.

    Args:
        file_path: Path to document file
        config_provider: Optional config provider (uses production if None)
        file_system_provider: Optional file system provider (uses production if None)

    Returns:
        Extracted text content
    """
    from .extractors_providers import create_config_provider, create_file_system_provider

    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Use injected providers or create defaults
    config = config_provider or create_config_provider()
    file_system = file_system_provider or create_file_system_provider()

    extractor = DocumentExtractor(config, file_system)
    result = extractor.extract_text(file_path)

    return result.text
