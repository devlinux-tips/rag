"""
Document text extraction for Croatian documents.
Supports PDF, DOCX, and TXT files with proper UTF-8 encoding.
"""

import logging
from pathlib import Path
from typing import Optional

import PyPDF2
from docx import Document

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extract text from various document formats with Croatian language support."""

    def __init__(self):
        """Initialize the document extractor."""
        self.supported_formats = {".pdf", ".docx", ".txt"}

    def extract_text(self, file_path: Path) -> str:
        """
        Extract text from a document file.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}. Supported: {self.supported_formats}")

        logger.info(f"Extracting text from {file_path}")

        if suffix == ".pdf":
            return self._extract_from_pdf(file_path)
        elif suffix == ".docx":
            return self._extract_from_docx(file_path)
        elif suffix == ".txt":
            return self._extract_from_txt(file_path)

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            text_content = []

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text.strip())

            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting from PDF {file_path}: {e}")
            raise

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(str(file_path))

            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())

            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from DOCX")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting from DOCX {file_path}: {e}")
            raise

    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file with proper UTF-8 encoding."""
        try:
            # Try UTF-8 first, then fallback to other encodings common for Croatian
            encodings = ["utf-8", "utf-8-sig", "cp1250", "iso-8859-2"]

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read().strip()
                    logger.info(f"Successfully read TXT file with {encoding} encoding")
                    logger.info(f"Extracted {len(text)} characters from TXT")
                    return text
                except UnicodeDecodeError:
                    continue

            raise ValueError(f"Could not decode file {file_path} with any supported encoding")

        except Exception as e:
            logger.error(f"Error extracting from TXT {file_path}: {e}")
            raise


def extract_document_text(file_path: str | Path) -> str:
    """
    Convenience function to extract text from a document.

    Args:
        file_path: Path to document file (string or Path object)

    Returns:
        Extracted text content
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    extractor = DocumentExtractor()
    return extractor.extract_text(file_path)
