"""Narodne Novine Metadata Extractor.

Extracts metadata from Croatian Official Gazette (Narodne Novine) HTML documents.
Uses ELI (European Legislation Identifier) metadata tags to extract structured information.

This module is feature-specific and only activates for narodne-novine feature scope.
"""

from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


def is_nn_document(html_content: str) -> bool:
    """Quick check if HTML contains Narodne Novine ELI metadata.

    Args:
        html_content: Raw HTML content

    Returns:
        True if document contains NN ELI metadata tags, False otherwise

    Example:
        >>> html = '<meta about="https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1" />'
        >>> is_nn_document(html)
        True
    """
    if not html_content:
        return False

    # Fast string search before parsing
    if "narodne-novine.nn.hr/eli" not in html_content.lower():
        return False

    try:
        soup = BeautifulSoup(html_content, "lxml")
        # Look for ELI metadata tag
        eli_meta = soup.find("meta", attrs={"about": lambda x: x and "narodne-novine.nn.hr/eli" in x})
        return eli_meta is not None
    except Exception:
        return False


def parse_issue_from_path(file_path: Path) -> Optional[str]:
    """Extract NN issue number from file path.

    Expected path structure: .../2024/001/2024_01_1_4.html
    Where: year=2024, issue=001 → "NN 1/2024"

    Args:
        file_path: Path to HTML file

    Returns:
        Issue string like "NN 1/2024" or None if pattern doesn't match

    Example:
        >>> from pathlib import Path
        >>> parse_issue_from_path(Path("data/features/narodne_novine/documents/hr/2024/001/file.html"))
        'NN 1/2024'
    """
    try:
        parts = file_path.parts
        # Look for pattern: .../YYYY/NNN/...
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 4:  # Year (2024)
                if i + 1 < len(parts):
                    issue_part = parts[i + 1]
                    if issue_part.isdigit():  # Issue number (001)
                        year = part
                        issue_num = issue_part.lstrip("0") or "0"  # Remove leading zeros
                        return f"NN {issue_num}/{year}"
        return None
    except Exception:
        return None


def extract_nn_metadata(html_content: str, file_path: Path) -> Optional[dict]:
    """Extract Narodne Novine metadata from HTML document.

    Extracts structured metadata from ELI (European Legislation Identifier) tags
    embedded in Narodne Novine HTML documents.

    Args:
        html_content: Raw HTML content of NN document
        file_path: Path to the HTML file (for issue number extraction)

    Returns:
        Dictionary with metadata or None if not a valid NN document:
        {
            "eli_url": "https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1",
            "title": "Uredba o utvrđivanju...",
            "issue": "NN 1/2024",
            "doc_number": "1",
            "pages": "1-2",
            "publisher": "Vlada Republike Hrvatske",
            "date_published": "2024-01-02",
            "document_type": "Uredba"
        }

    Example:
        >>> with open("2024_01_1_1.html") as f:
        ...     metadata = extract_nn_metadata(f.read(), Path("2024_01_1_1.html"))
        >>> metadata["title"]
        'Uredba o utvrđivanju najviših maloprodajnih cijena naftnih derivata'
    """
    if not is_nn_document(html_content):
        return None

    try:
        soup = BeautifulSoup(html_content, "lxml")

        # Find base ELI URL from 'about' attribute
        eli_meta = soup.find("meta", attrs={"about": lambda x: x and "narodne-novine.nn.hr/eli/sluzbeni" in x})
        if not eli_meta:
            return None

        eli_url = eli_meta.get("about")
        if not eli_url:
            return None

        metadata = {"eli_url": eli_url}

        # Note: Some meta tags use base ELI URL, others use language-specific (/hrv)
        # We match by checking if 'about' starts with our base ELI URL

        # Extract title (property contains "eli/ontology#title")
        # Title meta uses language-specific URL (/hrv)
        title_meta = soup.find(
            "meta",
            attrs={
                "about": lambda x: x and x.startswith(eli_url),
                "property": lambda x: x and "eli/ontology#title" in x,
            },
        )
        if title_meta and title_meta.get("content"):
            metadata["title"] = title_meta.get("content")

        # Extract document number (property contains "eli/ontology#number")
        number_meta = soup.find(
            "meta",
            attrs={
                "about": eli_url,
                "property": lambda x: x and "eli/ontology#number" in x,
            },
        )
        if number_meta and number_meta.get("content"):
            metadata["doc_number"] = number_meta.get("content")

        # Extract publication date (property contains "eli/ontology#date_publication")
        date_meta = soup.find(
            "meta",
            attrs={
                "about": eli_url,
                "property": lambda x: x and "eli/ontology#date_publication" in x,
            },
        )
        if date_meta and date_meta.get("content"):
            metadata["date_published"] = date_meta.get("content")

        # Extract document type from ELI URL or type_document property
        type_meta = soup.find(
            "meta",
            attrs={
                "about": eli_url,
                "property": lambda x: x and "eli/ontology#type_document" in x,
            },
        )
        if type_meta and type_meta.get("resource"):
            # Extract type from resource URL (e.g., .../document-type/UREDBA)
            type_url = type_meta.get("resource")
            if "/" in type_url:
                doc_type = type_url.split("/")[-1].capitalize()
                metadata["document_type"] = doc_type

        # Extract publisher/institution
        publisher_meta = soup.find(
            "meta",
            attrs={
                "about": eli_url,
                "property": lambda x: x and "eli/ontology#passed_by" in x,
            },
        )
        if publisher_meta and publisher_meta.get("resource"):
            institution_url = publisher_meta.get("resource")
            # Extract institution ID (e.g., .../nn-institutions/19560)
            if "/" in institution_url:
                institution_id = institution_url.split("/")[-1]
                # Map common institution IDs to names
                institution_names = {
                    "19560": "Vlada Republike Hrvatske",
                    "19591": "Ministarstvo financija",
                    # Add more mappings as needed
                }
                metadata["publisher"] = institution_names.get(institution_id, f"Institution ID: {institution_id}")

        # Extract page numbers from metadata table (if available)
        # Look for "Stranica tiskanog izdanja" in metadata table
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                key_span = cells[0].find("span", class_="key")
                if key_span and "Stranica" in key_span.get_text():
                    page_text = cells[1].get_text().strip()
                    if page_text:
                        metadata["pages"] = page_text

        # Extract issue number from file path
        issue = parse_issue_from_path(file_path)
        if issue:
            metadata["issue"] = issue

        # Return metadata only if we have essential fields
        if "title" in metadata and "eli_url" in metadata:
            return metadata

        return None

    except Exception as e:
        # Log error but don't fail processing
        # Caller should handle None gracefully
        print(f"Warning: Failed to extract NN metadata from {file_path}: {e}")
        return None
