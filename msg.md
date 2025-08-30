Implement Croatian document preprocessing pipeline with comprehensive test suite

Complete preprocessing system for Croatian RAG:
- Document extractors: PDF/DOCX/TXT with UTF-8 and Croatian encoding support
- Text cleaners: Croatian diacritic preservation, OCR error correction, document artifact removal
- Document chunkers: 3 strategies (sliding_window, sentence, paragraph) with Croatian sentence boundary detection
- Integration pipeline: Extract � Clean � Chunk workflow

Testing coverage:
- 70 test cases across unit and integration tests
- Croatian-specific features validated (diacritics, encoding, government doc processing)
- End-to-end pipeline tested with real Croatian documents
- 421 chunks successfully created from 5 Croatian government documents

Files created:
- src/preprocessing/{extractors,cleaners,chunkers,__init__}.py
- tests/test_{extractors,cleaners,chunkers,preprocessing_integration}.py
- data/processed/ with JSON chunk outputs and statistics
- LEARNING_NOTES.md for progress tracking

Configuration:
- Updated .env to use Claude Console auth
- Enhanced CLAUDE.md with testing commands and priorities

README.md updated for Quick start.
