Complete system overhaul: configuration, class rename, and PyTorch upgrade

- Fixed configuration loading issues and restructured config files
- Upgraded PyTorch to 2.8.0+cu128 (fixed security vulnerability)
- Renamed CroatianRAGSystem → RAGSystem for multilingual expansion
- Enhanced CLI with detailed JSON metadata output
- Added comprehensive device detection (CPU/CUDA/MPS support)
- Created centralized config loader and error handling
- Updated all tests and documentation

✅ BGE-M3 working, Croatian queries functional, ready for production
