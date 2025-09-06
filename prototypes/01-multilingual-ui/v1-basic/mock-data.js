// Mock data for multilingual RAG prototype
class MockDataService {
    constructor() {
        // Mock search results for different languages
        this.mockResults = {
            hr: [
                {
                    id: 'hr_001',
                    content: 'RAG (Retrieval-Augmented Generation) sustav je napredna tehnologija koja kombinira pretraživanje dokumenata s generiranjem odgovora. Sustav koristi vektorske baze podataka za pronalaženje relevantnih informacija i zatim generira kontekstualno odgovarajuće odgovore koristeći velike jezične modele.',
                    source: {
                        filename: 'uvod_u_rag.pdf',
                        page: 1,
                        language: 'hr'
                    },
                    metadata: {
                        relevance_score: 0.94,
                        language_confidence: 0.96,
                        chunk_language: 'hr',
                        last_updated: '2025-08-15'
                    }
                },
                {
                    id: 'hr_002',
                    content: 'Hrvatska implementacija RAG sustava mora uzeti u obzir posebnosti hrvatskog jezika, uključujući morfološke varijacije, dijakritike (č, ć, š, ž, đ) te kulturni kontekst. Važno je koristiti modele koji su optimizirani za hrvatski jezik poput BGE-M3 za embeddings.',
                    source: {
                        filename: 'hrvatski_nlp.docx',
                        page: 3,
                        language: 'hr'
                    },
                    metadata: {
                        relevance_score: 0.89,
                        language_confidence: 0.98,
                        chunk_language: 'hr',
                        last_updated: '2025-08-20'
                    }
                },
                {
                    id: 'hr_003',
                    content: 'Semantičko pretraživanje omogućuje pronalaženje relevantnih dokumenata na osnovu značenja, a ne samo ključnih riječi. To je posebno korisno za hrvatski jezik gdje ista riječ može imati različite oblike ovisno o padežu, broju ili vremenu.',
                    source: {
                        filename: 'semanticko_pretrazivanje.txt',
                        page: 1,
                        language: 'hr'
                    },
                    metadata: {
                        relevance_score: 0.85,
                        language_confidence: 0.94,
                        chunk_language: 'hr',
                        last_updated: '2025-08-18'
                    }
                }
            ],

            en: [
                {
                    id: 'en_001',
                    content: 'Retrieval-Augmented Generation (RAG) systems represent a breakthrough in AI technology by combining document retrieval with text generation. These systems use vector databases to find relevant information and then generate contextually appropriate responses using large language models like GPT or Claude.',
                    source: {
                        filename: 'rag_introduction.pdf',
                        page: 2,
                        language: 'en'
                    },
                    metadata: {
                        relevance_score: 0.92,
                        language_confidence: 0.99,
                        chunk_language: 'en',
                        last_updated: '2025-08-22'
                    }
                },
                {
                    id: 'en_002',
                    content: 'Vector embeddings are mathematical representations of text that capture semantic meaning. Modern embedding models like BGE-M3 or E5-large can create dense vectors that represent the meaning of text across multiple languages, making them ideal for multilingual RAG systems.',
                    source: {
                        filename: 'vector_embeddings_guide.docx',
                        page: 5,
                        language: 'en'
                    },
                    metadata: {
                        relevance_score: 0.88,
                        language_confidence: 0.97,
                        chunk_language: 'en',
                        last_updated: '2025-08-19'
                    }
                },
                {
                    id: 'en_003',
                    content: 'ChromaDB is a popular vector database that supports persistent storage, efficient similarity search, and metadata filtering. It\'s particularly useful for RAG applications because it can handle large collections of document embeddings with fast query times.',
                    source: {
                        filename: 'vector_databases.txt',
                        page: 1,
                        language: 'en'
                    },
                    metadata: {
                        relevance_score: 0.86,
                        language_confidence: 0.95,
                        chunk_language: 'en',
                        last_updated: '2025-08-17'
                    }
                }
            ],

            multilingual: [
                {
                    id: 'multi_001',
                    content: 'RAG sustavi omogućuju pretraživanje i generiranje odgovora na više jezika istovremeno. Multilingual RAG systems can process documents in Croatian, English, and other languages while maintaining semantic understanding across language boundaries.',
                    source: {
                        filename: 'multilingual_rag_overview.pdf',
                        page: 1,
                        language: 'multilingual'
                    },
                    metadata: {
                        relevance_score: 0.91,
                        language_confidence: 0.88,
                        chunk_language: 'multilingual',
                        last_updated: '2025-08-21'
                    }
                },
                {
                    id: 'multi_002',
                    content: 'Code-switching represents a significant challenge for multilingual NLP systems. Kada korisnici mijenjaju jezik usred rečenice, the system must maintain context and provide appropriate responses regardless of the language mixture.',
                    source: {
                        filename: 'code_switching_challenges.docx',
                        page: 2,
                        language: 'multilingual'
                    },
                    metadata: {
                        relevance_score: 0.84,
                        language_confidence: 0.82,
                        chunk_language: 'multilingual',
                        last_updated: '2025-08-16'
                    }
                }
            ]
        };

        // Mock language detection results
        this.mockLanguageDetection = {
            'što je rag sustav': { detected: 'hr', confidence: 0.95 },
            'what is rag system': { detected: 'en', confidence: 0.98 },
            'kako funkcionira semantic search': { detected: 'hr', confidence: 0.92 },
            'how does vector embedding work': { detected: 'en', confidence: 0.97 },
            'rag system kako works': { detected: 'multilingual', confidence: 0.75 },
            'pretraživanje documents using AI': { detected: 'multilingual', confidence: 0.78 }
        };

        // Mock upload status
        this.mockUploadStatus = {
            processing: {
                status: 'processing',
                progress: 45,
                message: 'Processing documents...',
                files_processed: 2,
                total_files: 4
            },
            completed: {
                status: 'completed',
                progress: 100,
                message: 'All documents processed successfully',
                files_processed: 4,
                total_files: 4,
                chunks_created: 127
            }
        };
    }

    // Simulate search with realistic delay
    async searchDocuments(query, inputLanguage, maxResults = 5) {
        // Simulate network delay
        await this.delay(100 + Math.random() * 300);

        const results = this.getSearchResults(query, inputLanguage, maxResults);
        const processingTime = Math.round(50 + Math.random() * 200);

        return {
            query,
            input_language: inputLanguage,
            results: results,
            summary: {
                total_results: results.length,
                languages_found: [...new Set(results.map(r => r.source.language))],
                processing_time_ms: processingTime
            }
        };
    }

    // Get search results based on query and language
    getSearchResults(query, inputLanguage, maxResults) {
        let availableResults = [];

        // Select results based on input language
        if (inputLanguage === 'multilingual') {
            availableResults = [
                ...this.mockResults.hr,
                ...this.mockResults.en,
                ...this.mockResults.multilingual
            ];
        } else {
            availableResults = this.mockResults[inputLanguage] || [];
            // Include multilingual results for any language
            availableResults = [...availableResults, ...this.mockResults.multilingual];
        }

        // Simple relevance scoring based on query keywords
        const queryLower = query.toLowerCase();
        const keywords = queryLower.split(' ').filter(word => word.length > 2);

        const scoredResults = availableResults.map(result => {
            let additionalScore = 0;
            const contentLower = result.content.toLowerCase();

            // Boost score based on keyword matches
            keywords.forEach(keyword => {
                if (contentLower.includes(keyword)) {
                    additionalScore += 0.05;
                }
            });

            // Boost for exact language match
            if (result.source.language === inputLanguage) {
                additionalScore += 0.1;
            }

            return {
                ...result,
                metadata: {
                    ...result.metadata,
                    relevance_score: Math.min(0.99, result.metadata.relevance_score + additionalScore)
                }
            };
        });

        // Sort by relevance and return top results
        return scoredResults
            .sort((a, b) => b.metadata.relevance_score - a.metadata.relevance_score)
            .slice(0, maxResults);
    }

    // Detect language of text
    detectLanguage(text) {
        const textLower = text.toLowerCase();

        // Check mock detection first
        if (this.mockLanguageDetection[textLower]) {
            return this.mockLanguageDetection[textLower];
        }

        // Simple heuristic detection
        const croatianWords = ['što', 'kako', 'gdje', 'kad', 'zašto', 'koji', 'sustav', 'pretraživanje'];
        const englishWords = ['what', 'how', 'where', 'when', 'why', 'which', 'system', 'search'];

        let hrScore = 0;
        let enScore = 0;

        const words = textLower.split(' ');
        words.forEach(word => {
            if (croatianWords.includes(word)) hrScore++;
            if (englishWords.includes(word)) enScore++;
        });

        if (hrScore > enScore) {
            return { detected: 'hr', confidence: 0.70 + (hrScore * 0.1) };
        } else if (enScore > hrScore) {
            return { detected: 'en', confidence: 0.70 + (enScore * 0.1) };
        } else {
            return { detected: 'multilingual', confidence: 0.60 };
        }
    }

    // Mock file upload
    async uploadFiles(files, language) {
        const uploadId = 'upload_' + Date.now();

        // Simulate initial upload response
        await this.delay(200);

        const fileInfos = Array.from(files).map((file, index) => {
            const detectedLang = language === 'auto' ?
                (Math.random() > 0.5 ? 'hr' : 'en') : language;

            return {
                filename: file.name,
                size: file.size,
                detected_language: detectedLang,
                confidence: 0.85 + Math.random() * 0.1,
                status: 'processing'
            };
        });

        return {
            upload_id: uploadId,
            status: 'processing',
            files: fileInfos,
            processing_time_estimate: '2-3 minutes'
        };
    }

    // Get upload status
    async getUploadStatus(uploadId) {
        await this.delay(100);

        // Simulate different stages of processing
        const stages = ['processing', 'completed'];
        const randomStage = stages[Math.floor(Math.random() * stages.length)];

        return this.mockUploadStatus[randomStage];
    }

    // Utility function for delays
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Get available language statistics
    getLanguageStats() {
        return {
            hr: {
                document_count: 156,
                chunk_count: 2340,
                last_updated: '2025-08-22'
            },
            en: {
                document_count: 89,
                chunk_count: 1567,
                last_updated: '2025-08-21'
            },
            multilingual: {
                document_count: 45,
                chunk_count: 678,
                last_updated: '2025-08-20'
            }
        };
    }

    // Generate random processing time for demo
    getRandomProcessingTime() {
        return (Math.random() * 0.5 + 0.05).toFixed(2) + 's';
    }
}

// Global mock data service instance
window.mockData = new MockDataService();
