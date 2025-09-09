// Main application logic for multilingual RAG prototype
class RAGApp {
    constructor() {
        this.currentQuery = '';
        this.currentInputLanguage = 'multilingual';
        this.searchResults = [];
        this.isSearching = false;

        // Initialize the application
        this.init();
    }

    init() {
        // Initialize i18n
        window.i18n.updateDOM();

        // Set up event listeners
        this.setupEventListeners();

        // Initialize interface language buttons
        this.updateLanguageSwitcher();

        // Load saved preferences
        this.loadPreferences();

        console.log('RAG App initialized with dark theme');
    }

    setupEventListeners() {
        // Interface language switcher
        document.querySelectorAll('.lang-flag').forEach(flag => {
            flag.addEventListener('click', (e) => {
                const flagElement = e.currentTarget;
                const lang = flagElement.dataset.lang;
                this.switchInterfaceLanguage(lang);
            });
        });

        // Input language selector
        const inputLanguageSelect = document.getElementById('inputLanguage');
        inputLanguageSelect.addEventListener('change', (e) => {
            this.currentInputLanguage = e.target.value;
            this.savePreferences();
            console.log('Input language changed to:', this.currentInputLanguage);
        });

        // Search functionality
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('searchInput');

        searchBtn.addEventListener('click', () => this.performSearch());

        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });

        searchInput.addEventListener('input', (e) => {
            this.currentQuery = e.target.value;
            // Optional: Auto-detect language and suggest input language
            if (this.currentQuery.length > 10) {
                this.detectAndSuggestLanguage();
            }
        });

        // File upload functionality
        const uploadDropzone = document.getElementById('uploadDropzone');
        const fileInput = document.getElementById('fileInput');

        // Drag and drop events
        uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDropzone.classList.add('dragover');
        });

        uploadDropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadDropzone.classList.remove('dragover');
        });

        uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDropzone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            this.handleFileUpload(files);
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Upload language selector
        const uploadLanguageSelect = document.getElementById('uploadLanguage');
        uploadLanguageSelect.addEventListener('change', (e) => {
            console.log('Upload language changed to:', e.target.value);
        });
    }

    switchInterfaceLanguage(lang) {
        // Update i18n
        window.i18n.setLanguage(lang);

        // Update active button
        this.updateLanguageSwitcher();

        // Save preference
        this.savePreferences();

        // If we have search results, update the results display
        if (this.searchResults.length > 0) {
            this.displayResults(this.searchResults);
        }

        console.log('Interface language switched to:', lang);
    }

    updateLanguageSwitcher() {
        const currentLang = window.i18n.getCurrentLanguage();
        document.querySelectorAll('.lang-flag').forEach(flag => {
            if (flag.dataset.lang === currentLang) {
                flag.classList.add('active');
            } else {
                flag.classList.remove('active');
            }
        });
    }

    async detectAndSuggestLanguage() {
        try {
            const detection = window.mockData.detectLanguage(this.currentQuery);
            console.log('Language detected:', detection);

            // Optionally update input language if confidence is high
            if (detection.confidence > 0.8 && detection.detected !== 'multilingual') {
                const inputLanguageSelect = document.getElementById('inputLanguage');
                if (inputLanguageSelect.value === 'multilingual') {
                    // Show subtle suggestion (could be implemented as a toast/hint)
                    console.log(`Suggestion: Switch to ${detection.detected} for better results`);
                }
            }
        } catch (error) {
            console.error('Language detection error:', error);
        }
    }

    async performSearch() {
        if (!this.currentQuery.trim() || this.isSearching) {
            return;
        }

        this.isSearching = true;
        this.showSearchStatus(true);

        try {
            // Show loading state
            this.updateSearchStatus('searching');

            // Perform search
            const results = await window.mockData.searchDocuments(
                this.currentQuery,
                this.currentInputLanguage,
                5
            );

            // Store results
            this.searchResults = results.results;

            // Update UI
            this.updateSearchStatus('completed', results.summary.processing_time_ms);
            this.displayResults(results);

            // Log for debugging
            console.log('Search completed:', results);

        } catch (error) {
            console.error('Search error:', error);
            this.updateSearchStatus('error');
            this.showError(window.i18n.t('errors.processingError'));
        } finally {
            this.isSearching = false;
        }
    }

    showSearchStatus(show) {
        const statusElement = document.getElementById('searchStatus');
        if (show) {
            statusElement.classList.remove('hidden');
        } else {
            statusElement.classList.add('hidden');
        }
    }

    updateSearchStatus(status, processingTime = null) {
        const statusElement = document.getElementById('searchStatus');
        const statusText = statusElement.querySelector('.status-text');
        const statusIndicator = statusElement.querySelector('.status-indicator');

        switch (status) {
            case 'searching':
                statusText.textContent = window.i18n.t('ui.loading');
                statusIndicator.style.backgroundColor = 'var(--warning-color)';
                break;
            case 'completed':
                statusText.textContent = `${window.i18n.t('results.processedIn')} ${processingTime}ms`;
                statusIndicator.style.backgroundColor = 'var(--success-color)';
                setTimeout(() => this.showSearchStatus(false), 3000);
                break;
            case 'error':
                statusText.textContent = window.i18n.t('errors.processingError');
                statusIndicator.style.backgroundColor = 'var(--error-color)';
                break;
        }
    }

    displayResults(searchResponse) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsList = document.getElementById('resultsList');
        const resultsCount = document.getElementById('resultsCount');
        const processingTime = document.getElementById('processingTime');

        // Show results section
        resultsSection.classList.remove('hidden');

        // Update stats
        const resultText = window.i18n.t('results.resultsFound');
        resultsCount.textContent = `${searchResponse.summary.total_results} ${resultText}`;
        processingTime.textContent = `${searchResponse.summary.processing_time_ms}ms`;

        // Clear previous results
        resultsList.innerHTML = '';

        // Display results
        searchResponse.results.forEach((result, index) => {
            const resultElement = this.createResultElement(result, index);
            resultsList.appendChild(resultElement);
        });

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    createResultElement(result, index) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-item';

        // Format relevance score as percentage
        const relevancePercent = Math.round(result.metadata.relevance_score * 100);

        // Get language display name
        const langMap = {
            'hr': window.i18n.t('languages.croatian'),
            'en': window.i18n.t('languages.english'),
            'multilingual': window.i18n.t('languages.multilingual')
        };

        const languageDisplay = langMap[result.source.language] || result.source.language;

        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="result-source">${result.source.filename}</div>
                <div class="result-language ${result.source.language}">${result.source.language.toUpperCase()}</div>
            </div>
            <div class="result-content">${result.content}</div>
            <div class="result-metadata">
                <span class="relevance-score">${relevancePercent}% ${window.i18n.t('results.relevance') || 'relevance'}</span>
                <span>Page ${result.source.page}</span>
                <span>${languageDisplay}</span>
                <span>${result.metadata.last_updated}</span>
            </div>
        `;

        // Add click handler for result interaction
        resultDiv.addEventListener('click', () => {
            console.log('Result clicked:', result);
            // Could implement result expansion or navigation
        });

        return resultDiv;
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;

        const uploadLanguage = document.getElementById('uploadLanguage').value;

        console.log(`Uploading ${files.length} files with language: ${uploadLanguage}`);

        try {
            // Show loading
            this.showLoading(true);

            // Mock upload
            const uploadResponse = await window.mockData.uploadFiles(files, uploadLanguage);

            // Show success message
            const message = window.i18n.getCurrentLanguage() === 'hr' ?
                `Uspješno učitano ${files.length} datoteka` :
                `Successfully uploaded ${files.length} files`;

            this.showNotification(message, 'success');

            console.log('Upload response:', uploadResponse);

        } catch (error) {
            console.error('Upload error:', error);
            this.showError(window.i18n.t('errors.processingError'));
        } finally {
            this.showLoading(false);

            // Reset file input
            document.getElementById('fileInput').value = '';
        }
    }

    showLoading(show) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (show) {
            loadingOverlay.classList.remove('hidden');
        } else {
            loadingOverlay.classList.add('hidden');
        }
    }

    showNotification(message, type = 'info') {
        // Simple console notification for prototype
        // In real implementation, would show toast/notification
        console.log(`Notification (${type}):`, message);

        // Could implement toast notifications here
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            background-color: var(--success-color);
            color: white;
            z-index: 1001;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => notification.style.opacity = '1', 10);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 3000);
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    savePreferences() {
        const preferences = {
            inputLanguage: this.currentInputLanguage,
            interfaceLanguage: window.i18n.getCurrentLanguage()
        };
        localStorage.setItem('ragAppPreferences', JSON.stringify(preferences));
    }

    loadPreferences() {
        try {
            const saved = localStorage.getItem('ragAppPreferences');
            if (saved) {
                const preferences = JSON.parse(saved);

                // Restore input language
                if (preferences.inputLanguage) {
                    this.currentInputLanguage = preferences.inputLanguage;
                    document.getElementById('inputLanguage').value = preferences.inputLanguage;
                }

                // Interface language is handled by i18n system
            }
        } catch (error) {
            console.error('Error loading preferences:', error);
        }
    }

    // Keyboard shortcuts
    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            document.getElementById('searchInput').focus();
        }

        // Ctrl/Cmd + L to switch interface language
        if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
            e.preventDefault();
            const currentLang = window.i18n.getCurrentLanguage();
            const newLang = currentLang === 'hr' ? 'en' : 'hr';
            this.switchInterfaceLanguage(newLang);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new RAGApp();

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => app.handleKeyboardShortcuts(e));

    // Global app instance for debugging
    window.ragApp = app;

    console.log('🚀 Multilingual RAG Prototype loaded successfully!');
    console.log('💡 Shortcuts: Ctrl+K (search), Ctrl+L (switch language)');
});
