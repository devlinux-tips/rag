// Internationalization (i18n) system for multilingual RAG prototype
class I18n {
    constructor() {
        this.currentLanguage = 'en';
        this.fallbackLanguage = 'en';

        // Translation strings
        this.translations = {
            hr: {
                app: {
                    title: "RAG Sustav",
                    subtitle: "Višejezični Prototip"
                },
                ui: {
                    interfaceLanguage: "Sučelje:",
                    loading: "Obrađuje se..."
                },
                search: {
                    title: "Inteligentna Pretraga",
                    placeholder: "Pretražite dokumente...",
                    button: "Pretraži"
                },
                results: {
                    title: "Rezultati Pretrage",
                    processedIn: "Obrađeno u",
                    resultsFound: "rezultata pronađeno"
                },
                upload: {
                    title: "Baza Znanja",
                    autoDetect: "🤖 Automatska Detekcija",
                    dragDrop: "Povucite datoteke ovdje ili kliknite za odabir",
                    supportedFormats: "Podržava PDF, DOCX, TXT datoteke"
                },
                languages: {
                    croatian: "🇭🇷 Hrvatski Fokus",
                    english: "🇬🇧 Engleski Fokus",
                    multilingual: "🌐 Pretraži Sve"
                },
                errors: {
                    noResults: "Nema rezultata za vašu pretragu.",
                    networkError: "Greška mreže. Molimo pokušajte ponovno.",
                    processingError: "Greška pri obradi. Molimo pokušajte ponovno."
                }
            },
            en: {
                app: {
                    title: "RAG System",
                    subtitle: "Multilingual Prototype"
                },
                ui: {
                    interfaceLanguage: "Interface:",
                    loading: "Processing..."
                },
                search: {
                    title: "Intelligent Search",
                    placeholder: "Search documents...",
                    button: "Search"
                },
                results: {
                    title: "Search Results",
                    processedIn: "Processed in",
                    resultsFound: "results found"
                },
                upload: {
                    title: "Knowledge Base",
                    autoDetect: "🤖 Auto-Detect",
                    dragDrop: "Drag files here or click to select",
                    supportedFormats: "Supports PDF, DOCX, TXT files"
                },
                languages: {
                    croatian: "🇭🇷 Croatian Focus",
                    english: "🇬🇧 English Focus",
                    multilingual: "🌐 Search Everything"
                },
                errors: {
                    noResults: "No results found for your search.",
                    networkError: "Network error. Please try again.",
                    processingError: "Processing error. Please try again."
                }
            }
        };

        // Initialize with browser language or saved preference
        this.init();
    }

    init() {
        // Check for saved language preference
        const savedLang = localStorage.getItem('interfaceLanguage');
        if (savedLang && this.translations[savedLang]) {
            this.currentLanguage = savedLang;
        } else {
            // Detect browser language
            const browserLang = navigator.language.split('-')[0];
            if (this.translations[browserLang]) {
                this.currentLanguage = browserLang;
            }
        }

        // Set HTML lang attribute
        document.documentElement.lang = this.currentLanguage;
    }

    // Get translated text for a key
    t(key, params = {}) {
        const keys = key.split('.');
        let value = this.translations[this.currentLanguage];

        // Navigate through nested object
        for (const k of keys) {
            if (value && typeof value === 'object') {
                value = value[k];
            } else {
                value = undefined;
                break;
            }
        }

        // Fallback to English if key not found
        if (value === undefined) {
            value = this.translations[this.fallbackLanguage];
            for (const k of keys) {
                if (value && typeof value === 'object') {
                    value = value[k];
                } else {
                    value = key; // Return key if no translation found
                    break;
                }
            }
        }

        // Replace parameters in translation
        if (typeof value === 'string' && Object.keys(params).length > 0) {
            return value.replace(/\{\{(\w+)\}\}/g, (match, param) => {
                return params[param] || match;
            });
        }

        return value;
    }

    // Change language
    setLanguage(lang) {
        if (this.translations[lang]) {
            this.currentLanguage = lang;
            localStorage.setItem('interfaceLanguage', lang);
            document.documentElement.lang = lang;
            this.updateDOM();
        }
    }

    // Update all DOM elements with translations
    updateDOM() {
        // Update elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            if (translation) {
                element.textContent = translation;
            }
        });

        // Update elements with data-i18n-placeholder attribute
        document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            const translation = this.t(key);
            if (translation) {
                element.placeholder = translation;
            }
        });

        // Update select options
        document.querySelectorAll('option[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            if (translation) {
                element.textContent = translation;
            }
        });
    }

    // Get current language
    getCurrentLanguage() {
        return this.currentLanguage;
    }

    // Check if language is RTL (for future expansion)
    isRTL(lang = this.currentLanguage) {
        const rtlLanguages = ['ar', 'he', 'fa', 'ur'];
        return rtlLanguages.includes(lang);
    }

    // Format numbers according to locale
    formatNumber(number, lang = this.currentLanguage) {
        const localeMap = {
            'hr': 'hr-HR',
            'en': 'en-US'
        };

        const locale = localeMap[lang] || 'en-US';
        return new Intl.NumberFormat(locale).format(number);
    }

    // Format dates according to locale
    formatDate(date, options = {}, lang = this.currentLanguage) {
        const localeMap = {
            'hr': 'hr-HR',
            'en': 'en-US'
        };

        const locale = localeMap[lang] || 'en-US';
        const defaultOptions = {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        };

        return new Intl.DateTimeFormat(locale, {...defaultOptions, ...options}).format(date);
    }

    // Get available languages
    getAvailableLanguages() {
        return Object.keys(this.translations).map(code => ({
            code,
            name: this.translations[code].languages[code === 'hr' ? 'croatian' : 'english'],
            nativeName: code === 'hr' ? 'Hrvatski' : 'English'
        }));
    }
}

// Global i18n instance
window.i18n = new I18n();
