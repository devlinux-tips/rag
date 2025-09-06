# Multilingual UI Architecture

**Created**: 2025-09-06
**Phase**: Phase 1A
**Priority**: High
**Component**: React Frontend

## 🎯 Overview

Design specification for multilingual user interface supporting Croatian and English languages with seamless language switching for both input and interface.

## 🌐 Language Architecture

### **Dual Language Selection Model**

#### **1. Interface Language**
Controls the UI text, labels, buttons, and messages
- **Options**: Croatian (`hr`) | English (`en`)
- **Storage**: localStorage + URL parameter
- **Default**: Browser language detection with fallback to English
- **Switching**: Top-right language switcher (HR/EN toggle)

#### **2. Input/Search Language**
Controls which language collection to search and how to process the query
- **Options**:
  - Croatian (`hr`) - Search Croatian documents
  - English (`en`) - Search English documents
  - Multilingual (`multilingual`) - Search across all languages
- **Storage**: Component state + query parameter
- **Default**: Same as interface language
- **Selection**: Dropdown in search interface

### **UI Component Structure**

```
App (i18n provider)
├── LanguageSwitcher (interface language)
├── Header (localized)
├── SearchInterface
│   ├── InputLanguageSelector (hr/en/multilingual)
│   ├── SearchBox (with language-appropriate placeholder)
│   └── SearchResults (with language indicators)
├── DocumentUpload
│   ├── LanguageSpecification (document language)
│   └── UploadProgress (localized messages)
└── Footer (localized)
```

## 🔧 Technical Implementation

### **i18n Configuration (react-i18next)**

```typescript
// locales/hr/common.json
{
  "search": {
    "placeholder": "Pretražite dokumente...",
    "button": "Pretraži",
    "inputLanguage": "Jezik pretrage:",
    "results": "rezultata pronađeno"
  },
  "upload": {
    "dragDrop": "Povucite datoteke ovdje ili kliknite za odabir",
    "processing": "Obrađuje se...",
    "success": "Uspješno učitano"
  },
  "languages": {
    "croatian": "Hrvatski",
    "english": "English",
    "multilingual": "Višejezično"
  }
}

// locales/en/common.json
{
  "search": {
    "placeholder": "Search documents...",
    "button": "Search",
    "inputLanguage": "Search language:",
    "results": "results found"
  },
  "upload": {
    "dragDrop": "Drag files here or click to select",
    "processing": "Processing...",
    "success": "Successfully uploaded"
  },
  "languages": {
    "croatian": "Croatian",
    "english": "English",
    "multilingual": "Multilingual"
  }
}
```

### **Language Selection Components**

```typescript
// InterfaceLanguageSwitcher.tsx
export const InterfaceLanguageSwitcher: React.FC = () => {
  const { i18n } = useTranslation();

  const switchLanguage = (lang: 'hr' | 'en') => {
    i18n.changeLanguage(lang);
    localStorage.setItem('interfaceLanguage', lang);
    // Update URL parameter for shareable links
    updateUrlParameter('ui_lang', lang);
  };

  return (
    <div className="language-switcher">
      <button onClick={() => switchLanguage('hr')}
              className={i18n.language === 'hr' ? 'active' : ''}>
        HR
      </button>
      <button onClick={() => switchLanguage('en')}
              className={i18n.language === 'en' ? 'active' : ''}>
        EN
      </button>
    </div>
  );
};

// InputLanguageSelector.tsx
export const InputLanguageSelector: React.FC<{
  value: 'hr' | 'en' | 'multilingual';
  onChange: (lang: 'hr' | 'en' | 'multilingual') => void;
}> = ({ value, onChange }) => {
  const { t } = useTranslation();

  return (
    <select value={value} onChange={(e) => onChange(e.target.value as any)}>
      <option value="hr">{t('languages.croatian')}</option>
      <option value="en">{t('languages.english')}</option>
      <option value="multilingual">{t('languages.multilingual')}</option>
    </select>
  );
};
```

### **API Integration Pattern**

```typescript
// SearchAPI integration with language parameters
const searchDocuments = async (
  query: string,
  inputLanguage: 'hr' | 'en' | 'multilingual',
  interfaceLanguage: 'hr' | 'en'
) => {
  const response = await fetch(`/api/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept-Language': interfaceLanguage
    },
    body: JSON.stringify({
      query,
      language: inputLanguage,
      ui_language: interfaceLanguage
    })
  });

  return response.json();
};
```

## 📱 User Experience Flow

### **First Visit**
1. Detect browser language (`navigator.language`)
2. Set interface language (hr if `hr-*`, otherwise en)
3. Set input language to match interface language
4. Show language options prominently

### **Language Switching**
1. **Interface Language**: Immediate UI update, no search refresh needed
2. **Input Language**: Clear current results, show language-appropriate placeholder
3. **State Persistence**: Save preferences in localStorage

### **Search Experience**
1. User selects input language (hr/en/multilingual)
2. Types query with appropriate keyboard layout support
3. Results show language indicators for each document
4. Error messages in selected interface language

### **Upload Experience**
1. User can specify document language during upload
2. Auto-detection suggestion shown if confident
3. Progress and status messages in interface language
4. Success/error handling localized

## 🎨 Visual Design Considerations

### **Language Indicators**
- **Croatian**: 🇭🇷 flag or "HR" badge
- **English**: 🇬🇧 flag or "EN" badge
- **Multilingual**: 🌐 globe or "ML" badge

### **Typography**
- **Croatian**: Support for diacritics (Č, Ć, Š, Ž, Đ)
- **English**: Standard Latin character set
- **Font**: Use system fonts with good diacritic support

### **Layout**
- **LTR Support**: Both Croatian and English are left-to-right
- **Responsive**: Language switchers work on mobile
- **Accessibility**: Proper ARIA labels in both languages

## 🔄 URL Structure

Support for shareable multilingual URLs:

```
# Interface in Croatian, searching Croatian documents
/search?q=rag+sustav&input_lang=hr&ui_lang=hr

# Interface in English, searching multilingual
/search?q=what+is+rag&input_lang=multilingual&ui_lang=en

# Upload page with Croatian interface
/upload?ui_lang=hr
```

## 🚀 Implementation Priority

### **Phase 1: Core Functionality**
1. ✅ i18n setup with react-i18next
2. ✅ Interface language switcher
3. ✅ Input language selector in search
4. ✅ Basic translations for common UI elements

### **Phase 2: Enhanced Experience**
1. 🔲 URL parameter synchronization
2. 🔲 Language detection for uploads
3. 🔲 Advanced error handling with localized messages
4. 🔲 Keyboard layout hints/support

### **Phase 3: Polish**
1. 🔲 Smooth language switching animations
2. 🔲 Language preference learning
3. 🔲 Cultural adaptations (date formats, etc.)
4. 🔲 Advanced typography optimizations

## ✅ Success Metrics

- [ ] Interface language switches instantly without page reload
- [ ] Input language selection affects search results appropriately
- [ ] All UI elements properly localized in both languages
- [ ] URLs are shareable with language preferences preserved
- [ ] Croatian diacritics display correctly throughout interface
- [ ] Language switching works seamlessly on mobile and desktop
- [ ] Error messages and status updates appear in selected interface language
- [ ] Document language indicators are clear and consistent
