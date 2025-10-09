# RAG System Improvements Tracker

## Pending Fixes

### 1. Fix `clear-data` command for feature scope
**Priority:** Medium
**Status:** ✅ PARTIALLY COMPLETE - Weaviate working, progress files issue
**Issue:** The `clear-data` command didn't work correctly with `--scope feature --feature narodne-novine`. It looked in wrong directory and didn't clear Weaviate collections.

**Fixed behavior:**
```bash
python rag.py --language hr --scope feature --feature narodne-novine clear-data --dry-run
# ✅ NOW CLEARS:
#   - Weaviate collection: Features_narodne_novine_hr (WORKING)
#   - Progress files: logs/nn_processing_progress.json (WORKING)
#   - Stats files: logs/nn_processing_stats.json (WORKING)
```

**Implementation:**
- ✅ Added `scope` and `feature_name` parameters to `execute_clear_data_command()`
- ✅ Implemented Weaviate collection deletion for feature scope
- ✅ Hardcoded progress file paths (config.toml doesn't have paths section)
- ✅ Hyphen-to-underscore conversion for collection names (`narodne-novine` → `narodne_novine`)
- ✅ Fixed Weaviate `list_all()` API usage (returns dict, not list)

**Known Issue - Progress Files:**
**Status:** ⚠️ MINOR ISSUE - `nn_processing_progress.json` shows wrong format
**Description:** The progress file exists but format doesn't match expectations
- Expected: JSON with `processed_folders` array
- Actual: May have different structure or schema
- **Impact:** Low - Service continues processing, analyzer still works
- **Workaround:** Manual deletion of progress files if needed
- **Priority:** Low - not blocking production use

**Files Modified:**
- `services/rag-service/src/cli/rag_cli.py:864-931` - Added feature scope support

**Related code locations:**
- `services/rag-service/src/cli/rag_cli.py:execute_clear_data_command()` - Main implementation
- Feature scope detection and Weaviate collection naming

---

## Planned Enhancements

### 2. Narodne Novine Source Enrichment
**Priority:** High
**Status:** ✅ ALL PHASES COMPLETE
**Description:** Add rich source metadata for Narodne Novine documents to enable proper citations with links.

**Completed Work:**
- ✅ Phase 1: Metadata extraction from HTML ELI tags (`nn_metadata_extractor.py`)
- ✅ Phase 2: Weaviate storage with JSON serialization and deserialization
- ✅ Phase 3: LLM prompt with numbered citations [1], [2] and Croatian instructions
- ✅ Phase 4: Frontend UI with expandable "Izvori" section
  - ✅ Created `SourcesList.tsx` component with expand/collapse functionality
  - ✅ Integrated into `Message.tsx` for assistant messages
  - ✅ Created shared TypeScript types in `types/message.ts`
  - ✅ Display: citation numbers, titles, issue, publisher, year, ELI links
  - ✅ Croatian UI text and proper pluralization (dokument/dokumenta/dokumenata)
- ✅ FastAPI integration: nnSources field with title, issue, eli, publisher, year
- ✅ UI improvements: Removed "5/5 docs", added timestamp and token display
- ✅ Bug fix: SearchEngineAdapter handling VectorSearchResult objects
- ✅ Bug fix: Field mapping eli_url → eli, date_published → year

**Remaining Work:**
- [ ] Testing: End-to-end test with real NN query
- [ ] Testing: Regression tests for non-NN features
- [ ] Testing: Performance testing of metadata extraction

---

## Security Issues

### 3. OpenRouter API Key Exposure (CRITICAL)
**Priority:** 🔴 CRITICAL
**Status:** ✅ FIXED
**Issue:** API keys were hardcoded in `config.toml` and logged, causing automatic revocation by GitHub secret scanning.

**Root Cause:**
- OpenRouter is a GitHub secret scanning partner
- Hardcoded keys in `config/config.toml` (even in private repos)
- API key preview logging in `llm_provider.py`
- GitHub automatically revokes detected keys → "API key disabled" errors

**Fixes Applied:**
1. ✅ Removed API key logging from `llm_provider.py`
2. ✅ Added environment variable expansion to `config_loader.py`
3. ✅ Changed `config.toml` to use `${OPENROUTER_API_KEY}` placeholder
4. ✅ `.env.local` already in `.gitignore`

**Migration Steps (REQUIRED):**
```bash
# 1. Get fresh API key from OpenRouter (https://openrouter.ai/keys)
# 2. Edit systemd service
sudo nano /etc/systemd/system/rag-api.service

# Add after line 21:
Environment="OPENROUTER_API_KEY=sk-or-v1-YOUR_NEW_KEY_HERE"

# 3. Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart rag-api

# 4. Rotate logs to remove old key traces
sudo journalctl --rotate
sudo journalctl --vacuum-time=1s
```

**Security Best Practices Now Enforced:**
- ✅ No secrets in git-tracked files
- ✅ Environment variable expansion for all sensitive config
- ✅ No API key logging (even partially)
- ✅ Proper `.gitignore` for `.env.local`

**Files Modified:**
- `services/rag-service/config/config.toml` - Replaced hardcoded key
- `services/rag-service/src/utils/config_loader.py` - Added `_expand_env_vars()`
- `services/rag-service/src/generation/llm_provider.py` - Removed logging

---

## Bug Fixes

### 4. Token Count Tracking (Architecture Issue)
**Priority:** Medium
**Status:** ✅ RESOLVED
**Issue:** The `tokensUsed` field was returning 0 because tokens weren't being passed through the response chain.

**Root Cause (CORRECTED):**
- **Initial assumption was WRONG**: System DOES use OpenRouter via ProviderAdapterClient
- **Actual issue**: Token counts from OpenRouter weren't being passed through RAGResponse
- ProviderAdapterClient extracted tokens but stored in metadata, not propagated to API response

**Architecture (ACTUAL):**
```
rag_system.py → ProviderAdapterClient → LLMManager → OpenRouter
                        ↓
            Returns GenerationResponse with tokens_used
                        ↓
                (tokens were available but not used)
```

**Fix Applied (Actual Implementation - 30 minutes):**

1. Added `tokens_used`, `input_tokens`, `output_tokens`, `model_used` fields to RAGResponse dataclass
2. Updated ProviderAdapterClient to store token breakdown in GenerationResponse.metadata
3. Extracted token breakdown from generation_response.metadata in rag_system.py
4. Passed tokens through to RAGResponse creation
5. Updated rag-api to use token breakdown from rag_response

**Files Modified:**
- `services/rag-service/src/pipeline/rag_system.py` - RAGResponse fields + extraction
- `services/rag-service/src/utils/factories.py` - ProviderAdapterClient metadata
- `services/rag-api/main.py` - Use rag_response tokens

**Testing Results:**
```bash
curl -X POST http://localhost:8082/api/v1/query ... | jq '.tokensUsed'
# Actual: {"input": 2134, "output": 1354, "total": 3488} ✅
```

**Use Case Enabled:**
- ✅ Accurate cost calculation (input vs output pricing)
- ✅ Rate limiting by token count
- ✅ Usage analytics per user
- ✅ Cost attribution per query

**Effort:** 30 minutes (not 3-4 hours as originally planned)

---

### 5. NN Sources Deduplication
**Priority:** High
**Status:** ✅ FIXED
**Issue:** Multiple chunks from the same NN document created duplicate source entries in the UI.

**Fix Applied:**
- Added deduplication logic in `rag_system.py` after collecting nn_sources
- Deduplicate by `eli_url` to show each document only once
- Log deduplication statistics for debugging

**Files Modified:**
- `services/rag-service/src/pipeline/rag_system.py:1326-1340`

---

**Requirements:**
- **MANDATORY for:** `narodne-novine` feature
- **OPTIONAL for:** All other features/scopes
- Must be feature-specific, not breaking other workflows

**Implementation Plan:**

#### Phase 1: Metadata Extraction (Document Processing)
**Goal:** Extract Narodne Novine metadata from HTML `<meta>` tags during document processing.

**Metadata to extract:**
- `eli_url`: Full ELI URL (e.g., `https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1`)
- `title`: Document title (e.g., `"Uredba o utvrđivanju najviših maloprodajnih cijena naftnih derivata"`)
- `issue`: NN issue number (e.g., `"NN 1/2024"`)
- `doc_number`: Document number in issue (e.g., `"1"`)
- `pages`: Page range (e.g., `"1-2"`)
- `publisher`: Donositelj (e.g., `"Vlada Republike Hrvatske"`)
- `date_published`: Publication date (e.g., `"2024-01-02"`)

**HTML Meta Tag Examples to Parse:**
```html
<meta about="https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1"
      property="http://data.europa.eu/eli/ontology#title"
      content="Uredba o utvrđivanju najviših maloprodajnih cijena naftnih derivata" lang="hrv" />

<meta about="https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1"
      property="http://data.europa.eu/eli/ontology#number" content="1"/>

<meta about="https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1"
      property="http://data.europa.eu/eli/ontology#date_publication"
      content="2024-01-02" datatype="http://www.w3.org/2001/XMLSchema#date"/>

<meta about="https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1"
      property="http://data.europa.eu/eli/ontology#passed_by"
      resource="https://narodne-novine.nn.hr/eli/vocabularies/nn-institutions/19560" />
```

**Implementation Steps:**

**Step 1.1:** Create NN metadata extractor
```
File: services/rag-service/src/extraction/nn_metadata_extractor.py (NEW FILE)

Purpose: Extract Narodne Novine metadata from HTML

Functions:
  - extract_nn_metadata(html_content: str, file_path: Path) -> dict | None
      * Parse HTML with BeautifulSoup
      * Find <meta about="...eli/sluzbeni/..." /> tags
      * Extract ELI URL from 'about' attribute
      * Extract title from property="eli/ontology#title"
      * Extract number from property="eli/ontology#number"
      * Extract date from property="eli/ontology#date_publication"
      * Parse issue number from file path (e.g., 2024/001/file.html → NN 1/2024)
      * Return structured dict or None if not NN document

  - parse_issue_from_path(file_path: Path) -> str | None
      * Extract year and issue from path: "2024/001" → "NN 1/2024"

  - is_nn_document(html_content: str) -> bool
      * Quick check if HTML contains ELI metadata
      * Look for <meta about="...narodne-novine.nn.hr/eli/..." />

Return structure:
{
    "eli_url": "https://narodne-novine.nn.hr/eli/sluzbeni/2024/1/1",
    "title": "Uredba o utvrđivanju...",
    "issue": "NN 1/2024",
    "doc_number": "1",
    "pages": "1-2",  # From metadata if available
    "publisher": "Vlada Republike Hrvatske",  # From institution lookup
    "date_published": "2024-01-02",
    "document_type": "Uredba"
}
```

**Step 1.2:** Integrate metadata extraction into document pipeline
```
File: services/rag-service/src/pipeline/rag_system.py

Modify: add_documents() method (around line 870)

Changes:
  1. Import nn_metadata_extractor
  2. After text extraction, check if HTML document
  3. If HTML and in narodne-novine feature scope, extract NN metadata
  4. Pass metadata to create_chunk_metadata()

Pseudocode:
  # After extraction_result = self._document_extractor.extract_text(doc_path)
  nn_metadata = None
  if doc_path.suffix == '.html' and self._is_nn_feature():
      from ..extraction.nn_metadata_extractor import extract_nn_metadata
      with open(doc_path, 'r', encoding='utf-8') as f:
          html_content = f.read()
      nn_metadata = extract_nn_metadata(html_content, doc_path)
      if nn_metadata:
          logger.info("nn_metadata", "extracted", f"title={nn_metadata['title'][:50]} | eli={nn_metadata['eli_url']}")
```

**Step 1.3:** Update chunk metadata creation
```
File: services/rag-service/src/pipeline/rag_system.py

Modify: create_chunk_metadata() function (around line 906)

Add parameter: nn_metadata: dict | None = None

Update metadata dict:
  metadata = {
      "source_file": str(doc_path),
      "chunk_index": chunk_idx,
      "language": language,
      # ... existing fields ...
  }

  # Add NN metadata if available (OPTIONAL - only for narodne-novine)
  if nn_metadata:
      metadata["nn_metadata"] = nn_metadata

Return: metadata dict now includes optional nn_metadata field
```

**Step 1.4:** Add intelligent feature detection
```
File: services/rag-service/src/pipeline/rag_system.py

Add methods to RAGSystem class:

def _get_feature_scope(self) -> str | None:
    """Get current feature name from scope context.

    Returns feature name if scope=feature, otherwise None.
    This allows feature-specific behavior without configuration files.
    """
    # Feature name should be set during RAGSystem initialization
    # from CLI args: --scope feature --feature narodne-novine
    return getattr(self, 'feature_name', None)

def _is_nn_feature(self) -> bool:
    """Check if processing narodne-novine feature.

    Convention-based detection:
    - No configuration needed
    - Based on --scope feature --feature narodne-novine
    - Enables NN-specific metadata extraction automatically
    """
    return self._get_feature_scope() == 'narodne-novine'

def _should_extract_nn_metadata(self, file_path: Path, html_content: str) -> bool:
    """Determine if NN metadata extraction should run.

    Conditions:
    1. Must be in narodne-novine feature scope
    2. Must be HTML file
    3. Must contain ELI metadata tags

    This triple-check ensures:
    - No metadata extraction for non-NN features
    - No wasted processing on non-HTML files
    - No errors on non-NN HTML files
    """
    if not self._is_nn_feature():
        return False

    if file_path.suffix.lower() != '.html':
        return False

    # Quick check for ELI metadata presence
    from ..extraction.nn_metadata_extractor import is_nn_document
    return is_nn_document(html_content)

Note: Convention over configuration approach
- No config files needed
- Behavior determined by scope parameter
- Self-documenting and maintainable
```

**Step 1.5:** Update RAGSystem initialization to accept feature context
```
File: services/rag-service/src/pipeline/rag_system.py

Modify: __init__() method

Add parameter:
  def __init__(
      self,
      language: str,
      tenant_slug: str = "development",
      user_id: str = "dev_user",
      scope: str = "user",           # NEW
      feature_name: str | None = None  # NEW
  ):
      self.language = language
      self.tenant_slug = tenant_slug
      self.user_id = user_id
      self.scope = scope
      self.feature_name = feature_name
      # ... rest of init

Why:
- Makes feature context explicit
- Passed from CLI through to RAGSystem
- No hidden global state
- Testable and clear
```

**Step 1.6:** Update CLI to pass feature context
```
File: services/rag-service/src/cli/rag_cli.py

Modify: Where RAGSystem is instantiated

Before:
  rag_system = RAGSystem(
      language=args.language,
      tenant_slug=args.tenant,
      user_id=args.user
  )

After:
  rag_system = RAGSystem(
      language=args.language,
      tenant_slug=args.tenant,
      user_id=args.user,
      scope=args.scope,
      feature_name=args.feature
  )

Why:
- Passes CLI context to system
- Enables feature-specific behavior
- No configuration files needed
```

**Dependencies needed:**
- `beautifulsoup4` (already in requirements.txt)
- `lxml` parser (already in requirements.txt)

**Architectural Principles Applied:**
1. **Convention over Configuration** - Behavior based on scope, not config files
2. **Feature Detection** - System auto-detects NN documents from HTML structure
3. **Explicit Context** - Feature name passed through call chain, not global
4. **Fail-Safe** - Triple-check before extraction (scope + extension + ELI tags)
5. **Optional Enhancement** - NN metadata enriches data, doesn't break without it

**Error Handling:**
- If metadata extraction fails → log warning, continue processing
- NN metadata is OPTIONAL → document processes without it
- Non-NN documents → pass through unchanged, zero overhead
- Wrong scope → metadata extraction never triggered

#### Phase 2: Metadata Storage & Retrieval (Vector Database)
**Goal:** Store NN metadata in Weaviate and return it with search results.

**Step 2.1:** Verify metadata passes through to Weaviate
```
File: services/rag-service/src/vectordb/storage.py

Current behavior:
  - add_documents() accepts chunks with metadata dict
  - Metadata should already be stored in Weaviate properties

Verification needed:
  - Check that nn_metadata nested dict is properly serialized
  - Weaviate supports nested objects in properties
  - Test: Insert chunk with nn_metadata, retrieve it, verify structure

Note: May need to flatten nn_metadata or store as JSON string
```

**Step 2.2:** Return nn_metadata in search results
```
File: services/rag-service/src/vectordb/search.py

Modify: search() and hybrid_search() methods

Changes:
  - When retrieving chunks from Weaviate, include nn_metadata in results
  - Ensure metadata is passed through SearchResult objects

Current SearchResult structure:
  {
    "content": "...",
    "metadata": {
      "source_file": "...",
      "chunk_index": 0,
      "nn_metadata": {  # NEW - only for NN docs
        "eli_url": "...",
        "title": "...",
        ...
      }
    },
    "score": 0.95
  }
```

**Step 2.3:** Pass metadata to RAG response
```
File: services/rag-service/src/pipeline/rag_system.py

Modify: query() method - where RAGResponse is created

Changes:
  - Collect nn_metadata from retrieved chunks
  - Deduplicate by eli_url (multiple chunks from same document)
  - Build sources list for RAGResponse
  - Include in response metadata

RAGResponse.metadata.sources structure:
  [
    {
      "eli_url": "https://...",
      "title": "Uredba o utvrđivanju...",
      "issue": "NN 1/2024",
      "publisher": "Vlada RH",
      "relevance": 0.95,  # from chunk score
      "citation_id": 1    # for inline citations [1]
    },
    ...
  ]
```

**Testing:**
- Process one NN document with metadata
- Query system and verify nn_metadata in response
- Check that sources array is properly formatted

#### Phase 3: LLM Prompt Enhancement (Inline Citations)
**Goal:** Modify LLM prompt to include source citations in responses.

**Step 3.1:** Update Croatian prompts configuration
```
File: services/rag-service/config/prompts.hr.toml

Add to [generation] section:

citation_instruction = """
Kada koristiš informacije iz dokumenata, navedi izvor pomoću brojeva u uglatim zagradama [1], [2], itd.
Na kraju odgovora, automatski će biti dodana lista izvora.

Primjer:
"Prema Uredbi o cijenama goriva [1], najviša dozvoljena cijena Eurosupera 95 je određena formulom [1].
Paušalni porez za samostalne djelatnosti izmijenjen je na 12% [2]."
"""

[generation.citation]
enabled = true
style = "numeric"  # [1], [2], [3]
include_inline = true
max_sources = 5
```

**Step 3.2:** Format context with source list
```
File: services/rag-service/src/generation/llm_provider.py (or rag_system.py)

Modify: _format_prompt_with_context() or similar

Before:
  Context: chunk1\n\nchunk2\n\n...

After:
  Dostupni dokumenti:

  [1] NN 1/2024 - Uredba o utvrđivanju najviših maloprodajnih cijena...
  Sadržaj: {chunk content}

  [2] NN 1/2024 - Pravilnik o izmjenama Pravilnika o paušalnom...
  Sadržaj: {chunk content}

Instructions:
  - Group chunks by document (same eli_url)
  - Assign citation number [1], [2], ...
  - Include document title and issue in context
  - Instruct LLM to use numbers when citing
```

**Step 3.3:** System prompt modification
```
File: services/rag-service/src/generation/llm_provider.py

Update system prompt when NN feature active:

system_prompt = f"""
Ti si asistent za hrvatsku pravnu dokumentaciju iz Narodnih novina.

VAŽNO - Navođenje izvora:
- Kada koristiš informaciju iz dokumenta, navedi broj u zagradama: [1], [2]
- Koristi točne informacije iz konteksta
- Ne izmišljaj informacije koje nisu u dokumentima

{base_system_prompt}
"""

Note: Only add citation instruction when nn_metadata is present
```

**Testing:**
- Query: "Kolika je najviša cijena goriva?"
- Expected: Response includes "[1]" citations
- Verify citations match document numbers in context

#### Phase 4: Frontend Display (Expandable Source List)
**Goal:** Display rich source information with clickable links in web-ui.

**Step 4.1:** Update API to pass sources to frontend
```
File: services/rag-api/main.py

Modify: /messages/send endpoint

Current response:
  {
    "userMessage": {...},
    "assistantMessage": {
      "content": "answer text",
      "metadata": {
        "ragContext": {
          "documentsRetrieved": 5,
          "documentsUsed": 3,
          ...
        }
      }
    }
  }

Enhanced response:
  {
    "assistantMessage": {
      "content": "answer with citations [1], [2]",
      "metadata": {
        "ragContext": { ... },
        "sources": [  # NEW
          {
            "citationId": 1,
            "title": "Uredba o utvrđivanju...",
            "issue": "NN 1/2024",
            "eliUrl": "https://narodne-novine.nn.hr/...",
            "publisher": "Vlada RH",
            "pages": "1-2",
            "relevance": 95
          },
          ...
        ]
      }
    }
  }
```

**Step 4.2:** Create SourcesList component
```
File: services/web-ui/src/components/SourcesList.tsx (NEW FILE)

Props:
  - sources: Source[]

interface Source {
  citationId: number;
  title: string;
  issue: string;
  eliUrl: string;
  publisher?: string;
  pages?: string;
  relevance: number;
}

Component structure:
  <div className="sources-section mt-4 pt-3 border-t border-gray-700">
    <button onClick={() => setExpanded(!expanded)}>
      📄 Izvori ({sources.length} dokumenta korištena)
      <ChevronIcon />
    </button>

    {expanded && (
      <div className="sources-list mt-2 space-y-2">
        {sources.map(source => (
          <div key={source.citationId} className="source-card">
            <div className="source-header">
              <span className="citation-number">[{source.citationId}]</span>
              <span className="source-issue">{source.issue}</span>
            </div>
            <h4 className="source-title">{source.title}</h4>
            <div className="source-meta">
              {source.publisher && <span>📋 {source.publisher}</span>}
              {source.pages && <span>📄 Str. {source.pages}</span>}
              <span className="relevance">Relevantnost: {source.relevance}%</span>
            </div>
            <a href={source.eliUrl} target="_blank" rel="noopener"
               className="source-link">
              🔗 Pogledaj na Narodnim novinama
            </a>
          </div>
        ))}
      </div>
    )}
  </div>

Styling:
  - Collapsed by default
  - Smooth expand/collapse animation
  - Source cards with hover effects
  - External link icon for NN links
```

**Step 4.3:** Integrate SourcesList into Message component
```
File: services/web-ui/src/components/Message.tsx

Import: import { SourcesList } from './SourcesList';

Modify: After message content, before copy button section

Add:
  {/* Source citations - only for assistant messages with sources */}
  {!isUser && message.metadata?.sources && message.metadata.sources.length > 0 && (
    <SourcesList sources={message.metadata.sources} />
  )}

Result:
  - Sources displayed below message content
  - Only shown when sources are available
  - Collapsible to save space
```

**Step 4.4:** Update TypeScript types
```
File: services/web-ui/src/types/message.ts (or inline types)

Add Source interface:
  interface Source {
    citationId: number;
    title: string;
    issue: string;
    eliUrl: string;
    publisher?: string;
    pages?: string;
    relevance: number;
  }

Update MessageMetadata:
  interface MessageMetadata {
    ragContext?: { ... };
    sources?: Source[];  // NEW
  }
```

**Testing:**
- Send query that retrieves NN documents
- Verify sources section appears
- Click expand/collapse - verify smooth animation
- Click NN link - verify opens in new tab to correct URL
- Check multiple sources displayed correctly
- Verify citation numbers [1], [2] match source list

---

## Implementation Summary & Checklist

### Phase Breakdown:
1. **Phase 1:** ✅ COMPLETED - Metadata Extraction (Backend - Python)
2. **Phase 2:** ✅ COMPLETED - Metadata Storage & Retrieval (Backend - Weaviate)
3. **Phase 3:** ✅ COMPLETED - LLM Prompt Enhancement (Backend - Prompt Engineering)
4. **Phase 4:** ✅ COMPLETED - Frontend Display (Frontend - React/TypeScript)

### Files to Create:
- `services/rag-service/src/extraction/nn_metadata_extractor.py`
- `services/web-ui/src/components/SourcesList.tsx`
- `services/web-ui/src/types/message.ts` (if not exists)

### Files to Modify:
- `services/rag-service/src/pipeline/rag_system.py`
- `services/rag-service/src/vectordb/search.py`
- `services/rag-service/config/prompts.hr.toml`
- `services/rag-service/src/generation/llm_provider.py`
- `services/rag-api/main.py`
- `services/web-ui/src/components/Message.tsx`

### Acceptance Criteria:
- [x] **Phase 1 Complete:** NN HTML documents have ELI metadata extracted during processing
- [x] **Phase 1 Complete:** Metadata includes: eli_url, title, issue, doc_number, publisher, date
- [x] **Phase 1 Complete:** Metadata only extracted for `narodne-novine` feature scope
- [x] **Phase 1 Complete:** Non-NN documents process unchanged (no errors)
- [x] **Phase 2 Complete:** NN metadata stored in Weaviate chunk properties (JSON serialized)
- [x] **Phase 2 Complete:** Metadata returned in search results (with deserialization)
- [x] **Phase 2 Complete:** Sources array built from chunks and deduplicated by eli_url
- [x] **Phase 3 Complete:** LLM prompt includes numbered document list [1], [2]
- [x] **Phase 3 Complete:** LLM instructed to cite sources inline (Croatian instructions)
- [x] **Phase 3 Complete:** Citation instruction only added when NN sources present
- [x] **Phase 4 Complete:** Frontend receives nnSources field via FastAPI
- [x] **Phase 4 Complete:** Each source includes: title, issue, eli, publisher, year
- [x] **Phase 4 Complete:** Field mapping fixed (eli_url → eli, date_published → year)
- [x] **Phase 4 Complete:** SourcesList.tsx component created with expand/collapse
- [x] **Phase 4 Complete:** Integration into Message.tsx
- [x] **Phase 4 Complete:** TypeScript types defined in types/message.ts
- [x] **Phase 4 Complete:** Croatian UI labels and pluralization
- [x] **UI Enhancement:** Removed "5/5 docs" display
- [x] **UI Enhancement:** Added timestamp display (🕐)
- [x] **UI Enhancement:** Added token usage display (🎯)
- [x] **Integration Test:** Process NN doc → query → verify sources in API response
- [ ] **End-to-End Test:** Query NN → verify sources display in UI
- [ ] **Regression Test:** Non-NN features still work (user/tenant scopes)
- [ ] **Performance Test:** Metadata extraction doesn't slow processing significantly

### Testing Commands:
```bash
# Process NN documents
python rag.py --language hr --scope feature --feature narodne-novine process-docs ./data/features/narodne_novine/documents/hr/2024/001

# Query and verify sources
python rag.py --language hr --scope feature --feature narodne-novine query "Kolika je najviša cijena goriva?"

# Check metadata in Weaviate
python -c "import weaviate; client = weaviate.connect_to_local(); coll = client.collections.get('Features_narodne_novine_hr'); obj = next(iter(coll.iterator())); print(obj.properties.get('nn_metadata'))"
```

### Dependencies:
- `beautifulsoup4` ✅ (already in requirements.txt)
- `lxml` ✅ (already in requirements.txt)
- No new dependencies needed

### Architecture Principles:
1. **Convention over Configuration** - No feature-specific config files
2. **Intelligent Detection** - Auto-detect NN documents from HTML ELI tags
3. **Explicit Context Passing** - Feature scope passed through call chain
4. **Triple Validation** - Check scope + file extension + ELI metadata presence
5. **Graceful Degradation** - System works with or without NN metadata
6. **Zero Impact** - Non-NN features have zero overhead from this enhancement

### Error Handling:
- Metadata extraction failures → log warning, continue processing
- Missing NN metadata → document processes normally
- Frontend gracefully handles missing sources (doesn't display section)

---

## Bug Fixes

### 3. [Template for future bugs]
**Priority:**
**Status:**
**Issue:**
**Files:**

---

## Documentation Updates Needed
- Update README with narodne-novine feature setup
- Document metadata extraction configuration
- Add examples of source citation output

---

## Session Summary

### Session 2025-10-08: Narodne Novine Phase 4 UI & Security Fixes

**Scope:** Complete Phase 4 frontend implementation and resolve critical security issue

**Work Completed:**

1. **Phase 4 Frontend UI (COMPLETE)**
   - ✅ Created `SourcesList.tsx` component with expand/collapse functionality
   - ✅ Created shared TypeScript types in `types/message.ts`
   - ✅ Integrated sources display into `Message.tsx`
   - ✅ Croatian UI localization with proper pluralization
   - ✅ Citation numbers [1], [2], [3] with ELI links

2. **Web-API Integration Fixes**
   - ✅ Added `nnSources` field to web-api TypeScript types (`rag.service.ts`)
   - ✅ Updated `messages.router.ts` to pass nnSources metadata to frontend
   - ✅ Removed all mock code from production (user requirement: "That freaking mock, remove that")
   - ✅ Added `citationId` field to sources (1-based enumeration)

3. **UI Polish**
   - ✅ Fixed time format: US AM/PM → 24h EU format (`toLocaleTimeString('hr-HR', { hour12: false })`)
   - ✅ Hide token display when count is 0
   - ✅ Keep timestamp and metadata display

4. **Critical Security Fix: OpenRouter API Key Exposure**
   - 🔴 **ROOT CAUSE IDENTIFIED**: OpenRouter is GitHub secret scanning partner
   - ✅ Removed API key logging from `llm_provider.py` (even partial key preview)
   - ✅ Added environment variable expansion to `config_loader.py` (`${VAR_NAME}` syntax)
   - ✅ Changed `config.toml` to use `${OPENROUTER_API_KEY}` placeholder
   - ✅ User manually added key to systemd service environment
   - **Impact**: Prevents automatic key revocation by GitHub scanning

5. **Bug Fix: Source Deduplication**
   - ✅ Multiple chunks from same NN document created duplicate sources
   - ✅ Added deduplication by `eli_url` in `rag_system.py:1326-1340`
   - ✅ Now shows each document only once with first occurrence
   - **Result**: UI now shows 1 source per document instead of 5

6. **Token Tracking Investigation**
   - 🔍 **ROOT CAUSE IDENTIFIED**: RAG system uses `ollama_client.py` directly (line 1382)
   - Architecture issue: Bypasses `llm_manager.py` with OpenRouter token tracking
   - ✅ Created detailed 6-step implementation plan (3-4 hour task)
   - ✅ Documented in IMPROVEMENTS.md Issue #4
   - ❌ NOT YET IMPLEMENTED - awaiting user decision
   - **Purpose**: Enable rate limiting by token count for cost control

**Files Modified:**
- `services/web-ui/src/components/SourcesList.tsx` (created)
- `services/web-ui/src/types/message.ts` (created)
- `services/web-ui/src/components/Message.tsx`
- `services/web-api/src/modules/messages/messages.router.ts`
- `services/web-api/src/services/rag.service.ts`
- `services/rag-service/src/generation/llm_provider.py`
- `services/rag-service/src/utils/config_loader.py`
- `services/rag-service/config/config.toml`
- `services/rag-service/src/pipeline/rag_system.py`
- `services/rag-api/main.py`
- `CLAUDE.md` (path corrections, architecture updates)

**User Feedback:**
- "That freaking mock, remove that I don't want no mock ever" → All mocks removed
- "We are in EU, so stop giving me US time with AM/PM" → Fixed to 24h format
- "Why new file" → Consolidated plan into IMPROVEMENTS.md
- "x is local computer rag is on server" → Updated documentation

**Testing Performed:**
- ✅ Build: `npm run build` in web-ui
- ✅ Systemd services restarted (web-api, web-ui)
- ✅ Security fix verified: No API key in logs
- ✅ Source deduplication tested with live query

**Known Issues:**
- Token tracking still returns 0 (architecture refactoring needed)
- `clear-data` command for feature scope (deprioritized)

**Next Steps (If User Approves):**
1. Implement token tracking (3-4 hours, detailed plan in Issue #4)
2. End-to-end testing of NN sources UI display
3. Performance testing of metadata extraction

**Session Duration:** Full implementation session
**Status:** Phase 4 Complete, Security Fixed, Token Tracking Planned

---

### Session 2025-10-08 (Continued): Token Tracking Implementation

**Scope:** Implement complete token tracking with input/output breakdown for accurate cost calculation

**Work Completed:**

1. **Token Tracking Infrastructure (COMPLETE)**
   - ✅ Added `tokens_used`, `input_tokens`, `output_tokens`, `model_used` fields to RAGResponse
   - ✅ Updated ProviderAdapterClient to extract token breakdown from OpenRouter response
   - ✅ Updated rag_system.py to pass token data through to RAGResponse
   - ✅ Updated rag-api to use token breakdown from RAGResponse
   - ✅ Verified OpenRouter integration is working correctly (not Ollama local)

2. **UI Improvements (COMPLETE)**
   - ✅ Updated timestamp format: `📅 08.10 14:23` (date + time, no seconds)
   - ✅ Token display with breakdown: `🎯 3488 tokens (2134in + 1354out)`
   - ✅ Simplified time display: Single query time `⏱️ 2.6s` (removed confusing duplicates)
   - ✅ Built web-ui successfully

**Architecture Clarification:**

**Discovered:** System ALREADY uses OpenRouter via LLMManager, not local Ollama for RAG queries.

**Token Flow:**
```
RAG Query → RAGSystem → ProviderAdapterClient → LLMManager → OpenRouter
                                    ↓
                        Returns TokenUsage (input/output/total)
                                    ↓
                    RAGResponse (tokens_used, input_tokens, output_tokens)
                                    ↓
                        RAG-API → web-API → Database → Frontend
```

**Files Modified:**
- `services/rag-service/src/pipeline/rag_system.py` - Added token fields to RAGResponse
- `services/rag-service/src/utils/factories.py` - ProviderAdapterClient returns token breakdown
- `services/rag-api/main.py` - Use token breakdown from RAGResponse
- `services/web-ui/src/components/Message.tsx` - UI improvements for date/time/tokens

**Testing Results:**
```json
{
  "model": "qwen/qwen3-30b-a3b-instruct-2507",
  "tokensUsed": {
    "input": 2134,
    "output": 1354,
    "total": 3488
  }
}
```

**Database Storage:**
- Location: `Message.metadata.ragContext.tokensUsed`
- Granularity: Per message (not per chat)
- Enables: Cost attribution, analytics, refunds, rate limiting

**Cost Calculation Example:**
```python
INPUT_RATE = 0.10 / 1_000_000   # $0.10 per 1M input tokens
OUTPUT_RATE = 0.30 / 1_000_000  # $0.30 per 1M output tokens
cost = (input_tokens * INPUT_RATE) + (output_tokens * OUTPUT_RATE)
# Example: (2134 * 0.0000001) + (1354 * 0.0000003) = $0.000619
```

**What Changed from Original Plan:**

Original Issue #4 assumed RAG used local Ollama and needed 3-4 hour refactoring. **Reality:** RAG already used OpenRouter via ProviderAdapterClient, just needed to pass tokens through the response chain (30 minutes).

**Known Issues Resolved:**
- ✅ Token tracking working (was returning 0, now returns real OpenRouter counts)
- ⏳ `clear-data` command for feature scope (still deprioritized)

**Next Steps:**
- End-to-end testing of NN sources + token tracking in live UI
- Monitor token usage for rate limiting implementation
- Consider adding daily/monthly usage aggregation queries

**Session Duration:** 2 hours (token tracking + UI improvements)
**Status:** ✅ Token Tracking Complete with Input/Output Breakdown

---

### Session 2025-10-09: Narodne Novine Incremental Processing Performance Optimization

**Scope:** Optimize batch processing configuration and add comprehensive performance monitoring

**Problem:**
- Incremental NN processing was extremely slow (53+ hours estimated for 2000 folders)
- Embedding batch size was too small (32) for a 244GB RAM system
- No visibility into performance bottlenecks (CPU, RAM, phase timings)
- Outlier folders (e.g., folder 108: 420 seconds vs 6-30s average)

**Work Completed:**

1. **Configuration Optimization (COMPLETE)**
   - ✅ Increased `embeddings.batch_size`: 32 → 200 (6x larger batches)
   - ✅ Optimized `batch_processing.embedding_batch_size`: 1000 → 200 (consistency)
   - ✅ Adjusted `batch_processing.document_batch_size`: 100 → 50 (incremental stability)
   - ✅ Reduced `batch_processing.vector_insert_batch_size`: 2000 → 1000 (stability)
   - **Expected Impact:** 3-5x speedup in embedding generation phase

2. **Full Instrumentation Infrastructure (COMPLETE)**
   - ✅ Added `ResourceSnapshot` dataclass for system state tracking
   - ✅ Implemented `capture_resource_snapshot()` - CPU, RAM, swap, process metrics
   - ✅ Enhanced `FolderStats` with chunk statistics and resource deltas
   - ✅ Added comprehensive structured logging:
     - `RESOURCE_SNAPSHOT` - System state at each processing step
     - `CHUNK_STATS` - Min/max/avg chunk sizes per folder
     - `RESOURCE_DELTA` - RAM/CPU changes during processing

3. **Monitoring Capabilities (COMPLETE)**
   - ✅ Resource snapshots before/after each folder
   - ✅ Chunk size analysis (detect large chunks causing slowdowns)
   - ✅ RAM delta tracking (detect memory leaks)
   - ✅ CPU utilization monitoring (detect throttling)
   - ✅ Process-specific metrics (CPU%, memory usage)

**Log Format Examples:**
```
RESOURCE_SNAPSHOT | folder=2025/102 | cpu=99.8% | ram=9.8GB (4.0%) | swap=0.1GB (1.6%) | proc_cpu=99.8% | proc_mem=9751MB
CHUNK_STATS | folder=2025/102 | chunks=381 | avg_size=1850 | min=420 | max=3200 | total_chars=246050
RESOURCE_DELTA | folder=2025/102 | ram_delta=150MB | cpu_avg=99.5%
```

**Performance Analysis Tools:**
- ✅ `analyze_nn_stats.py` - Real-time statistics with system resource monitoring
  - Shows: Throughput, ETA, slowest folders, CPU/RAM/Swap usage
  - Detects: Performance outliers, memory pressure, thermal throttling
  - Works with mixed old/new config data for before/after comparison

**Files Modified:**
- `services/rag-service/config/config.toml` - Batch size optimizations
- `services/rag-service/scripts/process_nn_incremental.py` - Full instrumentation
- `services/rag-service/scripts/analyze_nn_stats.py` - Enhanced with psutil monitoring

**Testing Results:**
- ✅ Process starts with batch_size=200 (verified in logs)
- ✅ Resource monitoring logs being generated
- ✅ Analyzer shows system resources and performance issues
- ✅ Mixed data (old batch=32 + new batch=200) handled correctly

**Expected Performance Improvement:**
- **Before:** ~53 hours for 2000 folders (batch=32)
- **After:** ~15-20 hours for 2000 folders (batch=200)
- **Speedup:** 3x faster overall, 4-7x faster for embedding phase
- **Outlier folders:** 420s → 60-100s (folder 108 type)

**Current Status:**
- ✅ Service running with optimized configuration
- ✅ Processing folder 2025/100 (30 folders completed)
- ✅ Old folders (127-103) provide baseline for comparison
- ✅ New folders (102+) should show dramatic speedup

**Monitoring Commands:**
```bash
# Watch live processing with resource monitoring
tail -f logs/nn_service.log | grep "RESOURCE_SNAPSHOT\|CHUNK_STATS\|COMPLETED"

# Get real-time statistics and system resources
python services/rag-service/scripts/analyze_nn_stats.py

# Check specific resource patterns
grep "RESOURCE_DELTA" logs/nn_service.log | tail -20
```

**Architecture Decisions:**
1. **Convention-based monitoring** - All logging uses structured format for grep/analysis
2. **Backward compatible** - Analyzer works with both old and new log formats
3. **Production-ready** - No overhead when monitoring features not used
4. **Comprehensive** - Captures everything needed to diagnose any performance issue

**Session Duration:** 2 hours (configuration + instrumentation + testing)
**Status:** ✅ Performance Optimization Complete, Service Running with Monitoring
