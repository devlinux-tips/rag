# RAG System Improvements Tracker

## Pending Fixes

### 1. Fix `clear-data` command for feature scope
**Priority:** Medium
**Status:** Not started
**Issue:** The `clear-data` command doesn't work correctly with `--scope feature --feature narodne-novine`. It looks in the wrong directory (`development/dev_user` instead of `features/narodne_novine`).

**Current behavior:**
```bash
python rag.py --language hr --scope feature --feature narodne-novine clear-data --dry-run
# Output: Would clear 0 paths for development/user:dev_user/hr
```

**Expected behavior:**
```bash
python rag.py --language hr --scope feature --feature narodne-novine clear-data --dry-run
# Should clear:
#   - Weaviate collection: Features_narodne_novine_hr
#   - Feature data directory: data/features/narodne_novine/
#   - Feature cache files
```

**Files to investigate:**
- `/home/rag/src/rag/services/rag-service/src/cli/rag_cli.py` - CLI command implementation
- `/home/rag/src/rag/services/rag-service/src/utils/data_manager.py` - Data clearing logic (if exists)
- Collection naming logic in vectordb module

**Related code locations:**
- `services/rag-service/src/cli/rag_cli.py:clear_data()` command handler
- Feature scope detection and path resolution

---

## Planned Enhancements

### 2. Narodne Novine Source Enrichment
**Priority:** High
**Status:** Planning
**Description:** Add rich source metadata for Narodne Novine documents to enable proper citations with links.

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
1. **Phase 1:** Metadata Extraction (Backend - Python)
2. **Phase 2:** Metadata Storage & Retrieval (Backend - Weaviate)
3. **Phase 3:** LLM Prompt Enhancement (Backend - Prompt Engineering)
4. **Phase 4:** Frontend Display (Frontend - React/TypeScript)

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
- [ ] **Phase 1 Complete:** NN HTML documents have ELI metadata extracted during processing
- [ ] **Phase 1 Complete:** Metadata includes: eli_url, title, issue, doc_number, publisher, date
- [ ] **Phase 1 Complete:** Metadata only extracted for `narodne-novine` feature scope
- [ ] **Phase 1 Complete:** Non-NN documents process unchanged (no errors)
- [ ] **Phase 2 Complete:** NN metadata stored in Weaviate chunk properties
- [ ] **Phase 2 Complete:** Metadata returned in search results
- [ ] **Phase 2 Complete:** Sources array built from chunks and deduplicated by eli_url
- [ ] **Phase 3 Complete:** LLM prompt includes numbered document list [1], [2]
- [ ] **Phase 3 Complete:** LLM instructed to cite sources inline
- [ ] **Phase 3 Complete:** Citation instruction only added when NN sources present
- [ ] **Phase 4 Complete:** Frontend displays expandable "Izvori" section
- [ ] **Phase 4 Complete:** Each source shows: title, issue, publisher, relevance, link
- [ ] **Phase 4 Complete:** Links open narodne-novine.nn.hr in new tab
- [ ] **Phase 4 Complete:** Inline citations [1], [2] match source list numbers
- [ ] **Integration Test:** Process NN doc → query → verify sources in UI
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
