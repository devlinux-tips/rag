-- Multi-Tenant RAG System Schema for SurrealDB
-- Supports tenant → users hierarchy with document scoping

-- ===============================================
-- TENANT MANAGEMENT
-- ===============================================

DEFINE TABLE tenant SCHEMAFULL;

DEFINE FIELD id ON TABLE tenant TYPE record<tenant>;
DEFINE FIELD name ON TABLE tenant TYPE string
  ASSERT string::len($value) > 0 AND string::len($value) <= 100;
DEFINE FIELD slug ON TABLE tenant TYPE string
  ASSERT string::len($value) > 0 AND string::len($value) <= 50
  ASSERT string::is::alphanum($value) OR string::contains($value, '_') OR string::contains($value, '-');
DEFINE FIELD description ON TABLE tenant TYPE option<string>;
DEFINE FIELD status ON TABLE tenant TYPE string DEFAULT 'active'
  ASSERT $value INSIDE ['active', 'suspended', 'inactive'];
DEFINE FIELD settings ON TABLE tenant TYPE object DEFAULT {};
DEFINE FIELD language_preference ON TABLE tenant TYPE string DEFAULT 'hr'
  ASSERT $value INSIDE ['hr', 'en', 'multilingual'];
DEFINE FIELD cultural_context ON TABLE tenant TYPE string DEFAULT 'croatian_business'
  ASSERT $value INSIDE ['croatian_business', 'croatian_academic', 'international', 'technical'];
DEFINE FIELD subscription_tier ON TABLE tenant TYPE string DEFAULT 'basic'
  ASSERT $value INSIDE ['basic', 'professional', 'enterprise'];
DEFINE FIELD created_at ON TABLE tenant TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON TABLE tenant TYPE datetime DEFAULT time::now();

-- Indexes for tenant management
DEFINE INDEX tenant_slug_idx ON TABLE tenant COLUMNS slug UNIQUE;
DEFINE INDEX tenant_status_idx ON TABLE tenant COLUMNS status;
DEFINE INDEX tenant_created_idx ON TABLE tenant COLUMNS created_at;

-- ===============================================
-- USER MANAGEMENT
-- ===============================================

DEFINE TABLE user SCHEMAFULL;

DEFINE FIELD id ON TABLE user TYPE record<user>;
DEFINE FIELD tenant_id ON TABLE user TYPE record<tenant>;
DEFINE FIELD email ON TABLE user TYPE string
  ASSERT string::is::email($value);
DEFINE FIELD username ON TABLE user TYPE string
  ASSERT string::len($value) >= 3 AND string::len($value) <= 50;
DEFINE FIELD full_name ON TABLE user TYPE option<string>;
DEFINE FIELD password_hash ON TABLE user TYPE string;
DEFINE FIELD role ON TABLE user TYPE string DEFAULT 'member'
  ASSERT $value INSIDE ['admin', 'member', 'viewer'];
DEFINE FIELD status ON TABLE user TYPE string DEFAULT 'active'
  ASSERT $value INSIDE ['active', 'suspended', 'inactive'];
DEFINE FIELD language_preference ON TABLE user TYPE string DEFAULT 'hr'
  ASSERT $value INSIDE ['hr', 'en', 'multilingual'];
DEFINE FIELD settings ON TABLE user TYPE object DEFAULT {};
DEFINE FIELD last_login_at ON TABLE user TYPE option<datetime>;
DEFINE FIELD created_at ON TABLE user TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON TABLE user TYPE datetime DEFAULT time::now();

-- User-Tenant relationship indexes
DEFINE INDEX user_email_idx ON TABLE user COLUMNS email UNIQUE;
DEFINE INDEX user_username_tenant_idx ON TABLE user COLUMNS [username, tenant_id] UNIQUE;
DEFINE INDEX user_tenant_idx ON TABLE user COLUMNS tenant_id;
DEFINE INDEX user_status_idx ON TABLE user COLUMNS status;

-- ===============================================
-- DOCUMENT MANAGEMENT
-- ===============================================

DEFINE TABLE document SCHEMAFULL;

DEFINE FIELD id ON TABLE document TYPE record<document>;
DEFINE FIELD tenant_id ON TABLE document TYPE record<tenant>;
DEFINE FIELD user_id ON TABLE document TYPE record<user>;
DEFINE FIELD title ON TABLE document TYPE string
  ASSERT string::len($value) > 0 AND string::len($value) <= 500;
DEFINE FIELD filename ON TABLE document TYPE string
  ASSERT string::len($value) > 0 AND string::len($value) <= 255;
DEFINE FIELD file_path ON TABLE document TYPE string;
DEFINE FIELD file_size ON TABLE document TYPE int DEFAULT 0
  ASSERT $value >= 0;
DEFINE FIELD file_type ON TABLE document TYPE string
  ASSERT $value INSIDE ['pdf', 'docx', 'txt', 'md', 'html'];
DEFINE FIELD language ON TABLE document TYPE string DEFAULT 'hr'
  ASSERT $value INSIDE ['hr', 'en', 'multilingual', 'auto'];
DEFINE FIELD scope ON TABLE document TYPE string DEFAULT 'user'
  ASSERT $value INSIDE ['user', 'tenant'];
DEFINE FIELD status ON TABLE document TYPE string DEFAULT 'processing'
  ASSERT $value INSIDE ['uploaded', 'processing', 'processed', 'failed', 'archived'];
DEFINE FIELD content_hash ON TABLE document TYPE option<string>;
DEFINE FIELD metadata ON TABLE document TYPE object DEFAULT {};
DEFINE FIELD categories ON TABLE document TYPE array<string> DEFAULT [];
DEFINE FIELD tags ON TABLE document TYPE array<string> DEFAULT [];
DEFINE FIELD chunk_count ON TABLE document TYPE int DEFAULT 0;
DEFINE FIELD processing_started_at ON TABLE document TYPE option<datetime>;
DEFINE FIELD processing_completed_at ON TABLE document TYPE option<datetime>;
DEFINE FIELD created_at ON TABLE document TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON TABLE document TYPE datetime DEFAULT time::now();

-- Document indexes for efficient querying
DEFINE INDEX document_tenant_idx ON TABLE document COLUMNS tenant_id;
DEFINE INDEX document_user_idx ON TABLE document COLUMNS user_id;
DEFINE INDEX document_scope_idx ON TABLE document COLUMNS [tenant_id, scope];
DEFINE INDEX document_language_idx ON TABLE document COLUMNS [tenant_id, language];
DEFINE FIELD document_status_idx ON TABLE document COLUMNS [tenant_id, status];
DEFINE INDEX document_categories_idx ON TABLE document COLUMNS categories;
DEFINE INDEX document_created_idx ON TABLE document COLUMNS created_at;
DEFINE INDEX document_hash_idx ON TABLE document COLUMNS content_hash;

-- ===============================================
-- DOCUMENT CHUNKS (Vector Storage Metadata)
-- ===============================================

DEFINE TABLE chunk SCHEMAFULL;

DEFINE FIELD id ON TABLE chunk TYPE record<chunk>;
DEFINE FIELD document_id ON TABLE chunk TYPE record<document>;
DEFINE FIELD tenant_id ON TABLE chunk TYPE record<tenant>;
DEFINE FIELD user_id ON TABLE chunk TYPE record<user>;
DEFINE FIELD scope ON TABLE chunk TYPE string DEFAULT 'user'
  ASSERT $value INSIDE ['user', 'tenant'];
DEFINE FIELD chunk_index ON TABLE chunk TYPE int
  ASSERT $value >= 0;
DEFINE FIELD content ON TABLE chunk TYPE string
  ASSERT string::len($value) > 0;
DEFINE FIELD content_length ON TABLE chunk TYPE int DEFAULT 0;
DEFINE FIELD language ON TABLE chunk TYPE string DEFAULT 'hr'
  ASSERT $value INSIDE ['hr', 'en', 'multilingual'];
DEFINE FIELD embedding_model ON TABLE chunk TYPE string DEFAULT 'bge-m3';
DEFINE FIELD vector_collection ON TABLE chunk TYPE string;
DEFINE FIELD vector_id ON TABLE chunk TYPE string;
DEFINE FIELD metadata ON TABLE chunk TYPE object DEFAULT {};
DEFINE FIELD categories ON TABLE chunk TYPE array<string> DEFAULT [];
DEFINE FIELD created_at ON TABLE chunk TYPE datetime DEFAULT time::now();

-- Chunk indexes for retrieval and management
DEFINE INDEX chunk_document_idx ON TABLE chunk COLUMNS document_id;
DEFINE INDEX chunk_tenant_scope_idx ON TABLE chunk COLUMNS [tenant_id, scope];
DEFINE INDEX chunk_collection_idx ON TABLE chunk COLUMNS vector_collection;
DEFINE INDEX chunk_categories_idx ON TABLE chunk COLUMNS categories;
DEFINE INDEX chunk_language_idx ON TABLE chunk COLUMNS [tenant_id, language];

-- ===============================================
-- SEARCH QUERIES AND ANALYTICS
-- ===============================================

DEFINE TABLE search_query SCHEMAFULL;

DEFINE FIELD id ON TABLE search_query TYPE record<search_query>;
DEFINE FIELD tenant_id ON TABLE search_query TYPE record<tenant>;
DEFINE FIELD user_id ON TABLE search_query TYPE record<user>;
DEFINE FIELD query_text ON TABLE search_query TYPE string
  ASSERT string::len($value) > 0 AND string::len($value) <= 1000;
DEFINE FIELD query_language ON TABLE search_query TYPE string DEFAULT 'hr'
  ASSERT $value INSIDE ['hr', 'en', 'multilingual', 'auto'];
DEFINE FIELD detected_language ON TABLE search_query TYPE option<string>;
DEFINE FIELD primary_category ON TABLE search_query TYPE option<string>;
DEFINE FIELD secondary_categories ON TABLE search_query TYPE array<string> DEFAULT [];
DEFINE FIELD retrieval_strategy ON TABLE search_query TYPE option<string>;
DEFINE FIELD scope_searched ON TABLE search_query TYPE array<string> DEFAULT ['user', 'tenant'];
DEFINE FIELD results_count ON TABLE search_query TYPE int DEFAULT 0;
DEFINE FIELD response_time_ms ON TABLE search_query TYPE int DEFAULT 0;
DEFINE FIELD satisfaction_rating ON TABLE search_query TYPE option<int>
  ASSERT $value >= 1 AND $value <= 5;
DEFINE FIELD metadata ON TABLE search_query TYPE object DEFAULT {};
DEFINE FIELD created_at ON TABLE search_query TYPE datetime DEFAULT time::now();

-- Search analytics indexes
DEFINE INDEX search_tenant_idx ON TABLE search_query COLUMNS tenant_id;
DEFINE INDEX search_user_idx ON TABLE search_query COLUMNS user_id;
DEFINE INDEX search_category_idx ON TABLE search_query COLUMNS primary_category;
DEFINE INDEX search_language_idx ON TABLE search_query COLUMNS [tenant_id, query_language];
DEFINE INDEX search_created_idx ON TABLE search_query COLUMNS created_at;

-- ===============================================
-- CATEGORIZATION TEMPLATES
-- ===============================================

DEFINE TABLE categorization_template SCHEMAFULL;

DEFINE FIELD id ON TABLE categorization_template TYPE record<categorization_template>;
DEFINE FIELD tenant_id ON TABLE categorization_template TYPE option<record<tenant>>;
DEFINE FIELD name ON TABLE categorization_template TYPE string
  ASSERT string::len($value) > 0 AND string::len($value) <= 100;
DEFINE FIELD category ON TABLE categorization_template TYPE string
  ASSERT $value INSIDE ['cultural', 'tourism', 'technical', 'legal', 'business', 'faq', 'educational', 'news', 'general'];
DEFINE FIELD language ON TABLE categorization_template TYPE string DEFAULT 'hr'
  ASSERT $value INSIDE ['hr', 'en', 'multilingual'];
DEFINE FIELD keywords ON TABLE categorization_template TYPE array<string> DEFAULT [];
DEFINE FIELD patterns ON TABLE categorization_template TYPE array<string> DEFAULT [];
DEFINE FIELD system_prompt ON TABLE categorization_template TYPE string;
DEFINE FIELD user_prompt_template ON TABLE categorization_template TYPE string;
DEFINE FIELD is_system_default ON TABLE categorization_template TYPE bool DEFAULT false;
DEFINE FIELD is_active ON TABLE categorization_template TYPE bool DEFAULT true;
DEFINE FIELD priority ON TABLE categorization_template TYPE int DEFAULT 0;
DEFINE FIELD created_at ON TABLE categorization_template TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON TABLE categorization_template TYPE datetime DEFAULT time::now();

-- Template indexes
DEFINE INDEX template_tenant_idx ON TABLE categorization_template COLUMNS tenant_id;
DEFINE INDEX template_category_idx ON TABLE categorization_template COLUMNS [category, language];
DEFINE INDEX template_system_idx ON TABLE categorization_template COLUMNS is_system_default;
DEFINE INDEX template_active_idx ON TABLE categorization_template COLUMNS is_active;

-- ===============================================
-- SYSTEM CONFIGURATION
-- ===============================================

DEFINE TABLE system_config SCHEMAFULL;

DEFINE FIELD id ON TABLE system_config TYPE record<system_config>;
DEFINE FIELD tenant_id ON TABLE system_config TYPE option<record<tenant>>;
DEFINE FIELD config_key ON TABLE system_config TYPE string
  ASSERT string::len($value) > 0 AND string::len($value) <= 100;
DEFINE FIELD config_value ON TABLE system_config TYPE string;
DEFINE FIELD config_type ON TABLE system_config TYPE string DEFAULT 'string'
  ASSERT $value INSIDE ['string', 'int', 'float', 'bool', 'json'];
DEFINE FIELD description ON TABLE system_config TYPE option<string>;
DEFINE FIELD is_system_config ON TABLE system_config TYPE bool DEFAULT false;
DEFINE FIELD created_at ON TABLE system_config TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON TABLE system_config TYPE datetime DEFAULT time::now();

-- Config indexes
DEFINE INDEX config_tenant_key_idx ON TABLE system_config COLUMNS [tenant_id, config_key] UNIQUE;
DEFINE INDEX config_system_idx ON TABLE system_config COLUMNS is_system_config;

-- ===============================================
-- INITIAL DATA SETUP
-- ===============================================

-- Create default development tenant
INSERT INTO tenant {
    id: tenant:development,
    name: 'Development Tenant',
    slug: 'development',
    description: 'Default tenant for development and testing',
    status: 'active',
    language_preference: 'hr',
    cultural_context: 'croatian_business',
    subscription_tier: 'enterprise',
    settings: {
        allow_user_document_promotion: true,
        auto_detect_language: true,
        enable_advanced_categorization: true,
        max_documents_per_user: 1000,
        max_total_documents: 10000
    }
};

-- Create default development user
INSERT INTO user {
    id: user:dev_user,
    tenant_id: tenant:development,
    email: 'dev@example.com',
    username: 'dev_user',
    full_name: 'Development User',
    password_hash: '$2b$12$dummy_hash_for_development',
    role: 'admin',
    status: 'active',
    language_preference: 'hr',
    settings: {
        preferred_categories: ['technical', 'business', 'cultural'],
        auto_categorize: true,
        search_both_scopes: true
    }
};

-- Insert system categorization templates for Croatian
INSERT INTO categorization_template {
    id: categorization_template:system_cultural_hr,
    tenant_id: NONE,
    name: 'Croatian Cultural Template',
    category: 'cultural',
    language: 'hr',
    keywords: ['kultura', 'tradicija', 'običaji', 'folklor', 'narod', 'baština', 'identitet'],
    patterns: ['kakva je.*kultura', 'tradicija.*', 'običaji.*', 'folklor.*'],
    system_prompt: 'Ti si stručnjak za hrvatsku kulturu i tradiciju. Odgovori na pitanja o hrvatskoj kulturi, tradicijama, običajima i identitetu.',
    user_prompt_template: 'Na temelju sljedećeg konteksta:\n\n{context}\n\nOdgovori na pitanje: {query}\n\nOdgovor daj na hrvatskom jeziku sa kulturnim kontekstom.',
    is_system_default: true,
    is_active: true,
    priority: 10
};

INSERT INTO categorization_template {
    id: categorization_template:system_technical_hr,
    tenant_id: NONE,
    name: 'Croatian Technical Template',
    category: 'technical',
    language: 'hr',
    keywords: ['tehnologija', 'programiranje', 'algoritam', 'kod', 'računalo', 'softver', 'API'],
    patterns: ['kako implementirati.*', 'što je.*algoritam', 'kako.*kod.*'],
    system_prompt: 'Ti si tehnički stručnjak. Odgovori na tehnička pitanja jasno i precizno.',
    user_prompt_template: 'Kontekst:\n\n{context}\n\nTehnično pitanje: {query}\n\nDaj jasan i precizan tehnički odgovor na hrvatskom jeziku.',
    is_system_default: true,
    is_active: true,
    priority: 10
};

-- Insert system configuration defaults
INSERT INTO system_config {
    id: system_config:default_embedding_model,
    tenant_id: NONE,
    config_key: 'default_embedding_model',
    config_value: 'BAAI/bge-m3',
    config_type: 'string',
    description: 'Default embedding model for all tenants',
    is_system_config: true
};

INSERT INTO system_config {
    id: system_config:default_chunk_size,
    tenant_id: NONE,
    config_key: 'default_chunk_size',
    config_value: '512',
    config_type: 'int',
    description: 'Default chunk size for document processing',
    is_system_config: true
};

INSERT INTO system_config {
    id: system_config:max_retrieval_results,
    tenant_id: NONE,
    config_key: 'max_retrieval_results',
    config_value: '10',
    config_type: 'int',
    description: 'Maximum number of results to retrieve per query',
    is_system_config: true
};

-- ===============================================
-- FUNCTIONS AND COMPUTED FIELDS
-- ===============================================

-- Function to get tenant collection name for ChromaDB
DEFINE FUNCTION fn::get_collection_name($tenant_id: string, $scope: string, $language: string) {
    RETURN string::concat($tenant_id, '_', $scope, '_', $language);
};

-- Function to check if user has access to document
DEFINE FUNCTION fn::user_can_access_document($user_id: record<user>, $document: record<document>) {
    LET $user_tenant = (SELECT tenant_id FROM $user_id)[0].tenant_id;
    LET $doc_tenant = $document.tenant_id;
    LET $doc_scope = $document.scope;
    LET $doc_owner = $document.user_id;

    RETURN (
        $user_tenant = $doc_tenant AND (
            $doc_scope = 'tenant' OR
            ($doc_scope = 'user' AND $user_id = $doc_owner)
        )
    );
};

-- ===============================================
-- EVENTS AND TRIGGERS
-- ===============================================

-- Update document updated_at on changes
DEFINE EVENT document_updated ON TABLE document WHEN $event = "UPDATE" THEN (
    UPDATE $after.id SET updated_at = time::now()
);

-- Update user updated_at on changes
DEFINE EVENT user_updated ON TABLE user WHEN $event = "UPDATE" THEN (
    UPDATE $after.id SET updated_at = time::now()
);

-- Update tenant updated_at on changes
DEFINE EVENT tenant_updated ON TABLE tenant WHEN $event = "UPDATE" THEN (
    UPDATE $after.id SET updated_at = time::now()
);

-- Auto-set chunk metadata when document is processed
DEFINE EVENT chunk_created ON TABLE chunk WHEN $event = "CREATE" THEN {
    LET $doc = (SELECT * FROM $after.document_id)[0];
    UPDATE $after.id SET
        tenant_id = $doc.tenant_id,
        user_id = $doc.user_id,
        scope = $doc.scope,
        language = $doc.language,
        categories = $doc.categories,
        vector_collection = fn::get_collection_name(
            <string>$doc.tenant_id,
            $doc.scope,
            $doc.language
        );
};
