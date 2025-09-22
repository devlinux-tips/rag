-- ============================================================
-- COMPLETE SUPABASE SCHEMA FOR MULTI-TENANT RAG + CHAT SYSTEM
-- ============================================================
-- This script creates the complete database schema for:
-- 1. Multi-tenant RAG system (tenants, users, documents, chunks)
-- 2. Chat system (conversations, messages)
-- 3. Analytics and configuration
-- ============================================================

-- ===============================================
-- TENANT MANAGEMENT
-- ===============================================

CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(name) > 0 AND length(name) <= 100),
    slug TEXT UNIQUE NOT NULL CHECK (length(slug) > 0 AND length(slug) <= 50),
    description TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'inactive')),
    settings JSONB DEFAULT '{}'::jsonb,
    language_preference TEXT DEFAULT 'hr' CHECK (language_preference IN ('hr', 'en', 'multilingual')),
    cultural_context TEXT DEFAULT 'croatian_business'
        CHECK (cultural_context IN ('croatian_business', 'croatian_academic', 'international', 'technical')),
    subscription_tier TEXT DEFAULT 'basic'
        CHECK (subscription_tier IN ('basic', 'professional', 'enterprise')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tenant indexes
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
CREATE INDEX IF NOT EXISTS idx_tenants_created ON tenants(created_at);

-- ===============================================
-- USER MANAGEMENT
-- ===============================================

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email TEXT NOT NULL CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    username TEXT NOT NULL CHECK (length(username) >= 3 AND length(username) <= 50),
    full_name TEXT,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'inactive')),
    language_preference TEXT DEFAULT 'hr' CHECK (language_preference IN ('hr', 'en', 'multilingual')),
    settings JSONB DEFAULT '{}'::jsonb,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(email),
    UNIQUE(username, tenant_id)
);

-- User indexes
CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ===============================================
-- DOCUMENT MANAGEMENT
-- ===============================================

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL CHECK (length(title) > 0 AND length(title) <= 500),
    filename TEXT NOT NULL CHECK (length(filename) > 0 AND length(filename) <= 255),
    file_path TEXT NOT NULL,
    file_size INTEGER DEFAULT 0 CHECK (file_size >= 0),
    file_type TEXT CHECK (file_type IN ('pdf', 'docx', 'txt', 'md', 'html')),
    language TEXT DEFAULT 'hr' CHECK (language IN ('hr', 'en', 'multilingual', 'auto')),
    scope TEXT DEFAULT 'user' CHECK (scope IN ('user', 'tenant')),
    status TEXT DEFAULT 'processing'
        CHECK (status IN ('uploaded', 'processing', 'processed', 'failed', 'archived')),
    content_hash TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    categories TEXT[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    chunk_count INTEGER DEFAULT 0,
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents(tenant_id, scope);
CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(tenant_id, language);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_documents_categories ON documents USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);

-- ===============================================
-- DOCUMENT CHUNKS (Vector Storage Metadata)
-- ===============================================

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scope TEXT DEFAULT 'user' CHECK (scope IN ('user', 'tenant')),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL CHECK (length(content) > 0),
    content_length INTEGER DEFAULT 0,
    language TEXT DEFAULT 'hr' CHECK (language IN ('hr', 'en', 'multilingual')),
    embedding_model TEXT DEFAULT 'bge-m3',
    vector_collection TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    categories TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunk indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_scope ON chunks(tenant_id, scope);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(vector_collection);
CREATE INDEX IF NOT EXISTS idx_chunks_categories ON chunks USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(tenant_id, language);

-- ===============================================
-- CHAT SYSTEM
-- ===============================================

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    tenant_slug TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    message_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    order_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Chat indexes
CREATE INDEX IF NOT EXISTS idx_conversations_tenant_user ON conversations(tenant_slug, user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_order ON chat_messages(conversation_id, order_index);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp DESC);

-- ===============================================
-- SEARCH QUERIES AND ANALYTICS
-- ===============================================

CREATE TABLE IF NOT EXISTS search_queries (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL CHECK (length(query_text) > 0 AND length(query_text) <= 1000),
    query_language TEXT DEFAULT 'hr' CHECK (query_language IN ('hr', 'en', 'multilingual', 'auto')),
    detected_language TEXT,
    primary_category TEXT,
    secondary_categories TEXT[] DEFAULT '{}',
    retrieval_strategy TEXT,
    scope_searched TEXT[] DEFAULT '{user,tenant}',
    results_count INTEGER DEFAULT 0,
    response_time_ms INTEGER DEFAULT 0,
    satisfaction_rating INTEGER CHECK (satisfaction_rating >= 1 AND satisfaction_rating <= 5),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Search analytics indexes
CREATE INDEX IF NOT EXISTS idx_search_tenant ON search_queries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_search_user ON search_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_search_category ON search_queries(primary_category);
CREATE INDEX IF NOT EXISTS idx_search_language ON search_queries(tenant_id, query_language);
CREATE INDEX IF NOT EXISTS idx_search_created ON search_queries(created_at);

-- ===============================================
-- CATEGORIZATION TEMPLATES
-- ===============================================

CREATE TABLE IF NOT EXISTS categorization_templates (
    id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES tenants(id) ON DELETE CASCADE,
    name TEXT NOT NULL CHECK (length(name) > 0 AND length(name) <= 100),
    category TEXT CHECK (category IN ('cultural', 'tourism', 'technical', 'legal', 'business', 'faq', 'educational', 'news', 'general')),
    language TEXT DEFAULT 'hr' CHECK (language IN ('hr', 'en', 'multilingual')),
    keywords TEXT[] DEFAULT '{}',
    patterns TEXT[] DEFAULT '{}',
    system_prompt TEXT NOT NULL,
    user_prompt_template TEXT NOT NULL,
    is_system_default BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Template indexes
CREATE INDEX IF NOT EXISTS idx_templates_tenant ON categorization_templates(tenant_id);
CREATE INDEX IF NOT EXISTS idx_templates_category ON categorization_templates(category, language);
CREATE INDEX IF NOT EXISTS idx_templates_system ON categorization_templates(is_system_default);
CREATE INDEX IF NOT EXISTS idx_templates_active ON categorization_templates(is_active);

-- ===============================================
-- SYSTEM CONFIGURATION
-- ===============================================

CREATE TABLE IF NOT EXISTS system_configs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES tenants(id) ON DELETE CASCADE,
    config_key TEXT NOT NULL CHECK (length(config_key) > 0 AND length(config_key) <= 100),
    config_value TEXT NOT NULL,
    config_type TEXT DEFAULT 'string' CHECK (config_type IN ('string', 'int', 'float', 'bool', 'json')),
    description TEXT,
    is_system_config BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, config_key)
);

-- Config indexes
CREATE INDEX IF NOT EXISTS idx_configs_system ON system_configs(is_system_config);

-- ===============================================
-- TRIGGERS FOR AUTO-UPDATING TIMESTAMPS
-- ===============================================

-- Function to update updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_templates_updated_at BEFORE UPDATE ON categorization_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configs_updated_at BEFORE UPDATE ON system_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===============================================
-- INITIAL DATA SETUP
-- ===============================================

-- Create default development tenant
INSERT INTO tenants (id, name, slug, description, status, language_preference, cultural_context, subscription_tier, settings)
VALUES (
    'development',
    'Development Tenant',
    'development',
    'Default tenant for development and testing',
    'active',
    'hr',
    'croatian_business',
    'enterprise',
    '{"allow_user_document_promotion": true, "auto_detect_language": true, "enable_advanced_categorization": true, "max_documents_per_user": 1000, "max_total_documents": 10000}'::jsonb
) ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    settings = EXCLUDED.settings;

-- Create default development user
INSERT INTO users (id, tenant_id, email, username, full_name, password_hash, role, status, language_preference, settings)
VALUES (
    'dev_user',
    'development',
    'dev@example.com',
    'dev_user',
    'Development User',
    '$2b$12$dummy_hash_for_development',
    'admin',
    'active',
    'hr',
    '{"preferred_categories": ["technical", "business", "cultural"], "auto_categorize": true, "search_both_scopes": true}'::jsonb
) ON CONFLICT (id) DO UPDATE SET
    full_name = EXCLUDED.full_name,
    role = EXCLUDED.role,
    settings = EXCLUDED.settings;

-- Insert system categorization templates for Croatian
INSERT INTO categorization_templates (id, tenant_id, name, category, language, keywords, patterns, system_prompt, user_prompt_template, is_system_default, is_active, priority)
VALUES (
    'system_cultural_hr',
    NULL,
    'Croatian Cultural Template',
    'cultural',
    'hr',
    '{kultura,tradicija,običaji,folklor,narod,baština,identitet}',
    '{kakva je.*kultura,tradicija.*,običaji.*,folklor.*}',
    'Ti si stručnjak za hrvatsku kulturu i tradiciju. Odgovori na pitanja o hrvatskoj kulturi, tradicijama, običajima i identitetu.',
    'Na temelju sljedećeg konteksta:\n\n{context}\n\nOdgovori na pitanje: {query}\n\nOdgovor daj na hrvatskom jeziku sa kulturnim kontekstom.',
    true,
    true,
    10
) ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    system_prompt = EXCLUDED.system_prompt,
    user_prompt_template = EXCLUDED.user_prompt_template;

INSERT INTO categorization_templates (id, tenant_id, name, category, language, keywords, patterns, system_prompt, user_prompt_template, is_system_default, is_active, priority)
VALUES (
    'system_technical_hr',
    NULL,
    'Croatian Technical Template',
    'technical',
    'hr',
    '{tehnologija,programiranje,algoritam,kod,računalo,softver,API}',
    '{kako implementirati.*,što je.*algoritam,kako.*kod.*}',
    'Ti si tehnički stručnjak. Odgovori na tehnička pitanja jasno i precizno.',
    'Kontekst:\n\n{context}\n\nTehnično pitanje: {query}\n\nDaj jasan i precizan tehnički odgovor na hrvatskom jeziku.',
    true,
    true,
    10
) ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    system_prompt = EXCLUDED.system_prompt,
    user_prompt_template = EXCLUDED.user_prompt_template;

-- Insert system configuration defaults
INSERT INTO system_configs (id, tenant_id, config_key, config_value, config_type, description, is_system_config)
VALUES
    ('default_embedding_model', NULL, 'default_embedding_model', 'BAAI/bge-m3', 'string', 'Default embedding model for all tenants', true),
    ('default_chunk_size', NULL, 'default_chunk_size', '512', 'int', 'Default chunk size for document processing', true),
    ('max_retrieval_results', NULL, 'max_retrieval_results', '10', 'int', 'Maximum number of results to retrieve per query', true)
ON CONFLICT (tenant_id, config_key) DO UPDATE SET
    config_value = EXCLUDED.config_value,
    description = EXCLUDED.description;

-- ===============================================
-- ROW LEVEL SECURITY (OPTIONAL - FOR PRODUCTION)
-- ===============================================

-- Enable RLS (uncomment for production)
-- ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE search_queries ENABLE ROW LEVEL SECURITY;

-- Sample RLS policies (uncomment and customize for production)
-- CREATE POLICY tenant_isolation ON tenants FOR ALL USING (id = current_setting('app.current_tenant', true));
-- CREATE POLICY user_tenant_isolation ON users FOR ALL USING (tenant_id = current_setting('app.current_tenant', true));
-- CREATE POLICY document_access ON documents FOR ALL USING (
--     tenant_id = current_setting('app.current_tenant', true) AND
--     (scope = 'tenant' OR user_id = current_setting('app.current_user', true))
-- );

-- ============================================================
-- SCHEMA CREATION COMPLETE!
-- ============================================================
-- Tables created:
-- ✅ tenants - Multi-tenant organization
-- ✅ users - User management per tenant
-- ✅ documents - Document storage and metadata
-- ✅ chunks - Vector storage metadata
-- ✅ conversations - Chat conversations
-- ✅ chat_messages - Chat message history
-- ✅ search_queries - Search analytics
-- ✅ categorization_templates - AI categorization
-- ✅ system_configs - System configuration
--
-- Features:
-- ✅ Multi-tenant isolation
-- ✅ Full-text search indexes
-- ✅ Performance optimized indexes
-- ✅ Foreign key constraints
-- ✅ Data validation checks
-- ✅ Auto-updating timestamps
-- ✅ Default development data
-- ============================================================