# Updated Phase Plan: Multi-Tenant RAG System Evolution

## Architecture Overview

### Multi-Tenant Data Hierarchy
```
Platform
├── Tenant A (e.g., "AcmeCorp")
│   ├── Users (tenant_user_1, tenant_user_2, ...)
│   ├── Tenant Documents (shared across all tenant users)
│   └── User Documents (private to each user)
├── Tenant B (e.g., "BetaCorp")
│   ├── Users (tenant_user_1, tenant_user_2, ...)
│   ├── Tenant Documents (shared)
│   └── User Documents (private)
```

### Document Scoping Strategy
1. **User Scope**: Private documents only accessible to the uploading user
2. **Tenant Scope**: Shared documents accessible to all users within the tenant
3. **Search Behavior**: Users search both their private docs + tenant shared docs
4. **Future Capability**: Users can promote private docs to tenant scope

## Updated Phase Plan

### Phase 1A: Multi-Tenant Foundation (2-3 weeks)
**Goal**: Establish multi-tenant data architecture while maintaining current functionality

#### **1.1 Data Architecture Design (3-5 days)**
- **SurrealDB Schema Enhancement**
  - Tenant table with metadata and settings
  - User table with tenant relationship
  - Document table with tenant_id and user_id scoping
  - Chunk table with proper tenant/user inheritance
  - Access control and permission models

- **Document Storage Strategy**
  - ChromaDB collection naming: `{tenant_id}_{scope}_{language}`
    - Example: `acme_user_hr`, `acme_tenant_en`, `beta_user_hr`
  - Vector namespace separation for complete tenant isolation
  - Metadata embedding for scope and ownership tracking

#### **1.2 Updated RAG System Architecture (4-6 days)**
- **Enhanced Categorization System**
  - Tenant-aware document categorization
  - Scope-specific retrieval strategies
  - Cross-scope search with proper filtering

- **Multi-Scope Retrieval Logic**
  - User scope retrieval: `{tenant_id}_user_{user_id}_{language}`
  - Tenant scope retrieval: `{tenant_id}_tenant_{language}`
  - Combined retrieval with relevance weighting
  - Permission-based result filtering

- **Context Management**
  - Tenant context injection for cultural/business-specific responses
  - User context for personalized results
  - Scope-aware prompt templates

#### **1.3 API and Security Layer (3-4 days)**
- **Multi-Tenant API Design**
  - Tenant identification via subdomain/header/path
  - User authentication and tenant membership validation
  - Document upload with scope selection (user vs tenant)
  - Search API with automatic scope aggregation

- **Data Isolation Guarantees**
  - Tenant-level data encryption (future)
  - Complete vector space isolation between tenants
  - Audit logging for cross-tenant data access attempts

### Phase 1B: Single Tenant Implementation (1-2 weeks)
**Goal**: Working end-to-end system with single tenant, single user for validation

#### **1.1 Development Environment Setup**
- Default tenant: "development"
- Default user: "dev_user"
- All existing functionality preserved with new scoping

#### **1.2 Migration Strategy**
- Migrate existing documents to `development_user_hr` collection
- Update existing tests to work with new tenant-user model
- Maintain backward compatibility with current RAG interface

#### **1.3 Validation and Testing**
- End-to-end testing with tenant-user scoping
- Performance validation (should match current system)
- Croatian language quality preservation

### Phase 2: Multi-User Within Tenant (1-2 weeks)
**Goal**: Multiple users within single tenant, with user and tenant document scopes

#### **2.1 User Management**
- User registration and authentication within tenant
- Document ownership and privacy controls
- Tenant shared document management

#### **2.2 Multi-Scope Search Implementation**
- Combined search across user private + tenant shared docs
- Relevance scoring with scope awareness
- Result presentation with scope indicators

#### **2.3 Document Promotion Workflow**
- User ability to promote private docs to tenant scope
- Approval workflow (if needed)
- Versioning and change tracking

### Phase 3: Full Multi-Tenancy (2-3 weeks)
**Goal**: Multiple tenants with complete isolation and scaling

#### **3.1 Tenant Onboarding**
- Tenant registration and configuration
- Custom tenant settings (language preferences, cultural context)
- Tenant-specific categorization patterns

#### **3.2 Advanced Multi-Tenant Features**
- Tenant administration interface
- Usage analytics per tenant
- Tenant-specific model fine-tuning (future)

#### **3.3 Performance and Scaling**
- Tenant-aware caching strategies
- Resource allocation per tenant
- Multi-tenant monitoring and alerting

## Data Model Changes

### Current Model Issues
The current system assumes single-user, single-language collections. We need:

1. **Collection Naming Strategy**
   ```python
   # Current: "croatian_documents", "english_documents"
   # New: "{tenant_id}_{scope}_{language}"

   # Examples:
   "acme_user_001_hr"      # User 001's Croatian docs in Acme tenant
   "acme_tenant_hr"        # Acme tenant shared Croatian docs
   "beta_user_005_en"      # User 005's English docs in Beta tenant
   ```

2. **Search Strategy**
   ```python
   async def search_user_and_tenant_docs(
       query: str,
       tenant_id: str,
       user_id: str,
       language: str
   ) -> CombinedResults:
       # Search user private documents
       user_results = await search_collection(f"{tenant_id}_user_{user_id}_{language}")

       # Search tenant shared documents
       tenant_results = await search_collection(f"{tenant_id}_tenant_{language}")

       # Combine and rerank with scope awareness
       return combine_and_rerank(user_results, tenant_results)
   ```

3. **Configuration Inheritance**
   ```toml
   [tenant.acme]
   name = "Acme Corporation"
   language_preference = "hr"
   cultural_context = "croatian_business"

   [tenant.acme.categorization]
   business_keywords = ["startup", "poslovanje", "tvrtka"]
   cultural_keywords = ["kultura", "tradicija", "običaji"]
   ```

## Implementation Strategy

### Option 1: Evolutionary Approach (Recommended)
- Keep current system working
- Add tenant-user layer gradually
- Migrate existing data with default tenant/user
- Lower risk, continuous functionality

### Option 2: Ground-Up Rebuild
- Start fresh with multi-tenant architecture
- Faster implementation of clean design
- Risk of regression and delayed delivery

**Recommendation**: Option 1 (Evolutionary) - maintain current functionality while adding multi-tenant capabilities.

## Success Criteria

### Phase 1A: Multi-Tenant Foundation
- [ ] SurrealDB schema supports tenant-user hierarchy
- [ ] ChromaDB collections properly namespaced by tenant/scope/language
- [ ] RAG system can handle tenant-user scoped queries
- [ ] All existing functionality preserved
- [ ] Croatian language quality maintained

### Phase 1B: Single Tenant Validation
- [ ] End-to-end system working with default tenant/user
- [ ] Document upload and search working with scoping
- [ ] Performance matches current system
- [ ] Comprehensive testing coverage

### Phase 2: Multi-User Within Tenant
- [ ] Multiple users can upload documents with privacy
- [ ] Users can search both private and tenant shared documents
- [ ] Document promotion workflow functional
- [ ] Proper access control enforcement

### Phase 3: Full Multi-Tenancy
- [ ] Complete tenant isolation verified
- [ ] Tenant onboarding and management working
- [ ] Performance scaling with multiple tenants
- [ ] Production-ready security and monitoring

## Risk Assessment

### High Risk
- **Data Migration Complexity**: Moving from single-user to multi-tenant collections
- **Performance Impact**: Multiple collection searches per query
- **Security Leaks**: Accidental cross-tenant data access

### Medium Risk
- **Configuration Complexity**: Tenant-specific settings management
- **Search Relevance**: Maintaining quality with multi-scope search
- **Development Timeline**: Significant architectural changes

### Mitigation Strategies
- **Incremental Implementation**: Maintain working system throughout transition
- **Comprehensive Testing**: Tenant isolation and data access verification
- **Performance Monitoring**: Benchmark at each phase to prevent regression
- **Rollback Planning**: Ability to revert to current system if needed

## Next Immediate Steps

1. **Design SurrealDB multi-tenant schema** (1-2 days)
2. **Plan ChromaDB collection migration strategy** (1 day)
3. **Update categorization system for tenant-user awareness** (2-3 days)
4. **Create development environment with default tenant/user** (1 day)
5. **Implement and test basic tenant-user scoped search** (3-4 days)

This plan balances the need for multi-tenant architecture with maintaining current system functionality and Croatian language quality.
