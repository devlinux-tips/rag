# Phase 1: Job-Centric Foundation

**Duration**: 4-6 weeks
**Status**: 🔲 Planning
**Progress**: 0% (0/6 milestones)
**Started**: TBD
**Target Completion**: TBD

## 🎯 Phase Objectives

Build job-orchestrated system around existing RAG (non-disruptive approach):
- Preserve existing Python RAG system completely untouched
- Add Elixir orchestration layer with Oban job queue
- Create basic React UI for RAG testing and validation
- Implement foundational rate limiting and feature flags
- Establish PostgreSQL foundation for jobs and configuration

## 📋 Milestones & Progress

### Milestone 1: Elixir API Foundation
**Status**: 🔲 Not Started
**Estimated**: 1 week
**Dependencies**: None

**Tasks**:
- [ ] Initialize Phoenix project (`services/platform-api/`)
- [ ] Set up PostgreSQL database with Ecto
- [ ] Configure Oban job queue with basic queues
- [ ] Create HTTP client module for Python RAG service
- [ ] Implement health checks for both services
- [ ] Add basic error handling and logging

**Deliverables**:
- Working Phoenix API that can communicate with existing RAG service
- Database schema for jobs, rate limits, and feature flags
- Basic integration tests proving Elixir ↔ Python communication
- Health check endpoints for both services

---

### Milestone 2: Job Orchestration System
**Status**: 🔲 Not Started
**Estimated**: 1.5 weeks
**Dependencies**: Milestone 1

**Tasks**:
- [ ] Implement DocumentProcessorWorker (Oban job)
- [ ] Create SearchQueryWorker for complex queries
- [ ] Add EmbeddingGeneratorWorker with retry logic
- [ ] Implement job progress tracking and status updates
- [ ] Add job monitoring and failure handling
- [ ] Create job queue management API endpoints

**Deliverables**:
- Document processing jobs with real-time progress tracking
- Search query jobs for complex operations
- Embedding generation jobs with exponential backoff retry
- Job monitoring dashboard endpoints
- Failed job recovery and dead letter queue handling

---

### Milestone 3: Rate Limiting System
**Status**: 🔲 Not Started
**Estimated**: 1 week
**Dependencies**: Milestone 1

**Tasks**:
- [ ] Implement API-level rate limiting (requests per minute per user)
- [ ] Add resource-level rate limiting (expensive ML operations)
- [ ] Create external API rate limiting (embedding services, LLM calls)
- [ ] Add rate limit monitoring and alerting
- [ ] Implement rate limit configuration via database
- [ ] Create rate limit status API endpoints

**Deliverables**:
- Multi-layer rate limiting protection preventing system overload
- Rate limit status API endpoints for monitoring
- Database-driven rate limit configuration
- Rate limit exceeded graceful handling with user feedback
- Monitoring and alerting for rate limit breaches

---

### Milestone 4: Feature Flags System
**Status**: 🔲 Not Started
**Estimated**: 0.5 weeks
**Dependencies**: Milestone 1

**Tasks**:
- [ ] Implement database-backed feature flags schema
- [ ] Create feature flag evaluation logic with caching
- [ ] Add runtime flag updates without application restart
- [ ] Create basic feature flag management API
- [ ] Add feature flag integration with job system
- [ ] Implement flag rollout strategies (percentage, user-based)

**Deliverables**:
- Feature flag storage and efficient evaluation
- Management API for toggling flags without restart
- Integration with job system for gradual feature rollout
- Documentation and examples for flag usage
- Caching layer for high-performance flag evaluation

---

### Milestone 5: Basic React UI
**Status**: 🔲 Not Started
**Estimated**: 1.5 weeks
**Dependencies**: Milestones 1-2

**Tasks**:
- [ ] Initialize React + Vite + TypeScript project (`services/user-frontend/`)
- [ ] Create search interface with Croatian/English input support
- [ ] Implement document upload with drag-and-drop functionality
- [ ] Add real-time job progress tracking via WebSocket
- [ ] Create basic error handling and loading states
- [ ] Implement responsive design for testing on different devices

**Deliverables**:
- Functional search interface for comprehensive RAG testing
- Document upload with real-time progress indication
- WebSocket integration for live job status updates
- Responsive design working on desktop and mobile
- Error handling with user-friendly messages
- Basic authentication preparation (UI only)

---

### Milestone 6: Integration & Testing
**Status**: 🔲 Not Started
**Estimated**: 0.5 weeks
**Dependencies**: All previous milestones

**Tasks**:
- [ ] End-to-end integration testing across all services
- [ ] Performance testing with rate limits under load
- [ ] Croatian language processing validation
- [ ] Documentation updates for new architecture
- [ ] Deployment preparation and environment setup
- [ ] Create setup scripts for development environment

**Deliverables**:
- Working end-to-end multilingual RAG platform
- Comprehensive integration test suite
- Performance benchmarks with rate limiting
- Updated documentation reflecting new architecture
- Development environment setup automation
- Ready for Phase 2 distributed features

## 🚧 Risks & Mitigation

**Risk: Python-Elixir Integration Complexity**
- *Impact*: High (could block entire phase)
- *Probability*: Medium (HTTP integration is well-understood)
- *Mitigation*: Start with simple HTTP communication, add complexity gradually, create integration tests early

**Risk: Job Queue Performance Under Load**
- *Impact*: Medium (could affect user experience)
- *Probability*: Medium (Oban is battle-tested but our use case is complex)
- *Mitigation*: Start with conservative job limits, monitor queue performance, implement backpressure

**Risk: UI Development Time Expansion**
- *Impact*: Low (UI is for testing, not production-ready)
- *Probability*: High (UI development often takes longer than expected)
- *Mitigation*: Focus on functionality over polish, iterate based on testing feedback

**Risk: Croatian Language Feature Complexity**
- *Impact*: Medium (core differentiator for the platform)
- *Probability*: Low (existing RAG system handles this well)
- *Mitigation*: Preserve existing RAG system unchanged, add Croatian-specific tests

## 📊 Success Criteria

- [ ] Can upload Croatian documents and see job progress in real-time
- [ ] Can search documents in Croatian and English getting accurate results
- [ ] Rate limiting prevents system overload during stress testing
- [ ] Feature flags enable/disable functionality without restart
- [ ] All services integrate smoothly with existing RAG system unchanged
- [ ] Performance meets targets: API < 200ms, job progress < 100ms updates
- [ ] Croatian diacritics preserved throughout the entire pipeline
- [ ] System handles concurrent requests (target: 10 simultaneous)

## 🔗 Dependencies

**This Phase Depends On**:
- Existing Python RAG system (✅ already working)
- PostgreSQL installation and setup
- Ollama service running locally for LLM generation
- Basic development environment (Elixir, Node.js, Python)

**This Phase Blocks**:
- Phase 2: Distributed & Advanced Jobs
- Phase 3: Multi-Tenancy & Scale
- All future platform features and enhancements

## 📈 Progress Tracking

**Week 1**:
- [ ] Milestone 1: Elixir API Foundation
- Current blockers: None identified
- Next week priorities: Job orchestration system

**Week 2**:
- [ ] Milestone 2: Job Orchestration System
- Current blockers: TBD
- Next week priorities: TBD

**Week 3**:
- [ ] Milestone 3: Rate Limiting System
- [ ] Milestone 4: Feature Flags System
- Current blockers: TBD
- Next week priorities: TBD

**Week 4-5**:
- [ ] Milestone 5: Basic React UI
- Current blockers: TBD
- Next week priorities: TBD

**Week 6**:
- [ ] Milestone 6: Integration & Testing
- Current blockers: TBD
- Next week priorities: Phase 2 planning

## 🔄 Next Phase Preview

**Phase 2: Distributed & Advanced Jobs (6-8 weeks)** will focus on:
- Distributed rate limiting across multiple nodes
- Advanced job workflows with dependencies and batching
- Enhanced user interface with advanced search features
- Phoenix LiveView admin dashboard for system monitoring
- Job analytics and performance optimization
- External integrations and webhook notifications

This builds directly on the foundation established in Phase 1, adding sophisticated distributed coordination while maintaining the job-centric architecture.
