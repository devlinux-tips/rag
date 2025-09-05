# Phase 1: Job-Centric Foundation

**Duration**: 4-6 weeks
**Status**: 🔄 Planning
**Progress**: 0% (0/6 milestones)
**Started**: TBD
**Target Completion**: TBD

## 🎯 Phase Objectives

Build job-orchestrated system around existing RAG (non-disruptive approach):
- Preserve existing Python RAG system completely
- Add Elixir orchestration layer with job queue
- Create basic React UI for RAG testing
- Implement foundational rate limiting and feature flags

## 📋 Milestones & Progress

### Milestone 1: Elixir API Foundation
**Status**: 🔲 Not Started
**Estimated**: 1 week
**Dependencies**: None

**Tasks**:
- [ ] Initialize Phoenix project with basic API structure
- [ ] Set up PostgreSQL database with Oban job queue
- [ ] Create basic HTTP client for Python RAG service
- [ ] Implement health checks and basic error handling

**Deliverables**:
- Working Phoenix API that can communicate with RAG service
- Database schema for jobs, rate limits, feature flags
- Basic integration tests

---

### Milestone 2: Job Orchestration System
**Status**: 🔲 Not Started
**Estimated**: 1.5 weeks
**Dependencies**: Milestone 1

**Tasks**:
- [ ] Implement Oban worker modules for document processing
- [ ] Create job queues for different operation types
- [ ] Add job retry logic with exponential backoff
- [ ] Implement job progress tracking and status updates

**Deliverables**:
- Document processing jobs with progress tracking
- Search query jobs for complex operations
- Embedding generation jobs with retry logic
- Job monitoring and failure handling

---

### Milestone 3: Rate Limiting System
**Status**: 🔲 Not Started
**Estimated**: 1 week
**Dependencies**: Milestone 1

**Tasks**:
- [ ] Implement API-level rate limiting (requests per minute)
- [ ] Add resource-level rate limiting (expensive ML operations)
- [ ] Create external API rate limiting (embedding services)
- [ ] Add rate limit monitoring and alerting

**Deliverables**:
- Multi-layer rate limiting protection
- Rate limit status API endpoints
- Configuration for different limit types
- Rate limit exceeded handling

---

### Milestone 4: Feature Flags System
**Status**: 🔲 Not Started
**Estimated**: 0.5 weeks
**Dependencies**: Milestone 1

**Tasks**:
- [ ] Implement database-backed feature flags
- [ ] Create feature flag evaluation logic
- [ ] Add runtime flag updates without restart
- [ ] Basic feature flag management API

**Deliverables**:
- Feature flag storage and evaluation
- API for managing flags
- Integration with job system
- Documentation for flag usage

---

### Milestone 5: Basic React UI
**Status**: 🔲 Not Started
**Estimated**: 1.5 weeks
**Dependencies**: Milestones 1-2

**Tasks**:
- [ ] Initialize React + Vite + TypeScript project
- [ ] Create search interface with Croatian/English input
- [ ] Implement document upload with drag-and-drop
- [ ] Add real-time job progress via WebSocket
- [ ] Basic error handling and loading states

**Deliverables**:
- Functional search interface for RAG testing
- Document upload with progress indication
- Real-time job status updates
- Responsive design for basic testing

---

### Milestone 6: Integration & Testing
**Status**: 🔲 Not Started
**Estimated**: 0.5 weeks
**Dependencies**: All previous milestones

**Tasks**:
- [ ] End-to-end integration testing
- [ ] Performance testing with rate limits
- [ ] Documentation updates
- [ ] Deployment preparation

**Deliverables**:
- Working end-to-end platform
- Integration test suite
- Updated documentation
- Ready for Phase 2

## 🚧 Risks & Mitigation

**Risk: Python-Elixir Integration Complexity**
- *Mitigation*: Start with simple HTTP communication, add complexity gradually

**Risk: Job Queue Performance**
- *Mitigation*: Start with conservative job limits, monitor and adjust

**Risk: UI Development Time**
- *Mitigation*: Focus on functionality over polish, iterate based on testing

## 📊 Success Criteria

- [ ] Can upload documents and see job progress in real-time
- [ ] Can search documents and get results through job system
- [ ] Rate limiting prevents system overload
- [ ] Feature flags work without restart
- [ ] All components integrate smoothly with existing RAG system

## 🔄 Next Phase Preview

**Phase 2** will focus on distributed coordination and advanced job orchestration, building on this foundation.
