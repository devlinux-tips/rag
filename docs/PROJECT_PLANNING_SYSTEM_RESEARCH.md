# Project Planning System Research: Human-AI Collaboration

**Research Date**: September 7, 2025
**Research Scope**: Modern project planning approaches optimized for human-AI collaboration, focusing on shared visibility and understanding between human architect and AI development workflow.

## Executive Summary

This document presents research findings on evolving the current Obsidian-based project management to a modern system that optimizes for **shared understanding** between human architect and AI, rather than real-time collaboration. The research recommends a **Git-Native Planning System** as the optimal approach for maintaining synchronized project state and clear development direction.

## Current Planning System Analysis

### Existing Obsidian Vault Structure

**Current Location**: `/obsidian-vault/`

**Existing Structure**:
```
obsidian-vault/
├── 01-Discovery/          # Requirements gathering and research
├── 02-Planning/           # High-level planning and PRDs
├── 03-Implementation/     # Active development tracking
├── 04-Specifications/     # Detailed implementation specs
├── 05-Progress/           # Progress tracking and milestones
└── Templates/             # Standardized templates
    ├── PRD-Template.md
    ├── Phase-Planning-Template.md
    └── Implementation-Specification-Template.md
```

**Current Template System Strengths**:
- **PRD Template**: Complete Product Requirements Document with Croatian language specifics
- **Phase Planning Template**: Detailed phase breakdown with milestone tracking and dependencies
- **Implementation Specification Template**: AI-optimized specification format with natural language test scenarios
- **AI-Accessible**: Claude Code can read/write all documents in the vault
- **Structured Approach**: Clear separation between discovery, planning, implementation, and progress

### Identified Limitations of Current System

**Human Architect Perspective**:
- **Manual Updates Required**: Progress tracking requires manual file updates
- **No Automated Synchronization**: Implementation status not automatically reflected in planning docs
- **Limited Cross-Reference**: Difficult to link planning documents to actual code changes
- **Static Templates**: Templates don't adapt based on project evolution

**AI Development Workflow**:
- **No API Access**: Cannot programmatically update progress or status
- **File-based Only**: Limited to reading/writing individual files
- **No Real-time State**: Cannot maintain persistent understanding of project state
- **Manual Coordination**: Requires human to update planning docs after AI completes work

**Shared Visibility Challenges**:
- **Sync Issues**: Human planning and AI implementation can become out of sync
- **Status Tracking**: Difficult to maintain current view of what's done vs what's next
- **Decision History**: Architectural decisions scattered across multiple documents
- **Progress Visibility**: No single source of truth for overall project status

## Research: Modern Planning Approaches

### Planning System Categories Analyzed

#### **1. Code-Native Documentation Systems**

**Dendron/Foam**:
- **Pros**: VSCode-based, graph connections, Git-backed
- **Cons**: Still requires manual updates, no automation layer

**Logseq**:
- **Pros**: Block-based structure, Git-backed, API available
- **Cons**: Complex for simple planning needs, learning curve

**Enhanced Obsidian with Automation**:
- **Pros**: Familiar interface, rich plugin ecosystem
- **Cons**: Limited API access, manual synchronization remains

#### **2. Developer-First Project Management**

**Linear**:
- **Pros**: Excellent API, keyboard-first design, GitHub integration
- **Cons**: Not local, external dependency, overkill for single developer + AI

**Plane (Open Source Linear Alternative)**:
- **Pros**: Self-hostable, API-first, modern interface
- **Cons**: Complex setup, more than needed for planning requirements

**GitHub Projects V2**:
- **Pros**: Repository-native, API access, automatic issue linking
- **Cons**: Web-based interface, less suitable for detailed specifications

#### **3. Living Documentation Systems**

**Notion with Automation**:
- **Pros**: Database-driven, excellent API, rich formatting
- **Cons**: Not local, requires internet, complex for straightforward needs

**Anytype**:
- **Pros**: Local-first, encrypted, block-based like Notion
- **Cons**: Still in development, limited automation capabilities

#### **4. Git-Native Approaches**

**Custom Git Workflow with Structured Markdown**:
- **Pros**: Version controlled, AI-accessible, simple automation
- **Cons**: Requires custom tooling, less visual than specialized tools

**GitLab/GitHub Issues + Boards**:
- **Pros**: Native repository integration, API access
- **Cons**: Issue-focused rather than document-focused planning

## Human-AI Collaboration Requirements Analysis

### Core Requirement: Shared Understanding

**Not Real-time Collaboration**: The goal is not simultaneous editing or real-time updates, but ensuring both human architect and AI have consistent, current understanding of:

- **Project Status**: What phase/milestone is currently active
- **Next Priorities**: What should be worked on next
- **Completed Work**: What has been finished and validated
- **Architectural Decisions**: Key technical choices and their rationale
- **Blockers and Dependencies**: What's preventing progress or requires attention

### Human Architect Needs

**Clear Visibility Into**:
- Current implementation progress vs planned milestones
- What AI has completed and what remains
- Architectural decisions made during implementation
- Any blockers or issues discovered during development

**Easy Navigation and Updates**:
- Familiar editing environment (Markdown, text editor, IDE)
- Quick updates to plans and priorities
- Historical view of how plans have evolved
- Search across all planning documentation

### AI Development Workflow Needs

**Context Awareness**:
- Current project phase and active milestones
- Previously completed work to avoid duplication
- Architectural constraints and decisions
- Coding standards and patterns established

**Progress Communication**:
- Ability to update status when tasks are completed
- Document discoveries and architectural decisions
- Flag blockers or issues for human attention
- Maintain accurate view of implementation status

### Shared State Management

**Single Source of Truth**:
- One location where both human and AI check current status
- Consistent view of project priorities and progress
- Historical record of decisions and changes
- Clear handoff points between human planning and AI execution

## Recommended Solution: Git-Native Planning System

### Architecture Overview

**Migration Strategy**: Evolve current Obsidian structure into Git-native system that maintains benefits while adding automation capabilities.

```
/planning/                     # Replaces obsidian-vault/
├── current-phase.md          # Single source of truth for current status
├── next-steps.md            # Clear priorities for upcoming work
├── phases/                   # Structured phase documentation
│   ├── phase-1-foundation.md
│   ├── phase-2-multi-tenant.md
│   └── phase-3-optimization.md
├── features/                 # Feature specifications
│   ├── user-authentication.md
│   ├── prompt-customization.md
│   └── real-time-updates.md
├── architecture/            # Technical decisions and ADRs
│   ├── database-selection.md
│   ├── phoenix-integration.md
│   └── api-design-patterns.md
├── progress/               # Automated progress tracking
│   ├── milestone-status.md
│   ├── weekly-updates/
│   └── completion-tracking.md
└── automation/            # GitHub Actions for maintenance
    ├── sync-status.yml
    └── update-progress.yml
```

### Structured Document Format

**YAML Frontmatter + Markdown Content**:
```markdown
---
phase: "2"
status: "in_progress"
priority: "high"
dependencies: ["phase-1-foundation"]
estimated_duration: "3 weeks"
last_updated: "2025-09-07"
updated_by: "ai"
completion_percentage: 65
blockers: []
---

# Phase 2: Multi-tenant User Management

## Current Status
✅ SurrealDB integration complete
✅ User authentication endpoints implemented
🔄 Multi-tenant namespace configuration in progress
⏳ Prompt template CRUD operations pending
⏳ User settings management pending

## Next Steps
1. **Priority 1**: Configure SurrealDB namespace routing
2. **Priority 2**: Implement user settings persistence
3. **Priority 3**: Add prompt template inheritance system

## Implementation Notes
SurrealDB namespaces provide natural multi-tenancy isolation:
```sql
USE NS tenant_acme DB rag_platform;
CREATE users SET email = "user@acme.com";
```

## Architectural Decisions Made
- **Decision**: Use SurrealDB native namespaces for tenant isolation
- **Rationale**: Simpler than application-level multi-tenancy
- **Impact**: Affects database connection management in Phoenix contexts

## Blockers/Issues Discovered
None currently identified.

## Testing Requirements
- [ ] Namespace isolation verification
- [ ] User data separation testing
- [ ] Performance testing with multiple tenants
```

### Key Features for Shared Understanding

#### **1. Current Status Dashboard (`current-phase.md`)**
```markdown
---
last_updated: "2025-09-07T10:30:00Z"
updated_by: "ai"
phase: "2"
milestone: "multi-tenant-setup"
---

# Current Project Status

## Active Work
**Phase 2: Multi-tenant User Management** (65% complete)

## What's Done This Week
- ✅ SurrealDB integration with Phoenix contexts
- ✅ User authentication endpoint implementation
- ✅ Basic namespace configuration

## What's Next (Priority Order)
1. **Multi-tenant document storage** (AI to implement)
2. **User settings persistence** (AI to implement)
3. **Prompt template customization UI** (Human to design)

## Waiting For
- Human review of authentication flow
- Design approval for settings interface

## Recent Architectural Decisions
- Using SurrealDB namespaces for tenant isolation (2025-09-07)
- Phoenix umbrella structure confirmed (2025-09-06)
```

#### **2. Next Steps Tracking (`next-steps.md`)**
```markdown
---
prioritized_by: "human"
last_reviewed: "2025-09-07"
---

# Next Development Priorities

## Immediate (This Week)
1. **Complete multi-tenant namespace routing**
   - **Owner**: AI
   - **Estimate**: 1-2 days
   - **Context**: Building on existing SurrealDB integration
   - **Acceptance**: User documents properly isolated by tenant

2. **User settings persistence**
   - **Owner**: AI
   - **Estimate**: 1 day
   - **Dependencies**: Multi-tenant routing complete
   - **Acceptance**: Users can save/retrieve custom prompt settings

## Next Week
3. **Prompt template UI design**
   - **Owner**: Human
   - **Estimate**: 2-3 days
   - **Context**: Need to design before AI implementation
   - **Deliverable**: UI mockups and component specifications

## Future (Prioritized Backlog)
4. Real-time job progress updates
5. Advanced prompt template inheritance
6. Analytics and usage reporting
```

#### **3. Automated Progress Updates**

**GitHub Actions Integration** (`automation/sync-status.yml`):
```yaml
name: Sync Project Status
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 9 * * 1'  # Monday morning updates

jobs:
  update-status:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Update completion percentages
        run: |
          # Parse commit messages for completed tasks
          # Update milestone completion tracking
          # Generate progress summary

      - name: Update current-phase.md
        run: |
          # Extract current implementation status
          # Update next steps based on recent work
          # Flag any blockers found in commits

      - name: Commit updates
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add planning/
          git commit -m "Auto-update: sync project status [skip ci]" || exit 0
          git push
```

### Human-AI Workflow Examples

#### **Scenario 1: Starting New Development Work**

**AI Process**:
1. Read `current-phase.md` to understand project status
2. Check `next-steps.md` for prioritized work
3. Review relevant feature specifications in `features/`
4. Begin implementation with full context

**Human Process**:
1. Review `progress/milestone-status.md` to see recent progress
2. Update `next-steps.md` with new priorities if needed
3. Check `architecture/` for any decisions needed
4. Let AI proceed with clear direction

#### **Scenario 2: AI Completes Implementation**

**AI Updates**:
```markdown
# AI updates current-phase.md after completing user auth
---
last_updated: "2025-09-07T15:45:00Z"
updated_by: "ai"
---

## What's Done This Week
- ✅ SurrealDB integration with Phoenix contexts
- ✅ User authentication endpoint implementation
- ✅ Basic namespace configuration
- ✅ **NEW: Multi-tenant document storage complete**

## What's Next (Priority Order)
1. **User settings persistence** (AI to implement - moved up from #2)
2. **Prompt template customization UI** (Human to design)

## Architectural Decisions Made
- **NEW Decision**: Document storage uses pattern `{tenant_id}_{collection_name}`
- **Rationale**: Cleaner than namespace per tenant for document collections
- **Code Location**: `lib/rag_platform/documents.ex:45-67`
```

**Human Review Process**:
1. See updated status in `current-phase.md`
2. Review implementation in code and architectural decisions
3. Update priorities in `next-steps.md` if needed
4. Approve progression to next milestone

#### **Scenario 3: Architectural Decision Required**

**Human Planning**:
```markdown
# Human adds to architecture/api-design-patterns.md
---
status: "decision_needed"
decision_date: "2025-09-08"
decision_by: "human"
---

# API Design Pattern Decision

## Question
Should we use Phoenix JSON API format or custom response format for RAG search results?

## Options
1. **Phoenix JSON API Standard**: Consistent with Phoenix conventions
2. **Custom Format**: Optimized for RAG response structure

## Recommendation Needed From
Human architect - affects frontend development patterns

## Context for AI
Once decided, implement across all search endpoints in `lib/rag_platform_web/controllers/`
```

**AI Acknowledgment**:
- Reads decision requirement
- Continues with other work items
- Waits for architectural decision before implementing search endpoints

### Benefits of Git-Native Approach

#### **For Human Architect**

**Familiar Tools**:
- Edit planning documents in VSCode, GitHub web, or any text editor
- Use standard Git operations (diff, history, blame) to track changes
- Search across all project documentation with standard tools

**Clear Visibility**:
- Single `current-phase.md` always shows current status
- `next-steps.md` provides clear priorities for AI work
- Git history shows how plans have evolved over time

**Low Maintenance**:
- Automated status updates reduce manual tracking work
- YAML frontmatter provides structured data without complex interfaces
- Standard Markdown remains human-readable and editable

#### **For AI Development**

**Context Awareness**:
- Always read `current-phase.md` before starting work for current context
- `next-steps.md` provides clear priority order for task selection
- Feature specifications include implementation requirements and acceptance criteria

**Progress Communication**:
- Update YAML frontmatter fields (`status`, `completion_percentage`, `last_updated`)
- Add completed items to status sections
- Document architectural decisions made during implementation

**Automated Assistance**:
- GitHub Actions help maintain current status across documents
- Structured data enables automated progress tracking
- Integration with commit messages for automatic status updates

#### **Shared Benefits**

**Single Source of Truth**:
- Both human and AI reference same planning documents
- Version control provides authoritative history of all changes
- No synchronization issues between planning and implementation

**Clear Handoffs**:
- `next-steps.md` clearly indicates who owns each priority
- Status tracking shows what's ready for review vs what's in progress
- Blocked items are clearly flagged for human attention

**Historical Context**:
- Git history preserves evolution of project planning
- Architectural decisions documented with rationale and date
- Milestone progression tracked with completion percentages

### Migration from Current Obsidian System

#### **Phase 1: Structure Migration (1-2 days)**

**Convert Existing Templates**:
```bash
# Convert current obsidian-vault structure
cp -r obsidian-vault/ planning/

# Add YAML frontmatter to existing documents
# Convert template structure to git-native format
# Set up initial current-phase.md and next-steps.md
```

**Template Conversion Example**:
```markdown
<!-- Current PRD-Template.md becomes -->
---
type: "prd"
phase: "TBD"
status: "template"
template_version: "1.0"
---

# PRD Template

## 🎯 Objective
Brief description of what we're building and why.

## 👥 User Stories
**As a** [user type]
**I want** [functionality]
**So that** [business value]

[... rest of template content ...]
```

#### **Phase 2: Automation Setup (1 day)**

**GitHub Actions Configuration**:
```yaml
# Set up automated status synchronization
# Configure progress tracking from commit messages
# Enable weekly progress report generation
```

**Integration Testing**:
- Verify AI can read and update planning documents
- Test automated progress updates
- Validate human editing workflow

#### **Phase 3: Workflow Validation (1 day)**

**Human Workflow Test**:
- Update priorities in `next-steps.md`
- Review AI progress in `current-phase.md`
- Make architectural decision in `architecture/`

**AI Workflow Test**:
- Read current status and priorities
- Complete implementation task
- Update progress and document decisions

### Implementation Timeline

#### **Week 1: Migration and Setup**
- **Days 1-2**: Convert Obsidian structure to Git-native planning
- **Day 3**: Set up automation and GitHub Actions
- **Days 4-5**: Test workflows and refine based on usage

#### **Week 2: Optimization and Validation**
- **Days 1-2**: Use system for actual development work
- **Days 3-4**: Refine automation and document formats
- **Day 5**: Document final workflows and best practices

### Success Metrics

**Shared Understanding Indicators**:
- Both human and AI reference same status information
- Minimal time spent on "what should I work on next" questions
- Clear visibility into project progress and blockers
- Reduced coordination overhead between planning and implementation

**Workflow Efficiency Measures**:
- Time from planning update to AI implementation start
- Frequency of misaligned work (AI working on wrong priorities)
- Human time spent on status updates and progress tracking
- Documentation consistency across planning and implementation

## Conclusion

The **Git-Native Planning System** provides an optimal evolution from the current Obsidian-based approach that maintains all current strengths while adding crucial automation and shared state management capabilities.

**Key Advantages**:
- **Builds on Existing Success**: Preserves template-based approach and AI-accessible documentation
- **Reduces Coordination Overhead**: Clear status tracking and priority management
- **Maintains Simplicity**: Standard tools (Git, Markdown, text editors) without complex interfaces
- **Enables Automation**: Structured data supports automated progress tracking and status updates
- **Future-Proof**: Version controlled system that scales with project complexity

**Implementation Strategy**:
- **Low-Risk Migration**: Gradual conversion of existing content with immediate benefits
- **Familiar Tools**: Uses existing developer toolchain (Git, Markdown, GitHub)
- **Immediate Value**: Better shared understanding and status tracking from day one

This approach addresses the core need for **shared understanding between human architect and AI** while building on the solid foundation of the current planning system. The result is a modern, automated planning workflow that maintains clarity and reduces coordination overhead throughout the development process.
