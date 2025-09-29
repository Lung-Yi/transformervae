
# Implementation Plan: TransformerVAE Modular Architecture Refactoring

**Branch**: `001-transformervae-1-config` | **Date**: 2025-09-26 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/home/lungyi/TransformerVAE/specs/001-transformervae-1-config/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Refactor existing TransformerVAE implementation into a highly modular, configuration-driven architecture that enables researchers to experiment with different model configurations, training parameters, and datasets without code modifications. The system will support MOSES and ZINC-15 datasets with configurable transformer encoder/decoder architectures, VAE sampling, and optional molecular property prediction heads.

## Technical Context
**Language/Version**: Python 3.12
**Primary Dependencies**: PyTorch 2.0+, RDKit, dataclasses, typing, YAML
**Storage**: File system for datasets (MOSES, ZINC-15), model checkpoints, configuration files
**Testing**: pytest for unit tests, integration tests for training workflows
**Target Platform**: Linux/macOS with CUDA support for GPU acceleration
**Project Type**: single - machine learning research library
**Performance Goals**: Maintain original TransformerVAE benchmarks (Valid, Novelty, FCD metrics)
**Constraints**: Memory efficient for large molecular datasets, reproducible experiments through seed management
**Scale/Scope**: Support molecular datasets up to millions of samples, configurable model architectures

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Configuration-Driven Architecture Check**:
- [x] All parameters externalized to YAML configuration files (see data-model.md, quickstart.md)
- [x] No hard-coded model or training parameters in source code (factory pattern enforces configuration)
- [x] Configuration schema documented and validated (comprehensive validation in contracts/configuration_api.py)

**Type Safety & Code Quality Check**:
- [x] All Python code uses dataclasses and type hints (LayerConfig, DetailedModelArchitecture, VAETrainingConfig)
- [x] Function signatures specify input/output types (see contracts for interface definitions)
- [x] Configuration parsing uses typed data structures (dataclass-based configuration system)

**Single Responsibility Design Check**:
- [x] Model components separated from training logic (models/ vs training/ modules)
- [x] Data preprocessing isolated from model architecture (data/ module separation)
- [x] Each module has clearly defined single responsibility (config/, models/, data/, training/, utils/)

**Research Reproducibility Check**:
- [x] Architecture maintains consistency with original TransformerVAE (documented in research.md)
- [x] Architectural changes documented with rationale (migration strategy in research.md)
- [x] Reproducible through configuration and seed management (comprehensive experiment tracking)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
transformervae/
├── config/
│   ├── __init__.py
│   ├── basic_config.py          # Main configuration dataclasses
│   ├── model_configs/           # Predefined model configurations
│   │   ├── __init__.py
│   │   ├── base_transformer.yaml
│   │   └── large_transformer.yaml
│   └── training_configs/        # Predefined training configurations
│       ├── __init__.py
│       ├── moses_config.yaml
│       └── zinc_config.yaml
├── models/
│   ├── __init__.py
│   ├── layer.py                 # Custom layer definitions
│   ├── model.py                 # TransformerVAE main model
│   └── utils.py                 # Model utility functions
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Dataset processing
│   └── tokenizer.py             # SMILES tokenization
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training logic
│   ├── evaluator.py             # Evaluation metrics
│   └── callbacks.py             # Training callbacks
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Molecular generation metrics
│   └── visualization.py         # Results visualization
└── __init__.py

tests/
├── contract/                    # Contract test validation
├── integration/                 # End-to-end training tests
├── unit/                       # Unit tests for components
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_layers.py
│   ├── test_training.py
│   └── test_data.py
└── fixtures/                   # Test data and configurations

main_train_VAE.py               # Main training script
requirements.txt
README.md
```

**Structure Decision**: Single project structure selected for machine learning research library. The transformervae/ package contains all core modules with clear separation of concerns: configuration management, model definitions, data processing, training workflows, and utilities. Tests are organized by type with comprehensive coverage of all components.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - research.md generated
- [x] Phase 1: Design complete (/plan command) - data-model.md, contracts/, quickstart.md, CLAUDE.md generated
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
