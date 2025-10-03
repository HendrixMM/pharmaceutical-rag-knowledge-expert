# Week 2 Implementation Plan – Refined for Production Readiness

---

Last Updated: 2025-10-03
Owner: Documentation Team
Review Cadence: Weekly

---

<!-- TOC -->
- [Phase 0 – Governance & Readiness (0.5h)](#phase-0--governance--readiness-05h)
- [Phase 1 – Critical Cleanup (4h)](#phase-1--critical-cleanup-4h)
  - [1. README Refactor (2.5h)](#1-readme-refactor-25h)
  - [2. Git History Cleanup (1.5h)](#2-git-history-cleanup-15h)
- [Phase 2 – Documentation Excellence (3h)](#phase-2--documentation-excellence-3h)
  - [1. Standard Repository Files (1.5h)](#1-standard-repository-files-15h)
  - [2. Enhanced Documentation (1.5h)](#2-enhanced-documentation-15h)
- [Phase 3 – CI/CD Enhancement (2h)](#phase-3--cicd-enhancement-2h)
  - [1. Security Workflow (1h)](#1-security-workflow-1h)
  - [2. Automation Enhancements (1h)](#2-automation-enhancements-1h)
- [Phase 4 – Developer Experience (2h)](#phase-4--developer-experience-2h)
  - [1. Makefile Enhancements (0.5h)](#1-makefile-enhancements-05h)
  - [2. Developer Tooling (1.5h)](#2-developer-tooling-15h)
- [Quality Gates & Metrics](#quality-gates--metrics)
- [Risk Matrix Updates](#risk-matrix-updates)
- [Audit Artefacts Checklist](#audit-artefacts-checklist)
- [Edge Case Coverage Summary](#edge-case-coverage-summary)
- [Timeline (Revalidated)](#timeline-revalidated)
- [Success Criteria (Updated)](#success-criteria-updated)
<!-- /TOC -->

This plan refines the original "Ultrathink" outline to emphasise production-grade practices, auditability, and edge-case coverage. Activities remain grouped by phase but add gating checklists, deliverables, and validation steps.

---

## Phase 0 – Governance & Readiness (0.5h)

1. **Create Coordination Ticket**
   - Document scope, owners, required approvals, and planned force-push window.
   - Notify stakeholders in Slack/Teams + create calendar hold for history rewrite.
2. **Baseline Snapshot**
   - Tag `backup/week2-start` and export dependency manifests (`pip freeze > backups/pip-freeze-<date>.txt`).
   - Capture current CI status screenshot/log for audit trail.

> **Exit Criteria**: Alignment documented, backups stored in `/backups/`, risks acknowledged by stakeholders.

---

## Phase 1 – Critical Cleanup (4h)

### 1. README Refactor (2.5h)

- **Information Architecture**
  - Use card-sort to cluster content; define top-level nav identical to docs sidebar structure.
  - Create `docs/` sub-pages with front-matter and consistent headers. Ensure cross-links via relative paths.
- **Implementation Steps**
  1. Stand-up `docs/_shared/toc.md` capturing target hierarchy for quick linting.
  2. Refactor README to <250 lines featuring: value prop, architecture overview (with diagram thumbnail), quick start, quality gates, support.
  3. Move deep-dive content to:
     - `docs/API_REFERENCE.md`
     - `docs/EXAMPLES.md`
     - `docs/FEATURES.md`
     - `docs/DEVELOPMENT.md`
     - `docs/DEPLOYMENT.md` (consolidate duplicates)
  4. Run `markdownlint` & `doctoc` (or `markdown-toc`) to ensure linted headings and consistent TOC.
  5. Add regression checklist ensuring all README links resolve (use `linkcheck` via `make docs-linkcheck`).

> **Exit Criteria**: README passes markdownlint, <250 lines, linkcheck passes, all docs have ownership metadata section.

### 2. Git History Cleanup (1.5h)

- **Preparation**
  - Identify secret fingerprints via `git log -p -- .env` and store patterns in `scripts/sensitive-patterns.txt`.
  - Ensure `.env` already added to `.gitignore`; enforce with pre-commit `check-added-large-files` + `detect-secrets` hook.
- **Execution** (performed in fresh mirror clone)
  1. Run BFG with allowlist of safe files: `bfg --delete-files .env --no-blob-protection`.
  2. Use `bfg --replace-text scripts/sensitive-patterns.txt` for specific keys.
  3. After rewrite, run `git log --stat -- .env` to confirm absence.
  4. Force push during agreed window; immediately rotate compromised keys and update secret manager entry.
  5. Add `docs/security/history-redaction.md` summarising process & approvals for audit.

> **Exit Criteria**: Secrets rotated, mirror repo clean, approval log stored, detect-secrets enforced.

---

## Phase 2 – Documentation Excellence (3h)

### 1. Standard Repository Files (1.5h)

- Implement missing files with templates customised to project domain:
  - `CHANGELOG.md` following [Keep a Changelog] with release automation integration notes.
  - `CODE_OF_CONDUCT.md` using Contributor Covenant v2.1 plus contact alias.
  - `ARCHITECTURE.md` featuring C4 level 1 diagram + data flow (Mermaid). Include tracing of guardrail components.
  - `.github/ISSUE_TEMPLATE/` (bug, feature, security disclosure) with mandatory reproduction checklist.
  - `.github/PULL_REQUEST_TEMPLATE.md` including security, docs, and data-protection checkboxes.
- Validate templates with GitHub Markdown preview locally (`gh pr create --fill --draft` dry-run) to catch formatting issues.

### 2. Enhanced Documentation (1.5h)

- Add ADR log (`docs/adr/0001-use-nemo-retriever.md`) and template for future decisions.
- Provide API usage snippets in Python + cURL; cover edge cases (missing API key, rate limiting).
- Capture performance benchmarks with metadata: dataset, hardware, date, command used.
- Troubleshooting decision tree using Mermaid `graph TD` with links to logs/metrics.
- Verify docs build using `mkdocs build --strict` (prep for Week 3 website) even if site not yet published.

> **Exit Criteria**: Documentation lint passes, ADR log introduced, benchmark data reproducible with command references.

---

## Phase 3 – CI/CD Enhancement (2h)

### 1. Security Workflow (1h)

- Create `.github/workflows/security.yml` with:
  - Trigger: PRs to `main`, nightly schedule.
  - Jobs: `pre-commit`, `dependency-scan`, `secret-scan` (using `gitleaks`), and `sast` (Bandit for Python).
  - Cache pip dependencies using `actions/cache`.
  - Upload artefacts (pre-commit log, pip-audit report) for traceability.
- Add branch protection rule draft referencing workflow.

### 2. Automation Enhancements (1h)

- Configure `dependabot.yml` for `pip`, `github-actions`, `docker` ecosystems; weekly schedule, security updates daily.
- Implement `release-drafter.yml` (auto-changelog on PR merge) in `.github`.
- Add `.github/workflows/pr-labeler.yml` leveraging `actions/labeler` with labels defined in `.github/labeler.yml`.
- Introduce `.github/workflows/stale.yml` with SLA-aware auto-triage (ignore security label).

> **Exit Criteria**: Workflows validated via `act` or `gh workflow run --dry-run`; documentation in `docs/ci/README.md`.

---

## Phase 4 – Developer Experience (2h)

### 1. Makefile Enhancements (0.5h)

- Add phony targets: `docs`, `docs-serve`, `release`, `security`, `audit`, `lint`, `format`.
- Ensure each target delegates to scripts under `scripts/` to keep Makefile thin.
- Document usage in `docs/DEVELOPMENT.md` including environment prerequisites.

### 2. Developer Tooling (1.5h)

- Create `.devcontainer/devcontainer.json` using base image with Python 3.11, pre-commit, pip-audit, mkdocs.
- Provide `.vscode/settings.json` for linting/formatting, `.vscode/extensions.json` for recommended tooling.
- Implement `scripts/setup_dev.sh` (idempotent, shellcheck clean) and `scripts/health_check.py` (pytest-compatible, returns non-zero on failure).
- Add CI step to execute `scripts/health_check.py` weekly to detect drift.

> **Exit Criteria**: Dev container builds locally, setup script passes `shellcheck`, health check integrated into docs & CI roadmap.

---

## Quality Gates & Metrics

- Track completion using checklist in `PROJECT_SUMMARY.md` (add Week 2 section).
- Define KPIs:
  - README bounce rate proxy: measure via onboarding survey (tracked in `docs/process/onboarding-metrics.md`).
  - Secret exposure MTTR < 2h (document rotations).
  - CI success rate ≥ 95% post changes.
- Schedule post-week retrospective capturing lessons learned and backlog for Week 3.

---

## Risk Matrix Updates

| Risk                  | Impact | Likelihood | Mitigation                                      | Owner     |
| --------------------- | ------ | ---------- | ----------------------------------------------- | --------- |
| Force push conflicts  | High   | Medium     | Freeze merges, communicate, follow runbook      | Tech Lead |
| CI tooling flakiness  | Medium | Medium     | Use dependency caching, pin versions            | DevOps    |
| Documentation drift   | Medium | High       | Adopt docs ownership table, add automated lint  | Docs Lead |
| Secret scanning noise | Low    | Medium     | Tune gitleaks config, baseline allowed patterns | Security  |

---

## Audit Artefacts Checklist

- `backups/` contains tags, dependency snapshots, and secret-rotation notes.
- `docs/security/history-redaction.md` completed with timestamps.
- Workflow run artefacts stored for ≥90 days (configure retention in workflow yaml).
- All new docs include `Last Updated`, `Owner`, `Review Cadence` metadata.

---

## Edge Case Coverage Summary

- README link validation and CI ensures future docs restructure doesn’t break navigation.
- Git history redaction runbook includes handling forks (notify maintainers, share patch files).
- CI security workflow handles projects without `requirements.txt` by falling back to `pyproject.toml` via conditional steps.
- `scripts/health_check.py` validates cache directories and API key presence but skips secret values in CI (uses dry-run mode).

---

## Timeline (Revalidated)

| Day   | Focus                              | Key Deliverables                                            |
| ----- | ---------------------------------- | ----------------------------------------------------------- |
| Day 1 | Phase 0 + README + History cleanup | Refactored README + docs, secrets scrubbed, audit log       |
| Day 2 | Documentation excellence           | Standard repo files, ADR, benchmarks, troubleshooting guide |
| Day 3 | CI/CD enhancement                  | Security workflow, automation configs, documentation        |
| Day 4 | Developer experience               | Makefile targets, dev container, setup & health scripts     |

> If schedule slips, defer Phase 4 except `security` Makefile target to maintain critical coverage.

---

## Success Criteria (Updated)

- ✅ README and docs restructure approved via PR review checklist with doc owners sign-off.
- ✅ Verified history clean via `detect-secrets scan --baseline .secrets.baseline` returning no incidents.
- ✅ Security workflow passing on a sample PR; artefacts accessible.
- ✅ Dev environment reproducible (devcontainer) and `scripts/setup_dev.sh` executes without manual steps.
- ✅ Audit artefacts stored and referenced in `PROJECT_SUMMARY.md`.

---

Last Updated: 2025-10-03
Owner: Documentation Team
Review Cadence: Weekly

---
