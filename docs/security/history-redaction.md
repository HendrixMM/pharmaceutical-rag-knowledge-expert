---
Last Updated: 2025-10-03
Owner: Security Team
Review Cadence: Monthly
Status: PENDING
---

# Git History Cleanup and Secret Redaction Runbook

<!-- TOC -->
- [Executive Summary](#executive-summary)
- [Incident Timeline](#incident-timeline)
- [Approval Workflow](#approval-workflow)
- [Exposed Secrets Inventory](#exposed-secrets-inventory)
- [Cleanup Process Runbook](#cleanup-process-runbook)
  - [Step 1: Preparation](#step-1-preparation)
  - [Step 2: Secret Identification](#step-2-secret-identification)
  - [Step 3: BFG Execution](#step-3-bfg-execution)
  - [Step 4: Force-Push](#step-4-force-push)
  - [Step 5: Key Rotation](#step-5-key-rotation)
  - [Step 6: Verification](#step-6-verification)
- [Verification Procedures](#verification-procedures)
- [Key Rotation Checklist](#key-rotation-checklist)
- [Team Coordination](#team-coordination)
- [Fork Handling](#fork-handling)
- [Lessons Learned](#lessons-learned)
- [Audit Artifacts](#audit-artifacts)
- [References](#references)
- [Appendix: Edge Cases](#appendix-edge-cases)
<!-- /TOC -->


This runbook documents the complete process to identify, remove, verify, and prevent secret exposure in git history. It provides an audit trail, roles and responsibilities, and step‑by‑step execution guidance.

## Executive Summary

- Incident: Historical commits contained exposed API credentials and emails (PII).
- Secret types: NVIDIA API keys (nvapi-*), PubMed E-utilities keys, Apify tokens, emails.
- Impact: Moderate to high; requires history rewrite and key rotation.
- Status: PENDING (working tree is clean; history cleanup to be performed per this runbook).

## Incident Timeline

- Discovery Date: [YYYY-MM-DD]
- Working Tree Cleanup: [YYYY-MM-DD]
- History Cleanup Initiated: [YYYY-MM-DD]
- Backup Created (backup/week2-start): [timestamp]
- BFG Execution: [timestamp]
- Force-Push Completed: [timestamp]
- Key Rotation Completed: [timestamp]
- Verification Completed: [timestamp]

## Approval Workflow

- Requested By: [Name, Role, Date]
- Approved By: [Tech Lead Name, Date, Signature]
- Stakeholders Notified: [List]
- Force-Push Window: [Date/Time, Duration]
- Communication Channels: [Slack/Teams/email]

## Exposed Secrets Inventory

| Secret Type         | First Exposed | Last Exposed | Commits Affected | Rotated | New Key ID        |
|---------------------|---------------|--------------|------------------|---------|-------------------|
| NVIDIA API Key      | YYYY-MM-DD    | YYYY-MM-DD   | N commits        | ☐/✅    | nvapi-xxx…        |
| PubMed API Key      | YYYY-MM-DD    | YYYY-MM-DD   | N commits        | ☐/✅    | [hash/prefix]     |
| Apify Token         | YYYY-MM-DD    | YYYY-MM-DD   | N commits        | ☐/✅    | [hash/prefix]     |

## Cleanup Process Runbook

### Step 1: Preparation
- Create coordination ticket and schedule force‑push window.
- Notify stakeholders about the merge freeze and required re‑clone.
- Create backups: tag `backup/week2-start`, `pip freeze` snapshot, branch backup.

### Step 2: Secret Identification
- Run history scanner:
  ```bash
  bash scripts/identify_secrets_in_history.sh --report backups/secret-scan-$(date +%Y%m%d).txt
  ```
- Review and sanitize `scripts/sensitive-patterns.txt` (no real secrets, only prefixes/fingerprints).

### Step 3: BFG Execution
- Dry-run:
  ```bash
  bash scripts/git_history_cleanup.sh --dry-run
  ```
- Execute cleanup:
  ```bash
  bash scripts/git_history_cleanup.sh
  ```
- Review logs and verify results.

### Step 4: Force-Push
- Coordinate with team; confirm freeze is active.
- Force-push cleaned history and tags per script output.
- Notify completion and re‑clone requirement.

### Step 5: Key Rotation
- Rotate NVIDIA, PubMed, Apify credentials.
- Update CI/CD secrets and `.env.example` placeholders.
- Track changes in `docs/security/key-rotation-tracker.md`.

### Step 6: Verification
- Run verification script:
  ```bash
  bash scripts/verify_history_cleanup.sh
  ```
- Ensure no `.env` in history, no key patterns, and detect‑secrets passes.

## Verification Procedures

- `git log --stat -- .env` → no results
- `git log -p --all | grep -i 'nvapi-'` → no results
- `git log -p --all | grep -i 'NVIDIA_API_KEY='` → no results
- `detect-secrets scan --baseline .secrets.baseline` → clean
- Review provider logs for unauthorized usage

## Key Rotation Checklist

- [ ] NVIDIA API key rotated (https://build.nvidia.com)
- [ ] PubMed API key rotated (https://www.ncbi.nlm.nih.gov/account/)
- [ ] Apify token rotated (https://console.apify.com)
- [ ] CI/CD secrets updated
- [ ] Team members notified
- [ ] `.env.example` refreshed with placeholders
- [ ] Old keys revoked; logs reviewed

## Team Coordination

Notification Template:
```
Subject: [ACTION REQUIRED] Git History Cleanup - Force Push Scheduled

Team,

We will be performing a git history cleanup to remove exposed API keys.

Force-Push Window: [Date/Time]
Duration: ~30 minutes
Impact: All team members must re-clone the repository after this window

Action Required:
1. Commit and push all work before [Date/Time]
2. Do not push during the force-push window
3. After completion, re-clone: git clone [repo-url]
4. Update your .env with new API keys (see updated .env.example)

Reason: Removing exposed API keys from git history for security compliance

Questions? Contact: [Security Team Contact]
```

## Fork Handling

- Identify known forks and notify maintainers.
- Provide patch instructions or assist with applying BFG to forks.

## Lessons Learned

- Root cause: [describe]
- Prevention: detect‑secrets hook, onboarding updates, regular audits.
- Process improvements: Automated scans in CI, rotation cadence.

## Audit Artifacts

- backup tag: `backup/week2-start`
- dependency snapshot: `backups/pip-freeze-<date>.txt`
- cleanup logs: `backups/cleanup-log-<date>.txt`
- verification report: `backups/verification-report-<date>.txt`
- secret scan report: `backups/secret-scan-report-<date>.txt`
- retention: ≥90 days

## References

- Security policy: ../SECURITY.md
- BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/
- detect‑secrets: https://github.com/Yelp/detect-secrets
- NVIDIA API key management: https://build.nvidia.com
- Incident response procedures: see SECURITY.md

## Appendix: Edge Cases

- Forks: coordinate separately; provide patch steps.
- CI failures: update secrets in CI before force‑push.
- Large repos: use `--no-blob-protection` with care; confirm with backups.
- Multiple env variants: scan `.env.local`, `.env.production`, etc.
