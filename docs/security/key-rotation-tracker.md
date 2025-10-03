---
Last Updated: 2025-10-03
Owner: Security Team
Review Cadence: Monthly
---

# API Key Rotation Tracker

<!-- TOC -->

- [Purpose](#purpose)
- [Rotation History](#rotation-history)
  - [NVIDIA API Key](#nvidia-api-key)
  - [PubMed E-utilities API Key](#pubmed-e-utilities-api-key)
  - [Apify Token](#apify-token)
- [Rotation Procedures](#rotation-procedures)
  - [Standard Rotation Process](#standard-rotation-process)
  - [Emergency Rotation (Suspected Compromise)](#emergency-rotation-suspected-compromise)
- [Rotation Schedule](#rotation-schedule)
- [Audit Trail](#audit-trail)
- [Compliance Notes](#compliance-notes)
- [References](#references)
<!-- /TOC -->

## Purpose

Tracks all API key rotations performed as part of git history cleanup and ongoing security maintenance. Provides an audit trail for compliance.

## Rotation History

### NVIDIA API Key

| Rotation Date | Reason              | Old Key ID | New Key ID | Rotated By | Verified | Notes                            |
| ------------- | ------------------- | ---------- | ---------- | ---------- | -------- | -------------------------------- |
| YYYY-MM-DD    | Git history cleanup | nvapi-xxx… | nvapi-yyy… | [Name]     | ✅ Yes   | Exposed in commits abc123-def456 |

**Current Key Status:**

- Active Key ID: nvapi-yyy…
- Created: YYYY-MM-DD
- Last Verified: YYYY-MM-DD
- Expiration: YYYY-MM-DD (90 days)
- Next Rotation Due: YYYY-MM-DD

**Verification Steps Completed:**

- [ ] Old key revoked at https://build.nvidia.com
- [ ] New key tested with `scripts/nvidia_build_api_test.py`
- [ ] CI/CD secrets updated (GitHub Secrets)
- [ ] Team members notified
- [ ] `.env.example` updated with placeholder
- [ ] Usage logs reviewed for unauthorized access

---

### PubMed E-utilities API Key

| Rotation Date | Reason              | Old Key ID | New Key ID | Rotated By | Verified | Notes                            |
| ------------- | ------------------- | ---------- | ---------- | ---------- | -------- | -------------------------------- |
| YYYY-MM-DD    | Git history cleanup | [hash]     | [hash]     | [Name]     | ✅ Yes   | Exposed in commits abc123-def456 |

**Verification Steps Completed:**

- [ ] Old key deleted at https://www.ncbi.nlm.nih.gov/account/
- [ ] New key tested with `scripts/pubmed_eutils_test.py`
- [ ] CI/CD secrets updated
- [ ] Team members notified
- [ ] `.env.example` updated with placeholder
- [ ] Rate limits verified (10 req/s with key)

---

### Apify Token

| Rotation Date | Reason              | Old Token ID | New Token ID | Rotated By | Verified | Notes                            |
| ------------- | ------------------- | ------------ | ------------ | ---------- | -------- | -------------------------------- |
| YYYY-MM-DD    | Git history cleanup | [hash]       | [hash]       | [Name]     | ✅ Yes   | Exposed in commits abc123-def456 |

**Verification Steps Completed:**

- [ ] Old token revoked at https://console.apify.com
- [ ] New token tested
- [ ] CI/CD secrets updated
- [ ] Team members notified
- [ ] `.env.example` updated with placeholder
- [ ] Usage logs reviewed

---

## Rotation Procedures

### Standard Rotation Process

1. Pre-Rotation: document reason, schedule, notify, backup.
2. Rotation: generate new key, test, update secrets, verify.
3. Post-Rotation: revoke old key, update tracker, confirm revocation, monitor.
4. Verification: run tests, check logs, confirm team updates.

### Emergency Rotation (Suspected Compromise)

1. Immediate: revoke compromised key, generate new, update production, notify.
2. Investigation: review logs, scope, timeline, impact.
3. Remediation: follow standard process; document; update procedures.

---

## Rotation Schedule

| Key/Token      | Frequency | Next Due Date | Owner         |
| -------------- | --------- | ------------- | ------------- |
| NVIDIA API Key | 90 days   | YYYY-MM-DD    | Security Team |
| PubMed API Key | 90 days   | YYYY-MM-DD    | Security Team |
| Apify Token    | 90 days   | YYYY-MM-DD    | Security Team |

---

## Audit Trail

**Git History Cleanup (Phase 1.2)**

- Discovery: [DATE]
- Cleanup: [DATE]
- Keys Rotated: NVIDIA, PubMed, Apify
- Approver: [Name]
- Documentation: `docs/security/history-redaction.md`

**Timeline:**

- [DATE TIME]: Exposure discovered
- [DATE TIME]: Cleanup initiated
- [DATE TIME]: Backup created (backup/week2-start)
- [DATE TIME]: BFG execution completed
- [DATE TIME]: Force-push completed
- [DATE TIME]: Keys rotated
- [DATE TIME]: Verification completed

---

## Compliance Notes

- Retain records ≥ 7 years (adjust per policy).
- Access: Security and Compliance teams.
- Review: Quarterly.

---

## References

- [Security Policy](../../SECURITY.md)
- [Git History Cleanup Documentation](history-redaction.md)
- [NVIDIA API Key Management](https://build.nvidia.com)
- [PubMed API Key Management](https://www.ncbi.nlm.nih.gov/account/)
- [Apify Token Management](https://console.apify.com)

Note: Only store key IDs (hashes/prefixes) here, never full values.
