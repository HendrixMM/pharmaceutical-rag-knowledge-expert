# Security Policy

## 🔒 Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability, please follow responsible disclosure practices.

### **DO NOT** Create Public Issues for Security Vulnerabilities

Public disclosure of security vulnerabilities puts all users at risk. Instead:

### How to Report

1. **Email security contact**: Send details to `security@yourorg.com`

2. **Include in your report**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)
   - Your contact information

3. **Expected response time**:
   - **Initial acknowledgment**: Within 24-48 hours
   - **Status update**: Within 7 days
   - **Fix timeline**: Depends on severity (see below)

### Severity Levels

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **Critical** | Immediate exploitation, data breach risk | 24-48 hours | Remote code execution, API key exposure |
| **High** | Significant impact, requires user action | 1 week | Authentication bypass, privilege escalation |
| **Medium** | Limited impact, specific conditions required | 2-4 weeks | XSS, CSRF with mitigation |
| **Low** | Minimal impact, theoretical risk | Next release | Information disclosure, minor DoS |

## 🛡️ Security Best Practices

### For Users

#### API Key Management

1. **Never commit API keys** to git
   - Use `.env` for local keys (gitignored)
   - Use `.env.example` for templates (tracked, placeholders only)
   - Use CI secrets for automation

2. **Rotate keys regularly**
   - NVIDIA API keys: Every 90 days
   - After any suspected exposure: Immediately

3. **Use environment variables**
   ```bash
   # Good
   export NVIDIA_API_KEY="nvapi-..."

   # Bad - don't hardcode
   api_key = "nvapi-..."
   ```

4. **Validate configuration**
   ```bash
   python scripts/validate_env.py
   ```

#### Network Security

1. **Use HTTPS only**
   - All NVIDIA API calls use HTTPS
   - Verify SSL certificates (default: enabled)

2. **Rate limiting**
   - Respect API rate limits
   - Implement backoff strategies
   - Monitor usage patterns

3. **Data handling**
   - Don't log sensitive data
   - Sanitize user inputs
   - Encrypt data at rest (production)

### For Contributors

#### Pre-commit Security Hooks

We use automated security scanning:

```bash
# Install hooks (required)
pre-commit install

# Manual scan
pre-commit run --all-files
```

**Blocked patterns**:
- API keys (nvapi-, Bearer tokens)
- Hardcoded credentials
- Environment variable assignments
- URL parameters with secrets

#### Code Security

1. **Input validation**
   ```python
   # Good
   def process_query(query: str) -> str:
       if not query or len(query) > 1000:
           raise ValueError("Invalid query")
       sanitized = sanitize_input(query)
       return process(sanitized)

   # Bad - no validation
   def process_query(query):
       return eval(query)  # NEVER do this
   ```

2. **Dependency scanning**
   ```bash
   # Check for known vulnerabilities
   pip-audit
   safety check
   ```

3. **Type safety**
   ```bash
   # Run type checker
   mypy src/
   ```

## 🔐 Authentication & Authorization

### NVIDIA API Key Security

1. **Storage**:
   - Environment variables (preferred)
   - Encrypted secrets management (production)
   - Never in source code

2. **Transmission**:
   - HTTPS only
   - Authorization headers (not URL params)

3. **Validation**:
   ```python
   from scripts.config_validator import validate_nvidia_api_key

   result = validate_nvidia_api_key(os.getenv('NVIDIA_API_KEY'))
   if not result.valid:
       raise ValueError(f"Invalid API key: {result.errors}")
   ```

### PubMed E-utilities

1. **API key** (optional but recommended):
   - Increases rate limits
   - Better performance
   - Tracked usage

2. **Rate limiting**:
   - Without key: 3 requests/second
   - With key: 10 requests/second
   - Respect NCBI guidelines

## 🚨 Security Incident Response

### If You Detect a Security Issue

1. **Stop the bleeding**:
   - Disable affected features
   - Rotate compromised credentials
   - Isolate affected systems

2. **Assess impact**:
   - What data was exposed?
   - How many users affected?
   - Duration of exposure?

3. **Report immediately**:
   - Email: security@yourorg.com
   - Include timeline, impact, actions taken

4. **Document**:
   - Incident timeline
   - Root cause analysis
   - Lessons learned

### If Your Keys Are Exposed

#### Immediate Actions (Within 1 Hour)

1. **Rotate NVIDIA API key**:
   - Visit https://build.nvidia.com
   - Go to API Keys → Revoke
   - Generate new key
   - Update your `.env`

2. **Rotate PubMed API key** (if applicable):
   - Visit https://www.ncbi.nlm.nih.gov/account/
   - Manage API keys → Delete
   - Create new key

3. **Check for unauthorized usage**:
   - Review API usage logs
   - Monitor for anomalies
   - Report suspicious activity

#### Follow-up Actions (Within 24 Hours)

4. **Clean git history** (if committed):
   ```bash
   # Use BFG Repo-Cleaner (safer than filter-branch)
   bfg --replace-text passwords.txt
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   git push --force
   ```

5. **Notify team**:
   - Alert collaborators
   - Update documentation
   - Review access controls

6. **Root cause analysis**:
   - Why did this happen?
   - How to prevent recurrence?
   - Update processes

## 🔍 Security Auditing

### Automated Scans

We run automated security scans:

- **Pre-commit hooks**: On every commit
- **CI/CD pipeline**: On every PR
- **Dependency scanning**: Weekly
- **Full security audit**: Quarterly

### Manual Review

Security-sensitive changes require:

1. **Code review** by maintainer
2. **Security checklist** completion
3. **Penetration testing** (for major changes)

### Security Checklist

For PRs touching security-sensitive code:

- [ ] Input validation implemented
- [ ] Output sanitization applied
- [ ] Authentication/authorization checked
- [ ] Secrets management reviewed
- [ ] Rate limiting considered
- [ ] Error handling doesn't leak info
- [ ] Logging doesn't contain secrets
- [ ] Dependencies updated
- [ ] Security tests added
- [ ] Documentation updated

## 📊 Supported Versions

| Version | Supported | Security Updates |
|---------|-----------|------------------|
| 2.x.x   | ✅ Yes    | Active |
| 1.x.x   | ⚠️ Limited | Critical only |
| < 1.0   | ❌ No     | Not supported |

## 🔗 Security Resources

### Internal

- [API Key Management Guide](README.md#-security--api-key-management)
- [NGC Deprecation Immunity](docs/NGC_DEPRECATION_IMMUNITY.md)
- [Deployment Security](docs/DEPLOYMENT_GUIDE.md#security)

### External

- [NVIDIA API Security](https://docs.api.nvidia.com/security)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)

## 📞 Security Contacts

- **Security Issues**: security@yourorg.com
- **General Security**: security-team@yourorg.com
- **Emergency (24/7)**: +1-XXX-XXX-XXXX

## 🏆 Security Acknowledgments

We recognize security researchers who responsibly disclose vulnerabilities:

### Hall of Fame

*(Contributors who have reported security issues will be listed here)*

### Bounty Program

*(Information about any security bounty program)*

---

**Last Updated**: 2025-10-02
**Policy Version**: 1.0

Thank you for helping keep our project secure! 🔒
