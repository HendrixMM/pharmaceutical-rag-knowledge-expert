## Description

<!-- Provide a clear and concise description of your changes -->

### Type of Change

<!-- Check all that apply -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] 🔧 Configuration change
- [ ] 🧪 Test update
- [ ] ♻️ Code refactoring
- [ ] 🔒 Security improvement
- [ ] 💊 Pharmaceutical/medical feature

### Related Issues

<!-- Link related issues using #issue_number -->

Closes #
Related to #

---

## Changes Made

<!-- Describe the changes in detail -->

### Summary

-
-
-

### Technical Details

<!-- Explain the technical approach, architecture decisions, or implementation details -->

---

## Testing

### Test Coverage

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Pharmaceutical benchmarks updated (if applicable)
- [ ] All tests pass locally (`make test`)
- [ ] Code coverage maintained or improved

### Manual Testing

<!-- Describe manual testing performed -->

**Test Environment:**

- OS:
- Python version:
- NVIDIA endpoint: [ ] Build [ ] Self-hosted

**Test Scenarios:**

1.
2.
3.

---

## Code Quality

### Quality Checks

- [ ] Code follows project style guidelines (`make quality` passes)
- [ ] Code is formatted with Black (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Type hints added for public APIs
- [ ] Docstrings added/updated (Google style)
- [ ] No new warnings introduced

### Security Checks

- [ ] No API keys or secrets committed
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Security scan passes (`make security`)
- [ ] Dependencies updated and scanned (`pip-audit`)
- [ ] Input validation implemented (if applicable)
- [ ] Output sanitization applied (if applicable)

---

## Documentation

### Documentation Updates

- [ ] README.md updated (if needed)
- [ ] API documentation updated (`docs/API_REFERENCE.md`)
- [ ] Examples added/updated (`docs/EXAMPLES.md`)
- [ ] CHANGELOG.md updated (maintainers will handle)
- [ ] Inline code comments added for complex logic
- [ ] Documentation links validated (`make docs-linkcheck`)
- [ ] Table of contents updated (`make docs`)

### Pharmaceutical Context

<!-- If this PR involves pharmaceutical/medical features -->

- [ ] Medical disclaimers added/updated
- [ ] Regulatory compliance considerations documented
- [ ] Drug interaction guidelines followed
- [ ] Clinical data sources cited
- [ ] Safety validation implemented

---

## Data Protection & Privacy

### Data Handling

- [ ] No PII (Personally Identifiable Information) exposed in logs
- [ ] No PHI (Protected Health Information) exposed
- [ ] Sensitive data sanitized in error messages
- [ ] API responses don't leak internal information
- [ ] Test data doesn't contain real patient information

### Compliance

- [ ] HIPAA considerations addressed (if applicable)
- [ ] FDA regulatory guidelines followed (if applicable)
- [ ] Data retention policies respected
- [ ] Audit trail maintained for sensitive operations

---

## Deployment Considerations

### Breaking Changes

<!-- If this is a breaking change, describe migration path -->

- [ ] Migration guide provided
- [ ] Backward compatibility maintained OR deprecation warnings added
- [ ] Environment variable changes documented
- [ ] API changes documented

### Performance Impact

- [ ] Performance impact assessed
- [ ] Benchmarks run (if performance-critical)
- [ ] No significant performance regression
- [ ] Caching strategy considered

---

## Checklist

### Before Submitting

- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
- [ ] I have read the [SECURITY.md](../SECURITY.md) policy
- [ ] My code follows the project's code style
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

### Pharmaceutical Domain (if applicable)

- [ ] Evidence-based approach used
- [ ] Medical sources cited
- [ ] Safety-first design principles followed
- [ ] Guardrails integration tested
- [ ] Medical disclaimers appropriate

---

## Screenshots/Demos

<!-- If applicable, add screenshots or demo videos -->

### Before

### After

---

## Additional Notes

<!-- Any additional information, context, or concerns -->

---

## Reviewer Guidance

<!-- Help reviewers focus on specific areas -->

## **Focus Areas:**

- **Questions for Reviewers:**

-
- ***

## Post-Merge Actions

<!-- Actions to take after merge -->

- [ ] Update related documentation
- [ ] Notify team of changes
- [ ] Update deployment guides
- [ ] Monitor for issues

---

**By submitting this pull request, I confirm that my contribution is made under the terms of the MIT License.**
