# GitHub Template Validation Guide

This document provides instructions for validating GitHub issue and pull request templates.

## Prerequisites

- GitHub CLI (`gh`) installed: https://cli.github.com/
- Authenticated with GitHub: `gh auth login`
- Repository cloned locally

## Validating Issue Templates

### Method 1: GitHub Web Interface

1. Navigate to the repository on GitHub
2. Click "Issues" → "New Issue"
3. Verify all templates appear:
   - Bug Report
   - Feature Request
   - Security Vulnerability Disclosure
4. Click each template and verify:
   - All fields render correctly
   - Required fields are marked
   - Dropdowns have correct options
   - Placeholders are helpful
   - Links work correctly

### Method 2: GitHub CLI Validation

```bash
# Validate YAML syntax
gh issue create --web

# This opens the browser to the issue creation page
# Verify templates load correctly
```

### Method 3: YAML Linting

```bash
# Install yamllint
pip install yamllint

# Validate all issue templates
yamllint .github/ISSUE_TEMPLATE/*.yml

# Expected output: No errors
```

## Validating Pull Request Template

### Method 1: Dry-Run PR Creation

```bash
# Create a test branch
git checkout -b test/template-validation

# Make a trivial change
echo "# Test" >> TEST.md
git add TEST.md
git commit -m "test: template validation"

# Push branch
git push origin test/template-validation

# Create draft PR (dry-run)
gh pr create --fill --draft

# This will:
# 1. Show the PR template in your editor
# 2. Allow you to verify all sections render
# 3. Create a draft PR for inspection

# After validation, close and delete the test PR and branches
gh pr close <PR_NUMBER> --delete-branch
git checkout main
```

### Method 2: Manual Inspection

```bash
# View the template
cat .github/PULL_REQUEST_TEMPLATE.md

# Verify:
# - Markdown renders correctly
# - All checkboxes are present
# - Links are valid
# - Sections are well-organized
```

### Method 3: Markdown Linting

```bash
# Lint the PR template
markdownlint .github/PULL_REQUEST_TEMPLATE.md

# Or use project's markdown linting
make docs-linkcheck
```

## Validation Checklist

### Issue Templates

- [ ] All templates appear in issue creation dropdown
- [ ] Bug report template:
  - [ ] All required fields marked
  - [ ] Dropdowns have correct options
  - [ ] Placeholders are helpful
  - [ ] Links to CONTRIBUTING.md, SECURITY.md work
- [ ] Feature request template:
  - [ ] All required fields marked
  - [ ] Use case dropdown includes pharmaceutical options
  - [ ] Links to FEATURES.md work
- [ ] Security disclosure template:
  - [ ] Warning banner displays prominently
  - [ ] Links to SECURITY.md work
  - [ ] Severity levels match SECURITY.md
- [ ] Config file:
  - [ ] Blank issues disabled
  - [ ] Contact links work
  - [ ] Links point to correct documentation

### Pull Request Template

- [ ] Template loads when creating PR
- [ ] All sections render correctly
- [ ] Checkboxes are functional
- [ ] Links to CONTRIBUTING.md, SECURITY.md work
- [ ] Pharmaceutical-specific sections present
- [ ] Data protection checkboxes present
- [ ] Security checkboxes present
- [ ] Documentation checkboxes present

### Code of Conduct

- [ ] File exists at root: `CODE_OF_CONDUCT.md`
- [ ] Contributor Covenant v2.1 attribution present
- [ ] Contact information updated (or placeholder noted)
- [ ] Pharmaceutical context additions present
- [ ] Links to CONTRIBUTING.md, SECURITY.md work

### Architecture Documentation

- [ ] C4 Level 1 diagram renders correctly
- [ ] Data flow diagram renders correctly
- [ ] Guardrail component tracing complete
- [ ] All Mermaid diagrams render on GitHub
- [ ] Links to guardrails files work
- [ ] Cross-references to other docs work

## Testing Mermaid Diagrams

### Method 1: GitHub Preview

1. Push changes to a branch
2. View the file on GitHub
3. Verify Mermaid diagrams render correctly

### Method 2: Mermaid Live Editor

1. Copy Mermaid code from ARCHITECTURE.md
2. Paste into https://mermaid.live/
3. Verify diagram renders correctly
4. Test different themes (light/dark)

### Method 3: VS Code Preview

1. Install "Markdown Preview Mermaid Support" extension
2. Open ARCHITECTURE.md
3. Use Markdown preview (Cmd+Shift+V or Ctrl+Shift+V)
4. Verify diagrams render

## Common Issues and Fixes

### Issue: Templates Don't Appear

**Cause**: YAML syntax error or incorrect file location

**Fix**:

```bash
# Validate YAML
yamllint .github/ISSUE_TEMPLATE/*.yml

# Ensure files are in correct location
ls -la .github/ISSUE_TEMPLATE/
```

### Issue: Mermaid Diagrams Don't Render

**Cause**: Syntax error or unsupported diagram type

**Fix**:

- Test in Mermaid Live Editor
- Check for syntax errors
- Ensure diagram type is supported by GitHub
- Use simpler diagram if needed

### Issue: Links Don't Work

**Cause**: Incorrect relative paths

**Fix**:

```bash
# Validate all documentation links
make docs-linkcheck

# Fix broken links in templates
```

### Issue: Required Fields Not Enforced

**Cause**: Missing `required: true` in YAML

**Fix**:

- Add `required: true` to field definition
- Test by creating issue without filling required field

## Automated Validation

### CI/CD Integration

Add to `.github/workflows/validate-templates.yml`:

```yaml
name: Validate Templates

on:
  pull_request:
    paths:
      - ".github/ISSUE_TEMPLATE/**"
      - ".github/PULL_REQUEST_TEMPLATE.md"
      - "CODE_OF_CONDUCT.md"
      - "docs/ARCHITECTURE.md"

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Validate YAML
        run: |
          pip install yamllint
          yamllint .github/ISSUE_TEMPLATE/*.yml

      - name: Validate Markdown
        run: |
          npm install -g markdownlint-cli
          markdownlint .github/PULL_REQUEST_TEMPLATE.md CODE_OF_CONDUCT.md

      - name: Validate Links
        run: |
          make docs-linkcheck
```

## Post-Validation Steps

1. **Commit templates**:

   ```bash
   git add .github/ CODE_OF_CONDUCT.md docs/ARCHITECTURE.md
   git commit -m "docs: add GitHub templates and enhanced architecture"
   ```

2. **Create PR**:

   ```bash
   gh pr create --fill
   ```

3. **Verify in PR**:

   - Check that PR template loaded correctly
   - Verify all checkboxes are present
   - Test links in PR description

4. **After merge**:
   - Create a test issue to verify templates work
   - Create a test PR to verify template works
   - Update team documentation

## Maintenance

- **Review templates**: Quarterly
- **Update contact information**: As needed
- **Sync with CONTRIBUTING.md**: When contribution guidelines change
- **Sync with SECURITY.md**: When security policy changes
- **Update pharmaceutical options**: As domain requirements evolve

## Resources

- [GitHub Issue Forms Syntax](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms)
- [GitHub PR Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository)
- [Contributor Covenant](https://www.contributor-covenant.org/)
- [Mermaid Documentation](https://mermaid.js.org/)
- [C4 Model](https://c4model.com/)

---

**Last Updated**: 2025-10-02
**Maintainer**: Documentation Team
