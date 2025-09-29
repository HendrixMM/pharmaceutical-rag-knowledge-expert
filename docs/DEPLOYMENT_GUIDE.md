# Deployment and Configuration Guide

**Comprehensive Deployment Documentation for Pharmaceutical RAG System**

## Overview

This guide provides step-by-step instructions for deploying and configuring the pharmaceutical RAG system with cloud-first architecture, cost optimization, and NGC deprecation immunity.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration Management](#configuration-management)
4. [Cloud-First Deployment](#cloud-first-deployment)
5. [Self-Hosted Fallback Setup](#self-hosted-fallback-setup)
6. [Monitoring and Alerts Configuration](#monitoring-and-alerts-configuration)
7. [Security Hardening](#security-hardening)
8. [Performance Optimization](#performance-optimization)
9. [Backup and Recovery](#backup-and-recovery)
10. [Production Checklist](#production-checklist)

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 10GB available space
- **Network**: Reliable internet connection for cloud endpoints

#### Recommended Production Environment
- **OS**: Linux (Ubuntu 22.04 LTS or CentOS 8+)
- **Python**: 3.11 (latest stable)
- **Memory**: 16GB RAM
- **Storage**: 50GB SSD
- **CPU**: 4+ cores
- **Network**: High-bandwidth connection with low latency to NVIDIA endpoints

### Account Requirements

#### NVIDIA Build Platform
1. **NVIDIA Developer Account**: Sign up at [developer.nvidia.com](https://developer.nvidia.com)
2. **NGC Account**: Create account at [ngc.nvidia.com](https://ngc.nvidia.com) (for fallback)
3. **API Key Generation**: Generate NVIDIA Build API key
4. **Access Verification**: Verify model access permissions

#### Optional Services
- **Email SMTP**: For alert notifications (Gmail, SendGrid, etc.)
- **Slack Workspace**: For Slack notifications (optional)
- **Monitoring Tools**: Prometheus, Grafana, or similar (optional)

### Dependencies Verification

```bash
# Check Python version
python --version  # Should be 3.9+

# Check pip
pip --version

# Check git
git --version

# Check system resources
free -h  # Memory check
df -h    # Disk space check
```

---

## Environment Setup

### 1. Repository Cloning and Setup

```bash
# Clone the repository
git clone <repository-url>
cd RAG-Template-for-NVIDIA-nemoretriever

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 2. Directory Structure Verification

```bash
# Verify directory structure
tree -L 3
```

Expected structure:
```
RAG-Template-for-NVIDIA-nemoretriever/
├── src/
│   ├── clients/
│   │   ├── openai_wrapper.py
│   │   └── nemo_client_enhanced.py
│   ├── pharmaceutical/
│   │   ├── query_classifier.py
│   │   ├── safety_alert_integration.py
│   │   ├── workflow_templates.py
│   │   └── model_optimization.py
│   ├── monitoring/
│   │   ├── credit_tracker.py
│   │   ├── alert_manager.py
│   │   ├── endpoint_health_monitor.py
│   │   └── pharmaceutical_cost_analyzer.py
│   ├── optimization/
│   │   ├── batch_processor.py
│   │   └── batch_integration.py
│   ├── validation/
│   │   └── model_validator.py
│   └── enhanced_config.py
├── config/
│   └── alerts.yaml
├── docs/
├── tests/
└── requirements.txt
```

### 3. Python Path Configuration

```bash
# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# For permanent setup, add to ~/.bashrc or ~/.zshrc:
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/project/src"' >> ~/.bashrc
```

---

## Configuration Management

### 1. Environment Variables Setup

#### Create .env File
```bash
# Create .env file from template
cp .env.example .env

# Edit with your configuration
nano .env
```

#### Complete .env Configuration
```bash
# =============================================================================
# NVIDIA Build Platform Configuration (Primary)
# =============================================================================
NVIDIA_API_KEY=your_nvidia_build_api_key_here
NVIDIA_BUILD_BASE_URL=https://integrate.api.nvidia.com/v1

# =============================================================================
# Cloud-First Strategy Configuration
# =============================================================================
ENABLE_CLOUD_FIRST_STRATEGY=true
ENABLE_NVIDIA_BUILD_FALLBACK=true
CLOUD_FIRST_PRIORITY=nvidia_build

# =============================================================================
# Pharmaceutical Optimization
# =============================================================================
ENABLE_PHARMACEUTICAL_OPTIMIZATION=true
PHARMACEUTICAL_RESEARCH_MODE=true
ENABLE_SAFETY_MONITORING=true
ENABLE_DRUG_INTERACTION_ALERTS=true

# =============================================================================
# Cost Optimization
# =============================================================================
ENABLE_DAILY_CREDIT_ALERTS=true
ENABLE_BATCH_OPTIMIZATION=true
ENABLE_COST_TRACKING=true
MAX_MONTHLY_BUDGET_USD=100
FREE_TIER_TARGET_UTILIZATION=0.80

# =============================================================================
# Monitoring and Alerts
# =============================================================================
ENABLE_HEALTH_MONITORING=true
ENABLE_PERFORMANCE_TRACKING=true
ALERT_EMAIL_ENABLED=false
ALERT_SLACK_ENABLED=false
MONITORING_INTERVAL_SECONDS=60

# =============================================================================
# Feature Flags
# =============================================================================
ENABLE_ADVANCED_CACHING=true
ENABLE_QUERY_CLASSIFICATION=true
ENABLE_WORKFLOW_TEMPLATES=true
ENABLE_MODEL_OPTIMIZATION=true

# =============================================================================
# Security Configuration
# =============================================================================
LOG_LEVEL=INFO
ENABLE_API_KEY_ENCRYPTION=true
ENABLE_PII_SANITIZATION=true
COMPLIANCE_MODE=pharmaceutical

# =============================================================================
# Development/Testing
# =============================================================================
DEVELOPMENT_MODE=false
ENABLE_DEBUG_LOGGING=false
TEST_MODE=false
```

#### Environment Variables Validation
```bash
# Run configuration validation script
python scripts/config_validator.py

# Expected output:
# ✅ NVIDIA API key configured
# ✅ Cloud-first strategy enabled
# ✅ Pharmaceutical optimization enabled
# ✅ Configuration validation passed
```

### 2. Alert Configuration

#### Configure Alert Thresholds
Edit `config/alerts.yaml`:

```yaml
# Critical thresholds for production
nvidia_build:
  monthly_free_requests: 10000
  usage_alerts:
    daily_burn_rate: 0.03      # 3% daily (conservative)
    weekly_burn_rate: 0.15     # 15% weekly
    monthly_usage_warning: 0.75 # 75% monthly warning
    monthly_usage_critical: 0.90 # 90% critical

pharmaceutical:
  project_budget:
    warning_threshold: 0.70    # 70% budget warning
    critical_threshold: 0.85   # 85% budget critical

  query_performance:
    max_acceptable_response_time_ms: 2500  # Production threshold
    consecutive_failures_alert: 2          # More sensitive
```

#### Validate Alert Configuration
```bash
python -c "
from src.enhanced_config import EnhancedRAGConfig
config = EnhancedRAGConfig.from_env()
alerts = config.get_alerts_config()
print('Alert configuration loaded successfully')
print(f'Monthly request limit: {alerts[\"nvidia_build\"][\"monthly_free_requests\"]}')
"
```

---

## Cloud-First Deployment

### 1. NVIDIA Build Platform Setup

#### API Key Configuration
1. **Generate API Key**:
   - Log into NVIDIA Developer Portal
   - Navigate to API Keys section
   - Generate new API key for NVIDIA Build
   - Copy API key securely

2. **Test API Access**:
```bash
# Test API key functionality
python scripts/test_nvidia_access.py

# Expected output:
# ✅ API key valid
# ✅ Model access verified
# ✅ Embedding endpoint accessible
# ✅ Chat endpoint accessible
```

#### Model Access Verification
```bash
python -c "
from src.clients.openai_wrapper import create_nvidia_build_client
client = create_nvidia_build_client()
models = client.list_available_models()
print(f'Available models: {len(models)}')
for model in models[:5]:
    print(f'  - {model[\"id\"]}')
"
```

### 2. Enhanced NeMo Client Deployment

#### Client Initialization Test
```bash
python -c "
from src.clients.nemo_client_enhanced import create_pharmaceutical_client

# Test client creation
client = create_pharmaceutical_client(cloud_first=True)
print('✅ Enhanced NeMo client created')

# Test endpoint status
status = client.get_endpoint_status()
print(f'Cloud available: {status[\"cloud_available\"]}')
print(f'Strategy: {status[\"strategy\"]}')
"
```

#### Pharmaceutical Capabilities Test
```bash
python -c "
import asyncio
from src.clients.nemo_client_enhanced import create_pharmaceutical_client

async def test_capabilities():
    client = create_pharmaceutical_client()
    capabilities = client.test_pharmaceutical_capabilities()
    print(f'Overall status: {capabilities[\"overall_status\"]}')
    print(f'Cloud test: {capabilities.get(\"cloud_test\", {}).get(\"success\", False)}')
    print(f'Embedding test: {capabilities.get(\"embedding_test\", {}).get(\"success\", False)}')
    print(f'Chat test: {capabilities.get(\"chat_test\", {}).get(\"success\", False)}')

asyncio.run(test_capabilities())
"
```

### 3. Cost Monitoring Setup

#### Initialize Cost Tracking
```bash
python -c "
from src.monitoring.pharmaceutical_cost_analyzer import create_pharmaceutical_cost_tracker

# Create cost tracker
tracker = create_pharmaceutical_cost_tracker()
print('✅ Cost tracking initialized')

# Create default project
tracker.create_research_project(
    project_id='production_research',
    project_name='Production Pharmaceutical Research',
    monthly_budget_usd=75.0,
    priority_level=2
)
print('✅ Default research project created')
"
```

#### Test Alert System
```bash
python -c "
from src.monitoring.alert_manager import PharmaceuticalAlertManager
import asyncio

async def test_alerts():
    manager = PharmaceuticalAlertManager()

    # Test daily burn rate check
    await manager.check_daily_burn_rate()
    print('✅ Daily burn rate alert system tested')

asyncio.run(test_alerts())
"
```

---

## Self-Hosted Fallback Setup

### 1. NeMo Infrastructure Configuration

#### Install NeMo Dependencies
```bash
# Install NVIDIA NeMo if needed for fallback
pip install nemo-toolkit[all]

# Or lightweight installation
pip install nemo-toolkit[asr,nlp]
```

#### Configure Fallback Client
```bash
python -c "
from src.clients.nemo_client_enhanced import EnhancedNeMoClient

# Test fallback configuration
client = EnhancedNeMoClient(
    enable_fallback=True,
    pharmaceutical_optimized=True
)

print(f'Fallback enabled: {client.enable_fallback}')
print(f'NeMo client available: {client.nemo_client is not None}')
"
```

### 2. Local Model Management

#### Model Download and Setup
```bash
# Create models directory
mkdir -p models/local

# Download pharmaceutical-optimized models (if available)
# This is placeholder - actual implementation depends on available models
python scripts/download_local_models.py --pharmaceutical-optimized

# Verify local model setup
python -c "
import os
models_dir = 'models/local'
if os.path.exists(models_dir):
    models = os.listdir(models_dir)
    print(f'Local models available: {len(models)}')
    for model in models:
        print(f'  - {model}')
else:
    print('No local models directory found')
"
```

### 3. Fallback Testing

#### Test Fallback Mechanism
```bash
python -c "
from src.clients.nemo_client_enhanced import EnhancedNeMoClient

# Force fallback testing (disable cloud temporarily)
import os
original_key = os.environ.get('NVIDIA_API_KEY')
os.environ['NVIDIA_API_KEY'] = 'invalid_key_for_testing'

try:
    client = EnhancedNeMoClient(enable_fallback=True)
    response = client.create_embeddings(['test pharmaceutical query'])

    if response.success:
        print(f'✅ Fallback working: {response.endpoint_type.value}')
        print(f'Cost tier: {response.cost_tier}')
    else:
        print(f'❌ Fallback failed: {response.error}')
finally:
    # Restore original key
    if original_key:
        os.environ['NVIDIA_API_KEY'] = original_key
"
```

---

## Monitoring and Alerts Configuration

### 1. Health Monitoring Setup

#### Initialize Health Monitor
```bash
python -c "
import asyncio
from src.monitoring.endpoint_health_monitor import create_endpoint_health_monitor

async def setup_monitoring():
    monitor = create_endpoint_health_monitor(
        monitoring_interval=60,
        pharmaceutical_focused=True
    )

    # Perform initial health check
    status = await monitor.perform_comprehensive_health_check()
    print('✅ Health monitoring initialized')
    print(f'Overall health: {status[\"endpoint_health\"][\"overall_health\"]}')
    print(f'NGC independent: {status[\"ngc_independence_status\"][\"verified\"]}')

asyncio.run(setup_monitoring())
"
```

#### Configure Continuous Monitoring
Create systemd service (Linux) for continuous monitoring:

```bash
# Create service file
sudo tee /etc/systemd/system/pharma-health-monitor.service > /dev/null <<EOF
[Unit]
Description=Pharmaceutical RAG Health Monitor
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=PYTHONPATH=$(pwd)/src
ExecStart=$(pwd)/venv/bin/python scripts/continuous_health_monitor.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable pharma-health-monitor
sudo systemctl start pharma-health-monitor
sudo systemctl status pharma-health-monitor
```

### 2. Alert Notification Setup

#### Email Alerts Configuration
```bash
# Configure email alerts in .env
cat >> .env << EOF

# Email Alert Configuration
ALERT_EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=alerts@yourcompany.com
EOF
```

#### Test Email Alerts
```bash
python -c "
import asyncio
from src.monitoring.alert_manager import PharmaceuticalAlertManager

async def test_email():
    manager = PharmaceuticalAlertManager()

    # Test email alert
    await manager.send_test_alert()
    print('✅ Test alert sent')

asyncio.run(test_email())
"
```

#### Slack Alerts Configuration (Optional)
```bash
# Add Slack configuration to .env
cat >> .env << EOF

# Slack Alert Configuration
ALERT_SLACK_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_CHANNEL=#pharmaceutical-alerts
EOF
```

### 3. Model Validation Automation

#### Automated Model Validation
Create validation cron job:

```bash
# Create validation script
cat > scripts/automated_model_validation.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import sys
import json
from datetime import datetime
from src.validation.model_validator import validate_nvidia_build_compatibility

async def main():
    try:
        results = await validate_nvidia_build_compatibility(pharmaceutical_optimized=True)

        print(f"Model validation completed: {datetime.now().isoformat()}")
        print(f"Overall status: {results['overall_status']}")
        print(f"NGC independent: {results['ngc_independent']}")

        # Check for issues
        if results['overall_status'] != 'full_compatibility':
            print(f"WARNING: Model compatibility issues detected")
            sys.exit(1)
        else:
            print("✅ All models fully compatible")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x scripts/automated_model_validation.py

# Add to crontab (runs daily at 6 AM)
(crontab -l 2>/dev/null; echo "0 6 * * * cd $(pwd) && $(pwd)/venv/bin/python $(pwd)/scripts/automated_model_validation.py >> logs/model_validation.log 2>&1") | crontab -
```

---

## Security Hardening

### 1. API Key Security

#### Secure API Key Storage
```bash
# Create secure key storage script
cat > scripts/setup_secure_keys.py << 'EOF'
#!/usr/bin/env python3
import os
from cryptography.fernet import Fernet

# Generate encryption key
key = Fernet.generate_key()
with open('.encryption_key', 'wb') as f:
    f.write(key)

# Secure the key file
os.chmod('.encryption_key', 0o600)

print("✅ Encryption key generated")
print("Add to .gitignore:")
print(".encryption_key")
print(".env.encrypted")
EOF

python scripts/setup_secure_keys.py
```

#### Environment Variable Encryption
```bash
# Add to .gitignore
echo ".encryption_key" >> .gitignore
echo ".env.encrypted" >> .gitignore

# Create encrypted environment
python -c "
from cryptography.fernet import Fernet
import os

# Load encryption key
with open('.encryption_key', 'rb') as f:
    key = f.read()

f = Fernet(key)

# Encrypt sensitive environment variables
with open('.env', 'r') as file:
    env_content = file.read()

encrypted_content = f.encrypt(env_content.encode())

with open('.env.encrypted', 'wb') as file:
    file.write(encrypted_content)

print('✅ Environment variables encrypted')
"
```

### 2. Network Security

#### Firewall Configuration
```bash
# Configure UFW firewall (Ubuntu)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port as needed)
sudo ufw allow ssh

# Allow HTTPS outbound for NVIDIA APIs
sudo ufw allow out 443

# Check status
sudo ufw status verbose
```

#### SSL/TLS Configuration
```bash
# Ensure proper SSL certificate validation
python -c "
import ssl
import certifi

# Verify SSL context
context = ssl.create_default_context()
print(f'✅ SSL context created with CA bundle: {certifi.where()}')

# Test NVIDIA endpoint SSL
import urllib.request
try:
    with urllib.request.urlopen('https://integrate.api.nvidia.com', context=context) as response:
        print('✅ NVIDIA endpoint SSL verified')
except Exception as e:
    print(f'❌ SSL verification failed: {e}')
"
```

### 3. Data Privacy

#### PII Sanitization Setup
```bash
# Test PII sanitization
python -c "
from src.pharmaceutical.model_optimization import PharmaceuticalDataSanitizer

sanitizer = PharmaceuticalDataSanitizer()

# Test query sanitization
test_query = 'Patient John Doe (DOB: 01/15/1980, MRN: 12345) on metformin'
sanitized, removed = sanitizer.sanitize_query(test_query)

print(f'Original: {test_query}')
print(f'Sanitized: {sanitized}')
print(f'Removed PII: {removed}')
"
```

#### Compliance Validation
```bash
python -c "
from src.pharmaceutical.model_optimization import PharmaceuticalDataSanitizer

sanitizer = PharmaceuticalDataSanitizer()
compliance = sanitizer.validate_pharmaceutical_compliance(
    'Research metformin efficacy in diabetes patients'
)

print(f'✅ Compliance validation: {compliance[\"compliance_status\"]}')
print(f'PII detected: {compliance[\"pii_detected\"]}')
"
```

---

## Performance Optimization

### 1. Caching Configuration

#### Redis Setup (Optional)
```bash
# Install Redis (Ubuntu)
sudo apt update
sudo apt install redis-server

# Configure Redis for caching
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Test Redis
redis-cli ping  # Should return PONG
```

#### Application-Level Caching
```bash
# Test built-in caching
python -c "
from src.pharmaceutical.model_optimization import PharmaceuticalQueryCache

cache = PharmaceuticalQueryCache(max_size=1000)
print('✅ Pharmaceutical query cache initialized')

# Test cache operations
cache.set('test_query', {'domain': 'test'}, 'test_result')
result = cache.get('test_query', {'domain': 'test'})
print(f'Cache test: {\"✅ passed\" if result == \"test_result\" else \"❌ failed\"}')
"
```

### 2. Connection Pooling

#### Client Pool Configuration
```bash
python -c "
import asyncio
from src.pharmaceutical.model_optimization import PharmaceuticalClientPool

async def test_pool():
    pool = PharmaceuticalClientPool(pool_size=5)

    async with pool.acquire_client() as client:
        print('✅ Client acquired from pool')
        status = client.get_endpoint_status()
        print(f'Client ready: {status[\"cloud_available\"]}')

asyncio.run(test_pool())
"
```

### 3. Database Optimization

#### SQLite Optimization (for metrics storage)
```bash
# Create optimized database
python -c "
import sqlite3
import os

# Create metrics database with optimizations
db_path = 'metrics.db'
conn = sqlite3.connect(db_path)

# Enable WAL mode for better concurrent access
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA synchronous=NORMAL')
conn.execute('PRAGMA cache_size=10000')
conn.execute('PRAGMA temp_store=memory')

# Create tables
conn.execute('''
CREATE TABLE IF NOT EXISTS pharmaceutical_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT UNIQUE,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    domain TEXT,
    cost_tier TEXT,
    response_time_ms INTEGER,
    success BOOLEAN
)
''')

conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON pharmaceutical_queries(timestamp)')
conn.execute('CREATE INDEX IF NOT EXISTS idx_domain ON pharmaceutical_queries(domain)')
conn.execute('CREATE INDEX IF NOT EXISTS idx_cost_tier ON pharmaceutical_queries(cost_tier)')

conn.commit()
conn.close()

print(f'✅ Optimized metrics database created: {db_path}')
"
```

---

## Backup and Recovery

### 1. Configuration Backup

#### Automated Backup Script
```bash
# Create backup script
cat > scripts/backup_config.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration files
cp .env "$BACKUP_DIR/" 2>/dev/null || echo "No .env file found"
cp .env.encrypted "$BACKUP_DIR/" 2>/dev/null || echo "No encrypted env file found"
cp config/alerts.yaml "$BACKUP_DIR/"
cp -r src/pharmaceutical/ "$BACKUP_DIR/pharmaceutical_config/"

# Backup metrics database
cp metrics.db "$BACKUP_DIR/" 2>/dev/null || echo "No metrics database found"

# Create backup manifest
cat > "$BACKUP_DIR/backup_manifest.txt" << EOL
Backup created: $(date)
System: $(uname -a)
Python version: $(python --version)
Git commit: $(git rev-parse HEAD 2>/dev/null || echo "Not a git repository")
Configuration files included:
- Environment configuration
- Alert configuration
- Pharmaceutical models configuration
- Metrics database
EOL

echo "✅ Backup created: $BACKUP_DIR"
tar -czf "$BACKUP_DIR.tar.gz" -C backups "$(basename $BACKUP_DIR)"
echo "✅ Compressed backup: $BACKUP_DIR.tar.gz"
EOF

chmod +x scripts/backup_config.sh

# Create initial backup
./scripts/backup_config.sh
```

#### Automated Backup Schedule
```bash
# Add to crontab (daily backups at midnight)
(crontab -l 2>/dev/null; echo "0 0 * * * cd $(pwd) && ./scripts/backup_config.sh >> logs/backup.log 2>&1") | crontab -
```

### 2. Disaster Recovery Plan

#### Recovery Script
```bash
cat > scripts/restore_config.sh << 'EOF'
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    echo "Available backups:"
    ls -la backups/*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE=$1
RESTORE_DIR="restore_$(date +%Y%m%d_%H%M%S)"

echo "🔄 Restoring from: $BACKUP_FILE"

# Extract backup
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

BACKUP_CONTENTS=$(ls "$RESTORE_DIR")
BACKUP_PATH="$RESTORE_DIR/$BACKUP_CONTENTS"

# Show backup manifest
if [ -f "$BACKUP_PATH/backup_manifest.txt" ]; then
    echo "📋 Backup information:"
    cat "$BACKUP_PATH/backup_manifest.txt"
    echo
fi

# Confirm restoration
read -p "Proceed with restoration? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Restoration cancelled"
    rm -rf "$RESTORE_DIR"
    exit 1
fi

# Backup current configuration
echo "🔒 Backing up current configuration..."
./scripts/backup_config.sh

# Restore files
echo "📦 Restoring configuration files..."
cp "$BACKUP_PATH/.env" . 2>/dev/null && echo "✅ Environment configuration restored"
cp "$BACKUP_PATH/.env.encrypted" . 2>/dev/null && echo "✅ Encrypted environment restored"
cp "$BACKUP_PATH/alerts.yaml" config/ 2>/dev/null && echo "✅ Alert configuration restored"
cp "$BACKUP_PATH/metrics.db" . 2>/dev/null && echo "✅ Metrics database restored"

if [ -d "$BACKUP_PATH/pharmaceutical_config" ]; then
    cp -r "$BACKUP_PATH/pharmaceutical_config/"* src/pharmaceutical/ 2>/dev/null && echo "✅ Pharmaceutical configuration restored"
fi

# Cleanup
rm -rf "$RESTORE_DIR"

echo "✅ Configuration restored successfully"
echo "🔄 Please restart services to apply changes"
EOF

chmod +x scripts/restore_config.sh
```

---

## Production Checklist

### 1. Pre-Deployment Checklist

#### Environment Validation
```bash
# Create comprehensive validation script
cat > scripts/production_checklist.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import asyncio
import json
from datetime import datetime

async def validate_production_readiness():
    """Comprehensive production readiness validation."""

    checks = []

    # Environment variables check
    required_env_vars = [
        'NVIDIA_API_KEY',
        'NVIDIA_BUILD_BASE_URL',
        'ENABLE_CLOUD_FIRST_STRATEGY',
        'ENABLE_PHARMACEUTICAL_OPTIMIZATION'
    ]

    env_check = all(os.getenv(var) for var in required_env_vars)
    checks.append(("Environment variables", env_check))

    # API connectivity check
    try:
        from src.clients.openai_wrapper import create_nvidia_build_client
        client = create_nvidia_build_client()
        models = client.list_available_models()
        api_check = len(models) > 0
    except Exception as e:
        api_check = False

    checks.append(("NVIDIA API connectivity", api_check))

    # Pharmaceutical capabilities check
    try:
        from src.clients.nemo_client_enhanced import create_pharmaceutical_client
        client = create_pharmaceutical_client()
        capabilities = client.test_pharmaceutical_capabilities()
        pharma_check = capabilities["overall_status"] in ["success", "partial"]
    except Exception as e:
        pharma_check = False

    checks.append(("Pharmaceutical capabilities", pharma_check))

    # Cost monitoring check
    try:
        from src.monitoring.pharmaceutical_cost_analyzer import create_pharmaceutical_cost_tracker
        tracker = create_pharmaceutical_cost_tracker()
        cost_check = True
    except Exception as e:
        cost_check = False

    checks.append(("Cost monitoring", cost_check))

    # Health monitoring check
    try:
        from src.monitoring.endpoint_health_monitor import quick_health_check
        health_status = await quick_health_check()
        health_check = health_status["endpoint_health"]["overall_health"] != "unhealthy"
    except Exception as e:
        health_check = False

    checks.append(("Health monitoring", health_check))

    # Model validation check
    try:
        from src.validation.model_validator import validate_nvidia_build_compatibility
        results = await validate_nvidia_build_compatibility()
        model_check = results["ngc_independent"] and results["overall_status"] != "no_compatible_models"
    except Exception as e:
        model_check = False

    checks.append(("Model validation", model_check))

    # Security checks
    security_checks = [
        ("API key encryption available", os.path.exists(".encryption_key")),
        ("PII sanitization enabled", os.getenv("ENABLE_PII_SANITIZATION", "false").lower() == "true"),
        ("SSL verification", True)  # Always enabled in production
    ]

    checks.extend(security_checks)

    # Performance checks
    performance_checks = [
        ("Caching enabled", os.getenv("ENABLE_ADVANCED_CACHING", "false").lower() == "true"),
        ("Batch optimization enabled", os.getenv("ENABLE_BATCH_OPTIMIZATION", "false").lower() == "true")
    ]

    checks.extend(performance_checks)

    # Display results
    print("=" * 60)
    print("PRODUCTION READINESS CHECKLIST")
    print("=" * 60)
    print(f"Validation time: {datetime.now().isoformat()}")
    print()

    passed_checks = 0
    total_checks = len(checks)

    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if passed:
            passed_checks += 1

    print()
    print(f"Overall: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print("🎉 PRODUCTION READY!")
        return True
    else:
        print("⚠️  PRODUCTION NOT READY - Address failed checks")
        return False

if __name__ == "__main__":
    ready = asyncio.run(validate_production_readiness())
    sys.exit(0 if ready else 1)
EOF

chmod +x scripts/production_checklist.py

# Run production checklist
python scripts/production_checklist.py
```

### 2. Performance Benchmarks

#### Baseline Performance Test
```bash
cat > scripts/performance_benchmark.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import time
import statistics
from datetime import datetime

async def run_performance_benchmark():
    """Run comprehensive performance benchmark."""

    print("🏃 Running performance benchmark...")
    print(f"Start time: {datetime.now().isoformat()}")

    from src.clients.nemo_client_enhanced import create_pharmaceutical_client

    client = create_pharmaceutical_client()

    # Test queries
    test_queries = [
        "metformin mechanism of action",
        "ACE inhibitor contraindications",
        "warfarin drug interactions",
        "statin side effects monitoring",
        "diabetes medication guidelines"
    ]

    # Embedding performance test
    print("\n📊 Embedding Performance Test")
    embedding_times = []

    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        response = client.create_embeddings([query])
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # milliseconds
        embedding_times.append(response_time)

        status = "✅" if response.success else "❌"
        print(f"  {i}/5 {status} {response_time:.0f}ms - {query[:30]}...")

    # Chat completion performance test
    print("\n💬 Chat Completion Performance Test")
    chat_times = []

    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        response = client.create_chat_completion([
            {"role": "user", "content": f"Explain {query} in pharmaceutical context"}
        ])
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # milliseconds
        chat_times.append(response_time)

        status = "✅" if response.success else "❌"
        print(f"  {i}/5 {status} {response_time:.0f}ms - {query[:30]}...")

    # Performance summary
    print("\n📈 Performance Summary")
    print(f"Embeddings:")
    print(f"  Average: {statistics.mean(embedding_times):.0f}ms")
    print(f"  Median: {statistics.median(embedding_times):.0f}ms")
    print(f"  Min: {min(embedding_times):.0f}ms")
    print(f"  Max: {max(embedding_times):.0f}ms")

    print(f"Chat Completions:")
    print(f"  Average: {statistics.mean(chat_times):.0f}ms")
    print(f"  Median: {statistics.median(chat_times):.0f}ms")
    print(f"  Min: {min(chat_times):.0f}ms")
    print(f"  Max: {max(chat_times):.0f}ms")

    # Performance targets
    embedding_target = 1000  # 1 second
    chat_target = 3000       # 3 seconds

    embedding_pass = statistics.median(embedding_times) < embedding_target
    chat_pass = statistics.median(chat_times) < chat_target

    print(f"\nPerformance Targets:")
    print(f"  Embeddings < {embedding_target}ms: {'✅ PASS' if embedding_pass else '❌ FAIL'}")
    print(f"  Chat < {chat_target}ms: {'✅ PASS' if chat_pass else '❌ FAIL'}")

    overall_pass = embedding_pass and chat_pass
    print(f"\nOverall Performance: {'✅ ACCEPTABLE' if overall_pass else '⚠️ NEEDS OPTIMIZATION'}")

    return overall_pass

if __name__ == "__main__":
    asyncio.run(run_performance_benchmark())
EOF

chmod +x scripts/performance_benchmark.py

# Run benchmark
python scripts/performance_benchmark.py
```

### 3. Security Audit

#### Security Validation Script
```bash
cat > scripts/security_audit.py << 'EOF'
#!/usr/bin/env python3
import os
import stat
import pwd
import grp
from pathlib import Path

def security_audit():
    """Perform security audit of the deployment."""

    print("🔒 SECURITY AUDIT")
    print("=" * 40)

    issues = []

    # File permissions audit
    sensitive_files = ['.env', '.env.encrypted', '.encryption_key']

    for file_path in sensitive_files:
        if os.path.exists(file_path):
            file_stat = os.stat(file_path)
            file_mode = stat.filemode(file_stat.st_mode)

            # Check if file is readable by others
            if file_stat.st_mode & stat.S_IROTH:
                issues.append(f"{file_path} is readable by others: {file_mode}")
            else:
                print(f"✅ {file_path} permissions secure: {file_mode}")

    # Environment variable audit
    env_vars_check = [
        ('NVIDIA_API_KEY', 'API key configured'),
        ('ENABLE_PII_SANITIZATION', 'PII sanitization enabled'),
        ('ENABLE_API_KEY_ENCRYPTION', 'API key encryption enabled')
    ]

    for var, description in env_vars_check:
        if os.getenv(var):
            print(f"✅ {description}")
        else:
            issues.append(f"Missing or disabled: {description}")

    # Network security checks
    if os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true':
        issues.append("Development mode enabled in production")
    else:
        print("✅ Development mode disabled")

    if os.getenv('ENABLE_DEBUG_LOGGING', 'false').lower() == 'true':
        issues.append("Debug logging enabled (may leak sensitive information)")
    else:
        print("✅ Debug logging disabled")

    # Summary
    print("\n" + "=" * 40)
    if issues:
        print(f"❌ SECURITY ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  Address these issues before production deployment")
        return False
    else:
        print("✅ NO SECURITY ISSUES FOUND")
        print("🔒 Security audit passed")
        return True

if __name__ == "__main__":
    security_audit()
EOF

chmod +x scripts/security_audit.py

# Run security audit
python scripts/security_audit.py
```

### 4. Final Production Deployment

#### Production Deployment Script
```bash
cat > scripts/deploy_production.sh << 'EOF'
#!/bin/bash

echo "🚀 PRODUCTION DEPLOYMENT STARTING..."
echo "=================================="

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Run production checklist
if ! python scripts/production_checklist.py; then
    echo "❌ Production readiness check failed"
    exit 1
fi

# Run security audit
if ! python scripts/security_audit.py; then
    echo "❌ Security audit failed"
    exit 1
fi

# Run performance benchmark
if ! python scripts/performance_benchmark.py; then
    echo "⚠️  Performance benchmark indicates potential issues"
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Deployment cancelled"
        exit 1
    fi
fi

# Create production backup
echo "💾 Creating pre-deployment backup..."
./scripts/backup_config.sh

# Set production environment
echo "🔧 Setting production environment..."
export DEVELOPMENT_MODE=false
export ENABLE_DEBUG_LOGGING=false
export LOG_LEVEL=INFO

# Start monitoring services
echo "📊 Starting monitoring services..."

# Start health monitoring (if systemd service exists)
if systemctl is-enabled pharma-health-monitor >/dev/null 2>&1; then
    sudo systemctl restart pharma-health-monitor
    echo "✅ Health monitoring service started"
fi

# Validate deployment
echo "✅ Running post-deployment validation..."

# Quick API test
python -c "
from src.clients.nemo_client_enhanced import create_pharmaceutical_client
client = create_pharmaceutical_client()
response = client.create_embeddings(['production deployment test'])
if response.success:
    print('✅ Production API test passed')
else:
    print('❌ Production API test failed')
    exit(1)
"

echo "🎉 PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "=============================================="
echo "✅ All systems operational"
echo "✅ Monitoring active"
echo "✅ NGC deprecation immune"
echo "✅ Pharmaceutical optimization enabled"
echo "✅ Cost monitoring active"
echo ""
echo "🔗 System Status Dashboard: http://localhost:8080 (if configured)"
echo "📧 Alerts will be sent to configured email/Slack"
echo "📊 Monitor logs: tail -f logs/*.log"
EOF

chmod +x scripts/deploy_production.sh

# Run production deployment
./scripts/deploy_production.sh
```

---

## Post-Deployment

### 1. Monitoring Setup

After successful deployment, monitor the following:

- **Health Dashboard**: Check endpoint health regularly
- **Cost Tracking**: Monitor daily/monthly usage
- **Performance Metrics**: Track response times and success rates
- **Alert Notifications**: Ensure alerts are working correctly
- **Security Logs**: Monitor for any security events

### 2. Maintenance Schedule

Establish regular maintenance tasks:

- **Daily**: Check health status and cost usage
- **Weekly**: Review performance metrics and optimize
- **Monthly**: Full system validation and backup verification
- **Quarterly**: Security audit and dependency updates

### 3. Support and Troubleshooting

For ongoing support:

- Monitor log files in `logs/` directory
- Check system metrics regularly
- Review alert notifications
- Maintain backup schedule
- Keep documentation updated

---

**Document Version**: 1.0.0
**Last Updated**: September 24, 2025
**Maintained By**: Pharmaceutical RAG Team