# Pharmaceutical RAG Template - NVIDIA NeMo Retriever

**Production-grade RAG system for pharmaceutical research with NGC deprecation immunity**

[![NGC Immune](https://img.shields.io/badge/NGC-Immune_March_2026-green)](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever)
[![Free Tier](https://img.shields.io/badge/NVIDIA_Build-10K_Free_Requests-blue)](https://build.nvidia.com)
[![License](https://img.shields.io/github/license/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever)](LICENSE)

_Cloud-first RAG system with pharmaceutical safety guardrails, PubMed integration, and drug interaction detection_

> Production-ready pharmaceutical research platform featuring NVIDIA NeMo Retriever embeddings, reranking, and medical compliance guardrails.

## 🔗 Quick Links

[Try It Now](#try-it-now) • [How It Works](#how-it-works) • [Capabilities](#business-capabilities) • [Impact](#business-impact) • [Industry](#industry-applications) • [Demo](#demo-pharmaceutical-research) • [Tech Overview](#tech-overview)

---

## 📈 Executive Summary

- Problem: Accurate and up-to-date information is critical in every domain—from pharmaceuticals to finance—but teams still rely on manual research across disconnected sources.
- Context: Retrieval-Augmented Generation (RAG) is now standard across industries, yet most implementations are closed, proprietary, and difficult to adapt.
- Solution: This project offers an open-source, enterprise-ready RAG template that organizations can easily customize with their own data, APIs, and compliance rules.
- Scale: Handle thousands of documents and live data streams with domain-specific optimization.
- Compliance: Built-in guardrails for regulatory and cultural alignment.
- Impact: Turn hours of research into minutes of analysis with transparent source traceability and audit confidence.

> [!TIP]
> Tech overview: See Architecture and Examples for system design and end-to-end workflows.

---

## ✨ Highlights

- Business-focused language with clear outcomes and value
- Open-source template designed for customization and control
- Compliance-ready guardrails with full source traceability
- Cross-industry scalability: pharma, finance, legal, consulting

---

## 🏢 Business Problem

Every industry depends on accurate, timely information to make decisions. Yet domain experts in healthcare, finance, law, and consulting still spend significant time manually reviewing reports, filings, and datasets to ensure completeness and compliance.

These manual processes slow down analysis, increase inconsistency, and create compliance risks — especially in regulated sectors where traceability and validation are essential.

---

## 🔍 How It Works

1️⃣ Feed – Upload document libraries (thousands of files) and connect relevant APIs
2️⃣ Customize – Configure industry rules, compliance constraints, and vocabulary
3️⃣ Query – Ask questions naturally, using your domain's language
4️⃣ Deliver – Generate concise expert briefings with full source traceability

---

## ⚡️ Try It Now

```bash
# 1) Clone and enter the project
git clone https://github.com/HendrixMM/pharmaceutical-rag-knowledge-expert.git
cd pharmaceutical-rag-knowledge-expert

# 2) Create a virtual environment (optional but recommended)
python -m venv venv && source venv/bin/activate

# 3) Install dependencies and prepare environment
pip install -r requirements.txt
cp .env.example .env

# 4) Set your NVIDIA API key in .env (required)
# NVIDIA_API_KEY=your_api_key_here

# 5) Add a few PDFs to the default docs folder
# mkdir -p Data/Docs && cp /path/to/*.pdf Data/Docs/

# Option A: Run the CLI assistant
python main.py

# Option B: Launch the Streamlit web app
streamlit run streamlit_app.py
```

---

## 🧩 Business Capabilities

| Category              | Capabilities                                                          |
| --------------------- | --------------------------------------------------------------------- |
| Data Integration      | Thousands of documents • Live API connections • Multi-source fusion   |
| Domain Optimization   | Custom vocabularies • Retrieval tuning • Quality filtering            |
| Enterprise Compliance | Regulatory alignment • Internal policy guardrails • Full audit trails |
| Real-Time Insights    | Continuous data updates • Change monitoring • Timely intelligence     |

---

## 🚀 Business Impact

| Business Outcome    | Manual Process           | AI Template          | Improvement             |
| ------------------- | ------------------------ | -------------------- | ----------------------- |
| Research Time       | Hours per review         | Minutes per review   | ⏱️ 10x faster           |
| Source Coverage     | Limited to a few sources | Hundreds of sources  | 🌐 20x broader          |
| Compliance Risk     | Manual checks            | Automated validation | 🛡️ Reduced risk         |
| Expert Availability | Limited capacity         | Always accessible    | 🔄 Continuous operation |

---

## 🌍 Industry Applications

- Pharma – Research archives + PubMed API + FDA compliance + safety validation
- Finance – Filings + market feeds + regulatory alignment + risk analysis
- Legal – Case databases + court APIs + jurisdiction rules + confidentiality checks
- Consulting – Market reports + industry APIs + client confidentiality + QA standards

---

## 💊 Demo: Pharmaceutical Research

Scenario: A pharmaceutical analyst needs the latest data on drug interactions and historical trial results.
Process: The platform connects to medical databases and uploaded research files, consolidating findings.
Output: An executive briefing with cited sources and safety notes.
Result: Informed decisions in minutes instead of days, with compliance transparency and source validation.

---

## 🧭 Extension Opportunities

This template can be easily extended to new domains:

- Finance – Integrate SEC filings, market data APIs, and compliance rules
- Legal – Connect case law databases and jurisdiction-specific policies
- Enterprise – Deploy across departments with custom data access and audit trails
- Regulatory Intelligence – Add live monitoring for policy or compliance updates

---

## 📚 Tech Overview

- Architecture: docs/ARCHITECTURE.md
- Examples: docs/EXAMPLES.md

---

## 📞 Contact

📧 hendrixmoreau123@gmail.com • 💼 LinkedIn • 🔗 Live Demo (Streamlit instructions above)
