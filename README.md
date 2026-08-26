# SupportAI – Intelligent Customer Support Ticket Triage and Response System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-SQLite-003B57.svg?logo=sqlite&logoColor=white)](https://sqlite.org/)
[![UI](https://img.shields.io/badge/UI-Bootstrap_5_%2B_Chart.js-purple.svg)](https://getbootstrap.com/)

> **Microsoft Skill-Based Internship Prototype**  
> Tracks Covered: **Azure Fundamentals (AZ-900)**, **Azure Administration (AZ-104)**, **Azure AI Fundamentals (AI-900)**, **Azure AI Applications & Agents (AI-103)**, and **AI-Assisted Development (AZ-2007)**.

---

## 1. Project Overview & Problem Statement

Modern enterprise customer support desks process thousands of incoming unstructured inquiries daily across multiple channels. Manual triage leads to:
- High response latency and missed Service Level Agreements (SLAs).
- Inconsistent categorization and routing errors.
- Support agent burnout on repetitive high-frequency questions.
- Inability to prioritize urgent high-risk tickets (e.g., security breaches, duplicate billing charges).

**SupportAI** is a full-stack, machine-learning-driven triage and auto-response system. A customer submits an issue in natural language; SupportAI instantly performs NLP preprocessing, feature extraction, and multi-dimensional classification to determine:
1. **Ticket Category** (7 domains: *Billing / Payment*, *Account / Login*, *Technical Issue*, *Order / Delivery*, *Product Issue*, *Cancellation / Refund*, *General Inquiry*).
2. **Priority Level** (*High*, *Medium*, *Low*) with urgency signal detection.
3. **Customer Sentiment** (*Positive*, *Neutral*, *Negative*).
4. **Context-Aware Suggested Response** incorporating domain troubleshooting, priority SLA commitments, and empathetic tone.

---

## 2. Key Objectives

- **Natural Language Processing (NLP):** Clean unstructured text, handle punctuation, stopwords, and normalize tokens.
- **TF-IDF Feature Representation:** Extract unigram and bigram numerical vectors with sublinear term-frequency scaling.
- **Empirical Algorithm Benchmarking:** Implement, train, and compare **Logistic Regression** versus **Random Forest Classifier** on real metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix).
- **Multi-Output Intelligence:** Classify category, predict priority level, and analyze customer sentiment tone.
- **Dynamic Response Synthesis:** Assemble context-specific, professional support replies without hardcoding or ungrounded generation.
- **Persistent Storage & Analytics:** SQLite transaction history and interactive Chart.js analytics dashboard.
- **Modular Agent-Ready Architecture:** Designed with modular abstraction for future cloud scaling to **Azure AI Agent Service / Azure OpenAI (AI-103)**.

---

## 3. System Architecture & Workflow

```
                        ┌─────────────────────────────────────────┐
                        │   Customer Ticket (Natural Language)    │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │        NLP Text Preprocessing           │
                        │ (Lowercasing, Regex, Token Normalizer)  │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │        TF-IDF Vectorization             │
                        │    (Unigrams + Bigrams, Sublinear TF)   │
                        └─────────┬──────────┬──────────┬─────────┘
                                  │          │          │
                 ┌────────────────┘          │          └────────────────┐
                 ▼                           ▼                           ▼
    ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
    │   Category Classifier   │ │   Priority Classifier   │ │   Sentiment Classifier  │
    │  (Logistic Regression   │ │ (ML Model + Urgency     │ │ (TF-IDF Multi-Class     │
    │   vs. Random Forest)    │ │  Heuristic Safeguard)   │ │  Sentiment Engine)      │
    └────────────┬────────────┘ └────────────┬────────────┘ └────────────┬────────────┘
                 │                           │                           │
                 └───────────────────┬───────┴───────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────────────────┐
                        │     Suggested Response Synthesizer      │
                        │ (Domain Actions + SLA + Empathy Tone)   │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │     SQLite Storage & Web UI Rendering   │
                        │   (Dashboard, History, Live Triage)     │
                        └─────────────────────────────────────────┘
```

---

## 4. Machine Learning Methodology & Evaluation

### 4.1 Dataset Description
The dataset (`data/support_tickets.csv`) contains **301 labeled customer support interactions** balanced across all 7 operational categories, 3 priority tiers, and 3 sentiment classes.

### 4.2 Feature Extraction (TF-IDF)
- **N-gram Range:** Unigrams & Bigrams `(1, 2)` to capture phrase context (e.g., *"not working"*, *"payment deducted"*, *"account locked"*).
- **Sublinear TF Scaling:** Reduces the disproportionate impact of high-frequency repetitive words.
- **Vocabulary Size:** Max 2,500 informative features.

### 4.3 Algorithm Comparison Results

Both classifiers were trained on an 80% stratified training set (240 samples) and evaluated on a held-out 20% test set (61 samples):

| Evaluation Metric | Logistic Regression (Linear) | Random Forest Classifier (Ensemble) | Delta (Advantage) | Best Algorithm |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy Score** | **81.97%** | 63.93% | **+18.04%** | **Logistic Regression** |
| **Weighted Precision** | **83.62%** | 69.41% | **+14.21%** | **Logistic Regression** |
| **Weighted Recall** | **81.97%** | 63.93% | **+18.04%** | **Logistic Regression** |
| **Weighted F1-Score** | **81.78%** | 64.77% | **+17.01%** | **Logistic Regression** |
| **Macro F1-Score** | **81.56%** | 64.12% | **+17.44%** | **Logistic Regression** |

> **Viva Insight:** *Why does Logistic Regression outperform Random Forest on this dataset?*  
> Text classification features are high-dimensional and sparse. Linear models like Logistic Regression calculate hyperplanes across TF-IDF features with continuous weights, generalizing smoothly on sparse n-gram spaces. In contrast, axis-aligned decision trees in Random Forest require deeper splits to isolate sparse orthogonal words, making them prone to overfitting on compact text corpora.

### 4.4 Auxiliary Classifiers
- **Priority Model:** Evaluates ticket urgency with an ML model combined with heuristic protection keywords (*"urgent"*, *"immediately"*, *"critical"*, *"cannot access"*, *"payment deducted"*, *"account locked"*, *"completely broken"*).
- **Sentiment Model:** Multinomial classification (*Positive*, *Neutral*, *Negative*) with 78.69% accuracy.

---

## 5. Technology Stack

- **Backend:** Python 3.9+, Flask (RESTful routing & Jinja2 templating)
- **Machine Learning & NLP:** scikit-learn, pandas, numpy, TF-IDF Vectorizer
- **Database:** SQLite3 (automatic schema initialization and persistence)
- **Frontend:** HTML5, CSS3, JavaScript (ES6), Bootstrap 5, Chart.js 4.4, FontAwesome 6
- **Architecture Standard:** Model-View-Controller (MVC) separation

---

## 6. Project Structure

```
supportai/
│
├── app.py                     # Main Flask web application & REST API routes
├── config.py                  # Environment and directory configuration
├── requirements.txt           # Python dependency specifications
├── README.md                  # Academic project documentation and guide
├── test_app.py                # Automated unit & integration test suite
│
├── data/
│   ├── support_tickets.csv    # 301 curated customer support ticket records
│   └── generate_data.py       # Dataset generator script
│
├── models/
│   ├── category_model.pkl     # Best trained production category classifier
│   ├── category_vectorizer.pkl# TF-IDF vectorizer for category features
│   ├── priority_model.pkl     # Priority prediction classifier
│   ├── priority_vectorizer.pkl# TF-IDF vectorizer for priority
│   ├── sentiment_model.pkl    # Sentiment analysis classifier
│   ├── sentiment_vectorizer.pkl# TF-IDF vectorizer for sentiment
│   └── metrics.json           # Serialized evaluation metrics & confusion matrices
│
├── ml/
│   ├── __init__.py
│   ├── preprocess.py          # Regex text cleaning & normalization pipeline
│   ├── train.py               # ML training and model comparison pipeline
│   ├── evaluate.py            # Precision, Recall, F1, and Confusion Matrix calculator
│   └── responder.py           # Contextual response synthesizer engine
│
├── database/
│   ├── __init__.py
│   └── db.py                  # SQLite database manager (CRUD & Analytics)
│
├── templates/
│   ├── base.html              # Core layout, navbar, footer, and CDN bundles
│   ├── index.html             # Dashboard with KPI cards and Chart.js graphs
│   ├── analyze.html           # Live ticket analysis form with 1-click presets
│   ├── history.html           # Historical tickets table with search/filter & modal
│   └── performance.html       # Side-by-side ML evaluation & Confusion Matrix
│
├── static/
│   ├── css/
│   │   └── style.css          # Custom Microsoft Azure-inspired styling
│   └── js/
│       ├── dashboard.js       # Chart.js initialization & dynamic updates
│       └── analyze.js         # Live AJAX ticket analysis & copy response logic
│
└── instance/
    └── supportai.db           # SQLite database file
```

---

## 7. Web Application Pages

### 1. Dashboard (`/`)
- **KPI Summary Cards:** Total Tickets, High Priority Count, Negative Sentiment Count, Most Common Category.
- **Interactive Visualizations:** Chart.js bar chart for category frequency, doughnut charts for priority and sentiment distributions.
- **Recent Tickets Feed:** Real-time log of the latest 5 triaged customer requests.

### 2. Analyze Ticket (`/analyze`)
- **Natural Language Text Area:** Allows support staff or customers to enter issues.
- **1-Click Demo Presets:** Pre-loaded with realistic scenarios (Payment failure, Password reset, 500 Server error, Missing package, Subscription refund, Product praise).
- **Live Output Cards:** Predicted Category with Confidence Score bar, Priority SLA badge, Sentiment analysis, and Synthesized Response with a 1-click **"Copy to Clipboard"** button.

### 3. Ticket History (`/history`)
- **Search & Filters:** Search by keyword, filter by category, priority level, or sentiment tone.
- **Detailed Modal:** Click any ticket to inspect the full original query, metadata, and generated response.
- **Management:** Individual ticket deletion with asynchronous UI updates.

### 4. Model Performance (`/performance`)
- **Side-by-side Benchmark Table:** Accuracy, Precision, Recall, Macro F1, and Weighted F1 comparing Logistic Regression vs Random Forest.
- **Visual Confusion Matrix:** Color-coded matrix showing true positives and misclassifications across all 7 classes.
- **Per-Class Metrics:** Granular breakdown for each category.

---

## 8. Installation & How to Run

### Step 1: Navigate to Project Directory
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\supportai
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Model Training Pipeline (Optional - Auto-runs if models are missing)
```powershell
python ml/train.py
```

### Step 4: Run Automated Verification Tests
```powershell
python test_app.py
```

### Step 5: Start the Flask Application
```powershell
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 9. Demonstration & Viva Guide

### Example Test Scenarios to Demonstrate:

1. **Billing & Payment Issue (High Priority / Negative Sentiment):**
   - **Input:** *"My payment was deducted from my account but my order was not placed. I need this resolved urgently."*
   - **Output:** Category: `Billing / Payment` | Priority: `High` | Sentiment: `Negative`
   - **Generated Response:** Apologizes for the critical inconvenience, logs financial audit of gateway transactions, and commits to expedited 1-2 hour priority escalation.

2. **Account Lockout (High Priority / Negative Sentiment):**
   - **Input:** *"My account is locked due to too many failed login attempts. Please unlock it."*
   - **Output:** Category: `Account / Login` | Priority: `High` | Sentiment: `Negative`

3. **Technical Bug (High Priority / Negative Sentiment):**
   - **Input:** *"The web application is throwing 500 internal server error when uploading large files."*
   - **Output:** Category: `Technical Issue` | Priority: `High` | Sentiment: `Negative`

4. **Product Compliment (Low Priority / Positive Sentiment):**
   - **Input:** *"The product exceeded my expectations! Build quality is fantastic and easy to use."*
   - **Output:** Category: `Product Issue` | Priority: `Low` | Sentiment: `Positive`

---

## 10. Future Scope & Azure Cloud AI Architecture (AI-103 / AZ-2007)

SupportAI is architected with modular abstraction layers so it can seamlessly scale into an enterprise **Azure AI Agent**:

```
                              ┌────────────────────────┐
                              │  Customer Ticket Input │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │    SupportAI Agent     │
                              │  (Azure AI Agent Host) │
                              └───────────┬────────────┘
                                          │
               ┌──────────────────────────┼──────────────────────────┐
               ▼                          ▼                          ▼
 ┌───────────────────────────┐ ┌─────────────────────┐ ┌───────────────────────────┐
 │   Azure Machine Learning  │ │   Azure AI Search   │ │      Azure OpenAI         │
 │  (Custom Fine-Tuned Model │ │ (RAG Knowledge Base │ │ (gpt-4o-mini dynamic grounded│
 │   for Category/Priority)  │ │  of Support Manuals)│ │  response synthesis)      │
 └───────────────────────────┘ └─────────────────────┘ └───────────────────────────┘
               │                          │                          │
               └──────────────────────────┼──────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Automated Resolution / │
                              │ Azure Logic App Ticket │
                              └────────────────────────┘
```

1. **Azure OpenAI Integration (AI-103):** Replace template synthesis with grounded generative responses using Retrieval-Augmented Generation (RAG) linked to product manuals in **Azure AI Search**.
2. **Azure Event Grid / Service Bus (AZ-104):** Ingest tickets asynchronously from email, WhatsApp, and Microsoft Teams.
3. **Azure Cosmos DB:** Globally distributed NoSQL store for high-throughput ticket ingestion.
4. **Human-in-the-Loop Feedback:** Allow support agents to thumbs-up/down generated responses, creating an active learning loop for periodic model re-training.

---

## 11. Limitations of Current Prototype

- **Demonstration Dataset Scope:** Model is trained on 301 structured examples; production deployments would fine-tune on historical enterprise ticketing databases (100k+ records).
- **Rule-Guided Priority:** Heuristic urgency keyword boosts are combined with ML to ensure safety on mission-critical keywords.
- **Language Support:** English-only in prototype; enterprise deployment would incorporate multilingual tokenizers (e.g., Azure AI Language translation).
