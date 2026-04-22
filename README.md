# Fraud Detection Capstone — Agentic AI System

**PG Diploma in Data Science & AI | IIT Bombay**

## Project overview
An end-to-end fraud detection system combining classical ML with an agentic AI investigation layer. The agent reasons over flagged transactions, produces explainable case reports, and mimics a real fraud analyst workflow.

## Dataset
[Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — Sparkov-simulated transactions with interpretable features.

## Project phases
- [x] Phase 1: Project setup
- [ ] Phase 2: EDA
- [ ] Phase 3: Preprocessing & feature engineering
- [ ] Phase 4: Model training (XGBoost + SMOTE)
- [ ] Phase 5: Agentic investigation layer (LangGraph + Claude)
- [ ] Phase 6: Evaluation & reporting

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your ANTHROPIC_API_KEY
```

## Structure
- `src/data/` — loading & preprocessing
- `src/models/` — training & evaluation
- `src/agents/` — agentic investigation layer
- `notebooks/` — exploratory analysis
- `data/raw/` — place dataset CSVs here (not committed to git)
