# 🎓 Student Performance & Predictive Analytics Dashboard

A **Streamlit** dashboard that trains a stacking ensemble (XGBoost + LightGBM + CatBoost → XGBoost meta-learner) on student data and surfaces real-time risk predictions with intervention suggestions.

---

## Features

| Section | What you get |
|---|---|
| 📊 Overview | KPI cards, gauges, outcome distribution, dropout rate by course |
| 🚨 At-Risk Students | Filterable high-risk table, low-engagement detection |
| 📈 Analytics | Feature importance, grade distributions, macroeconomic correlations |
| 🔍 Student Deep-Dive | Radar chart, probabilities, rule-based AI intervention tips per student |
| 🤖 Predict New Student | Form to input any student's attributes and get instant risk score |

---

## Quick Start

### 1 – Get the dataset

Download **`data.csv`** from either source and place it in this folder:

- **UCI / Kaggle (recommended):**  
  https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention  
  → File: `data.csv`

- **Kaggle competition (train.csv):**  
  https://www.kaggle.com/competitions/playground-series-s4e6/data  
  → Download `train.csv`, rename to `data.csv`

### 2 – Install dependencies

```bash
pip install -r requirements.txt
```

### 3 – Run locally

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.  
On first launch, click **"▶ Train Model Now"** in the sidebar.  
Training takes ~2–5 minutes (saved to `models/` for all future runs).

### 4 – Publish via Cloudflare (public URL)

#### Install `cloudflared`

| Platform | Command |
|---|---|
| macOS | `brew install cloudflare/cloudflare/cloudflared` |
| Linux | `curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb && sudo dpkg -i cloudflared.deb` |
| Windows | `winget install Cloudflare.cloudflared` |

#### Launch everything with one command

```bash
bash run.sh
```

This will:
1. Install dependencies
2. Start Streamlit on port 8501
3. Open a Cloudflare tunnel and print a public `*.trycloudflare.com` URL

> **No account or login required** — Cloudflare Quick Tunnels are ephemeral and free.

#### Alternative: ngrok

```bash
# terminal 1
streamlit run app.py

# terminal 2
ngrok http 8501
```

---

## File Structure

```
student_dashboard/
├── app.py              ← Streamlit UI (all pages)
├── model_trainer.py    ← Training pipeline & scoring logic
├── requirements.txt
├── run.sh              ← One-shot launcher + Cloudflare tunnel
├── data.csv            ← ⬅ YOU provide this (not included)
└── models/             ← Auto-created on first train
    ├── stacking_ensemble.joblib
    ├── label_encoder.joblib
    ├── feature_cols.joblib
    └── all_student_scores.csv
```

---

## Model Architecture

```
Data
 ├─ XGBoost   (Optuna-tuned)  ┐
 ├─ LightGBM  (Optuna-tuned)  ├─ 5-fold OOF probabilities
 └─ CatBoost  (Optuna-tuned)  ┘
                │
         Meta-learner
         XGBoost (Optuna-tuned)
                │
         Final Prediction
     (Dropout / Enrolled / Graduate)
```

Hyperparameters are taken directly from the Optuna-tuned values in the original notebook.

---

## Risk Levels

| Level | Dropout Probability |
|---|---|
| 🔴 High | > 60% |
| 🟡 Medium | 30–60% |
| 🟢 Low | < 30% |

Low-Engagement flag: `Enrolled` prediction **and** 0 units approved in 2nd semester.
