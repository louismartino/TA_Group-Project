# Crowdfunding Success Prediction from Kickstarter Blurbs

This repository contains a group project on predicting whether a crowdfunding campaign will succeed using campaign metadata and text from the project `name` and `blurb`.

The work moves from exploratory analysis and hand-built NLP features to embedding-based baselines, DistilBERT models, and a small web app that scores and rewrites campaign blurbs.

## Overview

The main dataset is `data/dataset.csv`. Each row is a campaign with fields such as:

- `name`
- `blurb`
- `state`
- `duration`
- `country`
- `category.parent_name`
- `category.name`
- `goal_usd`
- `pledged_usd`
- `backers_count`
- `CCI_index`

The core prediction task used across the repo is binary classification:

- `successful` -> `1`
- everything else -> `0` in the feature engineering script, though the modeling notebooks typically restrict to `successful` and `failed`

## Repository Structure

```text
.
├── data/
│   ├── dataset.csv
│   ├── dataset_raw.csv
│   ├── CCI.csv
│   ├── eda/
│   └── features/
├── feature_engineering/
├── baseline/
├── embeddings/
├── category_prediction/
├── LLM/
├── webapp/
│   ├── backend/
│   └── frontend/
├── improving_blurbs.ipynb
└── requirements.txt
```

### `data/`

- `dataset.csv`: main processed dataset used throughout the project.
- `dataset_raw.csv`: raw version kept alongside the processed dataset.
- `CCI.csv`: Consumer Confidence Index data used to create `CCI_index`.
- `eda/data_analysis.ipynb`: exploratory analysis of state, country, category, duration, goals, pledged amounts, backers, blurb length, and CCI.
- `eda/eda_slides.ipynb`: lighter notebook version prepared for presentation.
- `features/train.csv`, `val.csv`, `test.csv`: saved splits after feature engineering.
- `features/features_scale.txt`: numeric features scaled before modeling.
- `features/features_no_scale.txt`: categorical dummy columns and other unscaled features.

### `feature_engineering/`

- `feature_engineering.py`: main preprocessing script. It creates the target, category and country dummies, text-derived features, train/validation/test splits, and writes the outputs to `data/features/`.
- `nlp_features.ipynb`: notebook for modeling with the engineered NLP and structured features.

### `baseline/`

- `anchor_embeddings.ipynb`: embedding-based baseline using sentence embeddings and anchor similarity for success prediction.

### `embeddings/`

- `train_embeddings.pt`, `test_embeddings.pt`: saved embedding tensors used by the anchor-based experiments.

### `category_prediction/`

- `category_prediction.ipynb`: separate task that predicts campaign category from blurb embeddings using a two-level anchor similarity approach.

### `LLM/`

This folder contains the main model experiments:

- `nn_from_scratch.ipynb`: feedforward baselines using structured features only, text only, and text + structured inputs.
- `bert_frozen.ipynb`: frozen DistilBERT embeddings with logistic regression.
- `bert_fine_tuning.ipynb`: text-only DistilBERT fine-tuning for `successful` vs `failed`.
- `joint_finetuning.ipynb`: joint DistilBERT + structured features model.
- `ci_delong_distilbert_auc.ipynb`: paired DeLong test comparing text-only fine-tuned DistilBERT against the joint model.
- `distilbert_results/` and `joint_distilbert_structured_results/`: saved model checkpoints and tokenizer files.

### `webapp/`

The repo also includes a working demo app.

- `backend/main.py`: FastAPI app for scoring campaigns, generating recommendations, rewriting blurbs, and serving a simple leaderboard.
- `backend/bert_scorer.py`: inference wrapper for the joint DistilBERT + structured-features model.
- `backend/data/`: local copies of feature lists and train/validation/test tables used for inference.
- `backend/.env.example`: required environment variables for the backend.
- `backend/Dockerfile`: Dockerfile for the backend service.
- `frontend/`: Next.js frontend for entering project details, viewing scores and feature breakdowns, comparing rewritten blurbs, and submitting to the leaderboard.

### Root notebook

- `improving_blurbs.ipynb`: LLM-based rewriting experiment that rewrites blurbs with Gemini and rescales them with the joint model to measure predicted lift.

## Main Experiments / Models

The repo covers several modeling approaches:

1. Engineered NLP + structured features
   - Features include `blurb_length`, `sentiment_score`, `readability_score`, `name_blurb_similarity`, `log_goal`, `CCI_per_goal`, `z-score_log_goal`, category dummies, and country dummies.

2. Embedding baselines
   - Sentence embeddings are used for anchor-based similarity methods in both success prediction and category prediction.

3. DistilBERT text models
   - Frozen DistilBERT embeddings with a logistic regression classifier.
   - Fine-tuned DistilBERT on campaign text (`name + blurb`).

4. Joint text + structured model
   - DistilBERT text representation concatenated with a 29-dimensional structured feature vector.
   - This is the model reused in the backend and web app.

5. LLM rewriting experiment
   - Gemini rewrites blurbs using rules derived from the project analysis, then the rewritten blurbs are rescored with the joint model.

## Data / Features

The feature engineering script creates:

- Scaled features:
  - `duration`
  - `CCI_index`
  - `blurb_length`
  - `sentiment_score`
  - `readability_score`
  - `name_blurb_similarity`
  - `log_goal`
  - `CCI_per_goal`
- Unscaled features:
  - category dummies (`cat_*`)
  - country dummies (`country_*`)
  - `z-score_log_goal`

The text-derived features come from:

- VADER sentiment for `sentiment_score`
- `textstat` for readability
- spaCy similarity between project `name` and `blurb`

## Setup / Installation

### Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

The root `requirements.txt` is mainly for the notebooks and data work.

### Backend setup

From `webapp/backend/`:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
cp .env.example .env
```

Then set:

- `GEMINI_API_KEY`
- optionally `HF_MODEL_REPO` if you do not want to use the default `RoseCymbler/kickstarter-joint-bert`

### Frontend setup

From `webapp/frontend/`:

```bash
npm install
```

If needed, set `NEXT_PUBLIC_API_URL`. If it is not set, the frontend defaults to `http://localhost:8000`.

## How To Run

### Generate engineered features

Run the preprocessing script from inside `feature_engineering/` so its relative paths resolve correctly:

```bash
cd feature_engineering
python feature_engineering.py
```

This writes:

- `data/features/train.csv`
- `data/features/val.csv`
- `data/features/test.csv`
- `data/features/features_scale.txt`
- `data/features/features_no_scale.txt`

### Run notebooks

Open the notebooks you want to reproduce in Jupyter or VS Code. The main ones are:

- `data/eda/data_analysis.ipynb`
- `feature_engineering/nlp_features.ipynb`
- `baseline/anchor_embeddings.ipynb`
- `category_prediction/category_prediction.ipynb`
- `LLM/bert_frozen.ipynb`
- `LLM/bert_fine_tuning.ipynb`
- `LLM/joint_finetuning.ipynb`
- `LLM/ci_delong_distilbert_auc.ipynb`
- `improving_blurbs.ipynb`

Some notebooks are written so they can load the saved checkpoints already committed in the repo instead of retraining from scratch.

### Run the backend

From `webapp/backend/`:

```bash
uvicorn main:app --reload
```

The API exposes:

- `POST /score`
- `POST /rewrite`
- `POST /submit`
- `GET /leaderboard`
- `POST /leaderboard/reset`

### Run the frontend

From `webapp/frontend/`:

```bash
npm run dev
```

Then open `http://localhost:3000`.

## Results / Evaluation

The repo reports validation metrics directly inside the notebooks. A few key numbers:

- `LLM/nn_from_scratch.ipynb`
  - structured-only baseline: ROC-AUC `0.8273`
  - text-only embedding + MLP: ROC-AUC `0.7321`
  - text + structured embedding + MLP: ROC-AUC `0.8032`

- `LLM/bert_frozen.ipynb`
  - frozen DistilBERT text only: ROC-AUC `0.7975`
  - frozen DistilBERT + tabular features: ROC-AUC `0.8474`

- `LLM/bert_fine_tuning.ipynb`
  - fine-tuned DistilBERT text only: ROC-AUC `0.8391`

- `LLM/joint_finetuning.ipynb`
  - joint DistilBERT + structured features: ROC-AUC `0.8570`

- `LLM/ci_delong_distilbert_auc.ipynb`
  - compares the text-only fine-tuned DistilBERT model against the joint model with a paired DeLong test

- `improving_blurbs.ipynb`
  - for a sample of 100 rewritten campaigns, mean predicted success probability rises from `0.3853` to `0.6467`
  - `83/100` campaigns improve and `33` cross the `0.5` success threshold after rewriting

These numbers are notebook outputs, not a separate benchmark script.

## Web App

The web app is a simple interface around the joint model and the rewrite workflow.

What it does:

- takes a project name, blurb, category, country, goal, and duration
- predicts success probability
- shows a feature breakdown for sentiment, readability, blurb length, and name-blurb coherence
- gives short recommendations based on those features
- rewrites the blurb with Gemini and compares before/after scores
- lets users submit rewritten blurbs to an in-memory leaderboard

What it does not do:

- persist leaderboard entries across restarts
- return real similar projects yet (`similar_projects` is currently an empty list in the API)

## Notes / Limitations

- Most of the project work lives in notebooks, so reproduction is notebook-driven rather than packaged as a single training pipeline.
- The backend needs a valid `GEMINI_API_KEY` to start because the rewrite endpoint is enabled at app startup.
- The backend downloads model weights from Hugging Face by default, so first-time startup may require internet access.
- The root `requirements.txt` includes a couple of standard-library names (`os`, `warnings`), so it may need cleanup if you want a stricter environment file.
- Some experiments evaluate only on the saved train/validation/test splits already committed in the repo.
- The leaderboard is stored in memory only.
