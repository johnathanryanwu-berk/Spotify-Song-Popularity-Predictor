# Spotify Song Popularity Prediction

An end-to-end machine learning system that predicts whether a song will become popular (Top 10%) on Spotify using audio features and metadata.

This project compares linear and nonlinear modeling strategies, implements production-style ML pipelines, and deploys an interactive Spotify-themed dashboard for real-time inference.

Live Dashboard:  
https://spotify-song-popularity-prediction.onrender.com/predict

---

# Business Problem

Spotify assigns each track a popularity score (0–100).  
What actually drives a song into the top 10% of popularity?

This project answers:

Can we predict hit songs before they become hits using only audio characteristics and metadata?

Stakeholders who benefit:
- Artists and producers → optimize song characteristics
- Record labels → allocate marketing budget strategically
- Streaming platforms → improve recommendation ranking
- Product teams → understand feature importance behind engagement

---

# Dataset

Source: Kaggle – Spotify Tracks Dataset  
Size: ~114,000 tracks  

Features:
- Audio: danceability, energy, loudness, tempo, valence, acousticness, instrumentalness
- Metadata: genre, key, mode, time signature, explicit
- Duration (milliseconds)

Target Variable:
- popular = 1 if popularity is in the top 10%
- popular = 0 otherwise

Class imbalance: ~89% non-popular

---

# Modeling Strategy

Two parallel modeling pipelines were implemented to compare supervised feature selection versus unsupervised dimensionality reduction.

---

## LASSO-Based Pipeline (Feature Selection)

One-Hot Encoding  
→ Standardization  
→ L1 Logistic Regression (feature selection)  
→ CART / Random Forest / XGBoost  

Why LASSO:
- Reduces multicollinearity
- Removes noisy features
- Improves interpretability
- Reduces overfitting

---

## PCA-Based Pipeline (Dimensionality Reduction)

Ordinal Encoding  
→ Standardization  
→ PCA (retain 95% variance)  
→ CART / Random Forest / XGBoost  

Why PCA:
- Handles correlated features
- Compresses feature space
- Enables comparison with sparse models

---

# Model Performance

Due to severe class imbalance (~89% non-popular tracks), accuracy is not a meaningful metric.  
A model predicting all songs as “not popular” achieves ~0.89 accuracy.

Therefore, evaluation focuses on:

- ROC-AUC (ranking quality across thresholds)
- Recall (ability to detect actual hit songs)

---

## Performance Comparison

| Model | ROC-AUC | Recall (Hit Class) |
|-------|---------|-------------------|
| LASSO Logistic | 0.65 | 0.00 |
| CART + LASSO | 0.72 | 0.08 |
| Random Forest + LASSO | **0.81** | **0.19** |
| XGBoost + LASSO | 0.71 | 0.006 |

---

## Interpretation

### Baseline: LASSO Logistic Regression
- ROC-AUC: 0.65  
- Recall: 0.00  

The linear baseline can rank songs moderately well but fails to identify any hit songs at the default classification threshold.  
This highlights the limitation of linear decision boundaries in modeling complex audio interactions.

---

### CART
- ROC-AUC: 0.72  
- Recall: 0.08  

CART improves both ranking and minority class detection by introducing nonlinear splits.  
However, recall remains limited, indicating insufficient modeling capacity for deeper feature interactions.

---

### Random Forest (Best Model)
- ROC-AUC: 0.81  
- Recall: 0.19  

Random Forest trained on LASSO-selected features delivers the strongest overall performance.

Key improvements:
- 25% relative increase in ROC-AUC over baseline
- Detects 19% of actual hit songs (vs 0% for baseline)
- Captures nonlinear feature interactions (e.g., loudness × energy, genre × instrumentalness)

This model provides the best balance between ranking quality and actionable hit detection.

---

### XGBoost
- ROC-AUC: 0.71  
- Recall: 0.006  

Despite its theoretical advantages, XGBoost adopted an overly conservative strategy under class imbalance, failing to surface meaningful numbers of hit songs.

This suggests that in this dataset, bagging (Random Forest) outperforms boosting for minority class detection.

---

## Business Implication

In a music industry setting, missing a potential hit (false negative) is significantly more costly than incorrectly flagging a non-hit.

Random Forest reduces false negatives substantially compared to the baseline and provides interpretable feature importance rankings, enabling:

- Data-informed A&R decisions
- Marketing allocation optimization
- Feature-level insight into hit dynamics

For production deployment, Random Forest + LASSO is the recommended model.

---

# Key Insight

Linear models could rank songs but failed to identify actual hits.

Tree-based ensemble models significantly improved:
- Minority class detection
- Nonlinear interaction modeling
- Practical business usability

Despite XGBoost’s theoretical advantages, Random Forest performed best under heavy class imbalance.

---

# Interactive Dashboard

The project includes a Dash dashboard that allows users to:

- Adjust audio features
- Select trained model
- View real-time probability of popularity
- Compare model outputs

All models are deployed as serialized joblib pipelines.

Architecture:

Dash UI → FastAPI backend → ML pipeline inference

Routes:
- `/` → Dashboard
- `/api` → FastAPI
- `/api/docs` → Swagger documentation

---

# Tech Stack

- Python
- scikit-learn
- XGBoost
- pandas / numpy
- Dash
- FastAPI
- joblib
- Uvicorn
- Render (deployment)

---

# Run Locally

Install dependencies and run the full pipeline:

```bash
pip install -r requirements.txt

python -m src.data_prep \
  --input data/raw/spotify_dataset.csv \
  --output data/processed/spotify_processed.csv

python -m src.train --data data/processed/spotify_processed.csv

python -m src.evaluate --data data/processed/spotify_processed.csv

uvicorn main:app --host 0.0.0.0 --port 8000
```
---

## Future Improvements

- Threshold tuning to improve minority class recall
- Cost-sensitive learning for hit detection
- PR-AUC evaluation under heavy class imbalance
- Bayesian hyperparameter optimization
- Model calibration for probability reliability

---

## What This Project Demonstrates

- Handling severe class imbalance
- Feature selection vs dimensionality reduction tradeoffs
- Nonlinear ensemble modeling
- Production-ready ML pipelines
- Full-stack ML deployment (Dash + FastAPI)

---

## Author

Johnathan Wu  
UC Berkeley – Master of Analytics  
GitHub: https://github.com/johnathanryanwu-berk
