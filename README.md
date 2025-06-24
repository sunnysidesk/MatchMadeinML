# Match Made in ML – A Profile Matching Algorithm
_A machine learning system for profile matching using FAISS and BERT embeddings_

This machine learning project was developed as part of the graduate Machine Learning course at UC Davis by a team of three. It explores how artificial intelligence can improve online dating experiences through intelligent profile matching. 

We leverage **Facebook AI Similarity Search (FAISS)** and cosine similarity to build a high-performance recommender system using a publicly available OKCupid dataset (~60,000 profiles). Our solution ranks potential matches based on both structured and unstructured profile data.

## Project Objective

Design a scalable and accurate matching algorithm that pairs user profiles based on:
- Textual data (user bios and essays)
- Categorical features (e.g., job, religion)
- Numeric features (e.g., age, height)

## Methods

We implemented and compared three matching models:
1. **Baseline Cosine Similarity**
2. **FAISS Approximate Nearest Neighbor Search**
3. **Optimized Weighted FAISS** – our final model

The **Weighted FAISS model** used a grid search to tune the relative importance of each feature group:
- Essay = 2.0
- Categorical = 0.5
- Numerical = 0.5

## Results

- **MAP (Mean Average Precision):** 0.7699  
- **MRR (Mean Reciprocal Rank):** 0.9838

Our optimized Weighted FAISS model significantly outperformed the baseline, showing strong ranking quality while maintaining diversity and efficiency — critical for practical matchmaking applications.

## Insights

We discovered that:
- Essay embeddings contribute most to perceived compatibility
- Fine-tuning feature weights improves recommendation quality
- In matchmaking, **ranking order is crucial** — users may not scroll far, so highly compatible matches must appear early

This project demonstrates the power of advanced similarity search techniques in creating more relevant, personalized online dating experiences.

## Repository Contents

- `EDA_&_Preprocessing.ipynb` – Feature engineering and data prep
- `okcupid_Matching.ipynb` – Modeling pipeline and evaluation
- `Project Report.pdf` – Full write-up with methodology and findings

## Tools & Technologies

- Python, pandas, scikit-learn, FAISS, Hugging Face Transformers
- DistilBERT for essay embedding
- GridSearchCV for parameter optimization

## Authors

Developed by Sakshi Kumar, Amber Gonzalez-Pacheco, Abhay Padmanabhan @ UC Davis, 2025
