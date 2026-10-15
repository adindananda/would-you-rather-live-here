# Machine learning for social questions: 6 beginner lessons

Each script answers one question that matters for society, using one technique.
They're standalone: open one, read it from top to bottom, run it, then do the "TRY THIS" exercises at the bottom.

| # | Script | Social question | Technique | Type |
|---|--------|-----------------|-----------|------|
| 1 | `01_linear_regression_education_wages.py` | How much does an extra year of school raise wages? | Linear regression | predicts a number |
| 2 | `02_logistic_regression_food_insecurity.py` | Which households are at risk of going hungry? | Logistic regression | predicts yes/no + probability |
| 3 | `03_polynomial_regression_wealth_health.py` | Does life expectancy keep rising as countries get richer? | Polynomial regression | predicts a number (curved) |
| 4 | `04_kmeans_neighbourhood_types.py` | What types of neighbourhoods need different services? | K-means clustering | finds groups (no answer column) |
| 5 | `05_random_forest_student_dropout.py` | Which students are at risk of dropping out, and why? | Random forest classifier | predicts yes/no |
| 6 | `06_knn_air_quality_estimation.py` | How polluted is the air where there's no sensor? | K-nearest neighbours | predicts a number *and* yes/no |

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn
cd social_ml_lessons
python 01_linear_regression_education_wages.py
```

Each script prints its results and saves a chart to `outputs/`.

## The workflow every lesson follows

1. **Get data**: look at it before modelling.
2. **Split** it into train and test sets. Always grade the model on data it never saw.
3. **Fit** the model.
4. **Evaluate** it with metrics that fit the problem (R² and error for numbers; precision, recall and AUC for yes/no).
5. **Interpret** the result: turn the numbers into a sentence a policymaker can act on.
6. **Visualise** it, and question it.

## Why simulated data?

The data is generated in code with a fixed random seed, so:
- it runs offline and gives the same results every time;
- **you know the true answer**, so you can check whether the model found it (lesson 1 recovers the true $1.80/hour almost exactly);
- you can break things on purpose (add noise, remove features, shrink the sample) and watch what happens.

Once you're comfortable, swap in real data:
- Wages and education: [OECD Education at a Glance](https://www.oecd.org/education/education-at-a-glance/), or national labour surveys
- Food insecurity: [USDA Food Security Supplement](https://www.ers.usda.gov/data-products/food-security-in-the-united-states/)
- Wealth and health: [World Bank Open Data](https://data.worldbank.org/) (GDP per capita, life expectancy)
- Student dropout: UCI dataset *"Predict Students' Dropout and Academic Success"*
- Air quality: [OpenAQ](https://openaq.org/)

## Suggested study order

Do them in order 1 → 6. Each lesson introduces one new idea:
1. Fitting a line, train/test split, residuals
2. Probabilities, thresholds, precision vs recall
3. Underfitting vs overfitting (the most important idea in ML)
4. Unsupervised learning, scaling, choosing K
5. Ensembles, feature importance, interactions
6. Distance-based learning, cross-validation, coverage gaps
