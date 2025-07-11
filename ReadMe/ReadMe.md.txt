# 🩸 Breast Cancer Wisconsin Classification Project

## 📊 Project Objective
Accurately classify tumors as **malignant** or **benign** using the Breast Cancer Wisconsin dataset. The goal is to maximize recall for malignant cases (minimize false negatives) while ensuring the model remains explainable, robust, and production-ready.

---

## 📈 Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** 569
- **Features:** 30 numeric features (cell nuclei properties)
- **Target classes:** 0 = Malignant, 1 = Benign

---

## 🔍 Exploratory Data Analysis (EDA)
- Slight class imbalance (~37% malignant).
- Strong correlations: radius, perimeter, and area are highly related.
- Outliers detected & capped to reduce distortion.
- Skewness handled with log transformations on heavily skewed features.

**Key EDA Plots:**
- Class balance bar chart
- Feature histograms
- Boxplots for outliers
- Correlation heatmap

---

## 🧪 Modeling Pipeline
1. **Baseline Models:** Logistic Regression, KNN, Random Forest
2. **Resampling:** SMOTE used to address class imbalance.
3. **Cross-Validation:** Stratified K-Fold CV (n=10) to maintain class ratio.
4. **Model Selection:** Chose final model based on F1 and recall for malignant class.
5. **Threshold Tuning:** Optimized to reduce false negatives.
6. **Explainability:** SHAP used to interpret feature contributions.
7. **Feature Pruning:** Dropped features with zero mean absolute SHAP impact.

---

## 🧬 SHAP Feature Importance
**Top features driving predictions:**
- Worst texture
- Worst area
- Worst radius
- Worst concavity
- Mean concave points

**Dropped features:**
- `concave points error`
- `smoothness error`
(Mean absolute SHAP value = 0)

---

## ✅ Final Results

| Metric              | Value   |
|---------------------|---------|
| Accuracy (Test Set) | 97%     |
| F1-Score (Malignant)| ~0.97   |
| Recall (Malignant)  | 95%     |
| Confusion Matrix    | TP=42, FN=2, FP=1, TN=69 |

**Key Point:** High recall for malignant ensures we minimize undetected cancer cases.

---

## 📊 Tableau Dashboard (Optional)
Included:
- Target class breakdown
- Distributions for top SHAP features
- Correlation heatmap
- Final confusion matrix
- KPI highlight table for accuracy, recall, precision, F1

---

## 📑 Limitations & Next Steps
- Validate model on external dataset to test real-world generalization.
- Package as a deployable API for practical use.
- Explore advanced cost-sensitive models to balance false positives/negatives in line with medical risks.

---

## 👨‍💻 Project Files
- `notebooks/` — Full EDA, preprocessing, modeling, explainability.
- `reports/` — EDA summary, final model summary, Tableau export.
- `screenshots/` — Key plots: EDA, ROC-AUC, SHAP, confusion matrix.
- `README.md` — This file.

---

## 📌 How To Run
1. Clone repo
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks step-by-step

---

## 🙌 Author
*Built with love and SHAP values by [Your Name].*
