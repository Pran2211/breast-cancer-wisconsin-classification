#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer

bcd = load_breast_cancer(as_frame=True)
df = bcd.frame


# In[2]:


df.head()


# In[3]:


df.isna().sum()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df['target'].value_counts(normalize = True)
#0 - Malignant, 1 - Benign


# In[8]:


df.duplicated().sum()


# In[9]:


num_cols = [col for col in df if col != 'target']
for cols in num_cols:
    plt.figure(figsize = (20, 10))
    sns.boxplot(df[cols])
    plt.title(f"Outliers Detection of {cols}")
    plt.show()


# In[10]:


for cols in num_cols:
    if df[cols].skew() >= 1:
        print(f"Skew ({cols}): {df[cols].skew()}")


# In[11]:


for cols in num_cols:
    if df[cols].skew() >= 1:
        print(f"Minimum ({cols}): {df[cols].min()}")


# In[12]:


for cols in num_cols:
    if df[cols].skew() >= 1:
        df[cols] = np.log1p(df[cols])


# In[13]:


for cols in num_cols:
    if df[cols].skew() >= 1:
        print(f"Skew ({cols}): {df[cols].skew()}")


# In[14]:


# Outliers capped using IQR; recheck skewness afterward
Q3 = df.quantile(0.75)
Q1 = df.quantile(0.25)
IQR = Q3 - Q1

lower = Q1 - (1.5 * IQR)
upper = Q3 + (1.5 * IQR)

df[num_cols] = df[num_cols].clip(lower = lower, upper = upper, axis = 1)


# In[15]:


#Recheck Outliers - Result: Removed Successfully! 

for cols in num_cols:
    plt.figure(figsize = (20, 10))
    sns.boxplot(df[cols])
    plt.title(f"Outliers Detection (2nd): {cols}")
    plt.show()


# In[16]:


#Skew Re-Checking

for cols in num_cols:
    if df[cols].skew() >= 1:
        print(f"Skew ({cols}): {df[cols].skew()}")


# In[17]:


for cols in num_cols:
    print(f"Skew ({cols}): {df[cols].skew()}")


# In[18]:


#Great! Removing Outliers First corrected the skew by itself!
#BiVariate Analysis: How two Features Interact with the Target!
for cols in num_cols:
    plt.figure(figsize = (20, 10))
    sns.boxplot(x='target', y=cols, data=df)
    plt.title(f"Bi-Variate Analysis between {cols} and Target")
    plt.show()


# In[19]:


'''

Exploratory Data Analysis: Feature vs. Target (Malignant vs. Benign)
This analysis examines the distribution of various features across two target groups: '0' (Benign) and '1' (Malignant), 
as depicted by the provided boxplots.

1. Mean Radius vs. Target:
The mean radius is noticeably higher in the Malignant group (~17) compared to the Benign group (~12). 
Malignant tumors show a broader range (min: 11, max: 22), while Benign tumors range from 8 to 17. 
The interquartile range (IQR) in the Malignant group spans 15–19, indicating a tendency toward larger tumor sizes in cancerous cases.

2. Mean Texture vs. Target:
The mean texture also shows a clear distinction. 
The Malignant group has a higher mean texture (~21) compared to the Benign group (~18). 
The range for malignant tumors is wider (min: 16, max: 28) than for benign tumors (min: 10, max: 24). 
The IQR for malignant cases is approximately 19-23, suggesting that malignant tumors tend to have rougher textures.

3. Mean Perimeter vs. Target:
Similar to mean radius, the mean perimeter is significantly higher for the Malignant group (~110) compared to the Benign group (~78). 
Malignant tumors exhibit a larger spread (min: ~70, max: ~150) than benign tumors (min: ~50, max: ~100). 
The IQR for malignant cases, spanning roughly 95-125, reinforces that larger perimeters are associated with malignancy.

4. Mean Area vs. Target:
The mean area is substantially larger in the Malignant group (~900) compared to the Benign group (~500). 
The range for malignant tumors is extensive (min: ~350, max: ~1600), while benign tumors are more concentrated (min: ~200, max: ~800). 
The IQR for malignant tumors, ranging from approximately 700-1100, strongly indicates that larger areas are characteristic of malignant growths.

5. Mean Smoothness vs. Target:
Mean smoothness appears to be slightly higher in the Malignant group (~0.10) than in the Benign group (~0.09). 
Both groups show some overlap in their ranges, but the malignant cases generally have a slightly higher median and a wider upper quartile, 
suggesting a tendency towards more irregular cell borders.

6. Mean Compactness vs. Target:
The mean compactness is notably higher for the Malignant group (~0.14) compared to the Benign group (~0.08). Malignant tumors show a broader distribution (min: ~0.06, max: ~0.25) with a higher median, while benign tumors are more tightly clustered (min: ~0.04, max: ~0.14). The IQR for malignant cases (approximately 0.10-0.18) indicates greater cellular density in cancerous tissues.

7. Mean Concavity vs. Target:
Mean concavity is considerably higher in the Malignant group (~0.10) than in the Benign group (~0.03). The malignant distribution is much wider (min: ~0.01, max: ~0.20) and skewed towards higher values, with many outliers. The benign group is heavily concentrated near zero with a narrow range. This suggests that the presence and depth of concavities are strong indicators of malignancy.

8. Mean Concave Points vs. Target:
Similar to concavity, mean concave points are significantly higher in the Malignant group (~0.05) compared to the Benign group (~0.02). The malignant distribution is much broader (min: ~0.01, max: ~0.10) with a higher median, while the benign group is tightly clustered at lower values. This further emphasizes that the number of concave portions of the contour is a key differentiator.

9. Mean Symmetry vs. Target:
Mean symmetry shows a slight increase in the Malignant group (~0.18) compared to the Benign group (~0.17). While there's overlap, the malignant group's median is slightly higher, and it exhibits a wider upper range, suggesting that malignant cells might display more irregular or asymmetric shapes.

10. Mean Fractal Dimension vs. Target:
Mean fractal dimension appears to be slightly lower in the Malignant group (~0.06) than in the Benign group (~0.065), though the difference is less pronounced than other features. Both groups have similar ranges and IQRs, indicating this feature might be less discriminatory on its own.

Summary of Observations:
Features such as mean radius, mean texture, mean perimeter, mean area, mean compactness, mean concavity, and mean concave points show clear and substantial differences between the Malignant and Benign groups, with the malignant group consistently exhibiting higher values and often a wider spread. These features appear to be strong indicators for differentiating between the two target classes. Mean smoothness, mean symmetry, and mean fractal dimension show less distinct separation but still offer some insights into the characteristics of the tumors.
'''


# In[20]:


sns.pairplot(df)
plt.show()


# In[21]:


corr = df.corr()
plt.figure(figsize = (20, 10))
sns.heatmap(corr, cmap = 'coolwarm', annot = True)
plt.title("Co-Relation Matrix (Heatmap)")
plt.show()


# In[22]:


corr = pd.DataFrame(corr)


# In[23]:


corr


# In[24]:


corr.to_csv('Co-Relation Matrix Results', index = False)


# In[25]:


from sklearn.model_selection import train_test_split

x = df[num_cols]
y = df['target']
print(x.shape)
print(y.shape)


# In[26]:


from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
x_temp, x_val, y_temp, y_val = train_test_split(x, y, random_state = 42, test_size = 0.2)
x_train, x_test, y_train, y_test = train_test_split(x_temp, y_temp, random_state = 42, test_size = 0.25)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

smote = SMOTE(random_state = 42)
x_train_scaled_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter = 1000),
    'Random Forest': RandomForestClassifier(n_estimators = 100, max_depth = 100),
    'KNN': KNeighborsClassifier(n_neighbors = 100),
    'Decision Trees': DecisionTreeClassifier(max_depth = 100),
    'SVC': SVC(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(verbose = -1),
    'Ada Boost': AdaBoostClassifier(),
    'Cat Boost': CatBoostClassifier(verbose = 0)
}

for names, model in baseline_models.items():
    model.fit(x_train_scaled_resampled, y_train_resampled)
    preds = model.predict(x_val_scaled)
    accu = accuracy_score(y_val, preds)
    confu = confusion_matrix(y_val, preds)
    clsr = classification_report(y_val, preds)
    print(f"Model: {names}")
    print(f"Accuracy Score on val set: {accu}")
    print(f"Confusion matrix on val set: {confu}")
    print(f"Classification Report on val set: {clsr}")

best_model = LogisticRegression(max_iter = 1000)
best_model.fit(x_train_scaled_resampled, y_train_resampled)


# In[28]:


from sklearn.metrics import roc_curve, auc

y_proba = best_model.predict_proba(x_val_scaled)[:,0]
fpr, tpr, _ = roc_curve(y_val, y_proba, pos_label=0)
roc_auc = auc(fpr, tpr)

plt.figure(figsize = (20, 10))
plt.plot(fpr, tpr, label = (f"ROC Curve (AUC = {roc_auc:.3f}"), color = 'red')
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.grid()
plt.show()


# In[29]:


from sklearn.metrics import precision_recall_curve

y_proba = best_model.predict_proba(x_val_scaled)[:, 0]
precision, recall, thresholds = precision_recall_curve(y_val, y_proba, pos_label=0)
pr_auc = auc(recall, precision)

plt.figure(figsize = (20, 10))
plt.plot(recall, precision, label = f"Precision-Recall Curve (AUC = {pr_auc: .3f}", color = 'darkorange')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()


# In[30]:


for thresh in [0.2, 0.5, 0.8]:
    preds = (y_proba < thresh).astype(int)
    print(f"Threshold {thresh}:")
    print(classification_report(y_val, preds))


# In[31]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
import warnings
warnings.filterwarnings('ignore')
param_grids_lr = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.003, 0.005, 0.01, 0.1, 1],
    'solver': ['liblinear', 'saga'],
    'l1_ratio': [0, 1, 2, 3, 5],
    'max_iter': [100, 200, 300, 500, 1000],
    'fit_intercept': [True, False]
}

grid = GridSearchCV(cv = 10, param_grid = param_grids_lr, verbose = 1, estimator = best_model, scoring = 'f1', n_jobs = -1)
grid.fit(x_train_scaled, y_train)
print(f"Best Params: {grid.best_params_}")
print(f"Best F1 Score: {grid.best_score_}")


# In[32]:


final_model = LogisticRegression(C = 0.1, fit_intercept = True, max_iter = 100, penalty = 'l2', solver = 'saga')
final_model.fit(x_train_scaled_resampled, y_train_resampled)
preds = final_model.predict(x_val_scaled)
accu = accuracy_score(y_val, preds)
confm = confusion_matrix(y_val, preds)
clsfr = classification_report(y_val, preds)

print(f"Accuracy (Final Model) on Val Set: {accu}")
print(f"Classification Report (Final Model) on Val Set: {clsfr}")
print(f"Confusion Matrix (Final Model) on Val Set: {confm}")


# In[118]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import roc_curve, auc
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

f1s = []
aucs = []
for train_idx, val_idx in kfold.split(x_train_scaled, y_train):
    x_train_folds, y_train_folds = x_train_scaled[train_idx], y_train.iloc[train_idx]
    x_val_folds, y_val_folds = x_train_scaled[val_idx], y_train.iloc[val_idx]

    x_train_res, y_train_res = SMOTE().fit_resample(x_train_folds, y_train_folds)
    final_model.fit(x_train_res, y_train_res)
    preds_folds = final_model.predict(x_val_folds)
    f1s.append(f1_score(y_val_folds, preds_folds))
    
    y_proba_folds = final_model.predict_proba(x_val_folds)[:, 0]
    fpr, tpr, _ = roc_curve(y_val_folds, y_proba_folds, pos_label = 0)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    plt.plot(fpr, tpr, lw=1.5, label=f'Fold AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'Red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC-AUC Curvee Per Fold)")
plt.legend()
plt.grid(True)
plt.show()
print(f"Average F1 Score Across Each Fold: {np.mean(f1s): .3f}")
print(f"Average AUC Score Across Each Fold: {np.mean(aucs): .3f}")


# In[122]:


import shap

explainer = shap.Explainer(final_model, x_train_scaled)
shap_values = explainer(x_train_scaled)

# Then this will match dimensions:
shap.summary_plot(shap_values, x_train_scaled)


# In[135]:


explainer = shap.KernelExplainer(final_model.predict_proba, shap.sample(x_train_scaled, 100))
shap_values = explainer.shap_values(x_train_scaled)

shap.summary_plot(shap_values[..., 0], x_train_scaled)


# In[139]:


import numpy as np
import pandas as pd

shap_values_class0 = shap_values[..., 0]

# Take mean absolute SHAP value for each feature:
mean_abs_shap = np.abs(shap_values_class0).mean(axis=0)

# If you have feature names:
shap_scores = pd.DataFrame({
    'feature': x_train.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values(by='mean_abs_shap', ascending=False)

print(shap_scores)


# In[167]:


x_train_reduced = np.delete(x_train_scaled, [17, 14], axis=1)
x_test_reduced = np.delete(x_test_scaled, [17, 14], axis = 1)
x_val_scaled_reduced = np.delete(x_val_scaled, [17, 14], axis=1)
x_train_reduced_res, y_train_res = SMOTE().fit_resample(x_train_reduced, y_train)
final_model.fit(x_train_reduced_res, y_train_res)
preds_val_FR = final_model.predict(x_val_scaled_reduced)
accu = accuracy_score(y_val, preds_val_FR)
clfr = classification_report(y_val, preds_val_FR)
confm = confusion_matrix(y_val, preds_val_FR)
assert x_train_reduced.shape[1] == x_val_scaled_reduced.shape[1]
print(f"Accuracy After Feature Reduction: {accu:.2f}")
print(f"Classification Report After Feature Reduction: {clfr:}")
print(f"Confusion Matrix After Feature Reduction: {confm:}")


# In[203]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
f1s = []
precs = []
recs = []

for train_idx, val_idx in kfold.split(x_train_reduced, y_train):

    x_train_fold, y_train_fold = x_train_reduced[train_idx], y_train.iloc[train_idx]
    x_val_fold, y_val_fold = x_train_reduced[val_idx], y_train.iloc[val_idx]
    x_train_fold_res, y_train_fold_res = SMOTE().fit_resample(x_train_fold, y_train_fold)

    final_model.fit(x_train_fold_res, y_train_fold_res)
    preds_folds = final_model.predict(x_val_fold)
    
    accu = accuracy_score(y_val_fold, preds_folds)
    clfr = classification_report(y_val_fold, preds_folds)
    confm = confusion_matrix(y_val_fold, preds_folds)
    f1s.append(f1_score(y_val_fold, preds_folds))
    precs.append(precision_score(y_val_fold, preds_folds))
    recs.append(recall_score(y_val_fold, preds_folds))
print(f"Accuracy Across Each Fold: {np.mean(accu):.2f}")
print(f"Classification Report Across Each Fold: {clfr}")
print(f"Confusion Matrix Each Fold: {confm}")
print(f"Mean F1 Scores Across Folds: {np.mean(f1s): .2f}")

print("F1s Across Folds:", [f"{score:.2f}" for score in f1s])
print("Recalls Across Folds:", [f"{score:.2f}" for score in recs])
print("Precisions Across Folds:", [f"{score:.2f}" for score in precs])


# In[207]:


final_model.fit(x_train_reduced_res, y_train_res)
final_preds = final_model.predict(x_test_reduced)
accu = accuracy_score(y_test, final_preds)
clfr = classification_report(y_test, final_preds)
confm = confusion_matrix(y_test, final_preds)

print(f"Accuracy Score on ✅Final Test Set: {accu:.2f}")
print(f"Confusion Matrix on ✅Final Test Set: {confm}")
print(f"Classification Report on ✅Final Test Set: {clfr}")


# In[ ]:




