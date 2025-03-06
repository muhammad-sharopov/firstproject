import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit app title
st.title('Student Depression Prediction')

# Load dataset
data = pd.read_csv("Student Depression Dataset.csv")

# Show the data info
st.write('Data Info:')
st.write(data.info())

# Describe the data
st.write('Data Summary:')
st.write(data.describe())

# Show the data types
st.write('Data Types:')
st.write(data.dtypes)

# Show the unique counts
st.write('Unique Counts:')
st.write(data.nunique())

# Show the mode of the data
st.write('Mode:')
st.write(data.mode().iloc[0])

# Handle missing data in 'Financial Stress'
data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)

# Display missing data after imputation
st.write('Missing Data After Imputation:')
st.write(data.isnull().sum())

# Gender distribution pie chart
st.write('### Gender Distribution')
gender_counts = data['Gender'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
       colors=['#66b3ff', '#ff9999'], shadow=True, explode=(0.05, 0), textprops={'fontsize': 14})
ax.set_title('Gender Distribution', fontsize=16, fontweight='bold', pad=20)
st.pyplot(fig)

# Age distribution histogram
st.write('### Age Distribution')
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data['Age'], bins=15, color='skyblue', edgecolor='black')
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.grid(True)
st.pyplot(fig)

# Correlation matrix heatmap
st.write('### Correlation Matrix of Numerical Features')
numerical_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.set_style("white")
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    linecolor='black',
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 10, 'weight': 'bold'},
    ax=ax
)
ax.set_title('Correlation Matrix of Numerical Features', fontsize=18, fontweight='bold', pad=20)
st.pyplot(fig)

# Boxplot for top features correlated with Depression
st.write('### Boxplot for Top Features Correlated with Depression')
correlation_matrix = data.select_dtypes(include='number').corr()
top_features_corr = correlation_matrix['Depression'].abs().sort_values(ascending=False).head(6).index
top_features_corr = top_features_corr.drop('Depression')

fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=data[top_features_corr], palette='Set2', ax=ax)
ax.set_title('Boxplot for Top Features Correlated with Depression', fontsize=16, fontweight='bold', color='darkslategray')
ax.set_xlabel('Features', fontsize=14, fontweight='bold', color='darkslategray')
ax.set_ylabel('Values', fontsize=14, fontweight='bold', color='darkslategray')
st.pyplot(fig)

# Handle ordinal mapping
ordinal_mapping = {
    'Sleep Duration': {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 0},
    'Dietary Habits': {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 0}
}

for col, mapping in ordinal_mapping.items():
    data[col] = data[col].map(mapping)

# Handle binary columns
binary_columns = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0}) 

# Show data head
st.write('### Processed Data')
st.write(data.head())

# Drop unnecessary columns
data = data.drop(columns=['id','Age', 'Degree', 'Profession','Work Pressure','City', 'Gender'])

# Prepare data for model
X = data.drop(columns=['Depression']) 
y = data['Depression'] 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model initialization
models = {
    'Random Forest': RandomForestClassifier(n_estimators=80, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

# Results dataframe
results = pd.DataFrame(columns=['Model', 'Train ROC AUC', 'Test ROC AUC'])

# Training models and displaying cross-validation results
st.write('### Model Training and Evaluation')
for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    results = pd.concat([results, pd.DataFrame({
        'Model': [name],
        'Train ROC AUC': [train_roc_auc],
        'Test ROC AUC': [test_roc_auc]
    })], ignore_index=True)
    
    cross_validation_scores = cross_val_score(estimator=model,
                                              X=X_train,
                                              y=y_train,
                                              scoring='accuracy',
                                              n_jobs=-1,
                                              cv=5
                                              )
    st.write(f'Cross Validation {name}: {cross_validation_scores.mean()}')

st.write(results)

# ROC curve
st.write('### ROC Curve')
fig, ax = plt.subplots(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='black', linestyle='--')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.legend(loc='lower right')
ax.grid(True)

st.pyplot(fig)

# Feature importance plot
st.write('### Feature Importance')
model = RandomForestClassifier(n_estimators=80, random_state=42)
model.fit(X_train, y_train)
importance = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='purple')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importance')
ax.invert_yaxis() 
st.pyplot(fig)

# Prediction probability histogram
st.write('### Prediction Probability Histogram')
y_prob = model.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6, color='blue', label='Actual Yes')
ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6, color='red', label='Actual No')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Probability Histogram')
ax.legend(loc='upper center')
st.pyplot(fig)
