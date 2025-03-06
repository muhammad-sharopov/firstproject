import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Заголовок
st.title('Анализ депрессии студентов')

# Загрузка данных
DATA_PATH = "Student Depression Dataset.csv"
data = pd.read_csv(DATA_PATH)
st.write("## Первые строки датасета:")
st.write(data.head())

# Заполнение пропусков в 'Financial Stress'
data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)

# Визуализация распределения полов
st.subheader("Распределение полов")
fig, ax = plt.subplots()
gender_counts = data['Gender'].value_counts()
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
       colors=['#66b3ff', '#ff9999'], shadow=True, explode=(0.05, 0))
st.pyplot(fig)

# Гистограмма возрастов
st.subheader("Гистограмма возрастов")
fig, ax = plt.subplots()
ax.hist(data['Age'].dropna(), bins=15, color='skyblue', edgecolor='black')
ax.set_xlabel('Возраст')
ax.set_ylabel('Частота')
ax.set_title('Распределение возрастов')
st.pyplot(fig)

# Корреляционная матрица
st.subheader("Корреляционная матрица")
numerical_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)

# Топ-5 признаков по корреляции с 'Depression'
correlation_matrix = data.corr()
top_features_corr = correlation_matrix['Depression'].abs().sort_values(ascending=False).head(6).index
top_features_corr = top_features_corr.drop('Depression')
st.subheader("Топ-5 признаков по корреляции с депрессией")
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=data[top_features_corr], palette='Set2', ax=ax)
st.pyplot(fig)

# Кодирование категориальных данных
ordinal_mapping = {
    'Sleep Duration': {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 0},
    'Dietary Habits': {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 0}
}
for col, mapping in ordinal_mapping.items():
    data[col] = data[col].map(mapping)

data['Have you ever had suicidal thoughts ?'] = data['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
data['Family History of Mental Illness'] = data['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

# Удаление ненужных признаков
data = data.drop(columns=['id', 'Age', 'Degree', 'Profession', 'Work Pressure', 'City', 'Gender'])

# Разделение данных
X = data.drop(columns=['Depression'])
y = data['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Обучение моделей
models = {
    'Random Forest': RandomForestClassifier(n_estimators=80, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

results = pd.DataFrame(columns=['Модель', 'ROC AUC'])

st.subheader("ROC AUC для моделей")
fig, ax = plt.subplots(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    results = pd.concat([results, pd.DataFrame({'Модель': [name], 'ROC AUC': [roc_auc]})], ignore_index=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='black', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC-кривые')
ax.legend(loc='lower right')
st.pyplot(fig)

# Таблица результатов
st.write(results)

# Важность признаков
st.subheader("Важность признаков")
final_model = RandomForestClassifier(n_estimators=80, random_state=42)
final_model.fit(X_train, y_train)
importance = final_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Признак': features, 'Важность': importance}).sort_values(by='Важность', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Признак'], importance_df['Важность'], color='purple')
ax.set_xlabel('Важность')
ax.set_ylabel('Признаки')
ax.set_title('Значимость признаков')
ax.invert_yaxis()
st.pyplot(fig)

# Гистограмма вероятностей предсказаний
st.subheader("Гистограмма вероятностей предсказаний")
y_prob = final_model.predict_proba(X_test)[:, 1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6, color='blue', label='Депрессия')
ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6, color='red', label='Нет депрессии')
ax.set_xlabel('Предсказанная вероятность')
ax.set_ylabel('Частота')
ax.set_title('Гистограмма предсказаний')
ax.legend()
st.pyplot(fig)
