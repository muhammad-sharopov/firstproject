import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
@st.cache
def load_data():
    data = pd.read_csv("Student Depression Dataset.csv")
    return data

data = load_data()

# Показ информации о данных
st.title('Анализ данных о депрессии студентов')
st.write("Информация о данных:")
st.write(data.info())
st.write(data.describe())

# Обработка пропусков и заполнение медианой
data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)
st.write("Пропущенные значения после иммутатции:")
st.write(data.isnull().sum())

# График распределения по полу
st.subheader('Распределение по полу')
gender_counts = data['Gender'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
       colors=['#66b3ff', '#ff9999'], shadow=True, explode=(0.05, 0), textprops={'fontsize': 14})
ax.set_title('Распределение по полу', fontsize=16, fontweight='bold', pad=20)
st.pyplot(fig)

# Гистограмма возрастов
st.subheader('Распределение возрастов')
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data['Age'], bins=15, color='skyblue', edgecolor='black')
ax.set_title('Распределение возрастов')
ax.set_xlabel('Возраст')
ax.set_ylabel('Частота')
ax.grid(True)
st.pyplot(fig)

# Корреляционная матрица
st.subheader('Корреляционная матрица')
numerical_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, linecolor='black',
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 10, 'weight': 'bold'})
ax.set_title('Корреляционная матрица числовых признаков', fontsize=18, fontweight='bold', pad=20)
st.pyplot(fig)

# Топ-5 признаков по корреляции с депрессией
st.subheader('Топ-5 признаков, связанных с депрессией')
correlation_matrix = data.select_dtypes(include='number').corr()
top_features_corr = correlation_matrix['Depression'].abs().sort_values(ascending=False).head(6).index
top_features_corr = top_features_corr.drop('Depression')

fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=data[top_features_corr], palette='Set2', ax=ax)
ax.set_title('Топ признаки, связанные с депрессией', fontsize=16, fontweight='bold')
st.pyplot(fig)

# Преобразование данных
ordinal_mapping = {
    'Sleep Duration': {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 0},
    'Dietary Habits': {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 0}
}
for col, mapping in ordinal_mapping.items():
    data[col] = data[col].map(mapping)

binary_columns = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

data = data.drop(columns=['id', 'Age', 'Degree', 'Profession', 'Work Pressure', 'City', 'Gender'])
X = data.drop(columns=['Depression'])
y = data['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Определение моделей
models = {
    'Random Forest': RandomForestClassifier(n_estimators=80, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

# Оценка моделей
results = pd.DataFrame(columns=['Модель', 'Train ROC AUC', 'Test ROC AUC'])

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    results = pd.concat([results, pd.DataFrame({
        'Модель': [name],
        'Train ROC AUC': [train_roc_auc],
        'Test ROC AUC': [test_roc_auc]
    })], ignore_index=True)

st.write("Результаты моделей:")
st.write(results)

# ROC Curve
st.subheader('ROC-Кривая')
fig, ax = plt.subplots(figsize=(10, 8))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='black', linestyle='--')
ax.set_xlabel('Уровень ложных срабатываний')
ax.set_ylabel('Уровень истинных срабатываний')
ax.set_title('Кривая приема-отклонения (ROC)')
ax.legend(loc='lower right')
ax.grid(True)
st.pyplot(fig)

# Важность признаков
st.subheader('Важность признаков')
model = RandomForestClassifier(n_estimators=80, random_state=42)
model.fit(X_train, y_train)
importance = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='purple')
ax.set_xlabel('Важность')
ax.set_ylabel('Признаки')
ax.set_title('Важность признаков')
ax.invert_yaxis()
st.pyplot(fig)

# Гистограмма предсказанных вероятностей
st.subheader('Гистограмма предсказанных вероятностей')
y_prob = model.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6, color='blue', label='На самом деле Yes')
ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6, color='red', label='На самом деле No')
ax.set_xlabel('Предсказанная вероятность')
ax.set_ylabel('Частота')
ax.set_title('Гистограмма предсказанных вероятностей')
ax.legend(loc='upper center')
st.pyplot(fig)
