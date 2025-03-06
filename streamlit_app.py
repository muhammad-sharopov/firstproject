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

# Отключение предупреждений
warnings.filterwarnings('ignore')

st.title('Прогнозирование депрессии у студентов')

data = pd.read_csv("Student Depression Dataset.csv")

option = st.radio(
    "Выберите, что вы хотите увидеть:",
    ('Статистическое описание', 'Типы данных', 'Количество уникальных значений', 'Мода', 'Пропущенные значения')
)

if option == 'Статистическое описание':
    st.write('Статистическое описание данных:')
    st.write(data.describe())

elif option == 'Типы данных':
    st.write('Типы данных:')
    st.write(data.dtypes)

elif option == 'Количество уникальных значений':
    st.write('Количество уникальных значений:')
    st.write(data.nunique())

elif option == 'Мода':
    st.write('Мода:')
    st.write(data.mode().iloc[0])

elif option == 'Пропущенные значения':
    # Обработка пропущенных значений в столбце 'Financial Stress'
    data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)
    
    # Пропущенные значения после обработки
    st.write('Пропущенные значения после обработки:')
    st.write(data.isnull().sum())
       
# Диаграмма распределения по полу
st.write('### Распределение по полу')
gender_counts = data['Gender'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
       colors=['#66b3ff', '#ff9999'], shadow=True, explode=(0.05, 0), textprops={'fontsize': 14})
ax.set_title('Распределение по полу', fontsize=16, fontweight='bold', pad=20)
st.pyplot(fig)

# Гистограмма распределения по возрасту
st.write('### Распределение по возрасту')
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data['Age'], bins=15, color='skyblue', edgecolor='black')
ax.set_title('Распределение по возрасту')
ax.set_xlabel('Возраст')
ax.set_ylabel('Частота')
ax.grid(True)
st.pyplot(fig)

# Тепловая карта корреляции
st.write('### Тепловая карта корреляции для числовых признаков')
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
ax.set_title('Тепловая карта корреляции для числовых признаков', fontsize=18, fontweight='bold', pad=20)
st.pyplot(fig)

# Boxplot для признаков, наиболее связанных с депрессией
st.write('### Boxplot для признаков, наиболее связанных с депрессией')
correlation_matrix = data.select_dtypes(include='number').corr()
top_features_corr = correlation_matrix['Depression'].abs().sort_values(ascending=False).head(6).index
top_features_corr = top_features_corr.drop('Depression')

fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=data[top_features_corr], palette='Set2', ax=ax)
ax.set_title('Boxplot для признаков, наиболее связанных с депрессией', fontsize=16, fontweight='bold', color='darkslategray')
ax.set_xlabel('Признаки', fontsize=14, fontweight='bold', color='darkslategray')
ax.set_ylabel('Значения', fontsize=14, fontweight='bold', color='darkslategray')
st.pyplot(fig)

# Обработка порядковых признаков
ordinal_mapping = {
    'Sleep Duration': {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 0},
    'Dietary Habits': {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 0}
}

for col, mapping in ordinal_mapping.items():
    data[col] = data[col].map(mapping)

# Обработка бинарных признаков
binary_columns = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0}) 

# Удаление ненужных колонок
data = data.drop(columns=['id','Age', 'Degree', 'Profession','Work Pressure','City', 'Gender'])

st.write('### Обработанные данные')
st.write(data.head())

# Подготовка данных для модели
X = data.drop(columns=['Depression']) 
y = data['Depression'] 

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Инициализация моделей
models = {
    'Random Forest': RandomForestClassifier(n_estimators=80, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

# Результаты модели
results = pd.DataFrame(columns=['Модель', 'Train ROC AUC', 'Test ROC AUC'])

# Обучение моделей и вывод результатов кросс-валидации
st.write('### Обучение моделей и оценка')
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
    
    cross_validation_scores = cross_val_score(estimator=model,
                                              X=X_train,
                                              y=y_train,
                                              scoring='accuracy',
                                              n_jobs=-1,
                                              cv=5
                                              )
    st.write(f'Кросс-валидация {name}: {cross_validation_scores.mean()}')

st.write(results)

# ROC кривая
st.write('### ROC кривая')
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

# Важность признаков
st.write('### Важность признаков')
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

# Гистограмма предсказанных вероятностей
st.write('### Гистограмма предсказанных вероятностей')
y_prob = model.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6, color='blue', label='Actual Yes')
ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6, color='red', label='Actual No')
ax.set_xlabel('Предсказанная вероятность')
ax.set_ylabel('Частота')
ax.set_title('Гистограмма предсказанных вероятностей')
ax.legend(loc='upper center')
st.pyplot(fig)
