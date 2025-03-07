import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
    ('Количество столбцов и строк', 'Статистическое описание', 'Типы данных', 'Количество уникальных значений', 'Мода', 'Пропущенные значения')
)

if option == 'Статистическое описание':
    st.write('Статистическое описание данных:')
    st.write(data.describe())

elif option == 'Количество столбцов и строк':
    st.write(f'Количество строк: {data.shape[0]} столбцов: {data.shape[1]}')

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
    st.write('Пропущенные значения:')
    st.write(data.isnull().sum())

data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)

# Бинарные признаки
binary_columns = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

def plot_binary_column(col):
    st.write(f'### Распределение по признаку: {col}')
    binary_counts = data[col].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(binary_counts, labels=binary_counts.index, autopct='%1.1f%%', startangle=90,
           colors=['#66b3ff', '#ff9999'], shadow=True, explode=(0.05, 0), textprops={'fontsize': 14})
    ax.set_title(f'Распределение по признаку: {col}', fontsize=16, fontweight='bold', pad=20)
    st.pyplot(fig)

selected_column = st.sidebar.selectbox('Выберите бинарный признак для отображения:', binary_columns)
plot_binary_column(selected_column)

# Гистограмма возраста
st.write('### Распределение по возрасту')
fig_age = px.histogram(data, x='Age', nbins=15, title='Распределение по возрасту', color_discrete_sequence=['skyblue'])
st.plotly_chart(fig_age)

# Тепловая карта корреляции
numerical_data = data.select_dtypes(include='number')
num_features = st.sidebar.slider("Выберите количество признаков для отображения на тепловой карте", min_value=1, max_value=len(numerical_data.columns), value=5)

correlation_matrix = numerical_data.corr()
top_corr_features = correlation_matrix['Depression'].abs().nlargest(num_features).index
top_corr_matrix = correlation_matrix.loc[top_corr_features, top_corr_features]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
st.pyplot(fig)

# Обработка признаков
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

st.write('### Обработанные данные')
st.write(data.head())

X = data.drop(columns=['Depression']) 
y = data['Depression'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=80, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

results = pd.DataFrame(columns=['Модель', 'Train ROC AUC', 'Test ROC AUC'])

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    results = pd.concat([results, pd.DataFrame({
        'Модель': [name],
        'Train ROC AUC': [roc_auc_score(y_train, y_train_proba)],
        'Test ROC AUC': [roc_auc_score(y_test, y_test_proba)]
    })], ignore_index=True)

st.write(results)

# Важность признаков
st.write('### Важность признаков')
model = RandomForestClassifier(n_estimators=80, random_state=42)
model.fit(X_train, y_train)
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='purple')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importance')
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig)
