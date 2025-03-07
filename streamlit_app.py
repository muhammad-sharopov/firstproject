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
    st.write(f'Количество строк: {data.shape[0] } столбцов: {data.shape[1] } ')

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
    # Пропущенные значения после обработки
    st.write('Пропущенные значения:')
    st.write(data.isnull().sum())

data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)

# Диаграмма распределения по полу
binary_columns = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# Функция для отображения графиков
def plot_binary_column(col):
    st.write(f'### Распределение по признаку: {col}')
    binary_counts = data[col].value_counts()

    # Строим круговую диаграмму
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(binary_counts, labels=binary_counts.index, autopct='%1.1f%%', startangle=90,
           colors=['#66b3ff', '#ff9999'], shadow=True, explode=(0.05, 0), textprops={'fontsize': 14})
    ax.set_title(f'Распределение по признаку: {col}', fontsize=16, fontweight='bold', pad=20)
    st.pyplot(fig)

# Выбор бинарного признака с помощью выпадающего списка
selected_column = st.sidebar.selectbox('Выберите бинарный признак для отображения:', binary_columns)

# Отображаем график для выбранного признака
plot_binary_column(selected_column)

# Гистограмма распределения по возрасту
st.write('### Распределение по возрасту')
fig_age = px.histogram(data, x='Age', nbins=15, title='Распределение по возрасту',
                        labels={'Age': 'Возраст'}, color_discrete_sequence=['skyblue'])
st.plotly_chart(fig_age)

# Тепловая карта корреляции
numerical_data = data.select_dtypes(include='number')
num_features = st.sidebar.slider("Выберите количество признаков для отображения на тепловой карте", min_value=1, max_value=len(numerical_data.columns), value=5)

correlation_matrix = numerical_data.corr()

top_corr_features = correlation_matrix.abs().nlargest(num_features, 'Depression')

top_corr_matrix = correlation_matrix.loc[top_corr_features.index, top_corr_features.index]

fig, ax = plt.subplots(figsize=(12, 10))
sns.set_style("white")
sns.heatmap(
    top_corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    linecolor='black',
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 10, 'weight': 'bold'},
    ax=ax
)

# Настроим заголовок и отобразим тепловую карту
ax.set_title(f'Тепловая карта корреляции для топ {num_features} признаков', fontsize=18, fontweight='bold', pad=20)
st.pyplot(fig)

# Boxplot для признаков, наиболее связанных с депрессией
st.write('### Boxplot для признаков, наиболее связанных с депрессией')
correlation_matrix = data.select_dtypes(include='number').corr()
top_features_corr = correlation_matrix['Depression'].abs().sort_values(ascending=False).head(6).index
top_features_corr = top_features_corr.drop('Depression')

# Переводим данные в формат для plotly
data_top_features = data[top_features_corr]

# Переводим данные в длинный формат для plotly
data_long = data_top_features.melt(var_name='Feature', value_name='Value')

# Создаем интерактивный boxplot
fig_boxplot = px.box(data_long, x='Feature', y='Value', title='Boxplot для признаков, наиболее связанных с депрессией',
                     color='Feature', boxmode='overlay')
st.plotly_chart(fig_boxplot)

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
st.sidebar.write("### Выберите модели для отображения:")
selected_models = st.sidebar.multiselect(
    "Модели:", list(models.keys()), default=list(models.keys())
)

# Создание интерактивного ROC-графика
fig = go.Figure()

for name in selected_models:
    model = models[name]
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{name} (AUC = {roc_auc:.2f})',
        line=dict(width=2)
    ))

# Добавляем линию случайного угадывания
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Guess',
    line=dict(dash='dash', color='black')
))

# Настройки графика
fig.update_layout(
    title="Receiver Operating Characteristic (ROC) Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend=dict(x=0.8, y=0.2),
    template="plotly_white"
)

# Отображение графика
st.plotly_chart(fig)

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
