# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_laptop_prices.csv")  # use your cleaned dataset
    return df

df = load_data()

st.title("💻 Laptop Price Analytics & Prediction Dashboard")
st.markdown("An end-to-end advanced data analytics + ML project for pricing insights.")

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
st.header("Exploratory Data Analysis")

eda_choice = st.selectbox("Choose EDA visualization", 
                          ["Price Distribution", "Price by Company", "Price by Type", "Correlation Heatmap"])

if eda_choice == "Price Distribution":
    fig, ax = plt.subplots()
    sns.histplot(df['Price_inr'], bins=40, ax=ax, color='steelblue')
    ax.set_title("Price Distribution")
    st.pyplot(fig)

elif eda_choice == "Price by Company":
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Company', y='Price_inr', data=df.sort_values('Company'), ax=ax)
    ax.set_title("Price by Company")
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)

elif eda_choice == "Price by Type":
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='Type_clean', y='Price_inr', data=df, ax=ax)
    ax.set_title("Price by Type")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

elif eda_choice == "Correlation Heatmap":
    num_cols = ['Price_inr','Inches','Ram','Weight','ScreenW','ScreenH','CPU_freq','PrimaryStorage','SecondaryStorage','Pixels','PPI_proxy','TotalStorage']
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# -----------------------------
# Model Training
# -----------------------------
st.header("Model Training & Evaluation")

categorical_cols = ['Company','Type_clean','OS_clean','Screen','PrimaryStorageType','SecondaryStorageType','CPU_brand','GPU_brand']
numeric_cols = ['Inches','Ram','Weight','ScreenW','ScreenH','CPU_freq','PrimaryStorage','SecondaryStorage','Pixels','PPI_proxy','TotalStorage']

X = df[categorical_cols + numeric_cols]
y = df['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('cat', ohe, categorical_cols),
    ('num', scaler, numeric_cols)
])

rf = Pipeline(steps=[('prep', preprocessor),
                    ('model', RandomForestRegressor(n_estimators=400, random_state=42))])

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.write(f"**Random Forest Performance:** MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

# -----------------------------
# Feature Importance
# -----------------------------
st.header("Feature Importance")

prep = rf.named_steps['prep']
cat_features = prep.named_transformers_['cat'].get_feature_names_out(categorical_cols)
num_features = np.array(numeric_cols)
feature_names = np.concatenate([cat_features, num_features])

importances = rf.named_steps['model'].feature_importances_
fi = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8,10))
sns.barplot(y='feature', x='importance', data=fi, ax=ax, palette='viridis')
ax.set_title("Top 20 Feature Importance")
st.pyplot(fig)

# -----------------------------
# Pricing Simulator
# -----------------------------
st.header("Pricing Simulator")

st.markdown("Adjust laptop specs to simulate predicted price.")

ram = st.slider("RAM (GB)", 2, 64, 8)
storage = st.selectbox("Primary Storage Type", ["HDD","SSD","Hybrid","Flash Storage"])
cpu_freq = st.slider("CPU Frequency (GHz)", 0.9, 3.6, 2.5)
screen_def = st.selectbox("Screen Definition", df['Screen'].unique())

sample = df.iloc[0].copy()
sample['Ram'] = ram
sample['PrimaryStorageType'] = storage
sample['CPU_freq'] = cpu_freq
sample['Screen'] = screen_def

sample_df = pd.DataFrame([sample])[X.columns]
pred_price = rf.predict(sample_df)[0]

st.success(f"Predicted Price: €{pred_price:.2f}")

# -----------------------------
# Competitive Positioning
# -----------------------------
st.header("Competitive Positioning")

df['PriceBand'] = pd.cut(
    df['Price_inr'],
    bins=[0, 40000, 80000, 120000, 200000, df['Price_inr'].max()],
    labels=['Budget', 'Mid', 'Upper-Mid', 'Premium', 'Ultra']
)


fig, ax = plt.subplots(figsize=(12,6))
sns.countplot(x='PriceBand', hue='Company', data=df[df['Company'].isin(df['Company'].value_counts().head(6).index)], ax=ax)
ax.set_title("Brand Share Across Price Bands")
st.pyplot(fig)
