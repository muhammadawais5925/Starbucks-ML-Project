import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title='Starbucks Beverage Classifier',
    page_icon='☕',
    layout='wide',
    initial_sidebar_state='expanded'
)

css_style = """<style>
    .main-header {
        color: #00704A;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: bold;
    }
    .accent-bg {
        background-color: #f1f8f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00704A;
    }
</style>"""
st.markdown(css_style, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>☕ Starbucks Beverage Category Predictor & Dashboard</h1>", unsafe_allow_html=True)
st.markdown('Predict the category of any Starbucks beverage using its nutritional profile, or explore beverage statistics.')

@st.cache_resource
def load_model_package():
    model_path = 'model.pkl'
    if not os.path.exists(model_path):
        return None
    try:
        package = joblib.load(model_path)
        return package
    except Exception as e:
        st.error(f'Error loading model package: {e}')
        return None

model_package = load_model_package()

@st.cache_data
def load_data():
    csv_path = 'Starbucks.csv'
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f'Error loading dataset: {e}')
        return None

df = load_data()

if model_package is not None:
    pipeline = model_package['pipeline']
    label_encoder = model_package['label_encoder']
    features = model_package['features']

    st.sidebar.header('🔍 Beverage Nutritional Profile')
    st.sidebar.markdown('Adjust the parameters below to run classification predictions.')

    user_inputs = {}
    with st.sidebar.form(key='nutrition_form'):
        for feat in features:
            if df is not None and feat in df.columns:
                temp_col = df[feat].astype(str).str.replace('%', '', regex=False).str.strip()
                numeric_series = pd.to_numeric(temp_col, errors='coerce').dropna()
                min_val = float(numeric_series.min()) if not numeric_series.empty else 0.0
                max_val = float(numeric_series.max()) if not numeric_series.empty else 500.0
                mean_val = float(numeric_series.mean()) if not numeric_series.empty else 0.0
            else:
                min_val, max_val, mean_val = 0.0, 500.0, 0.0

            display_name = feat.replace('_', ' ').title()
            user_inputs[feat] = st.slider(
                label=display_name,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.1 if max_val <= 10.0 else 1.0
            )

        submit_button = st.form_submit_button(label='Classify Beverage ☕')

    tab1, tab2, tab3 = st.tabs(['🔮 Real-time Prediction', '📊 Historical Insights', '📋 Explore Dataset'])

    with tab1:
        st.subheader('Classification Outcome')
        if submit_button:
            input_df = pd.DataFrame([user_inputs])
            try:
                pred_encoded = pipeline.predict(input_df)[0]
                pred_label = label_encoder.inverse_transform([pred_encoded])[0]
                probabilities = pipeline.predict_proba(input_df)[0]
                pred_confidence = probabilities[pred_encoded]

                info_html = f"<div class='accent-bg'><h4>Predicted Starbucks Category: <b>{pred_label}</b></h4><p>Confidence Level: <b>{pred_confidence:.2%}</b></p></div>"
                st.markdown(info_html, unsafe_allow_html=True)

                prob_df = pd.DataFrame(
                    {'Category': label_encoder.classes_, 'Confidence (%)': probabilities * 100}
                ).sort_values(by='Confidence (%)', ascending=True)

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.barh(prob_df['Category'], prob_df['Confidence (%)'], color='#00704A')
                ax.set_xlabel('Confidence (%)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.markdown('#### Confidence Distribution across Categories')
                st.pyplot(fig)
            except Exception as e:
                st.error(f'An error occurred during prediction: {e}')
        else:
            st.info('👈 Modify nutrition parameters in the sidebar and click "**Classify Beverage**" to generate predictions.')

    with tab2:
        st.subheader('Nutritional Data Visualizations')
        if df is not None:
            clean_df = df.copy()
            clean_df['Beverage_category'] = clean_df['Beverage_category'].str.strip()
            for col in features:
                if col in clean_df.columns:
                    clean_df[col] = clean_df[col].astype(str).str.replace('%', '', regex=False).str.strip()
                    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

            chart_type = st.selectbox(
                'Choose a visualization:',
                ['Category Distribution', 'Calories vs. Sugar Correlation', 'Nutrient Correlation Heatmap']
            )
            if chart_type == 'Category Distribution':
                fig, ax = plt.subplots(figsize=(10, 5))
                order = clean_df['Beverage_category'].value_counts().index
                sns.countplot(data=clean_df, y='Beverage_category', order=order, palette='viridis', ax=ax)
                ax.set_title('Distribution of Beverages across Categories')
                ax.set_xlabel('Count')
                ax.set_ylabel('Beverage Category')
                plt.tight_layout()
                st.pyplot(fig)
            elif chart_type == 'Calories vs. Sugar Correlation':
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=clean_df, x='Sugars (g)', y='Calories', hue='Beverage_category', alpha=0.8, ax=ax)
                ax.set_title('Calories vs. Sugar Content')
                st.pyplot(fig)
            elif chart_type == 'Nutrient Correlation Heatmap':
                numeric_cols = [c for c in features if c in clean_df.columns]
                corr_matrix = clean_df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Map of Nutritional Values')
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning('Starbucks.csv dataset not found. Visualizations are disabled.')

    with tab3:
        st.subheader('Underlying Starbucks Nutritional Data')
        if df is not None:
            st.dataframe(df)
        else:
            st.warning('Starbucks.csv dataset not found.')
else:
    st.error("Unable to locate or decode 'model.pkl'. Please verify the model file matches your environment runtime configurations.")
