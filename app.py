# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
loaded_model = joblib.load('best_model.joblib')

st.title('Mobile Device User Behavior Prediction')

# Explanation of the app
st.write("""
This application predicts a user's mobile device behavior class based on various usage patterns and demographic information.
Understanding user behavior can help in tailoring services and optimizing device performance.
""")

# Define the order of columns as used during training
feature_columns = ['Device Model', 'Operating System', 'App Usage Time (min/day)',
                   'Screen On Time (hours/day)', 'Battery Drain (mAh/day)',
                   'Number of Apps Installed', 'Data Usage (MB/day)', 'Age',
                   'Gender', 'AgeGroup', 'App Usage Total (week)',
                   'App Usage to Screen Time Ratio', 'App Installation Category']

# Define the categorical columns and their original categories
categorical_features = {
    'Device Model': ['Google Pixel 5', 'iPhone 12', 'OnePlus 9', 'Samsung Galaxy S21', 'Xiaomi Mi 11'],
    'Operating System': ['Android', 'iOS'],
    'Gender': ['Female', 'Male'],
    'AgeGroup': ['Middle adult', 'old', 'young', 'young adult'],
    'App Installation Category': ['High', 'Low', 'Medium']
}

# Initialize and fit label encoders on the original categories
label_encoders = {}
for col, values in categorical_features.items():
    le = LabelEncoder()
    le.fit(values)
    label_encoders[col] = le

# Create and fit a dummy scaler for demonstration purposes.
# In a real application, save and load the fitted scaler from training.
dummy_data_for_scaler = pd.DataFrame(np.random.rand(10, len(feature_columns)), columns=feature_columns)
for col, values in categorical_features.items():
    dummy_data_for_scaler[col] = np.random.choice(values, 10)

dummy_data_for_scaler['App Usage Total (week)'] = dummy_data_for_scaler['App Usage Time (min/day)'] * 7
dummy_data_for_scaler['App Usage to Screen Time Ratio'] = dummy_data_for_scaler['App Usage Time (min/day)'] / (dummy_data_for_scaler['Screen On Time (hours/day)'] * 60)
dummy_data_for_scaler['App Usage to Screen Time Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
dummy_data_for_scaler['App Usage to Screen Time Ratio'].fillna(0, inplace=True)


for col, le in label_encoders.items():
    dummy_data_for_scaler[col] = le.transform(dummy_data_for_scaler[col])

scaler = StandardScaler()
scaler.fit(dummy_data_for_scaler)


# Add input components with explanations
input_data = {}
for col in feature_columns:
    if col in categorical_features:
        input_data[col] = st.selectbox(f'Select {col}', categorical_features[col], help=f'Choose the user\'s {col}.')
    elif col == 'App Usage Time (min/day)':
        input_data[col] = st.number_input(f'Enter {col}', min_value=0, value=100, help='Enter the average daily app usage time in minutes.')
    elif col == 'Screen On Time (hours/day)':
         input_data[col] = st.number_input(f'Enter {col}', min_value=0.0, value=5.0, help='Enter the average daily screen on time in hours.')
    elif col == 'Battery Drain (mAh/day)':
         input_data[col] = st.number_input(f'Enter {col}', min_value=0, value=1000, help='Enter the average daily battery drain in mAh.')
    elif col == 'Number of Apps Installed':
        input_data[col] = st.number_input(f'Enter {col}', min_value=0, value=50, help='Enter the total number of apps installed on the device.')
    elif col == 'Data Usage (MB/day)':
        input_data[col] = st.number_input(f'Enter {col}', min_value=0, value=500, help='Enter the average daily data usage in MB.')
    elif col == 'Age':
         input_data[col] = st.slider(f'Select {col}', min_value=18, max_value=60, value=30, help='Select the user\'s age.')
    elif col == 'App Usage Total (week)':
        # This is an engineered feature, not a direct input
        continue
    elif col == 'App Usage to Screen Time Ratio':
         # This is an engineered feature, not a direct input
         continue
    elif col == 'App Installation Category':
        # This is an engineered feature, not a direct input
        continue


# Add a button to trigger prediction
predict_button = st.button('Predict User Behavior Class')

if predict_button:
    # Prepare input data for prediction
    input_df = pd.DataFrame([input_data])

    # Calculate engineered features
    input_df['App Usage Total (week)'] = input_df['App Usage Time (min/day)'] * 7
    input_df['App Usage to Screen Time Ratio'] = input_df['App Usage Time (min/day)'] / (input_df['Screen On Time (hours/day)'] * 60)

    # Handle potential division by zero for engineered features
    input_df['App Usage to Screen Time Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
    input_df['App Usage to Screen Time Ratio'].fillna(0, inplace=True)


    # Apply label encoding to categorical features
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Ensure all feature columns are present before scaling
    # This is a safety check in case any engineered features were not added correctly
    for col in feature_columns:
        if col not in input_df.columns:
            # Add missing engineered columns with default values (e.g., 0) if needed
            input_df[col] = 0 # Or some other appropriate default

    # Reorder columns to match training data before scaling
    input_df = input_df[feature_columns]


    # Apply scaling to all features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = loaded_model.predict(input_scaled)

    st.subheader('Prediction:')
    st.write(f'The predicted User Behavior Class is: **{int(round(prediction[0]))}**') # Display as integer

    # Explanation of the output
    st.write("""
    The User Behavior Class is a numerical representation of the user's mobile device usage pattern.
    The classes generally represent different levels of intensity or type of usage.
    For example (based on typical interpretations, the exact meaning depends on the dataset):
    - Class 1: Very Low Usage
    - Class 2: Low Usage
    - Class 3: Moderate Usage
    - Class 4: High Usage
    - Class 5: Very High Usage
    """)

# Instructions on how to run the app
st.markdown("""
---
**How to run this application:**

1.  Save the code above as `app.py`.
2.  Make sure you have the `best_model.joblib` file in the same directory.
3.  Open your terminal or command prompt.
4.  Navigate to the directory where you saved `app.py` and `best_model.joblib`.
5.  Run the command: `streamlit run app.py`
6.  The application will open in your web browser.
""")
