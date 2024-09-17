import streamlit as st
import pandas as pd
import requests

# Title for the app
st.title('House Price Prediction')

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

# Display the uploaded file content
if uploaded_file is not None:
    st.write("Uploaded CSV file:")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # When the user clicks the Predict button
    if st.button('Predict'):
        # Send the file to FastAPI backend
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={"file": uploaded_file.getvalue()}
            )
            # Check if the request was successful
            if response.status_code == 200:
                predictions = response.json()['predictions']
                st.write("Predictions:")
                st.write(predictions)
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the API: {e}")
