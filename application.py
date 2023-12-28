import streamlit as st
import pandas as pd
#from src.utils import load_object
#from src.components.model_trainer import ModelTrainerConfig
#from src.components.data_transformation import DataTransformationConfig as preprocessing_obj
from src.logger import logging
import joblib
from joblib import dump, load
import os
import matplotlib.pyplot as plt


def main():
    st.title("Housing price prediction")

    # Upload file through Streamlit
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Read the file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)


  

               
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
