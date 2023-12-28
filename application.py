import streamlit as st
import pandas as pd
#from src.utils import load_object
#from src.components.model_trainer import ModelTrainerConfig
#from src.components.data_transformation import DataTransformationConfig as preprocessing_obj
from src.logger import logging
#import joblib
#from joblib import dump, load
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

        # Additional data analysis or visualization can be added here
            predict_button_enabled = False
            if uploaded_file is not None:
               predict_button_enabled = True
            # Create a button to trigger prediction
               predict_button = st.button("Predict", key="predict_button", disabled=not predict_button_enabled)
               if predict_button:
                 st.write("Clicked on predict")
                 #st.write(os.path.join('artifacts',"preprocessor.joblib"))
                 preproceesing_object=joblib.load(os.path.join('artifacts',"preprocessor.joblib"))
                 model=joblib.load(os.path.join('artifacts',"model.joblib"))
                 df.dropna(subset=['location'],axis=0,inplace=True)
                 input_feature_test_df=df.drop(['availability'],axis=1)
                 input_feature_test_arr=preproceesing_object.transform(input_feature_test_df)
                 test_arr = input_feature_test_arr.toarray()
                 prediction = model.predict(test_arr)
                 logging.info("Predictions Test")
                 logging.info(prediction)
                 df['PricePrediction'] = prediction
                 df.to_csv("TestPrediction.csv",index=False)
                 df = pd.read_csv("TestPrediction.csv")
                 st.write(df.head(5))
                 st.write()
                 plt.figure(figsize=(10, 6))
                 plt.plot(df['price'], label='price', marker='o')
                 plt.plot(df['PricePrediction'], label='New Price', marker='*')
                 plt.title('Old Price vs New Price')
                 plt.legend()
                 st.pyplot(plt)

  

               
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
