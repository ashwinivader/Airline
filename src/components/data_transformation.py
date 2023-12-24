#This is data transformation process(encoding)

import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, FunctionTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=  DataTransformationConfig()  


    @staticmethod
    def convert_sqft(sqft_range):
        #logging.info("in convert sqrt")
        if '-' in sqft_range:
           start, end = map(float, sqft_range.split('-'))
           return (start + end) / 2
        elif "Sq." in sqft_range : 
            no=str.split(sqft_range,"Sq.")[0] 
            return (no)
        elif "Acres" in sqft_range: 
             no=str.split(sqft_range,"Acres")[0] 
             return (no)
        elif "Guntha" in sqft_range: 
             no=str.split(sqft_range,"Guntha")[0] 
             return (float(no)*1089) 
        elif "Cents" in sqft_range: 
             no=str.split(sqft_range,"Cents")[0] 
             return (float(no)*435.56 ) 
        elif "Perch" in sqft_range: 
             no=str.split(sqft_range,"Perch")[0] 
             return (float(no)*272.25 ) 
        elif "Grounds" in sqft_range: 
            no=str.split(sqft_range,"Grounds")[0] 
            return (float(no)*2400 ) 
        else:
            return float(sqft_range)
 

    def data_transformer_function(train_data_path,test_data_path):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numeric_features=['bath', 'balcony','total_sqft']
            categorical_features = ['area_type', 'availability', 'location', 'size']
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

# Create a preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
            transformers=[
            ('convert_sqft', FunctionTransformer(lambda x: x.map(DataTransformation.convert_sqft)), ['total_sqft']),
            ])

            # Create a pipeline with preprocessing and linear regression
            #logging.info(train_df)
            X=train_df[['total_sqft']]
            y=train_df['price']

            pipeline = Pipeline([
             ('preprocessor', preprocessor),
             ('regressor', LinearRegression())
             ])

            pipeline.fit(X, y)
            logging.info("Information")
            logging.info(pipeline.predict(X))




        except Exception as e:
            raise CustomException(e,sys)
        
