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
    def convert_sqft1(df):
            for index, row in df.iterrows():
                sqft_range=df.at[index, 'total_sqft']
                if '-' in  str(sqft_range):
                   start, end = map(float, sqft_range.split('-'))
                   df.at[index, 'total_sqft']=(start + end) / 2
                elif "Sq." in str(sqft_range) : 
                   no=str.split(sqft_range,"Sq.")[0] 
                   df.at[index, 'total_sqft']=no
                elif "Acres" in str(sqft_range): 
                   no=str.split(sqft_range,"Acres")[0] 
                   df.at[index, 'total_sqft']=no
                elif "Guntha" in str(sqft_range): 
                   no=str.split(sqft_range,"Guntha")[0] 
                   df.at[index, 'total_sqft']=float(no)*1089
                elif "Cents" in str(sqft_range): 
                   no=str.split(sqft_range,"Cents")[0] 
                   df.at[index, 'total_sqft']=float(no)*435.56
                elif "Perch" in str(sqft_range): 
                   no=str.split(sqft_range,"Perch")[0] 
                   df.at[index, 'total_sqft']=float(no)*272.25
                elif "Grounds" in str(sqft_range): 
                   no=str.split(sqft_range,"Grounds")[0] 
                   df.at[index, 'total_sqft']=float(no)*2400
                else:
                   df.at[index, 'total_sqft']=float(sqft_range)
            return(df)            
  
    def custom_transformer_size(df):
       #one_BHK_avg=np.average(df[(df['size']=='1 Bedroom') | (df['size']=='1 BHK')]['total_sqft'])
       #two_BHK_avg=np.average(df[(df['size']=='2 Bedroom') | (df['size']=='2 BHK')]['total_sqft'])
       #three_BHK_avg=np.average(df[(df['size']=='3 Bedroom') | (df['size']=='3 BHK')]['total_sqft'])
       #four_BHK_avg=np.average(df[(df['size']=='4 Bedroom') | (df['size']=='4 BHK')]['total_sqft'])
       for i in df[df['size'].isna()].index:
          #logging.info(df.loc[i,'size'])
          #logging.info(df.loc[i,'total_sqft']) 
          #logging.info("IN to..0..")
          #logging.info(df.at[index, 'size'])
          if float(df.loc[i,'total_sqft']) < 1000:
              logging.info("IN to..1..")
              df.loc[i,'size']="1BHK"
          elif float(df.loc[i,'total_sqft']) < 1500:
              logging.info("IN to...2.")
              df.loc[i,'size']="2BHK"
          elif float(df.loc[i,'total_sqft']) < 2000:
              logging.info("IN to..3..")
              df.loc[i,'size']="3BHK"
          elif float(df.loc[i,'total_sqft']) < 2500:
              logging.info("IN to...4.")
              df.loc[i,'size']="4BHK"
          else:
              logging.info("IN to..5..")
              df.loc[i,'size']="5BHK"
          #logging.info("is any")    
          #logging.info(len(df[df['size'].isna()]))
                                

       for index, row in df.iterrows():
            if 'Bedroom' in str(df.at[index, 'size']):
               df.at[index, 'size']=float(str.split(str(df.at[index, 'size']),"Bedroom")[0].strip())
            elif "RK" in str(df.at[index, 'size']):
               df.at[index, 'size']=float(0)
            else:
               df.at[index, 'size']=float(str.split(str(df.at[index, 'size']),"BHK")[0].strip())
       return(df)
       

















 

    def data_transformer_function(train_data_path,test_data_path):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numeric_features=['bath', 'balcony','total_sqft']
            #categorical_features = ['area_type', 'availability', 'location', 'size']
            categorical_features = ['size']
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            

           # Create a preprocessor using ColumnTransformer
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(
            transformers=[
             ('cat', categorical_transformer,categorical_features),     
             ])


            logging.info(train_df.columns)

            X=train_df.drop(['ID', 'area_type', 'availability', 'location', 'society','bath', 'balcony'],axis=1)
            y=train_df['price']
            #logging.info(X)
            #logging.info(X.shape) 



            pipeline = Pipeline([
             ('convert_sqft', FunctionTransformer(DataTransformation.convert_sqft1,validate=False)),
             ('convert_size', FunctionTransformer(DataTransformation.custom_transformer_size, validate=False)),
             ('linear_regression', LinearRegression())
             ])
            
            logging.info(X)
            #check in between ouput of pipeline
            #pipeline.fit_transform(X)
            pipeline.fit(X,y)
            logging.info("Information")
            df_transformed = pipeline.named_steps['convert_size'].transform(X)
            logging.info(df_transformed)
            logging.info(df_transformed.columns)
            logging.info(df_transformed.isna().sum())
            logging.info(pipeline.predict(X))

            Xtest=test_df.drop(['ID', 'area_type', 'availability', 'location', 'society','bath', 'balcony'],axis=1)
            Ytest=pipeline.predict(Xtest)
            logging.info(Ytest)
        except Exception as e:
            raise CustomException(e,sys)
        
