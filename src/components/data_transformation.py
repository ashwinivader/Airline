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
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.joblib')


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
         #Handling nan of size
         for i in df[df['size'].isna()].index:
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
         #Removing Bedroom,RK,BHK string from size                 
         for index, row in df.iterrows():
            if 'Bedroom' in str(df.at[index, 'size']):
               df.at[index, 'size']=float(str.split(str(df.at[index, 'size']),"Bedroom")[0].strip())
            elif "RK" in str(df.at[index, 'size']):
               df.at[index, 'size']=float(0)
            else:
               df.at[index, 'size']=float(str.split(str(df.at[index, 'size']),"BHK")[0].strip())
         return(df)
    
      def drop_features(df):
         logging.info("Droping feature ID and society ")
         df.drop(['ID','society'],axis=1,inplace=True)
         return(df)
      
      def remove_NAN(df):
         logging.info("Removing nan for location ")
         df.dropna(subset=['location'],axis=0,inplace=True)
         df.dropna
         return(df)
  
  
      def custom_transformer_bath(df):   
            #Handling nan for column bat as per size   
            logging.info("In side custom_transformer_bath")  
            for i in df[df['bath'].isna()].index:
                   size=df.loc[i,'size']
                   if int(size)< 4:
                       df.loc[i,'bath']=size-1
                   else:
                       df.loc[i,'bath']=3
            return(df)
           
      def custom_transformer_balcony (df):
            #Handling nan for column bat as per size 
            logging.info("In side custom_transformer_balcony")    
            for i in df[df['balcony'].isna()].index:
                   size=df.loc[i,'size']
                   if int(size)< 4:
                       df.loc[i,'balcony']=size-1
                   else:
                       df.loc[i,'balcony']=3
            return(df)
      

      def remove_outlier(df,size):
            logging.info("In removal_outlier")    
            q1=df[df['size']==float(size)]['total_sqft'].quantile(0.25)
            q3=df[df['size']==float(size)]['total_sqft'].quantile(0.75)
            IQR=q3-q1
            lb=q1-(1.5*IQR)
            ub=q3+(1.5*IQR)
            filtered_df = df[(df['size'] == float(size)) & (df['total_sqft'] > ub)]
            df.drop(index=filtered_df.index,axis=0,inplace=True)  


      def custom_outlier_trasformation(df):
            logging.info("In side custom removal of outlier")    
            DataTransformation.remove_outlier(df,8.0)
            DataTransformation.remove_outlier(df,9.0) 
            return(df)  

        
      def get_transformer_object():
          '''
          This function is responsible for data trnasformation       
          '''
          try:
             #train_df=pd.read_csv(train_data_path)
             #test_df=pd.read_csv(test_data_path)

             
             categorical_features =['area_type', 'location']
             numerical_features = ['size', 'total_sqft','bath', 'balcony']


             preprocessor = ColumnTransformer(
             transformers=[
             ('num', StandardScaler(),numerical_features),
             ('cat', OneHotEncoder(), categorical_features)])

             pipeline_preprocesing = Pipeline([
             ('convert_sqft', FunctionTransformer(DataTransformation.convert_sqft1,validate=False)),
             ('convert_size', FunctionTransformer(DataTransformation.custom_transformer_size, validate=False)),
             ('Feature drop',FunctionTransformer(DataTransformation.drop_features,validate=False)),
             #('RemoveNaN',FunctionTransformer(DataTransformation.remove_NAN,validate=False)),
             ('BathNANhandling',FunctionTransformer(DataTransformation.custom_transformer_bath,validate=False)),
             ('BalconyNANhandling',FunctionTransformer(DataTransformation.custom_transformer_balcony,validate=False)),
             ('standardProcessing',preprocessor),
             #('OutlierRemoval',FunctionTransformer(DataTransformation.custom_outlier_trasformation,validate=True)),
             #('linear_regression', LinearRegression())
             ])
             logging.info("Out of get_transformer_object")
             return pipeline_preprocesing
          
          except Exception as e:
            raise CustomException(e,sys)
        
      def perform_transformation(train_data_path,test_data_path,raw_data_path):        
         try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            raw_df=pd.read_csv(raw_data_path)
            logging.info(train_df.columns)
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=DataTransformation.get_transformer_object()
            logging.info("Received preprocessing object")

            
            raw_df.dropna(subset=['location'],axis=0,inplace=True)
            #X=train_df.drop(['price','area_type', 'availability'],axis=1)
            input_feature_train_df=raw_df.drop(['price','availability'],axis=1)
            target_feature_train_df=raw_df['price']

            test_df.dropna(subset=['location'],axis=0,inplace=True)
            input_feature_test_df=test_df.drop(['price','availability'],axis=1)
            target_feature_test_df=test_df['price']

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(type(input_feature_train_arr.toarray))
            train_arr = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=DataTransformationConfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                DataTransformationConfig.preprocessor_obj_file_path
            )


            logging.info(Xtrain)
            #logging.info(type(Xtrain))

            #Sample model building
            #model=data_transformation_object.fit(X,y)
            #Xtest=test_df.drop(['price','availability'],axis=1)
            #ytest=test_df['price']
            #ytestPred=model.predict(Xtest)

     

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe.")
            #data_transformation_object.fit(X,y)
            #  1qaw  y_train_pred=data_transformation_object.predict(X)
            #logging.info(y_train_pred)
            #pipeline.fit_transform(X)
            #For model add ('linear_regression', LinearRegression()) into and then pipeline_preprocesing.fit(X,y)
            #pipeline_preprocesing.fit(X,y)
            #logging.info("Information")
            #df_transformed = pipeline_preprocesing.named_steps['convert_size'].transform(X)
            #logging.info(df_transformed)
            #logging.info(df_transformed.columns)
            #logging.info(df_transformed.isna().sum())
            #logging.info(pipeline_preprocesing.predict(X))
            #Xtest=test_df.drop([],axis=1)
            #Ytest=data_transformation_object.predict(test_df)
            #logging.info(Ytest)

 
         except Exception as e:
            raise CustomException(e,sys)
        
        
        
