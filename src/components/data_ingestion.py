#Data ingestion process
#train and test data
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses  import dataclass
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion process")
        #readinf the data...this can be normal csv or from any other data soure
        df=pd.read_csv("data/train.csv")
        logging.info("Read the dataset into dataframe")
        #Making directory with name "artifacts"
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
        #data.csv is original file
        df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
        logging.info("Train test split")

        #Train test split
        train,test=train_test_split(df,test_size=0.2,random_state=42)
        #save train data as artifacts/train.csv
        train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
        #save train data as artifacts/test.csv
        test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
        logging.info("Data injestion is completed")
        return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path,self.ingestion_config.raw_data_path)


        




    
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

    train_data,test_data,raw_data=obj.initiate_data_ingestion()
    print(train_data,test_data,raw_data)

    datatransformation=DataTransformation
    #datatransformation.data_transformer_function(train_data,test_data)
    datatransformation.perform_transformation(train_data,test_data,raw_data) 
    

    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))