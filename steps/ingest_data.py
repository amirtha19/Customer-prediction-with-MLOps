import logging
import pandas as pd
from zenml import step
class IngestData:
    def __init__(self,data_path:str):
        self.data_path =data_path
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
<<<<<<< HEAD
        excel_file = pd.ExcelFile("data/data.xlsx")
=======
        excel_file = pd.ExcelFile(self.data_path)
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2

        # Get the names of all sheets in the Excel file
        sheet_names = excel_file.sheet_names
        # Create a dictionary to store DataFrames
        dataframes = {sheet_name: excel_file.parse(sheet_name) for sheet_name in sheet_names}

        # Access the DataFrames using their names
        df_transaction = dataframes.get('Transaction Data')
        df_demo = dataframes.get('Customer Demographics')
        df = pd.merge(df_transaction,df_demo,on = "Customer_number")
        csv_path = "data/merged_data.csv"
        df['Term Deposit'] = df['Term Deposit'].map({'no': 2, 'yes': 1})
        df = df.dropna(subset=['Term Deposit','Annual Income'])
<<<<<<< HEAD
        df = df.drop(["balance",'Annual Income'],axis=1)
=======
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
        df.to_csv(csv_path, index=False)
        print(df.head())
        return df
    
@step
def ingest_df(data_path:str) -> pd.DataFrame:
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
    except Exception as e:
        logging.error("Failed to ingest data")
        raise e
