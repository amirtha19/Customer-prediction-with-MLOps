import logging
import pandas as pd
from src.data_cleaning import DataCleaning,DataPreProcessStrategy

def get_data_for_test():
    try:
        excel_file = pd.ExcelFile('data/data.xlsx')
        sheet_names = excel_file.sheet_names
        # Create a dictionary to store DataFrames
        dataframes = {sheet_name: excel_file.parse(sheet_name) for sheet_name in sheet_names}
        df_transaction = dataframes.get('Transaction Data')
        df_demo = dataframes.get('Customer Demographics')
        df = pd.merge(df_transaction,df_demo,on = "Customer_number")
        df['Term Deposit'] = df['Term Deposit'].map({'no': 2, 'yes': 1})
        df = df.dropna(subset=['Term Deposit','Annual Income','balance'])
        df = df.sample(n=100)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e