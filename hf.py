import pandas as pd
def get_data(self):
        excel_file = pd.ExcelFile(self.data_path)

        # Get the names of all sheets in the Excel file
        sheet_names = excel_file.sheet_names

        # Create a dictionary to store DataFrames
        dataframes = {sheet_name: excel_file.parse(sheet_name) for sheet_name in sheet_names}

        # Access the DataFrames using their names
        df_transaction = dataframes.get('Transaction Data')
        df_demo = dataframes.get('Customer Demographics')
        df = pd.merge(df_transaction, df_demo, on='common_column', how='inner')
        return df
    
    