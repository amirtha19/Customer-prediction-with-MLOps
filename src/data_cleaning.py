import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:

            # Drop unnecessary columns
            data = data.drop(columns=["Sno", "Customer_number"], errors='ignore')

            # Handle missing values and standardize 'contact' column
            data['contact'] = data['contact'].replace({'Mobile': 'cellular', 'Tel': 'telephone', '?': 'unknown'})
            data['contact'].fillna(data['contact'].mode()[0], inplace=True)

            # Handle missing values and standardize 'poutcome' column
            data['poutcome'] = data['poutcome'].replace({'?': 'unknown', '????': 'unknown', 'pending': 'unknown'})
            data['poutcome'].fillna(data['poutcome'].mode()[0], inplace=True)

            # Standardize 'job' column
            data['job'] = data['job'].replace({'blue collar': 'blue-collar', '????': 'unknown'})
            data['job'].fillna(data['job'].mode()[0], inplace=True)

            # Handle missing values in 'marital' column
            data['marital'].fillna(data['marital'].mode()[0], inplace=True)

            # Standardize and handle missing values in 'education' column
            data['education'] = data['education'].str.lower().str.strip().replace({'pri mary': 'primary', 'ter tiary': 'tertiary'})
            data['education'].replace('unknown', data['education'].mode()[0], inplace=True)
            data['education'].fillna(data['education'].mode()[0], inplace=True)

            # Standardize and handle outliers in 'balance' column
            data['balance'] = pd.to_numeric(data['balance'], errors='coerce').fillna(0).astype(int)
            data['balance'] = np.clip(data['balance'], data['balance'].quantile(0.25), data['balance'].quantile(0.75))

            # Replace entire rows with 0 where 'balance' column contains '/' or '?'
            data.loc[data['balance'].astype(str).str.contains(r'/|\?'), :] = 0

            # Convert 'balance' column to integer
            data['balance'] = data['balance'].astype(int)

            # Handle missing values in 'duration' column using KNN imputer
            data['duration'] = data['duration'].abs()
            imputer = KNNImputer(n_neighbors=5)
            data['duration'] = imputer.fit_transform(data['duration'].values.reshape(-1, 1))
            data['duration'] = data['duration'].round().astype(int)

            # Handle outliers in 'duration' column
            data['duration'] = np.clip(data['duration'], data['duration'].quantile(0.25), data['duration'].quantile(0.75))

            # Ensure 'Count_Txn' is non-negative and fill missing values with median
            data['Count_Txn'] = data['Count_Txn'].abs()
            data['Count_Txn'].fillna(data['Count_Txn'].median(), inplace=True)
            minmax_scaler = MinMaxScaler()
            numerical_features = ['campaign','age']
            data[numerical_features] = minmax_scaler.fit_transform(data[numerical_features])
            categorical_features = ['job', 'marital', 'education', 'poutcome','contact','last_contact_day']
            label_encoder = LabelEncoder()
            for feature in categorical_features:
                data[feature] = label_encoder.fit_transform(data[feature])
            
            data = pd.get_dummies(data, columns=['housing', 'loan','Gender','Insurance'], drop_first=True)
            return data

        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

class DataDivideStrategy(DataStrategy):
    def handle_data(self,data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Check for missing values in the target variable
            if data['Term Deposit'].isnull().any():
                print("Target variable contains missing values")
            else:
                print("Target variable does not contain missing values")

            data.dropna(subset=['Term Deposit'], inplace=True)

            X=data.drop(['Term Deposit','Annual Income'],axis=1)
            # Remove rows with missing values in the target variable
            


            y=data['Term Deposit']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
        
class DataCleaning:
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data = data
        self.strategy= strategy
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e

