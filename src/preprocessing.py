import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

class ClaimPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def fit_transform(self, df):
        df_processed = df.copy()
        
        # 1. Feature Engineering: Compute Risk & Policy Flags
        df_processed['Warranty_Expired_Flag'] = (df_processed['Remaining_Warranty_Months'] <= 0).astype(int)
        df_processed['Doc_Completeness_Score'] = (
            df_processed['Has_Receipt'].astype(int) + 
            df_processed['Has_Warranty_Card'].astype(int) + 
            df_processed['Has_Damage_Photo'].astype(int)
        )
        
        # 2. Separate Target Class
        y = self.label_encoder.fit_transform(df_processed['Claim_Class'])
        
        # 3. Categorical & Numerical Feature Columns
        num_cols = [
            'Purchase_Price', 'Product_Age_Months', 'Warranty_Duration_Months',
            'Remaining_Warranty_Months', 'Missing_Documents_Count', 'Doc_Completeness_Score'
        ]
        cat_cols = ['Product_Category', 'Brand', 'Fault_Type']
        bool_cols = [
            'Has_Receipt', 'Has_Warranty_Card', 'Has_Damage_Photo', 
            'Serial_Number_Match', 'Previous_Unauthorized_Repairs', 
            'Duplicate_Claim_Flag', 'Date_Contradiction_Flag', 'Warranty_Expired_Flag'
        ]
        
        # 4. Encode Features
        X_num = df_processed[num_cols].values
        X_bool = df_processed[bool_cols].astype(int).values
        
        # One-Hot Encoding for categorical features
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = self.ohe.fit_transform(df_processed[cat_cols])
        
        # Combine Features
        X = np.hstack([X_num, X_bool, X_cat])
        
        # Scale Numerical Features
        X = self.scaler.fit_transform(X)
        
        # Save feature name list for explainability
        cat_feature_names = self.ohe.get_feature_names_out(cat_cols)
        self.feature_names = num_cols + bool_cols + list(cat_feature_names)
        
        return X, y

    def transform(self, df):
        df_processed = df.copy()
        
        df_processed['Warranty_Expired_Flag'] = (df_processed['Remaining_Warranty_Months'] <= 0).astype(int)
        df_processed['Doc_Completeness_Score'] = (
            df_processed['Has_Receipt'].astype(int) + 
            df_processed['Has_Warranty_Card'].astype(int) + 
            df_processed['Has_Damage_Photo'].astype(int)
        )
        
        y = self.label_encoder.transform(df_processed['Claim_Class']) if 'Claim_Class' in df_processed else None
        
        num_cols = [
            'Purchase_Price', 'Product_Age_Months', 'Warranty_Duration_Months',
            'Remaining_Warranty_Months', 'Missing_Documents_Count', 'Doc_Completeness_Score'
        ]
        cat_cols = ['Product_Category', 'Brand', 'Fault_Type']
        bool_cols = [
            'Has_Receipt', 'Has_Warranty_Card', 'Has_Damage_Photo', 
            'Serial_Number_Match', 'Previous_Unauthorized_Repairs', 
            'Duplicate_Claim_Flag', 'Date_Contradiction_Flag', 'Warranty_Expired_Flag'
        ]
        
        X_num = df_processed[num_cols].values
        X_bool = df_processed[bool_cols].astype(int).values
        X_cat = self.ohe.transform(df_processed[cat_cols])
        
        X = np.hstack([X_num, X_bool, X_cat])
        X = self.scaler.transform(X)
        
        return X, y if y is not None else X

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    train_df = pd.read_csv('data/train/train_claims.csv')
    preprocessor = ClaimPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    joblib.dump(preprocessor, 'model/preprocessor.joblib')
    print("Preprocessor fitted and saved to model/preprocessor.joblib")