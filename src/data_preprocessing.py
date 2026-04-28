import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

def clean_data(df):
    #drop duplicates rows
    df.drop_duplicates(inplace=True)
    return df

def data_preprocess(df):
    #Apply labelencoder to target column
    label = LabelEncoder()
    df["Churn"] = label.fit_transform(df["Churn"])
    return df
    
def split_data(df):
    x = df.drop(columns='Churn',axis=1)
    y = df["Churn"]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=42)
    return x_train,x_test,y_train,y_test,y

#for scaling columns #onehot and standardscaler
def preprocessor():
    new_cat_cols = [ 'Partner', 'Dependents', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup','DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies','Contract',
            'PaperlessBilling', 'PaymentMethod']
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("trf1",OneHotEncoder(sparse_output=False,handle_unknown="ignore",drop="first"),new_cat_cols),
            ("trf2",StandardScaler(),num_cols)
        ]
    )
    return preprocessor

