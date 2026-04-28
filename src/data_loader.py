import pandas as pd

def load_data(path ="data/Raw/sample.csv"):
    df = pd.read_csv(path)
    df =  df.drop(columns="customerID")
    
    df["TotalCharges"] = df["TotalCharges"].replace(" ","0.0")
    df["TotalCharges"] = df["TotalCharges"].replace(" ","0.0")

    return df
    