import pandas as pd
import numpy as np


from train_pipeline import build_pipeline, train_model
from tune import tune_model
from evaluate import apply_threshold, evaluate_model
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import recall_score, precision_score
from utils import save_model


from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data
from data_preprocessing import data_preprocess,clean_data, split_data,preprocessor
from feature_engineering import create_features,FeatureEngineer
def main():
    #1. load data 
    #path = "C:\\Users\\Vikash\\Desktop\\customer_churn_model\\data\\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_data()

    #2. feature engineering
    df = create_features(df)

    #3.preprocess data
    df =  clean_data(df)
    df = data_preprocess(df)

    #4. Split
    x_train,x_test,y_train,y_test,y = split_data(df)



    # build pipeline
    pipeline = build_pipeline(y)

    # train
    pipeline = train_model(pipeline, x_train, y_train)



    #Hyperparimeter Tunning
    param_dist = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__scale_pos_weight": [
            len(y[y==0]) / len(y[y==1])
        ]
    }

    search = tune_model(pipeline, param_dist, x_train, y_train)


    # best model
    best_model = search.best_estimator_


    # threshold
    y_pred = apply_threshold(best_model, x_test)


    # evaluate
    evaluate_model(y_test, y_pred)


    #Saving Model

    save_model(best_model, x_train.columns)


    print("Model saved successfully!")

if __name__ == "__main__":
    main()