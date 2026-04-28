from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from feature_engineering import FeatureEngineer
from data_preprocessing import preprocessor

def build_pipeline(y):
    model_xg = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y[y==0]) / len(y[y==1]),
        random_state=42,
        eval_metric='logloss'
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", model_xg)
    ])
    return pipeline


def train_model(pipeline, x_train, y_train):
    pipeline.fit(x_train, y_train)
    return pipeline