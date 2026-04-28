import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin


def create_features(df):
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup','DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies','Contract',
            'PaperlessBilling', 'PaymentMethod']
    
    results = []
    
    for col in cat_cols:
        table = pd.crosstab(df[col], df["Churn"])
        stat, p, dof, exp = chi2_contingency(table)
    
        # Append col name + p-value
        results.append((col, p))
    
    # Create dataframe
    results_df = pd.DataFrame(results, columns=["feature","p_value"])
    
    # Add Decision column
    alpha = 0.05
    results_df["Decision"] = results_df["p_value"].apply(
        lambda p: "Accept H0 (Significant)" if p < alpha else "Fail to Reject H0 (Not Significant)"
    )
    # Sort by p-value
    results_df = results_df.sort_values("p_value")

    #remove these col because their p_val is greater than alpha
    df.drop(columns="gender",inplace = True)
    df.drop(columns="PhoneService",inplace = True)


    return df



class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.median_ = X["MonthlyCharges"].median()
        return self

    def transform(self, X):
        X = X.copy()

        # 🔥 Feature 1: tenure group
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 100],
            labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
        )

        # 🔥 Feature 2: avg monthly spend
        """X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1)

        # 🔥 Feature 3: high value customer
        X["is_high_value"] = (X["MonthlyCharges"] > self.median_).astype(int)"""

        return X 
