import pickle

def save_model(best_model,col):

    artifact = {
        "model": best_model,
        "threshold": 0.7,
        "columns": col
        }
    
    with open("model_Churn_2.pkl", "wb") as f:
        pickle.dump(artifact, f)
