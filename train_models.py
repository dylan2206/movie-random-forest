import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("movies_clean.csv")

# Features & Target
feature_cols = ['budget_log', 'votes_log', 'rating', 'duration', 'year', 'profit_log']
target = 'gross_log'

X = df[feature_cols]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42
)

rf.fit(X_train, y_train)

pickle.dump(rf, open("random_forest.pkl", "wb"))

print("Model Random Forest berhasil dibuat & disimpan sebagai random_forest.pkl!")
