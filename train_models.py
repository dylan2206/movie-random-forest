import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("final_dataset.csv")

df['gross_worldwide'] = df['gross_worldwide'].fillna(0)

# Convert release_date → year
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['year'] = df['year'].fillna(df['year'].median())

# Features & Target
feature_cols = ['budget', 'votes', 'rating', 'duration', 'year']
target = 'gross_worldwide'

X = df[feature_cols]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Random Forest
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42
)

# Train model
rf.fit(X_train, y_train)

# Save model to .pkl
pickle.dump(rf, open("random_forest.pkl", "wb"))

print("Model Random Forest berhasil dibuat & disimpan sebagai random_forest.pkl!")
