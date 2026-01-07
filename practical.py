import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("iris.csv")
print(df.head())

X = df.drop("species", axis=1)   
Y = df["species"]                

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
