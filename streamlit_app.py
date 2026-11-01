import streamlit as st
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

path = kagglehub.dataset_download("ahmedesso/brest-cancer", "data.csv")
df = pd.read_csv(path)

df["diagnosis"] = df["diagnosis"].map({"M":1, "B":-1})

x = df.drop(["diagnosis", "Unnamed: 32", id], axis=1)
y = df["diagnosis"]

xtrain, xtest, ytrain, ytest = train_text_split(x,y,random_state=42, test_size=0.2)

np.random.seed(42)
w=np.random.rand(len(x.columns))*10
b = np.random.rand()*10

