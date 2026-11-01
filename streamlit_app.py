import streamlit as st
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("ahmedesso/brest-cancer", "data.csv")
df = pd.read_csv(path)

df["diagnosis"] = df["diagnosis"].map({"M":1, "B":-1})

x = df.drop(["diagnosis", "Unnamed: 32", id], axis=1)
y = df["diagnosis"]
