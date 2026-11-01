import streamlit as st
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

path = kagglehub.dataset_download("ahmedesso/brest-cancer", "data.csv")
df = pd.read_csv(path)

df["diagnosis"] = df["diagnosis"].map({"M":1, "B":-1})

x = df.drop(["diagnosis", "Unnamed: 32", "id"], axis=1)
y = df["diagnosis"]

xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=42, test_size=0.2)

np.random.seed(42)
w=np.random.rand(len(x.columns))*10
b = np.random.rand()*10

z = lambda x: w.T@x+b

yhat = lambda x: (1 if z(x)>0 else -1)

hloss = lambda d: max(0,1-y.loc[d.name]*z(np.array(d)))

dw = lambda x,y: np.array(y*-x)
db = lambda x,y: -y

def train(x,y,a):
	global w,b
	weight = dw(x,y)
	bias = db(x,y)
	if yhat(x)*z(x)<1 or yhat(x)!=y:
		w-= a*weight
		b -= a*bias


