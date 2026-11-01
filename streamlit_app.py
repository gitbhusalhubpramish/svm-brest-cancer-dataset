import streamlit as st
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
st.write('''## Loss and Accuracy''')

lossandacc = st.empty()
lossacc = {"loss":[], "acc":[]}

for i in range(100):
	xtrain.apply(lambda x: train(x,ytrain.loc[x.name], 0.0001), axis=1)
	lossacc["loss"].append(sum(xtest.apply(lambda x:hloss(x), axis=1))/len(xtest))
	lossacc["acc"].append(accuracy_score(xtest.apply(lambda x:yhat(x), axis=1), ytest)*100)
	lossandacc.line_chart(pd.DataFrame(lossacc))
