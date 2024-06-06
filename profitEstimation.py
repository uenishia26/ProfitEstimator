import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

df = pd.read_csv("/Users/anamuuenishi/Desktop/dataEntryEnv/practiceCSVML/1000_Companies.csv")
print(df)


#Store as numpy array instead of Pandas df
X = df.iloc[:,[0,1,2,3]].values #X = df.iloc[:, :-1] Same syntax but shorter 
y = df.iloc[:, 4].values

"""
*** Quick Syntax recap 
- Handle_unknown: just ignores all unknwon categories in the Data
- Sparse matrix or dense array (Numpy / DataFrame)
    Dense array stores all the values even if its zero... Most languages use dense array 
    Sparse matrices only stores nonzero values 
- Setoutput is required to set the output or it will return Numpy matrix by default
""" 

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
oheDF = ohe.fit_transform(df[['State']]) #Creates three additional colums

#Keeping Profit column at the end of DF / Re organziing + adding columns 
profitColum = df[["Profit"]]
df = pd.concat([df, oheDF], axis=1).drop(columns=['State','Profit']) #axis=1 just means columns. 0=rows
df = pd.concat([df, profitColum], axis=1)


X = df.iloc[:,:-1] #All columns + rows except last column 
y = df.iloc[:,-1]

#Training regressor model 
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
linReg = LinearRegression()
linReg.fit(xtrain, ytrain) 

#Quality check 
ypred = linReg.predict(xtest)
print(r2_score(ytest, ypred))







