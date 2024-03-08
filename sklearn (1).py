import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('1Datasets/labeled.csv') # load our data with pandas

xtrain_data, xtest_data, ytrain, ytest = sklearn.model_selection.train_test_split(data['comment'], data['toxic'], test_size=0.2) # split data on train and test splits

token = TfidfVectorizer(max_features=5000) # create tokenizer object
xtrain = token.fit_transform(xtrain_data)  # tokenize train and test splits
xtest = token.transform(xtest_data)

model = LogisticRegression() # creating LogisticRegression model

model.fit(xtrain, ytrain) # train model with train splits

y_pred = model.predict(xtest) # testing model

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ytest, y_pred) # looking for accuracy in test split