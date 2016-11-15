import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

from sklearn import cross_validation

train, test, train_label, test_label = cross_validation.train_test_split(X, y, test_size=.8, random_state=None) 

reg1 = DecisionTreeRegressor()
reg1.fit(train, train_label)
treeMse = mse(test_label,reg1.predict(test))
print "Decision Tree mean absolute error: {:.2f}".format(treeMse)

reg2 = LinearRegression()
reg2.fit(train, train_label)
lrMse = mse(test_label,reg2.predict(test))
print "Linear regression mean absolute error: {:.2f}".format(lrMse)

results = {
 "Linear Regression": lrMse,
 "Decision Tree": treeMse
}