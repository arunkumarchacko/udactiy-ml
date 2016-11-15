# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv(r'C:\udacity\machine-learning\projects\titanic_survival_exploration\titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

train, test, train_label, test_label = cross_validation.train_test_split(X, y, test_size = 0.8, random_state = None)

clf1 = DecisionTreeClassifier()
clf1.fit(train,train_label)
treeScore = confusion_matrix(test_label,clf1.predict(test))
print "Confusion matrix for this Decision Tree:\n", treeScore

clf2 = GaussianNB()
clf2.fit(train,train_label)
nbScore = confusion_matrix(test_label,clf2.predict(test))
print "GaussianNB confusion matrix:\n",nbScore

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": nbScore,
 "Decision Tree": treeScore
}