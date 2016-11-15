# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv(r'C:\udacity\machine-learning\projects\titanic_survival_exploration\titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
train, test, train_label, test_label = cross_validation.train_test_split(X, y, test_size = 0.8, random_state = None)

clf1 = DecisionTreeClassifier()
clf1.fit(train, train_label)
treef1Score = f1_score(test_label, clf1.predict(test))
print "Decision Tree F1 score: {:.2f}".format(treef1Score)

clf2 = GaussianNB()
clf2.fit(train, train_label)
nbScore = f1_score(test_label, clf2.predict(test))
print "GaussianNB F1 score: {:.2f}".format(nbScore)

F1_scores = {
 "Naive Bayes": nbScore,
 "Decision Tree": treef1Score
}