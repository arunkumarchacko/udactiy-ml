# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv(r'C:\udacity\machine-learning\projects\titanic_survival_exploration\titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

train, test, train_label, test_label = cross_validation.train_test_split(X, y, test_size = 0.8, random_state = None)

clf1 = DecisionTreeClassifier()
clf1.fit(train, train_label)
treeRecall = recall(test_label,clf1.predict(test))
treePrecision = precision(test_label,clf1.predict(test))
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(treeRecall, treePrecision)

clf2 = GaussianNB()
clf2.fit(train, train_label)
nbRecall = recall(test_label,clf2.predict(test))
nbPrecision = precision(test_label,clf2.predict(test))
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(nbRecall, nbPrecision)

results = {
  "Naive Bayes Recall": nbRecall,
  "Naive Bayes Precision": nbPrecision,
  "Decision Tree Recall": treeRecall,
  "Decision Tree Precision": treePrecision
}