import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

with open('Unstructured_Models/test_labels.pickle', 'rb') as file1:
    test_labels = pickle.load(file1)
with open('Unstructured_Models/test_matrix.pickle', 'rb') as file2:
    test_matrix = pickle.load(file2).todense()
with open('Unstructured_Models/training_labels.pickle', 'rb') as file3:
    training_labels = pickle.load(file3)
with open('Unstructured_Models/training_matrix.pickle', 'rb') as file4:
    training_matrix = pickle.load(file4).todense()

sc = StandardScaler()
X_train = sc.fit_transform(training_matrix)
X_test = sc.transform(test_matrix)

regressor = RandomForestClassifier(n_estimators=50, random_state=42)
regressor.fit(X_train, training_labels)
y_pred = regressor.predict(X_test)

print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))
print(accuracy_score(test_labels, y_pred))
