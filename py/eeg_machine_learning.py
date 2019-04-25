import sys
import os
sys.path.append(os.environ["PYDIR"])
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from mne_erp import eeg_evoked_grand_avg
from mne import pick_types
import time

#TODO: classify every timepoint separately, to see which ones are the most important

# Step 1: load epochs and convert data into a pandas dataframe
epochs = eeg_evoked_grand_avg(subjects=["eegl01"], condition="Augenmitte", event_id=[1,2], include_chs=[])
df = epochs.to_data_frame(index=["epoch"], start=0.6, picks=picks) # columns contain channel data, indices are "epochs", "time", and "condition"
print(df.info()) #inspect data

# Step 2: separate data in response variable X and feature variable y
X = df.drop("condition", axis=1)
y = df["condition"]
# Step 3: randomly separate data into training and test set:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# SVM Classifier:
tic = time.time()
clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))
print("this took "+str(time.time()-tic)+" seconds")


"""
# Random Forest Classifier:
tic = time.time()
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
print("this took "+str(time.time()-tic)+" seconds")


# Neural Network:
mlpc = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict
"""
