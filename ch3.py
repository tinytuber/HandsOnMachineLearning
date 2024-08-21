#%%
# MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000],y[60000:]
# %%
# Visualize the numbers
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()
# %%
# %%
# Binary classifier for '5'
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
# %%
# Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# %%
# Using cross validation to check model accuracy
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# %%
# Look at confusion matrix instead
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

## Look at the confusion matrix of the predicted to actual
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)

## Look at precision and recall/sensitivity
## Precision = True positives out of all positives (TP + FP)
## Recall/Sensitivity = How many positives were really predicted (TP + FN)
## Specificity = True negatives out of all negatives (TN + FP)
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))
# %%
# Changing threshhold scores to affect precision and recall
y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
y_some_digit_prediction = (y_scores > threshold)

## Deciding decision threshold
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method='decision_function')

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plt.plot(thresholds, precisions[:-1], "b--")
plt.plot(thresholds, recalls[:-1], "b--")
plt.show()

idx_for_90_precision = (precisions>=.9).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
# %%
# ROC Curve to look at sensitivity versus 1-specificity
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.show()

## Calculate ROC AUC
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
# %%
# Random Forest classifier. In this case, it gives a much higher ROC AUC and precision/recall F1
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:,1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest
)
# %%
plt.plot(recalls_forest, precisions_forest, "b-")
plt.plot(recalls, precisions, "--")
plt.show()
# %%
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
print(f1_score(y_train_5, y_train_pred_forest))
print(roc_auc_score(y_train_5, y_scores_forest))
# %%
# Multiclass Classification
## One versus rest/all: Train a classifier for each class/number and see which is highest score
## One versus one: train a classifier on each pair of classes and see which class wins the most
## Ex: training an SVR on multiclass classification
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
svm_clf.predict([some_digit])
some_digit_scores = svm_clf.decision_function([some_digit]).round(2)

# %%
# OvR strategy
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
ovr_clf.predict([some_digit])

# %%
# SGD Classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train[:1000], y_train[:1000])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[:1000].astype("float64"))

## Confusion matrix for error analysis
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train[:1000], cv=3)
#ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
#plt.show()
# %%
# Error analysis pt 2: normalize the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_train[:1000], y_train_pred,
                                        normalize='true', values_format=".0%")
plt.show()
# %%
# Multilabel Classification
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
# %%
# Chain classification for processing serially
from sklearn.multioutput import ClassifierChain
chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

# %%
# Multioutput classification: each label can be multiclass
## Add noise to all images
np.random.seed(42)  
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

## Train classifier to clean image
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
# %%
# Exercise 1: create a MNIST classifier model that achieves over 97% accuracy on test set
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)

## Get data
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000],y[60000:]

'''
## Scale X data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''

## KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=3)
knn_clf.fit(X_train, y_train)

## Accuracy using CV
from sklearn.model_selection import cross_val_score
print(knn_clf.score(X_test, y_test))
#print(cross_val_score(knn_clf, X_test, y_test, cv=2, scoring='accuracy'))
# %%
from sklearn.model_selection import GridSearchCV

parameters = {'weights':('uniform', 'distance'), 'n_neighbors':[1,2,3]}
clf = GridSearchCV(knn_clf, parameters, verbose=10, cv=2)
clf.fit(X_train, y_train)
# %%
# Exeercise 2