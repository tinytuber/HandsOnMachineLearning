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

## Look at precision/specificity and recall/sensitivity
## Precision = True positives out of all positives (TP + FP)
## Recall = How many positives were really predicted (TP + FN)
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
# %%
