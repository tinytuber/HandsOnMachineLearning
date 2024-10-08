# Chapter 5: Support Vector Machine
#%%
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)

svm_clf = make_pipeline(StandardScaler(),
                        LinearSVC(C=1, random_state=42))
svm_clf.fit(X,y)
# %%
# Polynomial features to separate lower dimensioned data
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X,y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42)
)
polynomial_svm_clf.fit(X,y)
# %%
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1])
plt.show()
# %%
# Kernel trick
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(StandardScaler(),
                                    SVC(kernel='poly', degree=3, coef0=1, C=5))
poly_kernel_svm_clf.fit(X,y)
# %%
# Similarity Features: add dimensions using distance from landmarks
# Gamma acts as a technique for regularization. Higher gamma, more complex
rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
                                   SVC(kernel='rbf', gamma=5, C=0.001))
rbf_kernel_svm_clf.fit(X,y)
# %%
# Regression using SVM: fit as many data points in line without affecting regression
from sklearn.svm import LinearSVR
svm_reg = make_pipeline(StandardScaler(),
                        LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(X,y)
 
# %%
# Wine dataset
import sklearn
import numpy as np
wine = sklearn.datasets.load_wine()
X = wine.data
y = wine.target
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# %%
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
cats = np.unique(y_train)
classifiers = []
for i in cats:
    y_train_binary = (y_train == i).astype(int)
    svc_clf = make_pipeline(StandardScaler(),
                            SVC(kernel = "linear", probability=True))
    svc_clf.fit(X_train, y_train)
    classifiers.append(svc_clf)
# %%
from sklearn.metrics import accuracy_score
y_pred_proba = []
for classifier in classifiers:
    y_pred_proba.append(classifier.predict_proba(X_test))
y_pred = np.argmax(np.array(y_pred_proba).T, axis=1)

# %%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# %%
from sklearn.model_selection import cross_val_score
svm_clf = make_pipeline(StandardScaler(),
                        SVC(kernel="linear", gamma = 0.07969, C = 4.7454))
print(cross_val_score(svm_clf, X_train, y_train))
# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

param_distrib = {
    "svc__gamma": loguniform(0.001, 0.1),
    "svc__C": uniform(1, 10)
}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distrib, n_iter=100, cv=5,
                                   random_state=42)
rnd_search_cv.fit(X_train, y_train)
rnd_search_cv.best_estimator_
# %%
