# Decision Trees
#%% 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(X_iris,y_iris)
# %%
## Sample = number of instances that fit the criteria
## Value = number of instances from each category that fit
## Gini = purity of the sample (only 1 category)
'''
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file="iris_tree.dot",
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

from graphviz import Source

Source.from_file("iris_tree.dot")
'''
# %%
tree_clf.predict_proba([[5,1.5]])
# %%
#%%
# Regression with Decision Trees
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5  # a single random input feature
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

tree_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_reg.fit(X_quad, y_quad)
# %%
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

moons = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.2)
tree_clf = DecisionTreeClassifier()

params = {
    "max_depth": [5,8,12],
    "max_leaf_nodes": [20,40,80,150]
}
grid_search = GridSearchCV(tree_clf, params, cv=3, scoring="accuracy")
grid_search.fit(X_train,y_train)
# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)
# %%
tree_clf = DecisionTreeClassifier(max_depth=8,max_leaf_nodes=12)
tree_clf.fit(X_train,y_train)
tree_clf.score(X_test,y_test)
# %%
