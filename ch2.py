#%%
# Downloading the data
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
# %%
# Data summarization
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
# %%
# Data visualization
import matplotlib.pyplot as plt 
housing.hist(bins=50, figsize=(12,8))
plt.show()
# %%
# Train CV Test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Stratified sampling
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# Or use this
strat_train_set, strat_test_set = train_test_split(housing, stratify=housing['income_cat'], test_size=0.2, random_state=42)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# %%
# Data Visualization pt 2
housing = strat_train_set.copy()
plt.scatter(housing['longitude'], housing['latitude'], alpha=0.1)
plt.show()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
housing.drop('ocean_proximity', axis=1).corr()
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
# %%
# Data transformation
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
# %%
# Machine Learning cleaning
housing = strat_train_set.drop(["median_house_value"], axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Impute missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
## fitting the data just calculates the median for each column
imputer.fit(housing_num)
## transforming the data uses the fitted values in the dataset
## this creates a numpy array that will need to be wrapped into a dataframe
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
# %%
