import pandas as pd
import sklearn
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")

data.head()

data.describe()

data.info()

data["Life"].value_counts()

median = data["Schooling"].median()
data["Schooling"].fillna(median)
median = data["Alcohol"].median()
data["Alcohol"].fillna(median)
median = data["Adult Mortality"].median()
data["Adult Mortality"].fillna(median)
median = data["infant deaths"].median()
data["infant deaths"].fillna(median)
median = data["percentage expenditure"].median()
data["percentage expenditure"].fillna(median)
median = data["Hepatitis B"].median()
data["Hepatitis B"].fillna(median)
median = data["Polio"].median()
data["Polio"].fillna(median)
median = data["Total expenditure"].median()
data["Total expenditure"].fillna(median)
median = data["GDP"].median()
data["GDP"].fillna(median)
median = data["Population"].median()
data["Population"].fillna(median)
median = data["Income composition of resources"].median()
data["Income composition of resources"].fillna(median)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(data)

imputer.statistics_

X = imputer.transform(data)
data_tr = pd.DataFrame(X, columns = data.columns)

data_tr.describe()

from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(data_tr, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

data_tr = train_set.drop("Life", axis=1)
data_tr_labels = train_set["Life"].copy()

corr_matrix = data.corr()
corr_matrix['Life'].sort_values(ascending=False)

data.plot(kind="scatter", x="Schooling", y="Life", alpha=0.8)
data.plot(kind="scatter", x="SIncome composition of resources", y="Life", alpha=0.8)
data.plot(kind="scatter", x="Adult Mortality", y="Life", alpha=0.8)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])
data_num_tr = my_pipeline.fit_transform(data_tr) 
data_num_tr

data_num_tr.shape

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(data_tr, data_tr_labels)

some_data = data_tr.iloc[:5]
some_labels = data_tr_labels.iloc[:5]
model.predict(some_data)

some_labels

