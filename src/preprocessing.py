# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

np.random.seed(42)

# %% [markdown]
# # Math

# %%
mat = pd.read_csv("../data/student-mat.csv", sep=";")
mat.head()

# %%
# Categorical
enc = OrdinalEncoder()
mat.iloc[:, 15:23] = enc.fit_transform(mat.iloc[:, 15:23])
mat.iloc[:, 15:23].head()

# %%
# Numerical
scaler = RobustScaler()
mapping = {col: np.float64 for col in mat.columns[23:30]}
mat = mat.astype(mapping)
mat.iloc[:, 23:30] = scaler.fit_transform(mat.iloc[:, 23:30])
mat.iloc[:, 23:30].head()

# %%
X = mat.iloc[:, 0:30].values
y = mat.iloc[:, 30:]
X.shape, y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
mat_train = np.hstack((X_train, y_train))
mat_train = pd.DataFrame(mat_train, columns=mat.columns)
mat_train

# %%
mat_test = np.hstack((X_test, y_test))
mat_test = pd.DataFrame(mat_test, columns=mat.columns)
mat_test

# %%
mat_train.to_csv("../data/student-mat-train.csv", index=False)
mat_test.to_csv("../data/student-mat-test.csv", index=False)

# %% [markdown]
# # Portugese

# %%
por = pd.read_csv("../data/student-por.csv", sep=";")
por.head()

# %%
# Categorical
enc = OrdinalEncoder()
por.iloc[:, 15:23] = enc.fit_transform(por.iloc[:, 15:23])
por.iloc[:, 15:23].head()

# %%
# Numerical
scaler = RobustScaler()
mapping = {col: np.float64 for col in por.columns[23:30]}
por = por.astype(mapping)
por.iloc[:, 23:30] = scaler.fit_transform(por.iloc[:, 23:30])
por.iloc[:, 23:30].head()

# %%
X = por.iloc[:, 0:30].values
y = por.iloc[:, 30:]
X.shape, y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
por_train = np.hstack((X_train, y_train))
por_train = pd.DataFrame(por_train, columns=por.columns)
por_train

# %%
por_test = np.hstack((X_test, y_test))
por_test = pd.DataFrame(por_test, columns=por.columns)
por_test

# %%
por_train.to_csv("../data/student-por-train.csv", index=False)
por_test.to_csv("../data/student-por-test.csv", index=False)


