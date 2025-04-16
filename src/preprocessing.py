# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

# %% [markdown]
# # Math

# %%
cols = list(range(15, 30))
mat = pd.read_csv("../data/student-mat.csv", sep=";", usecols=cols)
mat.head()

# %%
# Categorical
enc = OrdinalEncoder()
mat.iloc[:, 0:8] = enc.fit_transform(mat.iloc[:, 0:8])
mat.iloc[:, 0:8].head()

# %%
# Numerical
scaler = RobustScaler()
mapping = {col: np.float64 for col in mat.columns[8:15]}
mat = mat.astype(mapping)
mat.iloc[:, 8:15] = scaler.fit_transform(mat.iloc[:, 8:15])
mat.iloc[:, 8:15].head()

# %%
mat.to_csv("../data/student-mat-processed.csv", index=False)

# %% [markdown]
# # Portugese

# %%
cols = list(range(15, 30))
por = pd.read_csv("../data/student-por.csv", sep=";", usecols=cols)
por.head()

# %%
# Categorical
enc = OrdinalEncoder()
por.iloc[:, 0:8] = enc.fit_transform(por.iloc[:, 0:8])
por.iloc[:, 0:8].head()

# %%
# Numerical
scaler = RobustScaler()
mapping = {col: np.float64 for col in por.columns[8:15]}
por = por.astype(mapping)
por.iloc[:, 8:15] = scaler.fit_transform(por.iloc[:, 8:15])
por.iloc[:, 8:15].head()

# %%
por.to_csv("../data/student-por-processed.csv", index=False)


