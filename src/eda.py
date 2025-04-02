# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Math

# %%
mat = pd.read_csv('../data/student-mat.csv', sep=';', usecols=list(range(15, 30)))
print(mat.shape)
mat.head()

# %%
mat.info()

# %%
# Binary variables
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8):
    sns.histplot(x=mat.columns[i], data=mat, ax=axes[i // 4, i % 4])
    axes[i // 4, i % 4].set_title(mat.columns[i])

# %%
# Categorical variables
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8, 15):
    j = i - 8
    sns.histplot(x=mat.columns[i], data=mat, ax=axes[j // 4, j % 4])
    axes[j // 4, j % 4].set_title(mat.columns[i])



# %%
