# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
mat = pd.read_csv('../data/student-mat.csv', sep=';')
por = pd.read_csv('../data/student-por.csv', sep=';')
mat.shape, por.shape

# %%
mat.head()

# %%
mat.info()

# %%
mat.describe()

# %%
por.head()



# %%
