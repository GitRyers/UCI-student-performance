# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Math

# %%
cols = list(range(15, 30)) + [32]
mat = pd.read_csv('../data/student-mat.csv', sep=';', usecols=cols)
print(mat.shape)
mat.head()

# %%
mat.info()

# %%
# Binary variables
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8):
    sns.violinplot(data=mat, x=mat.columns[i], y="G3", hue=mat.columns[i], palette="bright", inner="quartile", ax=axes[i // 4, i % 4])
    axes[i // 4, i % 4].set_title(mat.columns[i])

# %% [markdown]
# Summary: 
# - School Support
#   - A LOT Less 0's 
#   - Slightly less overall though (Yes Q3 == No Q2)
# - Family Support 
#   - Equal
# - Paid
#   - Yes has less 0's and variation, but otherwise equal grades
# - Extracurricular Activities
#   - Equal
# - Nursery
#   - Equal
# - Higher Education Goals
#   - Yes is significantly higher 
# - Internet
#   - Yes is significantly higher
# - Romantic
#   - No has significantly less 0's and slightly higher overall 

# %% [markdown]
# Significant Features:
# - High: Higher Education, Internet
# - Moderate: Romantic, Paid, School Support
# - Low/None: Family Support, Extracurricular Activities, Nursery

# %%
# Categorical variables
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i in range(8, 14):
    j = i - 8
    sns.violinplot(data=mat, x=mat.columns[i], y="G3", hue=mat.columns[i], palette="colorblind", inner="quartile", ax=axes[j // 3, j % 3])
    axes[j // 3, j % 3].set_title(mat.columns[i])
    axes[j // 3, j % 3].legend_.remove()


# %% [markdown]
# Summary:
# - Family Relationships 
#   - Better relationship == higher minimum grades
# - Free Time
#   - Desc: 2, 5, 4, 3, 1
# - Go Out
#   - Desc: 2, 1, 3, 4, 5
# - Dalc
#   - Strong negative association
# - Walc
#   - Slight negative association
# - Health
#   - Extremes are better

# %% [markdown]
# Significant Features:
# - High: Dalc 
# - Moderate: Walc, Family Relationships, Health
# - Low/None: Free Time, Go Out

# %%
sns.regplot(data=mat, x="absences", y="G3", robust=True)

# %% [markdown]
# Summary:
# - Weak negative association, even when doing robust analysis 

# %% [markdown]
# # Portugese

# %%
cols = list(range(15, 30)) + [32]
por = pd.read_csv('../data/student-por.csv', sep=';', usecols=cols)
print(por.shape)
por.head()

# %%
por.info()

# %%
# Binary variables
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8):
    sns.violinplot(data=por, x=por.columns[i], y="G3", hue=por.columns[i], palette="bright", inner="quartile", ax=axes[i // 4, i % 4])
    axes[i // 4, i % 4].set_title(por.columns[i])

# %% [markdown]
# Summary: 
# - School Support
#   - A LOT Less 0's 
#   - Slightly less overall though (Yes Q3 == No Q2)
# - Family Support 
#   - Yes is slightly greater
# - Paid
#   - No is greater? 
# - Extracurricular Activities
#   - Yes is greater
# - Nursery
#   - Yes is greater
# - Higher Education Goals
#   - Yes is significantly higher 
# - Internet
#   - Yes is significantly higher
# - Romantic
#   - No has significantly less 0's and slightly higher overall 
# 

# %% [markdown]
# Significant Features:
# - High: Higher Education, Internet
# - Moderate: Romantic, Paid, School Support, Extracurricular Activities, Nursery
# - Low/None: Family Support

# %%
# Categorical variables
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i in range(8, 14):
    j = i - 8
    sns.violinplot(data=por, x=por.columns[i], y="G3", hue=por.columns[i], palette="colorblind", inner="quartile", ax=axes[j // 3, j % 3])
    axes[j // 3, j % 3].set_title(por.columns[i])
    axes[j // 3, j % 3].legend_.remove()


# %% [markdown]
# Summary:
# - Family Relationships 
#   - Moderate Positive Association
# - Free Time
#   - Desc: 2, 1, 3, 4, 5
#   - Slight Positive Association
# - Go Out
#   - Desc: 2, 3, 4, 5, 1
#   - Slight negative association
# - Dalc
#   - Moderate negative association
#   - 4 has more failing students than 5, but could just be sample size
# - Walc
#   - Slight negative association
# - Health
#   - Slight negative association

# %% [markdown]
# Significant Features:
# - High: Dalc, Family Relationships
# - Moderate: Walc, Family Relationships, Health, Go Out
# - Low/None:

# %%
sns.regplot(data=por, x="absences", y="G3", robust=True)

# %% [markdown]
# Summary:
# - Slight negative association


