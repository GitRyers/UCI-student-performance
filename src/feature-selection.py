# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, r_regression, f_regression, mutual_info_regression, RFECV, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

np.random.seed(42)

# %% [markdown]
# # ETL

# %%
mat = pd.read_csv("../data/student-mat-train.csv")
mat.head()

# %%
X, y = mat.iloc[:, 15:30].values, mat.iloc[:, 32].values
X.shape, y.shape

# %% [markdown]
# # Correlation Analysis

# %%
def plot_corr(method, ax):
    corr = mat.iloc[:, 15:].corr(method=method)["G3"].sort_values(ascending=False)
    corr.drop(["G3", "G2", "G1"], inplace=True)
    sns.barplot(x=corr.index, y=corr, ax=ax)
    ax.set_xlabel("Features")
    ax.set_xticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.index, rotation=90)
    ax.set_title(f"{method.title()} Correlation")

# %%
N = 3
fig, ax = plt.subplots(1, N, figsize=(10, 8))

methods = ["pearson", "spearman", "kendall"]
for i in range(N):
    plot_corr(methods[i], ax[i])

plt.tight_layout()
plt.show()

# %%
n_features_to_select = 5

# %% [markdown]
# # Univariate Feature Selection 

# %% [markdown]
# ## Select K Best

# %% [markdown]
# ### $R^{2}$

# %%
best_r2 = SelectKBest(score_func=r_regression, k=n_features_to_select)
best_r2.fit_transform(X, np.ravel(y))
mat.columns[15:30][best_r2.get_support()]

# %% [markdown]
# ### $F_1$

# %%
best_f1 = SelectKBest(f_regression, k=n_features_to_select)
best_f1.fit_transform(X, np.ravel(y))
mat.columns[15:30][best_f1.get_support()]

# %% [markdown]
# ### Mutual Information

# %%
best_mut_info = SelectKBest(mutual_info_regression, k=n_features_to_select)
best_mut_info.fit_transform(X, np.ravel(y))
mat.columns[15:30][best_mut_info.get_support()]

# %% [markdown]
# # Sequential Feature Selection

# %% [markdown]
# ## Forward Selection

# %%
lr = LinearRegression()
forward_selector = SequentialFeatureSelector(lr, n_features_to_select=n_features_to_select, direction="forward")
forward_selector.fit(X, np.ravel(y))
mat.columns[15:30][forward_selector.get_support()]

# %% [markdown]
# ## Backward Selection

# %%
lr = LinearRegression()
backward_selector = SequentialFeatureSelector(lr, n_features_to_select=n_features_to_select, direction="backward")
backward_selector.fit(X, np.ravel(y))
mat.columns[15:30][backward_selector.get_support()]

# %% [markdown]
# # Recursive Selection

# %%
lr = LinearRegression()
recursive_selector = RFECV(lr, step=1, min_features_to_select=n_features_to_select, cv=5)
recursive_selector.fit(X, np.ravel(y))
mat.columns[15:30][recursive_selector.get_support()]

# %% [markdown]
# # Regularization 

# %% [markdown]
# ## Lasso (L1)

# %%
l1 = LassoCV()
l1.fit(X, np.ravel(y))
l1_coef = pd.Series(l1.coef_, index=mat.columns[15:30])
l1_coef = l1_coef[l1_coef != 0].map(lambda x: abs(x))
l1_coef.sort_values(ascending=False)

# %% [markdown]
# ## Ridge (L2)

# %%
l2 = RidgeCV()
l2.fit(X, np.ravel(y))
l2_coef = pd.Series(l2.coef_, index=mat.columns[15:30])
l2_coef = l2_coef.map(lambda x: abs(x))
l2_coef.sort_values(ascending=False)

# %% [markdown]
# ## ElasticNet

# %%
elastic = ElasticNetCV()
elastic.fit(X, np.ravel(y))
elastic_coef = pd.Series(elastic.coef_, index=mat.columns[15:30])
elastic_coef = elastic_coef.map(lambda x: abs(x))
elastic_coef.sort_values(ascending=False)

# %% [markdown]
# ## Tree Methods

# %%
tree = DecisionTreeRegressor()
tree.fit(X, np.ravel(y))
tree_coef = pd.Series(tree.feature_importances_, index=mat.columns[15:30])
tree_coef.sort_values(ascending=False)

# %%
xtree = ExtraTreesRegressor()
xtree.fit(X, np.ravel(y))
xtree_coef = pd.Series(xtree.feature_importances_, index=mat.columns[15:30])
xtree_coef.sort_values(ascending=False)

# %%
rf = RandomForestRegressor()
rf.fit(X, np.ravel(y))
rf_coef = pd.Series(rf.feature_importances_, index=mat.columns[15:30])
rf_coef.sort_values(ascending=False)


