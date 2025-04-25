# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, r_regression, f_regression, mutual_info_regression, RFECV, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

np.random.seed(42)

# %% [markdown]
# # ETL

# %% [markdown]
# ## Both

# %%
both = pd.read_csv('../data/v_both_processed.csv')
print(both.shape)
both.head()

# %%
both_mat = both.iloc[0:382, :]
print(both_mat.shape)
both_mat.head()

# %%
both_por = both.iloc[382:764, :]
print(both_por.shape)
both_por.head()

# %% [markdown]
# ## Math

# %%
mat_only = pd.read_csv('../data/v_only_mat_processed.csv')
print(mat_only.shape)
mat_only.head()

# %%
mat_all = pd.concat([mat_only, both_mat], axis=0)
print(mat_all.shape)
mat_all.head()

# %% [markdown]
# ## Portugese

# %%
por_only = pd.read_csv('../data/v_only_por_processed.csv')
print(por_only.shape)
por_only.head()

# %%
por_all = pd.concat([por_only, both_por], axis=0)
print(por_all.shape)
por_all.head()

# %% [markdown]
# # Correlation Analysis

# %% [markdown]
# ## Functions

# %%
def get_corr(data, target, method, thres):
    corr = data.corr(method=method)[target].sort_values(ascending=False)
    corr.drop(["G1", "G2", "G3"], inplace=True)
    return corr.loc[corr.abs() > thres]

# %%
def plot_corr(corr, target, method, ax):
    sns.barplot(x=corr.index, y=corr, ax=ax)
    ax.set_xlabel("Features")
    ax.set_xticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.index, rotation=90)
    ax.set_title(f"{method.title()} Correlation with {target}")

# %%
def plot_data(data):
    N = 3
    fig, ax = plt.subplots(N, N, figsize=(15, 15))

    targets = ["G1", "G2", "G3"]
    methods = ["pearson", "spearman", "kendall"]
    total_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
    for i in range(N):
        target_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
        for j in range(N):
            corr = get_corr(data, targets[i], methods[j], 0.05)
            plot_corr(corr, targets[i], methods[j], ax[i][j])  # 5% is a common threshold
            for feature in corr.index:
                score = np.sqrt(np.abs(corr[feature])) # sqrt since between 0 and 1 
                total_freq[feature] += score
                target_freq[feature] += score  
        target_freq = sorted(list(target_freq.items()), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top features for {targets[i]}: {[k for k, _ in target_freq]}")
    total_freq = sorted(list(total_freq.items()), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top features overall: {[k for k, _ in total_freq]}")
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Math 

# %%
plot_data(mat_all)

# %%
plot_data(mat_only)

# %%
plot_data(both_mat)

# %% [markdown]
# ## Portugese

# %%
plot_data(por_all)

# %%
plot_data(por_only)

# %%
plot_data(both_por)

# %% [markdown]
# - Promising Features
#   - failures
#   - higher
#   - school
#   - studytime 
#   - Dalc
#   - Walc
#   - Medu/Mjob
#   - Fedu/Fjob
#   - address 
#   - reason
#   - health
#   - sex

# %% [markdown]
# # Variance Analysis

# %% [markdown]
# ## Functions

# %%
def get_features_variance(data, threshold):
    X = data.drop(["G1", "G2", "G3"], axis=1)
    var_thres = VarianceThreshold(threshold=threshold)
    var_thres.fit(X.values)
    return X.columns[var_thres.get_support()]

# %%
def print_features_variance(data):
    thresholds = [0.05, 0.15, 0.25]
    for threshold in thresholds:
        print(f"Features with variance > {threshold}:")
        features = get_features_variance(mat_all, threshold)
        print(features.tolist())

# %% [markdown]
# ## Math

# %%
print_features_variance(mat_all)

# %%
print_features_variance(mat_only)

# %%
print_features_variance(both_mat)

# %% [markdown]
# ## Portugese

# %%
print_features_variance(por_all)

# %%
print_features_variance(por_only)

# %%
print_features_variance(both_por)

# %% [markdown]
# # Unsupervised Learning

# %% [markdown]
# ## Functions

# %%
def plot_pca(data):
    n_components = 5
    pca = PCA(n_components=n_components)
    X = data.drop(["G1", "G2", "G3"], axis=1)
    pca.fit(X.values)
    
    fig, ax = plt.subplots(n_components,1, figsize=(15, 20))
    for i in range(n_components):
        ax[i].bar(X.columns, pca.components_[i], color='purple')
        ax[i].set_xlabel('Features')
        ax[i].set_xticks(np.arange(len(X.columns)))
        ax[i].set_xticklabels(X.columns, rotation=90)
        ax[i].set_ylabel('Component Weight')
        ax[i].set_title(f'Feature Contributions to PC{i+1}')
        explained_var = pca.explained_variance_ratio_[i]
        ax[i].annotate(f'Explained Variance: {explained_var:.2%}', 
                        xy=(0.95, 0.9), 
                        xycoords='axes fraction', 
                        fontsize=10, 
                        ha='right', 
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightgrey'))

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## PCA

# %% [markdown]
# ### Math

# %%
plot_pca(mat_all)

# %%
plot_pca(mat_only)

# %%
plot_pca(both_mat)

# %% [markdown]
# ### Portugese

# %%
plot_pca(por_all)

# %%
plot_pca(por_only)

# %%
plot_pca(both_por)

# %% [markdown]
# # Select K Best

# %%
n_features_to_select = 10

# %% [markdown]
# ## Functions

# %%
def get_features_k_best(data, target, k, score_func):
    X = data.drop(["G1", "G2", "G3"], axis=1)
    y = data[target]
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X.values, y.values)
    return X.columns[selector.get_support()]

# %%
def print_features_k_best(data, k, score_func, method):
    feature_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
    for i in range(3):
        features = get_features_k_best(data, f"G{i+1}", k, score_func)
        for feature in features:
            feature_freq[feature] += 1
        print(f"Top {n_features_to_select} features for G{i+1} using {method}:")
        print(features.tolist())
        print("\n")
    feature_freq = sorted(list(feature_freq.items()), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top {n_features_to_select} features overall using {method}:")
    print([k for k, _ in feature_freq])

# %% [markdown]
# ## $R^{2}$

# %% [markdown]
# ### Math

# %%
print_features_k_best(mat_all, n_features_to_select, r_regression, "R^2")

# %%
print_features_k_best(mat_only, n_features_to_select, r_regression, "R^2")

# %%
print_features_k_best(both_mat, n_features_to_select, r_regression, "R^2")

# %% [markdown]
# ### Portugese

# %%
print_features_k_best(por_all, n_features_to_select, r_regression, "R^2")

# %%
print_features_k_best(por_only, n_features_to_select, r_regression, "R^2")

# %%
print_features_k_best(both_por, n_features_to_select, r_regression, "R^2")

# %% [markdown]
# ## $F_1$

# %% [markdown]
# ### Math

# %%
print_features_k_best(mat_all, n_features_to_select, f_regression, "F score")

# %%
print_features_k_best(mat_only, n_features_to_select, f_regression, "F score")

# %%
print_features_k_best(both_mat, n_features_to_select, f_regression, "F score")

# %% [markdown]
# ### Portugese

# %%
print_features_k_best(por_all, n_features_to_select, f_regression, "F score")

# %%
print_features_k_best(por_only, n_features_to_select, f_regression, "F score")

# %%
print_features_k_best(both_por, n_features_to_select, f_regression, "F score")

# %% [markdown]
# ## Mutual Information

# %% [markdown]
# ### Math

# %%
print_features_k_best(mat_all, n_features_to_select, mutual_info_regression, "Mutual Information")

# %%
print_features_k_best(mat_only, n_features_to_select, mutual_info_regression, "Mutual Information")

# %%
print_features_k_best(both_mat, n_features_to_select, mutual_info_regression, "Mutual Information")

# %% [markdown]
# ### Portugese

# %%
print_features_k_best(por_all, n_features_to_select, mutual_info_regression, "Mutual Information")

# %%
print_features_k_best(por_only, n_features_to_select, mutual_info_regression, "Mutual Information")

# %%
print_features_k_best(both_por, n_features_to_select, mutual_info_regression, "Mutual Information")

# %% [markdown]
# # Wrapper Based Methods

# %% [markdown]
# ## Functions

# %%
def get_features_sfs(data, target, n_features_to_select, direction='forward', cv=5):
    X = data.drop(["G1", "G2", "G3"], axis=1)
    y = data[target]
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features_to_select, direction=direction, cv=cv)
    sfs.fit(X.values, y.values)
    return X.columns[sfs.get_support()]

# %%
def print_features_sfs(data, k, direction='forward'):
    feature_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
    for i in range(3):
        features = get_features_sfs(data, f"G{i+1}", k, direction)
        for feature in features:
            feature_freq[feature] += 1
        print(f"Top {n_features_to_select} features for G{i+1} using {direction} Selection:")
        print(features.tolist())
        print("\n")
    feature_freq = sorted(list(feature_freq.items()), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top {n_features_to_select} features overall using {direction} Selection:")
    print([k for k, _ in feature_freq])

# %%
def get_features_rfecv(data, target, n_features_to_select, cv=5):
    X = data.drop(["G1", "G2", "G3"], axis=1)
    y = data[target]
    rfecv = RFECV(LinearRegression(), step=1, min_features_to_select=n_features_to_select, cv=cv)
    rfecv.fit(X.values, y.values)
    return X.columns[rfecv.get_support()]

# %%
def print_features_rfecv(data, k):
    feature_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
    for i in range(3):
        features = get_features_rfecv(data, f"G{i+1}", k)
        for feature in features:
            feature_freq[feature] += 1
        print(f"Top {n_features_to_select} features for G{i+1} using RFECV:")
        print(features.tolist())
        print("\n")
    feature_freq = sorted(list(feature_freq.items()), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top {n_features_to_select} features overall using RFECV:")
    print([k for k, _ in feature_freq])

# %% [markdown]
# ## Forward Selection

# %% [markdown]
# ### Math

# %%
print_features_sfs(mat_all, n_features_to_select, "forward")

# %%
print_features_sfs(mat_only, n_features_to_select, "forward")

# %%
print_features_sfs(both_mat, n_features_to_select, "forward")

# %% [markdown]
# ### Portugese

# %%
print_features_sfs(por_all, n_features_to_select, "forward")

# %%
print_features_sfs(por_only, n_features_to_select, "forward")

# %%
print_features_sfs(both_por, n_features_to_select, "forward")

# %% [markdown]
# ## Backward Selection

# %% [markdown]
# ### Math

# %%
print_features_sfs(mat_all, n_features_to_select, "backward")

# %%
print_features_sfs(mat_only, n_features_to_select, "backward")

# %%
print_features_sfs(both_mat, n_features_to_select, "backward")

# %% [markdown]
# ### Portugese

# %%
print_features_sfs(por_all, n_features_to_select, "backward")

# %%
print_features_sfs(por_only, n_features_to_select, "backward")

# %%
print_features_sfs(both_por, n_features_to_select, "backward")

# %% [markdown]
# ## Recursive Selection

# %% [markdown]
# ### Math

# %%
print_features_rfecv(mat_all, n_features_to_select)

# %%
print_features_rfecv(mat_only, n_features_to_select)

# %%
print_features_rfecv(both_mat, n_features_to_select)

# %% [markdown]
# ### Portugese

# %%
print_features_rfecv(por_all, n_features_to_select)

# %%
print_features_rfecv(por_only, n_features_to_select)

# %%
print_features_rfecv(both_por, n_features_to_select)

# %% [markdown]
# # Regularization 

# %% [markdown]
# ## Functions

# %%
def get_reg(data, target, l1_ratio, threshold = 0.1, cv=5):
    X = data.drop(["G1", "G2", "G3"], axis=1)
    y = data[target]
    en = ElasticNetCV(l1_ratio=l1_ratio, cv=cv)
    en.fit(X.values, y.values)
    en_coef = pd.Series(en.coef_, index=X.columns) 
    return en_coef[en_coef.abs() > threshold].sort_values(ascending=False)

# %%
def plot_reg(en_coef, l1_ratio, target, ax):
    sns.barplot(x=en_coef.index, y=en_coef, ax=ax)
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(en_coef.index)))
    ax.set_xticklabels(en_coef.index, rotation=90)
    ax.set_ylabel("Coefficients")
    ax.set_title(f"ElasticNet (L1 Ratio: {l1_ratio} Regularization for {target}")

# %%
def plot_features_regularization(data):
    _, ax = plt.subplots(4, 3, figsize=(25, 30))
    
    l1_ratios = [0.25, 0.5, 0.75, 1.0]
    total_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
    for i in range(4):
        target_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
        for j in range(3):
            en_coef = get_reg(data, f"G{j+1}", l1_ratios[i])
            plot_reg(en_coef, l1_ratios[i], f"G{j+1}", ax[i][j])
            for feature in en_coef.index:
                score = np.sqrt(np.abs(en_coef[feature])) # sqrt since between 0 and 1 
                total_freq[feature] += score
                target_freq[feature] += score  
        target_freq = sorted(list(target_freq.items()), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top features for L1 ratio: {l1_ratios[i]}: {[k for k, _ in target_freq]}")
    total_freq = sorted(list(total_freq.items()), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top features overall: {[k for k, _ in total_freq]}")
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Math

# %%
plot_features_regularization(mat_all)

# %%
plot_features_regularization(mat_only)

# %%
plot_features_regularization(both_mat)

# %% [markdown]
# ## Portugese

# %%
plot_features_regularization(por_all)

# %%
plot_features_regularization(por_only)

# %%
plot_features_regularization(both_por)

# %% [markdown]
# # Tree Methods

# %% [markdown]
# ## Functions

# %%
def get_features_tree(data, target, tree_model, n_features = 10):
    X = data.drop(["G1", "G2", "G3"], axis=1)
    y = data[target]
    tree = tree_model()
    tree.fit(X.values, y.values)
    feature_importances = pd.Series(tree.feature_importances_, index=X.columns)
    return feature_importances.nlargest(n_features).sort_values(ascending=False)

# %%
def plot_tree(tree_coef, tree_model, target, ax):
    sns.barplot(x=tree_coef.index, y=tree_coef, ax=ax)
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(tree_coef.index)))
    ax.set_xticklabels(tree_coef.index, rotation=90)
    ax.set_ylabel("Coefficients")
    ax.set_title(f"{tree_model.__name__} Feature Importances for {target}")

# %%
def plot_features_tree(data):
    _, ax = plt.subplots(3, 3, figsize=(25, 30))
    
    tree_models = [DecisionTreeRegressor, ExtraTreesRegressor, RandomForestRegressor]
    total_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
    for i in range(3):
        target_freq = {data.columns[i]: 0 for i in range(len(data.columns) - 3)}
        for j in range(3):
            tree_model = tree_models[i]
            tree_coef = get_features_tree(data, f"G{j+1}", tree_model)
            plot_tree(tree_coef, tree_model, f"G{j+1}", ax[i][j])
            for feature in tree_coef.index:
                score = np.sqrt(np.abs(tree_coef[feature])) # sqrt since between 0 and 1 
                total_freq[feature] += score
                target_freq[feature] += score  
        target_freq = sorted(list(target_freq.items()), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top features for {tree_model.__name__}: {[k for k, _ in target_freq]}")
    total_freq = sorted(list(total_freq.items()), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top features overall: {[k for k, _ in total_freq]}")
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Math

# %%
plot_features_tree(mat_all)

# %%
plot_features_tree(mat_only)

# %%
plot_features_tree(both_mat)

# %% [markdown]
# ## Portugese

# %%
plot_features_tree(por_all)

# %%
plot_features_tree(por_only)

# %%
plot_features_tree(both_por)


