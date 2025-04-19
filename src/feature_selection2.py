# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

mat_processed = pd.read_csv("../data/student-mat-processed.csv")
mat_processed.head()

# %%
mat_features = mat_processed.iloc[:, :15]
mat_targets = mat_processed.iloc[:, -3:]


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
correlation = {}

X = mat_features  
y_G = mat_targets  

for target_col in y_G.columns:
    y = y_G[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    correlation[target_col] = importances
    
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Feature Importances for Predicting {target_col}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


# %%
mi_results = {}

for target in mat_targets.columns:
    mi = mutual_info_regression(mat_features, mat_targets[target], random_state=42)
    mi_series = pd.Series(mi, index=mat_features.columns).sort_values(ascending=False)
    mi_results[target] = mi_series
    

    sns.barplot(x=mi_series.values, y=mi_series.index)
    plt.title(f"Mutual Information between Features and {target}")
    plt.show()


# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(mat_features, mat_targets["G3"])

importance = pd.Series(tree.feature_importances_, index=mat_features.columns)
top_features = importance.nlargest(3).index.tolist()
print("Decision Tree top features:", top_features)

# %%
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

X = mat_features
y = mat_targets["G3"]

tree = DecisionTreeRegressor(
    max_depth=3,        
    min_samples_leaf=5,
    random_state=42
)
tree.fit(X, y)  

plot_tree(
    tree,
    feature_names=X.columns.tolist(), 
    filled=True,
    rounded=True,
    proportion=True
)

plt.title("Decision Tree for Predicting Final Grade (G3)", pad=20)
plt.tight_layout() 
plt.show()

importances = pd.Series(tree.feature_importances_, index=X.columns)
importances[importances > 0].sort_values().plot(kind='barh')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Decision Tree Feature Importance", pad=15, fontsize=14)
plt.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.show()


