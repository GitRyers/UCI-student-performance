# %%
# Call datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
np.random.seed(42)


# %%
both = pd.read_csv('../src/v_both_processed.csv')
both_mat = both.iloc[0:382, :]
both_por = both.iloc[382:764, :]

# %%
mat_only = pd.read_csv('../src/v_only_mat_processed.csv')
por_only = pd.read_csv('../src/v_only_por_processed.csv')
mat_all = pd.concat([mat_only, both_mat], axis=0)
por_all = pd.concat([por_only, both_por], axis=0)


# %%
features = ['failures','higher','studytime','Medu','Fedu','Mjob','Fjob','schoolsup','Dalc','Walc','absences','famrel','age','sex','school','traveltime']


# %%
#Logistic Regression
datasets = [mat_all, por_all, mat_only, por_only, both_mat, both_por]
dataset_names = ['mat_all', 'por_all', 'mat_only', 'por_only', 'both_mat', 'both_por']

for data, name in zip(datasets, dataset_names):
    print(f"\n{name}")
    for grade in ['G1', 'G2', 'G3']:
        X = data[features]
        y = (data[grade] >= 10).astype(int)  #logistic regression is binary
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        print(f"{grade}: {acc:.4f}")
  

# %%
def get_accuracy(y):
    return (y >= 10).astype(int)

# %%


grades = ["G1", "G2", "G3"]
datasets = {'mat_all': mat_all,'por_all': por_all,'mat_only': mat_only,'por_only': por_only,'both_mat': both_mat,'both_por': both_por}

for name, data in datasets.items():
    print(name)
    
    X = data[features]
    
    for grade in grades:
        y = data[grade]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        acc = accuracy_score(get_accuracy(y_test), get_accuracy(y_pred))
        
        print(f"ElasticNet - {grade} - Accuracy: {acc:.4f}")
        print(f"Grade: {grade} - MSE: {rmse:.4f}")
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Grade")
    plt.ylabel("Predicted Grade")


# %%
for dataset in datasets:
    for grade in grades:
        X = data[features]  
        y = data[grade] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        acc = accuracy_score(get_accuracy(y_test), get_accuracy(y_pred))
        
        print(f"Randomforest - {grade} - Accuracy: {acc:.4f}")

        print(f"Dataset: {dataset} | Grade: {grade} | RMSE: {rmse}")
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Grade")
    plt.ylabel("Predicted Grade")


