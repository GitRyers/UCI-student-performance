# %%

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wandb
from wandb.sklearn import plot_summary_metrics, plot_feature_importances

# %%
both = pd.read_csv('../src/v_both_processed.csv')
both_mat = both.iloc[0:382, :]
both_por = both.iloc[382:764, :]
mat_only = pd.read_csv('../src/v_only_mat_processed.csv')
por_only = pd.read_csv('../src/v_only_por_processed.csv')
mat_all = pd.concat([mat_only, both_mat], axis=0)
por_all = pd.concat([por_only, both_por], axis=0)

# %%
features = ['failures','higher','studytime','Medu','Fedu','Mjob','Fjob','schoolsup','Dalc','Walc','absences','famrel','age','sex','school','traveltime']
grades = ["G1", "G2", "G3"]
datasets = {'mat_all': mat_all,'por_all': por_all,'mat_only': mat_only,'por_only': por_only,'both_mat': both_mat,'both_por': both_por}

# %%
#MLP Regressor
for dataset, df in datasets.items():
    for grade in ["G1", "G2", "G3"]:
        X = df.drop(["G1", "G2", "G3"], axis=1)
        y = df[grade]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                           max_iter=1000, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)
    
    
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        
  

# %%
import wandb
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


wandb.login(key="795a7761141cd0d468a1377ef15963ba25683480")



def train():
    wandb.init(project="student_grades_mlp222")
    config = wandb.config  
    
    X = datasets['mat_all'][features]
    y = datasets['mat_all']['G3']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        alpha=config.alpha,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    wandb.log({"mse": mse})

sweep_config = {
    'method': 'grid',
    'metric': {'name': 'mse', 'goal': 'minimize'},
    'parameters': {
        'hidden_layer_sizes': {'values': [(50,), (100,), (50,50)]},
        'activation': {'values': ['relu', 'tanh']},
        'alpha': {'values': [0.0001, 0.001, 0.01]},
        'learning_rate_init': {'values': [0.001, 0.01]},
        'max_iter': {'values': [200, 300]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="student_grades_mlp222")
wandb.agent(sweep_id, function=train)  


# %%
wandb.finish()

# %%



