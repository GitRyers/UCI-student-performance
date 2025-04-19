# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

# %%
cols = list(range(1, 14))  # Selects columns 15-29
mat = pd.read_csv('/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/data/student+performance/student/student-mat.csv', sep=';')
por = pd.read_csv('/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/data/student+performance/student/student-por.csv', sep=';')
mat.head()

# %%
mat_features = mat.iloc[:, :15].copy()
mat_targets = mat.iloc[:, -3:].copy()

por_features = por.iloc[:, :15].copy()
por_targets = por.iloc[:, -3:].copy()

# %%
mat_cat_cols = mat_features.select_dtypes(include='object').columns
mat_num_cols = mat_features.select_dtypes(include='number').columns

mat_encoder = OrdinalEncoder()
mat_features[mat_cat_cols] = pd.DataFrame(mat_encoder.fit_transform(mat_features[mat_cat_cols]), columns=mat_cat_cols, index=mat_features.index)

mat_scaler = RobustScaler()
mat_features[mat_num_cols] = pd.DataFrame(mat_scaler.fit_transform(mat_features[mat_num_cols]), columns=mat_num_cols, index=mat_features.index)

# %%
por_cat_cols = por_features.select_dtypes(include='object').columns
por_num_cols = por_features.select_dtypes(include='number').columns

por_encoder = OrdinalEncoder()
por_features[por_cat_cols] = pd.DataFrame(por_encoder.fit_transform(por_features[por_cat_cols]), columns=por_cat_cols, index=por_features.index)

por_scaler = RobustScaler()
por_features[por_num_cols] = pd.DataFrame(por_scaler.fit_transform(por_features[por_num_cols]), columns=por_num_cols, index=por_features.index)

# %%
mat_processed = pd.concat([mat_features, mat_targets], axis=1)
por_processed = pd.concat([por_features, por_targets], axis=1)

mat_processed.to_csv("../data/student-mat-processed.csv", index=False)
por_processed.to_csv("../data/student-por-processed.csv", index=False)

# %%
for grade in mat_targets.columns:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(mat_cat_cols):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(x=mat_features[col], y=mat_targets[grade])
        plt.title(f"{col} vs {grade}")
    plt.tight_layout()
    plt.show()


# %%
for grade in mat_targets.columns:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(mat_num_cols):
        plt.subplot(3, 3, i + 1)
        sns.scatterplot(x=mat_features[col], y=mat_targets[grade])
        plt.title(f"{col} vs {grade}")
    plt.tight_layout()
    plt.show()

# %%
mat_full = pd.concat([mat_features, mat_targets], axis=1)
plt.figure(figsize=(10, 8))
sns.heatmap(mat_full.corr()[mat_targets.columns].sort_values(by='G3', ascending=False),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation of Features with Grades (mat)")
plt.show()


