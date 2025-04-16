# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

# %%
por = pd.read_csv('/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/data/student+performance/student/student-por.csv', sep=';')
por.head()

# %%
por_features = por.iloc[:, :15].copy()
por_targets = por.iloc[:, -3:].copy()

# %%
por_cat_cols = por_features.select_dtypes(include='object').columns
por_num_cols = por_features.select_dtypes(include='number').columns

por_encoder = OrdinalEncoder()
por_features[por_cat_cols] = pd.DataFrame(por_encoder.fit_transform(por_features[por_cat_cols]), columns=por_cat_cols, index=por_features.index)

por_scaler = RobustScaler()
por_features[por_num_cols] = pd.DataFrame(por_scaler.fit_transform(por_features[por_num_cols]), columns=por_num_cols, index=por_features.index)

# %%
por_processed = pd.concat([por_features, por_targets], axis=1)
por_processed.to_csv("../data/student-por-processed.csv", index=False)

# %%
for grade in por_targets.columns:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(por_cat_cols):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(x=por_features[col], y=por_targets[grade])
        plt.title(f"{col} vs {grade}")
    plt.tight_layout()
    plt.show()

# %%
for grade in por_targets.columns:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(por_num_cols):
        plt.subplot(3, 3, i + 1)
        sns.scatterplot(x=por_features[col], y=por_targets[grade])
        plt.title(f"{col} vs {grade}")
    plt.tight_layout()
    plt.show()

# %%
por_full = pd.concat([por_features, por_targets], axis=1)
plt.figure(figsize=(10, 8))
sns.heatmap(por_full.corr()[por_targets.columns].sort_values(by='G3', ascending=False),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation of Features with Grades (por)")
plt.show()


