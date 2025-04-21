# %%
import pandas as pd

mat = pd.read_csv("/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/UCI-student-performance/data/student-mat.csv", sep=';')
por = pd.read_csv("/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/UCI-student-performance/data/student-por.csv", sep=';')
print(mat.shape)
print(por.shape)





# %%
import pandas as pd
import numpy as np

id_cols = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet'
]

merged = pd.merge(mat, por, on=id_cols, how='outer', suffixes=('_mat', '_por'), indicator=True)

only_mat = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
only_por = merged[merged['_merge'] == 'right_only'].drop(columns=['_merge'])
both     = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])

print(f"only in mat: {len(only_mat)}")
print(f"only in por: {len(only_por)}")
print(f"in both:     {len(both)}")

both_clean = both[id_cols].copy()

shared_cols = [col for col in mat.columns if col in por.columns and col not in id_cols]

for col in shared_cols:
    col_mat = col + '_mat'
    col_por = col + '_por'

    if col_mat in both.columns and col_por in both.columns:
        if np.issubdtype(both[col_mat].dtype, np.number):
            both_clean[col] = round((both[col_mat] + both[col_por]) / 2)
        else:
            both_clean[col] = both[col_mat]  
    else:
        raise IndexError

only_mat_cols = [col + '_mat' for col in mat.columns if col not in id_cols]
only_mat_clean = only_mat[id_cols + only_mat_cols].copy()

only_mat_clean.columns = id_cols + [col.replace('_mat', '') for col in only_mat_cols]

only_por_cols = [col + '_por' for col in por.columns if col not in id_cols]
only_por_clean = only_por[id_cols + only_por_cols].copy()
only_por_clean.columns = id_cols + [col.replace('_por', '') for col in only_por_cols]


# Final shapes
print("only_mat_clean:", only_mat_clean.shape)
print("only_por_clean:", only_por_clean.shape)
print("both_clean:", both_clean.shape)


# %%
both_clean.head

# %%
import numpy as np
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

target_cols = ['G1', 'G2', 'G3']

def preprocess_features(df):
    features = df.drop(columns=target_cols)
    targets = df[target_cols]

    cat_cols = features.select_dtypes(include='object').columns
    num_cols = features.select_dtypes(include='number').columns

    encoder = OrdinalEncoder()
    features[cat_cols] = pd.DataFrame(
        encoder.fit_transform(features[cat_cols]),
        columns=cat_cols,
        index=features.index
    )

    scaler = RobustScaler()
    features[num_cols] = pd.DataFrame(
        scaler.fit_transform(features[num_cols]),
        columns=num_cols,
        index=features.index
    )

    return pd.concat([features, targets], axis=1)

only_mat_processed = preprocess_features(only_mat_clean.copy())
only_por_processed = preprocess_features(only_por_clean.copy())
both_processed = preprocess_features(both_clean.copy())

only_mat_processed.to_csv('../data/student-mat-processed.csv', index=False)
only_por_processed.to_csv('../data/student-por-processed.csv', index=False)
both_processed.to_csv('../data/student-both-processed.csv', index=False)
print(both_processed.head)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
mat_features = only_mat_processed.drop(columns=['G1', 'G2', 'G3'])
mat_targets = only_mat_processed[['G1', 'G2', 'G3']]

mat_full = pd.concat([mat_features, mat_targets], axis=1)

# Plot heatmap of correlation for mat dataset
plt.figure(figsize=(10, 8))
sns.heatmap(mat_full.corr()[mat_targets.columns].sort_values(by='G3', ascending=False),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation of Features with Grades (mat)")
plt.show()


