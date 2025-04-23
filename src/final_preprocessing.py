# %%
import pandas as pd

mat = pd.read_csv("/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/UCI-student-performance/data/student-mat.csv", sep=';')
por = pd.read_csv("/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/UCI-student-performance/data/student-por.csv", sep=';')
print(mat.shape)
print(por.shape)


# %%
### Horizontal Merging

id_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus','Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']

h_both = pd.merge(mat, por, on=id_cols, suffixes=('_mat', '_por'), how='inner')

grade_cols = ['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por']
non_grade_cols = [col for col in h_both.columns if col not in grade_cols]
h_both = h_both[non_grade_cols + grade_cols]

mat_only = pd.merge(mat, h_both[id_cols], on=id_cols, how='left', indicator=True)
h_mat_only = mat_only[mat_only['_merge'] == 'left_only'].drop(columns=['_merge'])

por_only = pd.merge(por, h_both[id_cols], on=id_cols, how='left', indicator=True)
h_por_only = por_only[por_only['_merge'] == 'left_only'].drop(columns=['_merge'])

print(h_mat_only.shape)
print(h_por_only.shape)
print(h_both.shape)
print(h_both.head)



# %% [markdown]
# 

# %%
#### Vertical Merge

import numpy as np
id_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus','Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']

merged = pd.merge(mat, por, on=id_cols, suffixes=('_mat', '_por'), how='outer', indicator=True)

both_mask = merged['_merge'] == 'both'

both_mat_rows = merged[both_mask].copy()
both_por_rows = merged[both_mask].copy()

mat_cols = id_cols + [col for col in mat.columns if col not in id_cols]
por_cols = id_cols + [col for col in por.columns if col not in id_cols]

both_mat_rows = both_mat_rows[[col if col in id_cols else f"{col}_mat" for col in mat.columns]]
both_mat_rows.columns = mat.columns

both_por_rows = both_por_rows[[col if col in id_cols else f"{col}_por" for col in por.columns]]
both_por_rows.columns = por.columns

v_both = pd.concat([both_mat_rows, both_por_rows], ignore_index=True)

mat_only = mat.merge(v_both, on=mat.columns.tolist(), how='outer', indicator=True)
v_mat_only = mat_only.query('_merge == "left_only"').drop(columns=['_merge'])

por_only = por.merge(v_both, on=por.columns.tolist(), how='outer', indicator=True)
v_por_only = por_only.query('_merge == "left_only"').drop(columns=['_merge'])

print(v_mat_only.shape)
print(v_por_only.shape)
print(v_both.shape)
print(v_both.head)

# %%
import numpy as np
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

target_cols = ['G1', 'G2', 'G3']
h_both_target = ['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por']

def preprocess_features(df):
    if df.shape[1]==53:
        features = df.drop(columns=h_both_target)
        targets = df[h_both_target]
    else:
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

h_only_mat_processed = preprocess_features(h_por_only.copy())
h_only_por_processed = preprocess_features(h_mat_only.copy())
h_both_processed = preprocess_features(h_both.copy())

v_only_mat_processed = preprocess_features(v_por_only.copy())
v_only_por_processed = preprocess_features(v_mat_only.copy())
v_both_processed = preprocess_features(v_both.copy())

h_only_mat_processed.to_csv("h_only_mat_processed.csv", index=False)
h_only_por_processed.to_csv("h_only_por_processed.csv", index=False)
h_both_processed.to_csv("h_both_processed.csv", index=False)

v_only_mat_processed.to_csv("v_only_mat_processed.csv", index=False)
v_only_por_processed.to_csv("v_only_por_processed.csv", index=False)
v_both_processed.to_csv("v_both_processed.csv", index=False)


