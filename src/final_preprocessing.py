# %%
import pandas as pd

math_df = pd.read_csv("/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/UCI-student-performance/data/student-mat.csv", sep=';')
por_df = pd.read_csv("/Users/choijunyeol/Desktop/JHU spring 2025/Gateway_Data_Science/UCI-student-performance/data/student-por.csv", sep=';')
print(math_df.shape)
print(por_df.shape)


# %%
id_columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus','Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']
##
math_df['student_id'] = math_df[id_columns].astype(str).agg('-'.join, axis=1)
por_df['student_id'] = por_df[id_columns].astype(str).agg('-'.join, axis=1)

math_ids = set(math_df['student_id'])
por_ids = set(por_df['student_id'])

both_ids = math_ids & por_ids
only_math_ids = math_ids - por_ids
only_por_ids = por_ids - math_ids

only_math_df = math_df[math_df['student_id'].isin(only_math_ids)].copy()
only_por_df = por_df[por_df['student_id'].isin(only_por_ids)].copy()
math_both_df = math_df[math_df['student_id'].isin(both_ids)].copy()
por_both_df = por_df[por_df['student_id'].isin(both_ids)].copy()

math_both_df = math_both_df.sort_values('student_id').reset_index(drop=True)
por_both_df = por_both_df.sort_values('student_id').reset_index(drop=True)

#Averaging
grades_math = math_both_df[['G1', 'G2', 'G3']]
grades_por = por_both_df[['G1', 'G2', 'G3']]
grades_avg = ((grades_math + grades_por) / 2).round()

both_df = math_both_df.copy()
both_df[['G1', 'G2', 'G3']] = grades_avg

only_math_df.drop(columns='student_id', inplace=True)
only_por_df.drop(columns='student_id', inplace=True)
both_df.drop(columns='student_id', inplace=True)

only_math_df.to_csv("only_mat_final.csv", index=False)
only_por_df.to_csv("only_por_final.csv", index=False)
both_df.to_csv("both_final.csv", index=False)

print("✔️ only_math_df:", only_math_df.shape)
print("✔️ only_por_df:", only_por_df.shape)
print("✔️ both_df (with averaged grades):", both_df.shape)


# %%
# import pandas as pd
# import numpy as np

# id_cols = [
#     'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
#     'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet'
# ]

# merged = pd.merge(mat, por,on=id_cols, how='outer', suffixes=('_mat', '_por'), indicator=True)

# only_mat = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
# only_por = merged[merged['_merge'] == 'right_only'].drop(columns=['_merge'])
# both     = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])

# print(f"only in mat: {len(only_mat)}")
# print(f"only in por: {len(only_por)}")
# print(f"in both:     {len(both)}")

# both_clean = both[id_cols].copy()

# shared_cols = [col for col in mat.columns if col in por.columns and col not in id_cols]

# for col in shared_cols:
#     col_mat = col + '_mat'
#     col_por = col + '_por'

#     if col_mat in both.columns and col_por in both.columns:
#         if np.issubdtype(both[col_mat].dtype, np.number):
#             both_clean[col] = round((both[col_mat] + both[col_por]) / 2)
#         else:
#             both_clean[col] = both[col_mat]  
#     else:
#         raise IndexError

# only_mat_cols = [col + '_mat' for col in mat.columns if col not in id_cols]
# only_mat_clean = only_mat[id_cols + only_mat_cols].copy()

# only_mat_clean.columns = id_cols + [col.replace('_mat', '') for col in only_mat_cols]



# only_por_cols = [col + '_por' for col in por.columns if col not in id_cols]
# only_por_clean = only_por[id_cols + only_por_cols].copy()
# only_por_clean.columns = id_cols + [col.replace('_por', '') for col in only_por_cols]


# # Final shapes
# print("only_mat_clean:", only_mat_clean.shape)
# print("only_por_clean:", only_por_clean.shape)
# print("both_clean:", both_clean.shape)


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

only_mat_processed = preprocess_features(only_math_df.copy())
only_por_processed = preprocess_features(only_por_df.copy())
both_processed = preprocess_features(both_df.copy())


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


