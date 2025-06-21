import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv('../data/biomass_lignin_removal.csv')

# Rename target column
df = df.rename(columns={"Delignification (%)": "lignin_removal"})

# Handle missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = cat_imputer.fit_transform(df[[col]])
    else:
        df[col] = num_imputer.fit_transform(df[[col]])

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_columns = df.select_dtypes(include='object').columns
encoded_data = encoder.fit_transform(df[cat_columns])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_columns))

# Combine with numerical data
df_final = pd.concat([df.drop(cat_columns, axis=1), encoded_df], axis=1)

# Save processed data
df_final.to_csv('../data/processed_biomass_lignin_removal.csv', index=False)
print("Data preprocessing complete. Saved to ../data/processed_biomass_lignin_removal.csv")
