# Preprocessing script placeholder
import pandas as pd

# Load the dataset
df = pd.read_csv('../data/biomass_lignin_removal.csv')

# Rename the target column for consistency and code safety
df = df.rename(columns={"Delignification (%)": "lignin_removal"})

# (Optional) Show the first 5 rows for inspection
print(df.head())

# Save the cleaned data
df.to_csv('../data/cleaned_biomass_lignin_removal.csv', index=False)
