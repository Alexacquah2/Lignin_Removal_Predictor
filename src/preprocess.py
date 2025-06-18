# Preprocessing script placeholder
import pandas as pd

# Load the dataset
df = pd.read_csv('../data/biomass_lignin_removal.csv')

# Show the first 5 rows for inspection
print(df.head())

# OPTIONAL: Rename the column for consistency (recommended)
df = df.rename(columns={"Delignification (%)": "lignin_removal"})

# Save a cleaned version
df.to_csv('../data/cleaned_biomass_lignin_removal.csv', index=False)
