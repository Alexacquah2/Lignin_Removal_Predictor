# Preprocessing script placeholder
import pandas as pd

# Load the dataset
df = pd.read_csv('../data/biomass_lignin_removal.csv')

# Show the first 5 rows (for initial inspection)
print(df.head())

# Save a cleaned version (for now, just a copy)
df.to_csv('../data/cleaned_biomass_lignin_removal.csv', index=False)
