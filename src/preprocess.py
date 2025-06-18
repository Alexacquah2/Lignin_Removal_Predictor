# Preprocessing script placeholder
import pandas as pd

# Load the dataset
df = pd.read_csv('../data/biomass_lignin_removal.csv')

# Rename the target column for consistency and code safety
df = df.rename(columns={"Delignification (%)": "lignin_removal"})

# Set pandas to display all rows
pd.set_option('display.max_rows', None)

# Show all rows for inspection
print(df)

# Save the cleaned data
df.to_csv('../data/cleaned_biomass_lignin_removal.csv', index=False)
