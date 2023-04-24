import pandas as pd

# Load the dataset
data = pd.read_csv('output//dataset_merged.csv')

# Remove empty values (NaNs)
data = data.dropna()

# Remove outliers using the IQR method
Q1 = data.quantile(0.25, numeric_only=True)
Q3 = data.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
filtered_data = data.copy()

for col in data.columns:
    if col in lower_bound.index and col in upper_bound.index:
        mask = (data[col] >= lower_bound[col]) & (data[col] <= upper_bound[col])
        filtered_data = filtered_data.loc[mask]

# Save the cleaned dataset to a new file
filtered_data.to_csv('output//dataset_merged_st.csv', index=False)