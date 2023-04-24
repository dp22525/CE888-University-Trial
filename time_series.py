import os
import pandas as pd
import matplotlib.pyplot as plt

PATH = 'output'

# Load the dataset
data = pd.read_csv(os.path.join(PATH, 'dataset_merged_st.csv'))

# Drop the 'new_id' column
data = data.drop('new_id', axis=1)

data['datetime'] = pd.to_datetime(data['datetime'])

# Set the custom date and time range
start_datetime = '2021-01-01 00:00:00'
end_datetime = '2021-01-02 00:00:00'

# Set the candidate ids
candidate_1_id = "S1"
candidate_2_id = "S2"

# Filter the data within the specified date and time range for each candidate
filtered_data_1 = data[(data['id'] == candidate_1_id) & (data['datetime'] >= start_datetime) & (data['datetime'] <= end_datetime)]
filtered_data_2 = data[(data['id'] == candidate_2_id) & (data['datetime'] >= start_datetime) & (data['datetime'] <= end_datetime)]

# Plot the time series of the heart rate data for both candidates within the custom date and time range
plt.figure(figsize=(12, 6))
plt.plot(filtered_data_1['datetime'], filtered_data_1['HR'], label=f'Candidate {candidate_1_id}')
plt.plot(filtered_data_2['datetime'], filtered_data_2['HR'], label=f'Candidate {candidate_2_id}')
plt.xlabel('Datetime')
plt.ylabel('Heart Rate')
plt.title('Heart Rate Time Series')
plt.legend()
plt.show()