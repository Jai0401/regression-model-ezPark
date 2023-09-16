import numpy as np
import pandas as pd
from random import randint, uniform

# Set a random seed for reproducibility
np.random.seed(0)

# Number of examples in the dataset
num_examples = 300

# Generate random data for the features
traffic_data = np.random.uniform(0, 1, num_examples)  # Simulated traffic data (e.g., congestion level)
geological_distance = np.random.uniform(0, 10, num_examples)  # Simulated geological distance in miles
time_of_day = np.random.uniform(0, 24, num_examples)  # Simulated time of day (in hours)
peak_hour = np.random.randint(0, 2, num_examples)  # Simulated peak hour (0 or 1)
weekend = np.random.randint(0, 2, num_examples)  # Simulated weekend status (0 or 1)
available_slots = np.random.randint(0, 101, num_examples)  # Simulated number of available parking slots

# Generate the price feature based on the specified conditions
# Prices are higher on weekends, during peak hours, and when fewer slots are available
price = (weekend + 0.3) * (peak_hour + 0.3) * (1 - available_slots / 100) * np.random.uniform(10, 50, num_examples)

# Create a DataFrame to store the dataset
data = pd.DataFrame({
    'Traffic_Data': traffic_data,
    'Geological_Distance': geological_distance,
    'Time_of_Day': time_of_day,
    'Peak_Hour': peak_hour,
    'Weekend': weekend,
    'Available_Slots': available_slots,
    'Price': price  # Add the Price feature
})

# Display the first few rows of the dataset
print(data.head())

# Save the dataset to a CSV file (optional)
data.to_csv('smart_parking_dataset_with_price_influence.csv', index=False)
