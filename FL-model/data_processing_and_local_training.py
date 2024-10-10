import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Load the CSV data
data = pd.read_csv('combined_router_data.csv')

# Data Preprocessing
data.replace("N/A", np.nan, inplace=True)
data.dropna(inplace=True)

# Encode the 'Status' column
data['Status'] = data['Status'].map({'Received': 1, 'Lost': 0})

# Convert necessary columns to numeric types
columns_to_numeric = ['Delay', 'Throughput (Mbps)', 'CW Value', 'Router ID', 'Configured Data Rate (Mbps)', 'Connected Devices']
data[columns_to_numeric] = data[columns_to_numeric].apply(pd.to_numeric)

# Add a column to indicate packet loss
data['Lost'] = data['Status'].apply(lambda x: 1 if x == 'Lost' else 0)

# Calculate the number of lost packets for each router
lost_packet_counts = data.groupby('Router ID')['Lost'].sum().reset_index()
lost_packet_counts.columns = ['Router ID', 'Lost Packet Count']

# Merge the lost packet counts back into the original data
data = pd.merge(data, lost_packet_counts, on='Router ID', how='left')

# Feature selection including the number of lost packets
features = ["Delay", "Throughput (Mbps)", "Configured Data Rate (Mbps)", "Connected Devices", "Lost Packet Count"]
target = 'CW Value'

# Normalize features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Define the set of possible CWmin values
cwmin_values_set = np.array([15, 31, 63, 127, 255, 511, 1023])

# Function to create the model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)  # Predict CW Value
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to train and evaluate a model for a specific router
def train_model_for_router(router_id, data, global_model_path):
    router_data = data[data['Router ID'] == router_id]
    X = router_data[features]
    y = router_data[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    model = create_model(len(features))
    
    # Load existing global model weights if available
    if os.path.exists(global_model_path):
        model.load_weights(global_model_path)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=1)
    
    # Evaluate the model
    val_loss, val_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Router {router_id} - Validation Loss: {val_loss}, Validation MAE: {val_mae}")
    
    return model

# Function to find the nearest CWmin value in the predefined set
def find_nearest_cwmin(value, cwmin_values_set):
    index = np.abs(cwmin_values_set - value).argmin()
    return cwmin_values_set[index]

# Global model path
global_model_path = 'global_model.h5'

# Train models for each router
router_ids = data['Router ID'].unique()
local_models = {}
predicted_cwmins = []

for router_id in router_ids:
    # Train and save the local model
    model = train_model_for_router(router_id, data, global_model_path)
    local_models[router_id] = model
    model.save(f'model_router_{router_id}.h5')
    
    # Use the model to predict optimized CWmin values
    example_data = data[data['Router ID'] == router_id][features].mean().values.reshape(1, -1)
    predicted_cwmin = model.predict(example_data)[0][0]
    nearest_cwmin = find_nearest_cwmin(predicted_cwmin, cwmin_values_set)
    predicted_cwmins.append((router_id, nearest_cwmin))
    print(f"Predicted CWmin for router {router_id}: {nearest_cwmin}")

# Write the predicted CWmin values to a text file
with open('predicted_cwmin_values.txt', 'w') as f:
    for router_id, cwmin in predicted_cwmins:
        f.write(f"router {router_id} : {int(cwmin)}\n")

print("Training completed for all routers.")
