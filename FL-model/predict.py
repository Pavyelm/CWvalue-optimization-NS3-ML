import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Set of allowed CWmin values
VALID_CWMIN_VALUES = np.array([15, 31, 63, 127, 255, 511, 1023])

def load_data(file_path):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    
    # Data Preprocessing
    data.replace("N/A", np.nan, inplace=True)
    data.dropna(inplace=True)
    
    data['Status'] = data['Status'].map({'Received': 1, 'Lost': 0})
    
    columns_to_numeric = ['Delay', 'Throughput (Mbps)', 'CW Value', 'Router ID', 'Configured Data Rate (Mbps)', 'Connected Devices']
    
    data[columns_to_numeric] = data[columns_to_numeric].apply(pd.to_numeric)
    
    data['Lost'] = data['Status'].apply(lambda x: 1 if x == 'Lost' else 0)
    
    lost_packet_counts = data.groupby('Router ID')['Lost'].sum().reset_index()
    lost_packet_counts.columns = ['Router ID', 'Lost Packet Count']
    
    data = pd.merge(data, lost_packet_counts, on='Router ID', how='left')
    
    return data

def prepare_input_data(data, router_id, features, scaler):
    """Prepare the input data for prediction."""
    router_data = data[data['Router ID'] == int(router_id)]
    
    # Check if router_data is empty
    if router_data.empty:
        print(f"No data available for router {router_id}")
        return None
    
    print(f"Data for router {router_id}: {router_data.shape}")
    scaled_features = scaler.transform(router_data[features])
    input_data = np.mean(scaled_features, axis=0).reshape(1, -1)
    return input_data

def map_to_valid_cwmin(predicted_value):
    """Map the predicted CWmin value to the nearest valid value."""
    return VALID_CWMIN_VALUES[np.argmin(np.abs(VALID_CWMIN_VALUES - predicted_value))]

def predict_cwmin(model_paths, data, features, scaler):
    """Predict the optimized CWmin values for each router."""
    predictions = {}
    for path in model_paths:
        router_id = path.split('_')[-1].split('.')[0]  # Extract router ID from path
        model = tf.keras.models.load_model(path)
        input_data = prepare_input_data(data, router_id, features, scaler)
        
        # Skip prediction if input_data is None
        if input_data is None:
            continue
        
        predicted_cwmin = model.predict(input_data)[0][0]
        mapped_cwmin = map_to_valid_cwmin(predicted_cwmin)
        predictions[router_id] = mapped_cwmin
        print(f"Predicted CWmin for router {router_id}: {predicted_cwmin}, Mapped CWmin: {mapped_cwmin}")
    return predictions

def save_predictions_to_file(predictions, file_path):
    """Save the predicted CWmin values to a text file."""
    with open(file_path, 'w') as file:
        for router_id, cwmin in predictions.items():
            file.write(f"router {router_id} : {cwmin}\n")

# Example usage:

# Path to the updated local models (same paths as used in distribution script)
model_paths = [
    'model_router_0.h5',
    'model_router_1.h5',
    'model_router_2.h5'
    # Add paths for all routers
]

# Path to the CSV data file
data_file_path = 'combined_router_data.csv'

# Load and preprocess the data
data = load_data(data_file_path)

# Print unique router IDs to debug
print("Unique Router IDs in data:", data['Router ID'].unique())

# Features used for prediction (same as in training script)
features = ["Delay", "Throughput (Mbps)", "Configured Data Rate (Mbps)",  "Connected Devices", "Lost Packet Count"]

# Fit the scaler on the entire dataset
scaler = StandardScaler()
scaler.fit(data[features])

# Predict optimized CWmin values
optimized_cwmin_values = predict_cwmin(model_paths, data, features, scaler)

# Save the predictions to a text file
output_file_path = 'predicted_cwmin_values.txt'
save_predictions_to_file(optimized_cwmin_values, output_file_path)

# Print completion message
print(f"Optimized CWmin values predicted and saved to {output_file_path}")
