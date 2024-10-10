import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

def federated_averaging(weight_list):
    """Perform Federated Averaging on the weights of the models."""
    new_weights = []
    
    # Assuming all models have the same structure and number of layers
    for weights in zip(*weight_list):
        new_weights.append(np.mean(weights, axis=0))
    
    return new_weights

def load_local_model_weights(model_paths):
    """Load model weights from given paths."""
    weight_list = []
    for path in model_paths:
        model = tf.keras.models.load_model(path)
        weight_list.append(model.get_weights())
    return weight_list

def create_global_model(input_shape, model_weights):
    """Create a new global model and set its weights."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)  # Predict CW Value
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.set_weights(model_weights)
    return model

# Paths to the local models saved by each router
model_paths = [
    'model_router_0.h5',
    'model_router_1.h5',
    'model_router_2.h5'
    # Add paths for all routers
]

# Step 1: Collecting Model Weights
local_weights = load_local_model_weights(model_paths)

# Step 2: Aggregating Weights using Federated Averaging
global_weights = federated_averaging(local_weights)

# Step 3: Create a Global Model and Set Aggregated Weights
input_shape = 5  # Number of features
global_model = create_global_model(input_shape, global_weights)

# Save the global model
global_model.save('global_model.h5')

# Differential Privacy parameters
noise_multiplier = 1.1
l2_norm_clip = 1.0
num_microbatches = 1
learning_rate = 0.15

# Step 4: Apply Differential Privacy
def apply_differential_privacy(global_model):
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate
    )

    global_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Dummy data for one step of DP-SGD to apply differential privacy
    dummy_input = np.random.randn(64, input_shape).astype(np.float32)  # Adjust shape according to your model's input
    dummy_output = np.random.randn(64, 1).astype(np.float32)
    
    global_model.fit(dummy_input, dummy_output, epochs=1, batch_size=64)
    
    return global_model

# Apply Differential Privacy to the global model
global_model_with_dp = apply_differential_privacy(global_model)

# Save the differentially private global model
global_model_with_dp.save('global_model_with_dp.h5')

# Print completion message
print("Global model with differential privacy created and saved as 'global_model_with_dp.h5'")
