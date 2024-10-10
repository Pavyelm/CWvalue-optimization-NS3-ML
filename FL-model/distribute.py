import tensorflow as tf

def load_global_model(global_model_path):
    """Load the global model from the specified path."""
    return tf.keras.models.load_model(global_model_path)

def distribute_global_weights(global_model, model_paths):
    """Update local models with the global model weights."""
    global_weights = global_model.get_weights()
    for path in model_paths:
        local_model = tf.keras.models.load_model(path)
        local_model.set_weights(global_weights)
        local_model.save(path)  # Overwrite the local model with updated weights
        print(f"Updated local model at {path} with global weights")

# Example usage:

# Path to the global model
global_model_path = 'global_model.h5'

# Paths to the local models saved by each router
model_paths = [
    'model_router_0.h5',
    'model_router_1.h5',
    'model_router_2.h5'
    # Add paths for all routers
]

# Step 1: Load the Global Model
global_model = load_global_model(global_model_path)

# Step 2: Distribute Global Weights to Local Models
distribute_global_weights(global_model, model_paths)

# Print completion message
print("All local models updated with global weights")
