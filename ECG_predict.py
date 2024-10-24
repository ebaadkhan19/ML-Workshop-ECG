import streamlit as st
import tensorflow as tf
import numpy as np

# Define the predict function
def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.cast(tf.math.less(loss, threshold), dtype=tf.int32)

# Load your ECG anomaly detection model
# Replace 'your_model_path' with the actual path to your model
def load_model():
    model = tf.keras.models.load_model('ECG_model')
    return model

# Normalize the input data
def normalize_data(data):
    min_val = -6.2808752
    max_val = 7.4021031
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# Streamlit app
def main():
    st.title("ECG Anomaly Detection")

    # Load the model
    model = load_model()

    # Get user input
    input_data = st.text_area("Enter your ECG data (comma-separated):")

    # Convert input string to numpy array
    try:
        # Assuming your input data is stored in a variable named 'input_data'
        data = np.array([float(value.strip()) for value in input_data.split(",")])
        data = normalize_data(data)  # Normalize the input data
        data = np.expand_dims(data, axis=0)  # Add batch dimension
        data = data[:, :140]  # Keep only the first 140 elements of each sample
    except ValueError:
        st.error("Invalid input. Please enter comma-separated numerical values.")
        return

    # Make prediction
    result = predict(model, data, 0.3)

    # Display result
    st.write("Prediction:", "Normal ECG" if result.numpy()[0] == 1 else "Anomaly ECG")

if __name__ == "__main__":
    main()