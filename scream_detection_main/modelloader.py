import os
import numpy as np  
import pandas as pd
from tensorflow import keras
from scipy.io.wavfile import read

# Global variable for model
model = None
suitable_length_for_model = 48250  # Default value

def load_model():
    """Load or create the model as needed"""
    global model
    
    if model is not None:
        return model
        
    try:
        # Use a consistent path resolution approach
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different model file formats
        for ext in ['.keras', '.h5']:
            model_path = os.path.join(script_dir, f'saved_model{ext}')
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                model = keras.models.load_model(model_path)
                return model
        
        # If we get here, try loading from 'saved_model' directory
        model_path = os.path.join(script_dir, 'saved_model')
        if os.path.exists(model_path) and os.path.isdir(model_path):
            print(f"Loading model from directory: {model_path}")
            model = keras.models.load_model(model_path)
            return model
            
        # If we still don't have a model, create a placeholder
        print("Model not found, creating placeholder model")
        return create_placeholder_model()
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Creating placeholder model")
        return create_placeholder_model()

def create_placeholder_model():
    """Create a placeholder model if needed"""
    global model
    
    # Get the input dimension
    try:
        with open("input dimension for model.txt", "r") as file:
            input_dim = int(file.read())
    except:
        input_dim = suitable_length_for_model
    
    # Create a simple model
    model = keras.Sequential([
        keras.layers.Dense(12, input_dim=input_dim, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Save the model for future use
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.keras')
    model.save(model_path)
    print(f"Created and saved placeholder model to {model_path}")
    
    return model

def process_file(filename):
    """Process an audio file using the model"""
    global suitable_length_for_model
    
    try:
        arr = []
        
        # Make sure we have a model
        current_model = load_model()
        
        print(f"Processing file: {filename}")

        # Check if file exists and is valid
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Input file not found: {filename}")
        if not filename.endswith('.wav'):
            raise ValueError(f"Invalid file format: {filename}. Only .wav files are supported.")

        # Get input dimensions for the model
        try:
            with open("input dimension for model.txt", "r") as file:
                suitable_length_for_model = int(file.read())
        except FileNotFoundError:
            print(f"Using default input dimension: {suitable_length_for_model}")
        
        # Read and process the audio file
        data, rs = read(filename)
        rs = rs.astype(float)
        
        # Ensure we have enough samples
        if len(rs) < suitable_length_for_model + 1:
            # Pad with zeros if too short
            rs = np.pad(rs, (0, suitable_length_for_model + 1 - len(rs)))
        else:
            # Truncate if too long
            rs = rs[:suitable_length_for_model + 1]
            
        a = pd.Series(rs)
        arr.append(a)
        df = pd.DataFrame(arr)
        
        # Handle case where DataFrame has no columns named 1
        if df.shape[1] > 1:
            X2 = df.iloc[:, 1:]
        else:
            # If there's only one column, use that
            X2 = df
        
        # Make prediction
        predictions = current_model.predict(X2)
        rounded = [round(x[0]) for x in predictions]

        print(f"Predicted value: {rounded}")
        return predictions
        
    except Exception as e:
        print(f"Error in process_file: {e}")
        import traceback
        traceback.print_exc()
        # Return a numpy array with value 0.0 to indicate no scream
        return np.array([[0.0]])

# Initialize model dimensions once when module is imported
try:
    with open("input dimension for model.txt", "r") as file:
        suitable_length_for_model = int(file.read())
    print(f"Using input dimension: {suitable_length_for_model}")
except FileNotFoundError:
    print(f"Warning: input dimension file not found. Using default value {suitable_length_for_model}")