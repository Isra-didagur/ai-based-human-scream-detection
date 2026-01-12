
import pickle
import numpy as np
from scipy.io.wavfile import read

def svm_process(filename):
    try:
        # Load the model
        with open('svm_based_model/phase1_model.sav', 'rb') as file:
            model = pickle.load(file)
            
        # Process the audio file
        data, rs = read(filename)
        rs = rs.astype(float)
        
        # Extract features (simplified)
        features = extract_features(rs)
        
        # Make prediction
        prediction = model.predict([features])
        return prediction[0] == 1
    except Exception as e:
        print(f"SVM model error: {e}")
        return False
        
def extract_features(audio_data):
    # Simplified feature extraction
    # In a real implementation, you would extract specific features for SVM
    return np.mean(audio_data[:min(len(audio_data), 1000)])
