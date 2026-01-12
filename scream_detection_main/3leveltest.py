import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Use TensorFlow's Keras API directly
Sequential = keras.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint

class AutoRiskLevelDetector:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 3
        self.n_mfcc = 40
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled
        except Exception as e:
            print(f"  Error loading {os.path.basename(audio_path)}: {e}")
            return None
    
    def extract_intensity_features(self, audio_path):
        """Extract intensity-based features for risk classification"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Calculate various intensity metrics
            rms = np.sqrt(np.mean(audio**2))  # Root Mean Square (loudness)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))  # Zero crossing rate
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            
            # Get MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            # Combine features
            intensity_score = rms * 1000  # Scale up for better visibility
            
            return {
                'mfcc': mfccs_mean,
                'rms': rms,
                'zcr': zcr,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'intensity_score': intensity_score
            }
        except Exception as e:
            print(f"  Error processing {os.path.basename(audio_path)}: {e}")
            return None
    
    def assign_risk_levels(self, positive_dir):
        """Automatically assign risk levels to positive files based on intensity"""
        print("="*100)
        print(" "*20 + "ANALYZING POSITIVE FILES FOR RISK LEVEL ASSIGNMENT")
        print("="*100)
        print()
        
        if not os.path.exists(positive_dir):
            print(f"❌ Directory not found: {positive_dir}")
            return {}
        
        positive_files = [f for f in os.listdir(positive_dir) 
                         if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        print(f"Analyzing {len(positive_files)} positive files for intensity...")
        print()
        
        file_intensities = []
        file_data = []
        
        for i, file in enumerate(positive_files, 1):
            file_path = os.path.join(positive_dir, file)
            features = self.extract_intensity_features(file_path)
            
            if features is not None:
                intensity = features['intensity_score']
                file_intensities.append(intensity)
                file_data.append({
                    'file': file,
                    'intensity': intensity,
                    'rms': features['rms'],
                    'features': features
                })
            
            if i % 20 == 0:
                print(f"  Analyzed {i}/{len(positive_files)} files...")
        
        if len(file_intensities) == 0:
            return {}
        
        # Use percentiles to divide into 3 risk levels
        intensities = np.array(file_intensities)
        low_threshold = np.percentile(intensities, 33)  # Bottom 33%
        high_threshold = np.percentile(intensities, 67)  # Top 33%
        
        print()
        print(f"Intensity Analysis Complete:")
        print(f"  Min intensity: {np.min(intensities):.4f}")
        print(f"  Max intensity: {np.max(intensities):.4f}")
        print(f"  Low/Medium threshold: {low_threshold:.4f}")
        print(f"  Medium/High threshold: {high_threshold:.4f}")
        print()
        
        # Assign risk levels
        risk_assignments = {}
        low_count = 0
        medium_count = 0
        high_count = 0
        
        for data in file_data:
            if data['intensity'] < low_threshold:
                risk_level = 'low'
                low_count += 1
            elif data['intensity'] < high_threshold:
                risk_level = 'medium'
                medium_count += 1
            else:
                risk_level = 'high'
                high_count += 1
            
            risk_assignments[data['file']] = {
                'risk_level': risk_level,
                'intensity': data['intensity'],
                'features': data['features']
            }
        
        print(f"Risk Level Assignment:")
        print(f"  🟢 Low Risk (quiet screams):      {low_count} files")
        print(f"  🟡 Medium Risk (moderate screams): {medium_count} files")
        print(f"  🔴 High Risk (loud/intense screams): {high_count} files")
        print()
        
        return risk_assignments
    
    def load_data_with_auto_risk(self, negative_dir, positive_dir):
        """Load data and automatically assign 3 risk levels"""
        
        print("="*100)
        print(" "*20 + "LOADING AUDIO DATA WITH AUTO RISK CLASSIFICATION")
        print("="*100)
        print()
        
        # First, analyze positive files and assign risk levels
        risk_assignments = self.assign_risk_levels(positive_dir)
        
        features = []
        labels = []
        file_names = []
        label_details = []
        
        # Load NEGATIVE files (all are Low Risk)
        print("Loading NEGATIVE (safe) audio files as LOW RISK...")
        if os.path.exists(negative_dir):
            negative_files = [f for f in os.listdir(negative_dir) 
                            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
            print(f"  Found {len(negative_files)} negative files")
            
            for i, file in enumerate(negative_files, 1):
                file_path = os.path.join(negative_dir, file)
                feature = self.extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(0)  # Low risk (0)
                    file_names.append(file)
                    label_details.append('negative_low')
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(negative_files)} files...")
        
        print()
        
        # Load POSITIVE files with assigned risk levels
        print("Loading POSITIVE audio files with assigned risk levels...")
        if os.path.exists(positive_dir):
            positive_files = [f for f in os.listdir(positive_dir) 
                            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
            print(f"  Found {len(positive_files)} positive files")
            
            for i, file in enumerate(positive_files, 1):
                if file not in risk_assignments:
                    continue
                
                file_path = os.path.join(positive_dir, file)
                feature = self.extract_features(file_path)
                
                if feature is not None:
                    risk_info = risk_assignments[file]
                    risk_level = risk_info['risk_level']
                    
                    # Assign numerical labels: 0=Low, 1=Medium, 2=High
                    if risk_level == 'low':
                        label = 1  # Medium-low (positive but quiet)
                    elif risk_level == 'medium':
                        label = 1  # Medium
                    else:  # high
                        label = 2  # High risk
                    
                    features.append(feature)
                    labels.append(label)
                    file_names.append(file)
                    label_details.append(f'positive_{risk_level}')
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(positive_files)} files...")
        
        print()
        print(f"✓ Total files loaded: {len(features)}")
        print(f"  - Low Risk (Negative): {labels.count(0)}")
        print(f"  - Medium Risk (Positive-Moderate): {labels.count(1)}")
        print(f"  - High Risk (Positive-Intense): {labels.count(2)}")
        print()
        
        if len(features) == 0:
            print("❌ ERROR: No audio files were loaded!")
            return None, None, None, None, None, None, None
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test, files_train, files_test, details_train, details_test = train_test_split(
            X, y, file_names, label_details, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Data split:")
        print(f"  Training set: {len(X_train)} files")
        print(f"  Test set: {len(X_test)} files")
        print()
        
        return X_train, X_test, y_train, y_test, files_train, files_test, details_test
    
    def build_model(self, input_shape, num_classes):
        """Build neural network model"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the model"""
        
        print("="*100)
        print(" "*20 + "TRAINING AUTO 3-LEVEL RISK CLASSIFICATION MODEL")
        print("="*100)
        print()
        
        # Build model
        num_classes = 3  # Low, Medium, High
        self.model = self.build_model(X_train.shape[1], num_classes)
        
        print("Model Architecture:")
        self.model.summary()
        print()
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        checkpoint = ModelCheckpoint('auto_risk_model.h5', 
                                    monitor='val_accuracy', 
                                    save_best_only=True, 
                                    verbose=1)
        
        # Train
        print("Starting training...")
        print()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        print()
        print("✓ Training completed!")
        print(f"✓ Model saved as 'auto_risk_model.h5'")
        print()
        
        return history
    
    def test_and_evaluate(self, X_test, y_test, files_test, details_test):
        """Test the model and show detailed risk level results"""
        
        print("="*100)
        print(" "*20 + "TESTING AUTO 3-LEVEL RISK CLASSIFICATION")
        print("="*100)
        print()
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
        
        # Separate results by predicted risk level
        results_by_level = {
            0: [],  # Low
            1: [],  # Medium
            2: []   # High
        }
        
        for i, (true_label, pred_label, probs, filename, detail) in enumerate(zip(y_test, y_pred, y_pred_prob, files_test, details_test)):
            result = {
                'file': filename,
                'true_label': risk_levels[true_label],
                'pred_label': risk_levels[pred_label],
                'confidence': probs[pred_label],
                'correct': true_label == pred_label,
                'probs': probs,
                'detail': detail
            }
            
            results_by_level[pred_label].append(result)
        
        # Print results grouped by PREDICTED risk level
        print("="*100)
        print(" "*25 + "RESULTS GROUPED BY PREDICTED RISK LEVEL")
        print("="*100)
        print()
        
        for level_idx in [0, 1, 2]:
            results = results_by_level[level_idx]
            
            if len(results) == 0:
                continue
            
            print("─"*100)
            print(f"FILES CLASSIFIED AS {risk_levels[level_idx].upper()} ({len(results)} files)")
            print("─"*100)
            print()
            
            for i, res in enumerate(results, 1):
                status = "✓ CORRECT" if res['correct'] else "✗ WRONG"
                prob_str = f"L:{res['probs'][0]*100:4.1f}% M:{res['probs'][1]*100:4.1f}% H:{res['probs'][2]*100:4.1f}%"
                
                # Show original category
                origin = "Negative" if 'negative' in res['detail'] else "Positive"
                
                print(f"{i:3d}. {res['file'][:35]:<35} | Origin: {origin:<8} | "
                      f"True: {res['true_label']:<12} | Pred: {res['pred_label']:<12} | "
                      f"{prob_str} | {status}")
            
            print()
        
        # Print final statistics
        self._print_final_results(y_test, y_pred, results_by_level)
    
    def _print_final_results(self, y_test, y_pred, results_by_level):
        """Print final conclusion"""
        
        print()
        print("="*100)
        print(" "*25 + "FINAL RISK CLASSIFICATION RESULTS")
        print("="*100)
        print()
        
        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
        
        # Count predictions by risk level
        print("RISK LEVEL DISTRIBUTION IN TEST SET:")
        print("─"*100)
        
        for level_idx in [0, 1, 2]:
            count = len(results_by_level[level_idx])
            percentage = (count / len(y_test) * 100) if len(y_test) > 0 else 0
            
            icon = "🟢" if level_idx == 0 else ("🟡" if level_idx == 1 else "🔴")
            print(f"\n{icon} {risk_levels[level_idx].upper()}:")
            print(f"  Files classified:          {count}")
            print(f"  Percentage:                {percentage:.1f}%")
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_test, y_pred) * 100
        total_correct = sum(y_test == y_pred)
        
        print()
        print("="*100)
        print("OVERALL MODEL PERFORMANCE:")
        print("="*100)
        print(f"\n  📊 Total Audio Files Tested:     {len(y_test)}")
        print(f"  ✓ Total Correct Predictions:     {total_correct}")
        print(f"  ✗ Total Wrong Predictions:       {len(y_test) - total_correct}")
        print(f"\n  🎯 OVERALL ACCURACY ACHIEVED:    {overall_accuracy:.2f}%")
        print()
        
        if overall_accuracy >= 90:
            print("  ⭐⭐⭐ EXCELLENT! Auto risk classification is working very well!")
        elif overall_accuracy >= 80:
            print("  ⭐⭐ GOOD! Auto risk classification is performing well!")
        elif overall_accuracy >= 70:
            print("  ⭐ MODERATE! Consider fine-tuning the intensity thresholds!")
        else:
            print("  ⚠ NEEDS IMPROVEMENT!")
        
        print()
        print("="*100)
        print()
        
        # Detailed report
        print("DETAILED CLASSIFICATION REPORT:")
        print("─"*100)
        print(classification_report(y_test, y_pred, 
                                   target_names=risk_levels,
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nCONFUSION MATRIX:")
        print("─"*100)
        print(f"\n{'':20}", end='')
        for label in risk_levels:
            print(f"{label:>15}", end='')
        print("\n" + "─"*80)
        
        for i, true_label in enumerate(risk_levels):
            print(f"{true_label:20}", end='')
            for j in range(3):
                print(f"{cm[i][j]:>15}", end='')
            print()
        
        print()
        print("="*100)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n")
    print("*"*100)
    print("*" + " "*98 + "*")
    print("*" + " "*10 + "AUTO 3-LEVEL RISK CLASSIFICATION - NEGATIVE & POSITIVE FOLDERS ONLY" + " "*20 + "*")
    print("*" + " "*98 + "*")
    print("*"*100)
    print("\n")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    NEGATIVE_DIR = "C:/Users/Admin/Desktop/MAIN PPROJECT/MAIN PPROJECT/Human_Scream_Detection_using_ml_and_deep_learning-main/scream_detection_main/negative"
    POSITIVE_DIR = "C:/Users/Admin/Desktop/MAIN PPROJECT/MAIN PPROJECT/Human_Scream_Detection_using_ml_and_deep_learning-main/scream_detection_main/positive"
    
    # ========================================================================
    
    # Check if directories exist
    if not os.path.exists(NEGATIVE_DIR):
        print(f"❌ ERROR: Negative directory not found: {NEGATIVE_DIR}\n")
        exit(1)
    
    if not os.path.exists(POSITIVE_DIR):
        print(f"❌ ERROR: Positive directory not found: {POSITIVE_DIR}\n")
        exit(1)
    
    # Initialize detector
    detector = AutoRiskLevelDetector()
    
    # Step 1: Load data with automatic risk assignment
    result = detector.load_data_with_auto_risk(NEGATIVE_DIR, POSITIVE_DIR)
    
    if result[0] is None:
        print("❌ Failed to load data. Exiting...")
        exit(1)
    
    X_train, X_test, y_train, y_test, files_train, files_test, details_test = result
    
    # Step 2: Train model
    history = detector.train(X_train, y_train, X_test, y_test)
    
    # Step 3: Test and evaluate
    detector.test_and_evaluate(X_test, y_test, files_test, details_test)
    
    print("\n✅ Complete! Auto 3-level risk classification finished!")
    print("✅ Model saved as: auto_risk_model.h5")
    print("\n📊 The model automatically classified your files into:")
    print("   🟢 Low Risk - Negative/safe sounds")
    print("   🟡 Medium Risk - Moderate intensity screams")
    print("   🔴 High Risk - High intensity screams")
    print("\n" + "*"*100 + "\n")