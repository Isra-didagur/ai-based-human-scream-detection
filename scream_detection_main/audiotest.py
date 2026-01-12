import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

class AudioScreamDetector:
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
    
    def load_data(self, negative_dir, positive_dir):
        """Load and prepare data from negative and positive directories"""
        
        print("="*100)
        print(" "*30 + "LOADING AUDIO DATA")
        print("="*100)
        print()
        
        features = []
        labels = []
        file_names = []
        
        # Load NEGATIVE files
        print("Loading NEGATIVE (safe) audio files...")
        if os.path.exists(negative_dir):
            negative_files = [f for f in os.listdir(negative_dir) 
                            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
            print(f"  Found {len(negative_files)} negative files")
            
            for i, file in enumerate(negative_files, 1):
                file_path = os.path.join(negative_dir, file)
                feature = self.extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append('negative')
                    file_names.append(file)
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(negative_files)} files...")
        else:
            print(f"  ❌ Directory not found: {negative_dir}")
        
        print()
        
        # Load POSITIVE files
        print("Loading POSITIVE (scream) audio files...")
        if os.path.exists(positive_dir):
            positive_files = [f for f in os.listdir(positive_dir) 
                            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
            print(f"  Found {len(positive_files)} positive files")
            
            for i, file in enumerate(positive_files, 1):
                file_path = os.path.join(positive_dir, file)
                feature = self.extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append('positive')
                    file_names.append(file)
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(positive_files)} files...")
        else:
            print(f"  ❌ Directory not found: {positive_dir}")
        
        print()
        print(f"✓ Total files loaded: {len(features)}")
        print(f"  - Negative: {labels.count('negative')}")
        print(f"  - Positive: {labels.count('positive')}")
        print()
        
        if len(features) == 0:
            print("❌ ERROR: No audio files were loaded!")
            return None, None, None, None, None, None
        
        # Convert to numpy arrays
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
            X, y, file_names, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Data split:")
        print(f"  Training set: {len(X_train)} files")
        print(f"  Test set: {len(X_test)} files")
        print()
        
        return X_train, X_test, y_train, y_test, files_train, files_test
    
    def build_model(self, input_shape):
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
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the model"""
        
        print("="*100)
        print(" "*30 + "TRAINING THE MODEL")
        print("="*100)
        print()
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        print("Model Architecture:")
        self.model.summary()
        print()
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('scream_detector_model.h5', 
                                    monitor='val_accuracy', 
                                    save_best_only=True, 
                                    verbose=1)
        
        # Train
        print("Starting training...")
        print()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        print()
        print("✓ Training completed!")
        print(f"✓ Model saved as 'scream_detector_model.h5'")
        print()
        
        return history
    
    def test_and_evaluate(self, X_test, y_test, files_test, negative_dir, positive_dir):
        """Test the model and show detailed results"""
        
        print("="*100)
        print(" "*30 + "TESTING THE MODEL")
        print("="*100)
        print()
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Separate results by true label
        negative_results = []
        positive_results = []
        
        for i, (true_label, pred_label, prob, filename) in enumerate(zip(y_test, y_pred, y_pred_prob, files_test)):
            result = {
                'file': filename,
                'true_label': 'Negative' if true_label == 0 else 'Positive',
                'pred_label': 'Negative' if pred_label == 0 else 'Positive',
                'confidence': prob[0] if pred_label == 1 else (1 - prob[0]),
                'correct': true_label == pred_label
            }
            
            if true_label == 0:
                negative_results.append(result)
            else:
                positive_results.append(result)
        
        # Print NEGATIVE files results
        print("─"*100)
        print("TESTING NEGATIVE FILES (Safe/Non-Scream)")
        print("─"*100)
        print()
        
        for i, res in enumerate(negative_results, 1):
            status = "✓ CORRECT" if res['correct'] else "✗ WRONG"
            print(f"{i:3d}. {res['file'][:45]:<45} | True: {res['true_label']:<8} | "
                  f"Pred: {res['pred_label']:<8} | Conf: {res['confidence']*100:5.1f}% | {status}")
        
        neg_correct = sum(1 for r in negative_results if r['correct'])
        neg_total = len(negative_results)
        neg_accuracy = (neg_correct / neg_total * 100) if neg_total > 0 else 0
        
        print()
        print(f"Negative Files: {neg_correct}/{neg_total} correct ({neg_accuracy:.1f}% accuracy)")
        
        # Print POSITIVE files results
        print()
        print("─"*100)
        print("TESTING POSITIVE FILES (Scream)")
        print("─"*100)
        print()
        
        for i, res in enumerate(positive_results, 1):
            status = "✓ CORRECT" if res['correct'] else "✗ WRONG"
            print(f"{i:3d}. {res['file'][:45]:<45} | True: {res['true_label']:<8} | "
                  f"Pred: {res['pred_label']:<8} | Conf: {res['confidence']*100:5.1f}% | {status}")
        
        pos_correct = sum(1 for r in positive_results if r['correct'])
        pos_total = len(positive_results)
        pos_accuracy = (pos_correct / pos_total * 100) if pos_total > 0 else 0
        
        print()
        print(f"Positive Files: {pos_correct}/{pos_total} correct ({pos_accuracy:.1f}% accuracy)")
        
        # Overall results
        self._print_final_results(y_test, y_pred, neg_correct, neg_total, 
                                 pos_correct, pos_total)
    
    def _print_final_results(self, y_test, y_pred, neg_correct, neg_total, 
                            pos_correct, pos_total):
        """Print final conclusion"""
        
        print()
        print()
        print("="*100)
        print(" "*30 + "FINAL RESULTS & ACCURACY")
        print("="*100)
        print()
        
        # Category-wise results
        print("RESULTS BY CATEGORY:")
        print("─"*100)
        
        neg_accuracy = (neg_correct / neg_total * 100) if neg_total > 0 else 0
        print(f"\nNEGATIVE FILES (Safe/Non-Scream):")
        print(f"  Total files tested:        {neg_total}")
        print(f"  ✓ Correctly predicted:     {neg_correct}")
        print(f"  ✗ Wrongly predicted:       {neg_total - neg_correct}")
        print(f"  Accuracy:                  {neg_accuracy:.2f}%")
        
        pos_accuracy = (pos_correct / pos_total * 100) if pos_total > 0 else 0
        print(f"\nPOSITIVE FILES (Scream):")
        print(f"  Total files tested:        {pos_total}")
        print(f"  ✓ Correctly predicted:     {pos_correct}")
        print(f"  ✗ Wrongly predicted:       {pos_total - pos_correct}")
        print(f"  Accuracy:                  {pos_accuracy:.2f}%")
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_test, y_pred) * 100
        total_files = len(y_test)
        total_correct = neg_correct + pos_correct
        
        print()
        print("="*100)
        print("OVERALL MODEL PERFORMANCE:")
        print("="*100)
        print(f"\n  📊 Total Audio Files Tested:     {total_files}")
        print(f"  ✓ Total Correct Predictions:     {total_correct}")
        print(f"  ✗ Total Wrong Predictions:       {total_files - total_correct}")
        print(f"\n  🎯 OVERALL ACCURACY ACHIEVED:    {overall_accuracy:.2f}%")
        print()
        
        # Performance rating
        if overall_accuracy >= 90:
            print("  ⭐⭐⭐ EXCELLENT! Your model is performing very well!")
        elif overall_accuracy >= 80:
            print("  ⭐⭐ GOOD! Your model is performing well!")
        elif overall_accuracy >= 70:
            print("  ⭐ MODERATE! Consider adding more training data!")
        else:
            print("  ⚠ NEEDS IMPROVEMENT! Try collecting more diverse audio samples!")
        
        print()
        print("="*100)
        print()
        
        # Detailed metrics
        print("DETAILED CLASSIFICATION REPORT:")
        print("─"*100)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Negative', 'Positive'],
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nCONFUSION MATRIX:")
        print("─"*100)
        print(f"\n                    Predicted Negative    Predicted Positive")
        print(f"Actual Negative            {cm[0][0]:>6}                {cm[0][1]:>6}")
        print(f"Actual Positive            {cm[1][0]:>6}                {cm[1][1]:>6}")
        print()
        print("="*100)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n")
    print("*"*100)
    print("*" + " "*98 + "*")
    print("*" + " "*20 + "SCREAM DETECTION - TRAINING & TESTING SYSTEM" + " "*34 + "*")
    print("*" + " "*98 + "*")
    print("*"*100)
    print("\n")
    
    # ========================================================================
    # CONFIGURATION - YOUR AUDIO FOLDERS
    # ========================================================================
    
    NEGATIVE_DIR = "C:/Users/Admin/Desktop/MAIN PPROJECT/MAIN PPROJECT/Human_Scream_Detection_using_ml_and_deep_learning-main/scream_detection_main/negative"
    POSITIVE_DIR = "C:/Users/Admin/Desktop/MAIN PPROJECT/MAIN PPROJECT/Human_Scream_Detection_using_ml_and_deep_learning-main/scream_detection_main/positive"
    
    # ========================================================================
    
    # Check if directories exist
    if not os.path.exists(NEGATIVE_DIR):
        print(f"❌ ERROR: Negative directory not found: {NEGATIVE_DIR}")
        print(f"\nPlease update NEGATIVE_DIR in the script.\n")
        exit(1)
    
    if not os.path.exists(POSITIVE_DIR):
        print(f"❌ ERROR: Positive directory not found: {POSITIVE_DIR}")
        print(f"\nPlease update POSITIVE_DIR in the script.\n")
        exit(1)
    
    # Initialize detector
    detector = AudioScreamDetector()
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test, files_train, files_test = detector.load_data(
        NEGATIVE_DIR, POSITIVE_DIR
    )
    
    if X_train is None:
        print("❌ Failed to load data. Exiting...")
        exit(1)
    
    # Step 2: Train model
    history = detector.train(X_train, y_train, X_test, y_test)
    
    # Step 3: Test and evaluate
    detector.test_and_evaluate(X_test, y_test, files_test, NEGATIVE_DIR, POSITIVE_DIR)
    
    print("\n✅ Complete! Your model has been trained and tested!")
    print("✅ Model saved as: scream_detector_model.h5")
    print("\n" + "*"*100 + "\n")