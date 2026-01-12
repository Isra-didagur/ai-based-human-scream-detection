# dataset for this model can be easily prepare by datasetmaker.py file
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense # type: ignore
import librosa
import numpy as np
from modelloader import load_model
# Removed duplicate import of Dense from tensorflow.keras.layers

try:
    df = pd.read_csv('newresources.csv', index_col=0, engine='c')
    try:
        with open("begining index of testing files.txt","r") as file:
            data1 = int(file.read())
        row_num_for_verification_of_model = data1
        X = df.iloc[:row_num_for_verification_of_model,1:]  #independent variables columns
        print(row_num_for_verification_of_model)
        X2 = df.iloc[row_num_for_verification_of_model:,1:]
        
        with open("input dimension for model.txt","r") as file:
            data2 = int(file.read())
        print(data2)
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        exit(1)
    except Exception as e:
        print(f"Error processing files: {e}")
        exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)
total_number_of_column_required_for_prediction = data2
column_number_of_csv_having_labels = 0
y = df.iloc[:data1,column_number_of_csv_having_labels] # dependent variable column
# # define the keras model
model = Sequential()
model.add(Dense(12, input_dim=total_number_of_column_required_for_prediction, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
history = model.fit(X, y, validation_split=0.33, epochs=150, batch_size=50)


# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))

# make probability predictions with the model
predictions = model.predict(X2)

# round predictions
rounded = [round(x[0]) for x in predictions]

print("predicted value is"+str(rounded))
print("actual value was"+str(list(df.iloc[row_num_for_verification_of_model:,column_number_of_csv_having_labels])))

model.save('saved_model.keras')



def process_file(filepath):
    try:
        # Load the audio file
        audio, sr = librosa.load(filepath, sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)  # shape = (40,)

        # Reshape to match model input (1 sample, 40 features)
        model_input = np.expand_dims(mfccs_processed, axis=0)

        # Load the trained model
        model = load_model()

        # Predict
        prediction = model.predict(model_input)
        predicted_label = int(np.round(prediction[0][0]))  # binary: 0 or 1

        return predicted_label
    
    except Exception as e:
        print(f"❌ Error in process_file: {e}")
        return -1  # indicates error
