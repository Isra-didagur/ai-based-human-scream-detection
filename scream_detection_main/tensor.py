import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

print("TF Version:", tf.__version__)
model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)),
    Dense(1, activation='sigmoid')
])
print("Model created successfully!")

