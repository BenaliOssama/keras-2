import keras
from keras.layers import Dense

# Build multi-class classification network
model = keras.Sequential()
model.add(Dense(16, input_shape=(5,), activation='sigmoid'))  # Hidden layer 1: 16 neurons
model.add(Dense(8, activation='sigmoid'))                     # Hidden layer 2: 8 neurons
model.add(Dense(5, activation='softmax'))                     # Output layer: 5 neurons (one per class)

print(model.summary())
