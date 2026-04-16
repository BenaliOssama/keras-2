import keras
from keras.layers import Dense

# Build the network from Exercise 3
model = keras.Sequential()
model.add(Dense(16, input_shape=(5,), activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(5, activation='softmax'))

# Compile for multi-class classification
model.compile(
    loss='categorical_crossentropy',  # Multi-class loss
    optimizer='adam',                  # Same optimizer
    metrics=['accuracy']               # Accuracy instead of MAE
)

print(model.summary())
