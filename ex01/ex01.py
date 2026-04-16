import keras
from keras.layers import Dense

# Build the regression neural network (from Exercise 0)
model = keras.Sequential()
model.add(Dense(8, input_shape=(5,), activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

# Compile the model - set up the optimization
model.compile(
    optimizer='adam',
    loss='mse',           # Mean Squared Error for regression
    metrics=['mae']       # Mean Absolute Error to monitor during training
)

print(model.summary())
