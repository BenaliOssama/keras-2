import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model year', 'origin', 'car name']
data = pd.read_csv(url, sep='\s+', names=column_names, na_values='?')

# Drop unnecessary columns
data = data.drop(['model year', 'origin', 'car name'], axis=1)

# Handle missing values (drop rows with NaN in horsepower)
data = data.dropna()

# Separate features (X) and target (y)
X = data.drop('mpg', axis=1)
y = data['mpg']

# Split without shuffling: 20% test, 80% train
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scale using StandardScaler (fit on train, apply to both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Convert back to DataFrames to preserve column names (optional but clean)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)



# Build the neural network
model = Sequential()
model.add(Dense(30, input_dim=5, activation='sigmoid'))  # 30 neurons, sigmoid
model.add(Dense(30, activation='sigmoid'))               # 30 neurons, sigmoid
model.add(Dense(1))                                      # 1 output, linear (default)



# Compile the model
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])


# Train the model
model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, verbose=0)

# Evaluate on the scaled test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test MAE: {test_mae:.3f}")


# Make predictions
y_pred = model.predict(X_test_scaled, verbose=0)
print(f"Sample predictions (first 5): {y_pred[:5].flatten()}")
print(f"Sample actuals (first 5): {y_test.iloc[:5].values}")


