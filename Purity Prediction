import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

zeolite_13X_error = pd.read_csv("zeolite_13X_error.csv", delimiter=",")

zeolite_13X_error_testset = zeolite_13X_error.copy()

# ------------- Filter points, Recovery rate and Purity - Test Set Only ---------------------
# zeolite_13X_error_testset = zeolite_copy[zeolite_copy.Recovery > 0.7]
# zeolite_13X_error_testset = zeolite_13X_error_testset[zeolite_13X_error_testset.Purity > 0.7]
# zeolite_13X_error_testset = zeolite_13X_error_testset[zeolite_13X_error_testset.Recovery < 1.0]

# Shuffle all the data points
zeolite_13X_error_testset = zeolite_13X_error_testset.reindex(
    np.random.permutation(zeolite_13X_error_testset.index))
zeolite_13X_error_testset = zeolite_13X_error_testset.values
# --------------------------------------------------------------------------------------------

# -------------- Normal Data Set, No Filtration - Training Set -------------------------------
# Shuffle all the data points
# zeolite_13X_error = zeolite_13X_error[zeolite_13X_error.Recovery > 0]
# zeolite_13X_error = zeolite_13X_error[zeolite_13X_error.Recovery < 1]
zeolite_13X_error = zeolite_13X_error.reindex(np.random.permutation(zeolite_13X_error.index))
zeolite_13X_error = zeolite_13X_error.values
# --------------------------------------------------------------------------------------------

# Now, train the model with 200 raw data points
train_features = zeolite_13X_error[0:200, 0:8]
train_targets = zeolite_13X_error[0:200, 10]

# Normalisation
train_targets = np.reshape(train_targets, (-1, 1))
scaler_x = MinMaxScaler()
print(scaler_x.fit(train_features))
# The printing line is very important cause it contains .fit function
scaler_y = MinMaxScaler()
print(scaler_y.fit(train_targets))
xscale = scaler_x.transform(train_features)
yscale = scaler_y.transform(train_targets)

# Split into training set and test set
X_train, X_validation, y_train, y_validation = train_test_split(xscale, yscale)

# Build the Neural Network
model = Sequential()
model.add(Dense(16, input_dim=8, kernel_initializer='normal', activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(12, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
hist = model.fit(X_train, y_train, epochs=150, batch_size=50, verbose=1, validation_split=0.2)

# visualise the respective losses
print(hist.history.keys())

# visualise Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss V.S. Val Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# After the training, use filtered data points to evaluate the accuracy of the model

# Predictions - use the rest of data in the file to test the accuracy of predictions
test_features = zeolite_13X_error_testset[1:11, 0:8]
test_targets = zeolite_13X_error_testset[1:11, 10]
# Remember that using this way, all the values extrapolated will not form a vector length automatically

test_features_normalisation = scaler_x.transform(test_features)
predicted_test_targets = model.predict(test_features_normalisation)

# Reverse Normalisation
Final_predicted_test_targets = scaler_y.inverse_transform(predicted_test_targets)

# Reshape the test targets to the same shape as the prediction
length = predicted_test_targets.shape[0]
targets = Final_predicted_test_targets.reshape((length, 1))

test_targets = test_targets.reshape((length, 1))

# Compute precision in fraction
error = abs(test_targets - targets)
print('Absolute errors between predicted purities and real purities \n', error, '\n')
Precision = abs(1 - error/targets)
print('Estimated Precision of model in purity predicting \n', Precision, '\n')
print(Precision)

# Compute the average precision
ave = np.mean(Precision)
print("Average Precision of Productivity Prediction is:", 100*ave, "%")

# Build a function to take the columns as inputs and target column as output
# combine the file and generate one function - iteration
