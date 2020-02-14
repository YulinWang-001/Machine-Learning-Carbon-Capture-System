import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow import keras


# Read all the data points in the file
zeolite_13X_error = pd.read_csv("zeolite_13X_error.csv", delimiter=",")
zeolite_copy = zeolite_13X_error.copy()

# ------------- Filter points, Recovery rate and Purity - Test Set Only --------------------
zeolite_13X_error_testset = zeolite_copy[zeolite_copy.Recovery > 0.7]
zeolite_13X_error_testset = zeolite_13X_error_testset[zeolite_13X_error_testset.Purity > 0.7]
zeolite_13X_error_testset = zeolite_13X_error_testset[zeolite_13X_error_testset.Recovery < 1.0]

# Shuffle all the data points in test set
zeolite_13X_error_testset = zeolite_13X_error_testset.reindex(
    np.random.permutation(zeolite_13X_error_testset.index))
zeolite_13X_error_testset = zeolite_13X_error_testset.values
# --------------------------------------------------------------------------------------------

# -------------- Normal Data Set, No Filtration - Training Set -------------------------------
# Shuffle all the data points
zeolite_13X_error = zeolite_13X_error[zeolite_13X_error.Recovery < 1]
zeolite_13X_error = zeolite_13X_error[zeolite_13X_error.Recovery > 0]
zeolite_13X_error = zeolite_13X_error.reindex(np.random.permutation(zeolite_13X_error.index))
zeolite_13X_error = zeolite_13X_error.values
# --------------------------------------------------------------------------------------------

# Split into features and targets
train_features = zeolite_13X_error[0:200, 0:8]
train_targets = zeolite_13X_error[0:200, 13]

# Normalisation
train_targets = np.reshape(train_targets, (-1, 1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
# this line is very important cause it contains '.fit'
print(scaler_x.fit(train_features))
xscale = scaler_x.transform(train_features)
# this line is very important cause it contains '.fit'
print(scaler_y.fit(train_targets))
yscale = scaler_y.transform(train_targets)

# Split into validation set and training set
X_train, X_validation, y_train, y_validation = train_test_split(xscale, yscale)

# Build Neural Network
model = Sequential()
model.add(Dense(16, input_dim=8, kernel_initializer='normal', activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
hist = model.fit(X_train, y_train, epochs=150, batch_size=None,  verbose=1, validation_split=0.2)

# visualise the respective losses
print(hist.history.keys())

# visualise Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss V.S. Val Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

# Prediction - use another small data set whose size is quite similar to validation set as a test set
test_features = zeolite_13X_error_testset[1:11, 0:8]
test_targets = zeolite_13X_error_testset[1:11, 13]
# Remember that using this way, all the values extrapolated will not form a vector length automatically

test_features_modify = scaler_x.transform(test_features)
predicted_test_targets = model.predict(test_features_modify)

# Invert Normalisation
Final_test_targets = scaler_y.inverse_transform(predicted_test_targets)

# Reshape the test targets to the same shape as the prediction
length = predicted_test_targets.shape[0]
targets = Final_test_targets.reshape((length, 1))
test_targets = test_targets.reshape((length, 1))

# Compute precision in fraction
error = abs(targets - test_targets)
print('Absolute errors between predicted Productivity and real Productivity \n', error, '\n')
Precision = abs(1 - error/targets)
print('Estimated Precision of model in Productivity Predicting \n', Precision, '\n')

# Compute the average precision
ave = np.mean(Precision)
print("Average Precision of Productivity Prediction is:", 100*ave, "%")




# Get rid of regularization - DONE
#  purity and recovery > 85%
# Currently, if we train the neural network with raw dataset (only remove those non-sensible data points)
# then even using 1000 data points, the accuracy remains low - around 70%
# SHALL we just use filtered data points to train the model, could potentially improve its performance
# 200 - 400 - 600
