import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow import keras

"""------------- Read all the data points in the file ----------------------------------------"""
zeolite_13X_error = pd.read_csv("zeolite_13X_error.csv", delimiter=",")  # Used for training set
zeolite_copy = zeolite_13X_error.copy()  # Used for test set
"""--------------------------------------------------------------------------------------------"""


"""------------- Filter points, Recovery rate and Purity - Test Set Only ---------------------"""
zeolite_13X_error_testset = zeolite_copy[zeolite_copy.Recovery > 0.7]
zeolite_13X_error_testset = zeolite_13X_error_testset[zeolite_13X_error_testset.Purity > 0.7]
zeolite_13X_error_testset = zeolite_13X_error_testset[zeolite_13X_error_testset.Recovery < 1.0]

# Shuffle all the data points in test set
# Anaconda 3.7 Python

zeolite_13X_error_testset = zeolite_13X_error_testset.reindex(
    np.random.permutation(zeolite_13X_error_testset.index))
# search random package, use seed package
zeolite_13X_error_testset = zeolite_13X_error_testset.values
"""--------------------------------------------------------------------------------------------"""


"""-------------- Normal Data Set, No Filtration - Training Set -------------------------------"""
"""Shuffle all the data points"""
zeolite_13X_error = zeolite_13X_error[zeolite_13X_error.Recovery < 1]
zeolite_13X_error = zeolite_13X_error[zeolite_13X_error.Recovery > 0]
zeolite_13X_error = zeolite_13X_error.reindex(np.random.permutation(zeolite_13X_error.index))
zeolite_13X_error = zeolite_13X_error.values
"""--------------------------------------------------------------------------------------------"""


"""-------------------- Split into features and targets - training set -------------------------"""
train_features = zeolite_13X_error[0:200, 0:8]
recovery_train_targets = zeolite_13X_error[0:200, 11]
purity_train_targets = zeolite_13X_error[0:200, 10]
specific_energy_train_targets = zeolite_13X_error[0:200, 12]
productivity_train_targets = zeolite_13X_error[0:200, 13]
"""-----------------------------------------------------------------------------------------------"""

# lst = []
# lst.append(zeolite_13X_error)
# for i in range(10, 14):


"""-------------------- Split into features and targets - test set -------------------------------"""
test_features = zeolite_13X_error_testset[1:11, 0:8]
recovery_test_targets = zeolite_13X_error_testset[1:11, 11]
purity_test_targets = zeolite_13X_error_testset[1:11, 10]
specific_energy_test_targets = zeolite_13X_error_testset[1:11, 12]
productivity_test_targets = zeolite_13X_error_testset[1:11, 13]
"""------------------------------------------------------------------------------------------------"""


"""----------------------- Normalisation ---------------------------------------------------------"""
recovery_train_targets = np.reshape(recovery_train_targets, (-1, 1))
purity_train_targets = np.reshape(purity_train_targets, (-1, 1))
specific_energy_train_targets = np.reshape(specific_energy_train_targets, (-1, 1))
productivity_train_targets = np.reshape(productivity_train_targets, (-1, 1))
"""------------------------------------------------------------------------------------------------"""


"""--------------------------------- Scale and Fit ------------------------------------------------"""
scaler_x = MinMaxScaler()
scaler_y_recovery = MinMaxScaler()
scaler_y_purity = MinMaxScaler()
scaler_y_specific_energy = MinMaxScaler()
scaler_y_productivity = MinMaxScaler()
""" This line is very important as this contains '.fit' """
print(scaler_x.fit(train_features))
xscale = scaler_x.transform(train_features)
print(scaler_y_recovery.fit(recovery_train_targets))
print(scaler_y_purity.fit(purity_train_targets))
print(scaler_y_specific_energy.fit(specific_energy_train_targets))
print(scaler_y_productivity.fit(productivity_train_targets))
recovery_yscale = scaler_y_recovery.transform(recovery_train_targets)
purity_yscale = scaler_y_purity.transform(purity_train_targets)
specific_yscale = scaler_y_specific_energy.transform(specific_energy_train_targets)
productivity_yscale = scaler_y_productivity.transform(productivity_train_targets)
"""---------------------------------------------------------------------------------------------------"""


"""--------------------------- Split the data into training set and test set -------------------------"""
recovery_X_train, recovery_X_validation, recovery_Y_train, recovery_Y_validation = \
    train_test_split(xscale, recovery_yscale)
purity_X_train, purity_X_validation, purity_Y_train, purity_Y_validation = \
    train_test_split(xscale, purity_yscale)
specific_energy_X_train, specific_energy_X_validation, specific_energy_Y_train, specific_energy_Y_validation = \
    train_test_split(xscale, specific_yscale)
productivity_X_train, productivity_X_validation, productivity_Y_train, productivity_Y_validation = \
    train_test_split(xscale, recovery_yscale)

"""-----------------------------------------------------------------------------------------------------"""


"""--------------------------------- Build Neural Network -----------------------------------------------"""


def build_model():
    model = Sequential()
    model.add(Dense(16, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(8, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    return model


"""--------------------------------------------------------------------------------------------------------"""

"""-------------------------- Compile Model and Assign them to each model ---------------------------------"""
model_recovery = build_model()
model_purity = build_model()
model_specific_energy = build_model()
model_productivity = build_model()

model_recovery.summary()
model_purity.summary()
model_specific_energy.summary()
model_productivity.summary()
"""-----------------------------------------------------------------------------------------------------------"""

"""------------------------------------ Train the each model with corresponding values -----------------------"""
recovery_hist = model_recovery.fit(recovery_X_train, recovery_Y_train, epochs=150,
                                   batch_size=None, verbose=1, validation_split=0.2)
purity_hist = model_purity.fit(purity_X_train, purity_Y_train, epochs=150,
                               batch_size=None, verbose=1, validation_split=0.2)
specific_energy_hist = model_specific_energy.fit(specific_energy_X_train, specific_energy_Y_train, epochs=150,
                                                 batch_size=None, verbose=1, validation_split=0.2)
productivity_hist = model_productivity.fit(productivity_X_train, productivity_Y_train, epochs=150,
                                           batch_size=None, verbose=1, validation_split=0.2)
"""--------------------------------------------------------------------------------------------------------"""

"""------------------- visualise loss ----------------------------------------------------------------------"""
plt.figure(1)
plt.plot(recovery_hist.history['loss'])
plt.plot(recovery_hist.history['val_loss'])
plt.title('Recovery Loss V.S. Recovery Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()
plt.close()

plt.figure(2)
plt.plot(purity_hist.history['loss'])
plt.plot(purity_hist.history['val_loss'])
plt.title('Purity Loss V.S. Purity Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()
plt.close()

plt.figure(3)
plt.plot(specific_energy_hist.history['loss'])
plt.plot(specific_energy_hist.history['val_loss'])
plt.title('specific_energy_hist Loss V.S. specific_energy_hist Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()
plt.close()

plt.figure(4)
plt.plot(productivity_hist.history['loss'])
plt.plot(productivity_hist.history['val_loss'])
plt.title('productivity_hist Loss V.S. productivity_hist Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()
plt.close()
"""--------------------------------------------------------------------------------------------------------"""

"""-------------------------------------- Prediction - Use the test set - normalisation --------------------"""
test_features_normalisation = scaler_x.transform(test_features)
Recovery_predicted_test_targets = model_recovery(test_features_normalisation)
Purity_predicted_test_targets = model_purity(test_features_normalisation)
SpecificEnergy_predicted_test_targets = model_specific_energy(test_features_normalisation)
Productivity_predicted_test_targets = model_productivity(test_features_normalisation)
""" ----------------------------------------------------------------------------------------------------------"""
# print those inverse numbers
""" --------------------------------------- Invert Normalisation --------------------------------------------"""
Recovery_final_test_targets = scaler_y_recovery.inverse_transform(Recovery_predicted_test_targets)
Purity_final_test_targets = scaler_y_purity.inverse_transform(Purity_predicted_test_targets)
SpecificEnergy_final_test_targets = scaler_y_specific_energy.inverse_transform(SpecificEnergy_predicted_test_targets)
Productivity_final_test_targets = scaler_y_productivity.inverse_transform(Productivity_predicted_test_targets)
""" ---------------------------------------------------------------------------------------------------------"""

"""--------------------------------- Reshape all the values into the same dimensions -----------------------
Although I believe that they all have the same length, just in case, still define four 'XXX_length'"""
Recovery_length = Recovery_predicted_test_targets.shape[0]
Purity_length = Purity_predicted_test_targets.shape[0]
SpecificEnergy_length = SpecificEnergy_predicted_test_targets.shape[0]
Productivity_length = Productivity_predicted_test_targets.shape[0]

Recovery_targets = Recovery_final_test_targets.reshape((Recovery_length, 1))
Purity_targets = Purity_final_test_targets.reshape((Purity_length, 1))
SpecificEnergy_targets = SpecificEnergy_final_test_targets.reshape((SpecificEnergy_length, 1))
Productivity_targets = Productivity_final_test_targets.reshape((Productivity_length, 1))

## Reshape the test targets
recovery_test_targets_s = recovery_test_targets.reshape((Recovery_length, 1))
purity_test_targets_s = purity_test_targets.reshape((Recovery_length, 1))
specific_energy_test_targets_s = specific_energy_test_targets.reshape((Recovery_length, 1))
productivity_test_targets_s = productivity_test_targets.reshape((Recovery_length, 1))
"""--------------------------------------------------------------------------------------------------------"""

"""------------------------------------ Compute absolute precision and fractional precision ---------------"""
print(Recovery_targets.shape)
print(recovery_test_targets_s.shape)

Recovery_error = abs(Recovery_targets - recovery_test_targets_s)
print('Absolute errors - Recovery \n', Recovery_error, '\n')
Purity_error = abs(Purity_targets - purity_test_targets_s)
print('Absolute errors - Purity \n', Purity_error, '\n')
SpecificEnergy_error = abs(SpecificEnergy_targets - specific_energy_test_targets_s)
print('Absolute errors - Specific Energy \n', SpecificEnergy_error, '\n')
Productivity_error = abs(Productivity_targets - productivity_test_targets_s)
print('Absolute errors - Productivity \n', Productivity_error, '\n')

Recovery_precision = abs(1 - Recovery_error / Recovery_targets)
print('Estimated Precision - Recovery \n', Recovery_precision, '\n')
Purity_precision = abs(1 - Purity_error / Purity_targets)
print('Estimated Precision - Purity \n', Purity_precision, '\n')
SpecificEnergy_precision = abs(1 - SpecificEnergy_error / SpecificEnergy_targets)
print('Estimated Precision - Specific Energy \n', SpecificEnergy_precision, '\n')
Productivity_precision = abs(1 - Productivity_error / Productivity_targets)
print('Estimated Precision - Productivity \n', Productivity_precision, '\n')
"""--------------------------------------------------------------------------------------------------------"""

"""------------------------------------------ Get the average precision ----------------------------------"""
Recovery_ave = np.mean(Recovery_precision)
print("Average Precision - Recovery:", 100 * Recovery_ave, "%", '\n')
Purity_ave = np.mean(Purity_precision)
print("Average Precision - Purity:", 100 * Purity_ave, "%", '\n')
Specific_Energy_ave = np.mean(SpecificEnergy_precision)
print("Average Precision - Specific Energy:", 100 * Specific_Energy_ave, "%", '\n')
Productivity_ave = np.mean(Productivity_precision)
print("Average Precision - Productivity:", 100 * Productivity_ave, "%")
# -------------------------------------------------------------------------------------------------------
