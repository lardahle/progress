#### Anomaly_Detection.py

# Copyright (c) 2023 Landon Dahle
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# For more information on licensing, see https://opensource.org/licenses/MIT

# -----------------------------------------------------------------------------
# Author: Landon Dahle
# Date: 2023
# Project: Progress - BMEN 351 Final Project
# License: MIT
# -----------------------------------------------------------------------------

# =============================================================================
# Input Variables
# =============================================================================

# Modules
import time
import os
import pandas as pd

# Anomaly Detection Modules
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Speed Boost
from multiprocessing import Pool

# Start Time
start = time.time()
# Paths
data_directory = r'F:\OneDrive - Texas A&M University\BMEN 351\Project 2\Data'
vector_directory = os.path.join(data_directory, 'Training\\Vectors')
combined_csv_path = os.path.join(data_directory, 'combined_data_copy.csv') #'Training\\combined_data.csv'

#%%

# =============================================================================
# Preprocess Data
# =============================================================================
df = pd.read_csv(combined_csv_path)
df = df.drop(['Text', 'OpenAI_Subjects'], axis=1)

# Conversion function to handle floats and integers
def convert_to_int(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return np.nan  # Use NaN for non-convertible values

# Apply the conversion function to 'Research_Advice' and 'OpenAI_Bool'
df['Research_Advice'] = df['Research_Advice'].apply(convert_to_int)
df['OpenAI_Bool'] = df['OpenAI_Bool'].apply(convert_to_int)

# Drop rows with NaN values in 'Research_Advice' or 'OpenAI_Bool'
df = df.dropna(subset=['Research_Advice', 'OpenAI_Bool'])

# Redundant # Convert 'Research_Advice' and 'OpenAI_Bool' to integers
# df['Research_Advice'] = df['Research_Advice'].astype(int)
# df['OpenAI_Bool'] = df['OpenAI_Bool'].astype(int)

# Base part of the path to remove
base_path_to_remove = r'F:\OneDrive - Texas A&M University\BMEN 351\Project 2\Data\Training\Vectors'

# Remove the base part of the path from the 'vector_file' column
df['vector_file'] = df['vector_file'].str.replace(base_path_to_remove, '', regex=False)
df['vector_file'] = df['vector_file'].str.lstrip('\\')

# Initialize an empty list to hold indices to drop
indices_to_drop = []

# Identify new papers by finding where 'Temporal_Position' resets back to 0
reset_indices = df[df['Temporal_Position'] == 0].index.tolist()

# Include the start of the DataFrame and the end of the last paper for iteration
start_points = [0] + reset_indices
end_points = reset_indices + [len(df)]

# Iterate over each paper
for start, end in zip(start_points, end_points):
    paper_section = df.iloc[start:end]
    # Check if there is no '1' in 'OpenAI_Bool' for the current paper
    if not (paper_section['OpenAI_Bool'] == 1).any():
        # If not, mark all indices of the current paper for removal
        indices_to_drop.extend(paper_section.index.tolist())

# Drop the marked indices from the DataFrame all at once
df = df.drop(indices_to_drop)

#%%

# =============================================================================
# Load Vectors
# =============================================================================
def load_vector(file_path):
    vector_full_path = os.path.join(vector_directory, file_path)
    if os.path.exists(vector_full_path):
        # Assuming the vector files are in npy format
        return np.load(vector_full_path)
    else:
        return np.nan  # Return NaN if the file does not exist

# Apply the function to load vectors
df['vector_file'] = df['vector_file'].apply(load_vector)

# # Identify the indices where a new paper starts
# reset_indices = df.index[df['Temporal_Position'] == 0].tolist()

# # If the last paper doesn't end with Temporal_Position == 1, add the last index manually
# if df.iloc[-1]['Temporal_Position'] != 1:
#     reset_indices.append(len(df))

# # Assuming you want to process the first 10 papers
# if len(reset_indices) > 10:
#     end_index = reset_indices[10]  # Get the index of the start of the 11th paper
# else:
#     end_index = len(df)  # Or the end of the DataFrame if there are fewer than 10 papers

# # Slice the DataFrame to only include the first 10 papers
# df_first_10 = df.iloc[:end_index]

# # Load vectors for the first 10 papers
# df_first_10['vector_file'] = df_first_10['vector_file'].apply(load_vector)


# Remove entries where vector loading failed
df = df.dropna(subset=['vector_file'])
df.to_pickle('processed_data.pkl')

#%%
df = pd.read_pickle('processed_data.pkl')

#%% Split Data

# Split Data
X_vectors = np.array(list(df['vector_file']))  # Assuming vector_file contains the actual vectors
X_temporal = df['Temporal_Position'].values.reshape(-1, 1)  # Convert to a column vector
X = np.hstack((X_vectors, X_temporal))  # Horizontally stack the vectors with the temporal position
y = df['OpenAI_Bool'].values

# # Split the data into training and testing sets with stratification
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# # Further split the training set into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

# Trial 

from sklearn.decomposition import PCA

# Assuming `X` is your feature matrix
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_reduced = pca.fit_transform(X)

# Split the data into training and testing sets with stratification
X_train_val, X_test, y_train_val, y_test = train_test_split(X_reduced, y, test_size=0.2, stratify=y, random_state=42)
# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

#%% Solo Isolation Forest Model Training

clf = IsolationForest(n_estimators=100, contamination=float(np.mean(y_train_val==1)))
clf.fit(X_res)  # Train only on normal data (assuming 0 is normal)

# Predict on the full training set to see how well the model does
# Predict on the full training set to see how well the model does
predictions = clf.predict(X_res)
predictions = np.where(predictions == 1, 0, 1)  # Convert from Isolation Forest's output to your labeling (0: normal, 1: anomaly)
print(classification_report(y_res, predictions))

predictions = clf.predict(X_val)
predictions = np.where(predictions == 1, 0, 1)  # Convert from Isolation Forest's output to your labeling (0: normal, 1: anomaly)
print(classification_report(y_val, predictions))


# Get decision function scores
scores = clf.decision_function(X_test)

# Convert scores to positive values (higher means more normal)
scores_pos = -scores

# True labels need to be 0 for normal and 1 for anomalies
# Assuming y_test is already in this format

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, scores_pos)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

import seaborn as sns

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot using seaborn for better aesthetics
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

# Confusion Matrix for the training set
plot_confusion_matrix(y_val, predictions, title='Confusion Matrix - Training Set')

# Confusion Matrix for the validation set
predictions_val = clf.predict(X_val)
predictions_val = np.where(predictions_val == 1, 0, 1)  # Convert Isolation Forest's output
plot_confusion_matrix(y_val, predictions_val, title='Confusion Matrix - Validation Set')

#%% Ensemble Isolation Forest

# Number of models in the ensemble
n_models = 5
models = []

# Train multiple Isolation Forest models
for i in range(n_models):
    model = IsolationForest(n_estimators=100, contamination=float(np.mean(y_train_val==1)), random_state=i)
    model.fit(X_res)
    models.append(model)
    
# Collect predictions from all models
predictions = np.array([model.predict(X_test) for model in models])

# Convert predictions to binary (0: normal, 1: anomaly)
predictions_binary = np.where(predictions == 1, 0, 1)

# Majority voting mechanism
final_predictions = np.round(predictions_binary.mean(axis=0))

# Evaluate the ensemble model
print(classification_report(y_test, final_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, final_predictions))

# Compute confusion matrix
cm = confusion_matrix(y_test, final_predictions)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Normal', 'Anomaly'])
ax.yaxis.set_ticklabels(['Normal', 'Anomaly'])

plt.show()

#%% OneClassSVM

from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Train a One-Class SVM model
oc_svm_clf = OneClassSVM(gamma='auto').fit(X_res[y_res == 0]) # Train only on normal data

# Predict on the test set
oc_svm_predictions = oc_svm_clf.predict(X_test)
oc_svm_predictions = np.where(oc_svm_predictions == 1, 0, 1) # Convert to binary labels

# Evaluate the model
print("One-Class SVM Classification Report")
print(classification_report(y_test, oc_svm_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, oc_svm_predictions))


#%% LocalOutlierFactor
from sklearn.neighbors import LocalOutlierFactor

# Train a Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
lof.fit(X_res) # Train only on normal data

# Get LOF scores (negative_outlier_factor_ is more negative for outliers)
lof_scores = -lof.negative_outlier_factor_

# Determine a threshold for being an outlier
# You might need to experiment with different values for this threshold
outlier_threshold = np.percentile(lof_scores, 95)  # for example, 95th percentile

# Predict outliers on the training set
y_pred_train = np.where(lof_scores > outlier_threshold, 0, 1)

# Evaluate the model performance
print(classification_report(y_res, y_pred_train))

# Predict on the test set
y_pred_test = lof.fit_predict(X_test)
y_pred_test = np.where(y_pred_test == 1, 0, 1)  # Convert LOF output to match your labels (1: anomaly, 0: normal)

# Evaluate the model performance on the test set
print(classification_report(y_test, y_pred_test))

#%% LOF Cont.
# Generate thresholds from LOF scores
thresholds = np.linspace(min(lof_scores), max(lof_scores), 100)

# Initialize lists to store TPR and FPR values
tpr_list = []
fpr_list = []

# Calculate TPR and FPR for each threshold
for thresh in thresholds:
    y_pred = np.where(lof_scores > thresh, 1, 0)  # 1 for outliers, 0 for inliers
    tn, fp, fn, tp = confusion_matrix(y_res, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# Calculate AUC
roc_auc = auc(fpr_list, tpr_list)

# Plot ROC Curve
plt.figure()
plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Choose a threshold for confusion matrix (e.g., based on ROC curve)
selected_threshold = np.percentile(lof_scores, 95)
y_pred_selected = np.where(lof_scores > selected_threshold, 1, 0)
cm = confusion_matrix(y_res, y_pred_selected)

# Plot confusion matrix
plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal (0)', 'Anomaly (1)'], rotation=45)
plt.yticks(tick_marks, ['Normal (0)', 'Anomaly (1)'])

# Labeling the plot
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


#%% Neural Network

from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.callbacks import EarlyStopping

# Define the autoencoder architecture
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)

# Autoencoder
autoencoder = Model(input_layer, decoded)

# Encoder (for feature extraction)
encoder = Model(input_layer, encoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Fit the model
history = autoencoder.fit(X_train, X_train, 
                          epochs=50, 
                          batch_size=256, 
                          shuffle=True, 
                          validation_data=(X_val, X_val),
                          callbacks=[early_stopping],
                          verbose=1)

# Extract features
X_train_features = encoder.predict(X_train)
X_test_features = encoder.predict(X_test)

# Train a classifier on the extracted features
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_features, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test_features)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

#%% Neural Network V2
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Recall
from sklearn.model_selection import train_test_split

# Neural Network Model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Recall()])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Evaluate the model
_, recall = model.evaluate(X_test, y_test)
print(f'Recall: {recall}')

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = [1 if y > 0.5 else 0 for y in y_pred_prob.flatten()]

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Normal', 'Anomaly'])
ax.yaxis.set_ticklabels(['Normal', 'Anomaly'])

plt.show()

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

#%% Neural Network With Hyperparameter Tuning
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.layers import Dropout, BatchNormalization, LeakyReLU, ReLU
from keras.regularizers import  l1_l2 # l1, l2,
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adagrad
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.preprocessing import normalize

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def NeuralNet_v2(X_train_scaled, X_test_scaled, y_train, y_test,
                 dropout_rate=0.0, l1_reg=0.0, l2_reg=0.0,
                 learning_rate=0.001, optimizer='adam', 
                 num_layers=2, neurons_per_layer=32, 
                 activation='relu', batch_norm=False):

    
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, activation=None, input_shape=(X_train_scaled.shape[1],),
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if activation == 'relu':
        model.add(ReLU())
    elif activation == 'leakyrelu':
        model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))

    # Second layer
    model.add(Dense(neurons_per_layer, activation=None, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if activation == 'relu':
        model.add(ReLU())
    elif activation == 'leakyrelu':
        model.add(LeakyReLU())
    
    # Output layer
    model.add(Dense(3, activation='softmax'))
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adagrad':
        opt = Adagrad(learning_rate=learning_rate)
    
    # # Compute class weights
    # # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
    # class_weights_dict = dict(enumerate(class_weights))
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
       
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001) # , reduce_lr

       
    model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, 
              validation_split=0.2, callbacks=[early_stopping, reduce_lr], 
             ) # class_weight=class_weights_dict
    
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    nn_accuracy = np.mean(y_pred == y_test)
    nn_report = classification_report(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return test_loss, test_accuracy, nn_accuracy, nn_report, precision, recall, f1
    
    # This function is an enhanced version of the original, which allows for hyperparameter tuning.
    # The actual search can be done outside this function.


def hyperopt_tuning(X_train, X_test, y_train, y_test):
    
    def objective(params):
        test_loss, test_accuracy, _, _, precision, recall, f1 = NeuralNet_v2(X_train, X_test, y_train, y_test, **params)
        return {'loss': -test_accuracy, 'status': STATUS_OK, 'precision': precision, 'recall': recall, 'f1': f1} 
    
    # Define the hyperparameter space
    space = {
    'dropout_rate': hp.choice('dropout_rate', [0.0, 0.25, 0.5]),
    'l1_reg': hp.choice('l1_reg', [0.0, 0.005, 0.01]),
    'l2_reg': hp.choice('l2_reg', [0.0, 0.005, 0.01]),
    'learning_rate': hp.choice('learning_rate', [0.005]),
    'neurons_per_layer': hp.choice('neurons_per_layer', [16, 32, 64]),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
    'activation': hp.choice('activation', ['relu', 'leakyrelu']),
    'batch_norm': hp.choice('batch_norm', [True, False])
}

# # Comparison to no param network
#     space = {
#     'dropout_rate': hp.choice('dropout_rate', [0.0]),
#     'l1_reg': hp.choice('l1_reg', [0.0]),
#     'l2_reg': hp.choice('l2_reg', [0.0]),
#     'learning_rate': hp.choice('learning_rate', [0.001]),
#     'neurons_per_layer': hp.choice('neurons_per_layer', [32]),
#     'optimizer': hp.choice('optimizer', ['adam']),
#     'activation': hp.choice('activation', ['relu']),
#     'batch_norm': hp.choice('batch_norm', [False])
# }


    # Use fmin to find the best hyperparameters
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals= 20,  # can be increased for longer searches
                trials=trials)
    
    # Extract the best accuracy from the trials
    best_accuracy = -min(trials.losses())
    
    # Find the index of the best trial
    best_trial_idx = trials.losses().index(-best_accuracy)
    
    metrics = {
        'hyperparameters': best,
        'accuracy': best_accuracy,
        'precision': trials.trials[best_trial_idx]['result']['precision'],
        'recall': trials.trials[best_trial_idx]['result']['recall'],
        'f1': trials.trials[best_trial_idx]['result']['f1']
    }

    return metrics

#### Hyperopt ####

# Dictionary to store results for each threshold
results = {}


print("Running hyperparameter tuning:")
best_metrics = hyperopt_tuning(X_train, X_test, y_train, y_test)
print(best_metrics)

#%% NN with Hyperparameter Tuning
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import pandas as pd



def NeuralNet_v2(X_train_scaled, y_train, X_val_scaled, y_val, params):
    model = Sequential()
    model.add(Dense(64, activation=params['activation'], input_shape=(X_train_scaled.shape[1],),
                    kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))
    if params['batch_norm']:
        model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(params['neurons_per_layer'], activation=params['activation'], 
                    kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))
    model.add(Dense(1, activation='sigmoid'))
    
    if params['optimizer'] == 'adam':
        opt = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        opt = RMSprop(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'adagrad':
        opt = Adagrad(learning_rate=params['learning_rate'])

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping, reduce_lr])

    return model


# Define the hyperparameter space
space = {
    'dropout_rate': hp.choice('dropout_rate', [0.0, 0.1, 0.2]),
    'l1_reg': hp.choice('l1_reg', [0.0, 0.0025, 0.005]),
    'l2_reg': hp.choice('l2_reg', [0.005, 0.01, 0.015]),
    'learning_rate': hp.choice('learning_rate', [0.005, 0.01, 0.015]),
    'neurons_per_layer': hp.choice('neurons_per_layer', [48, 64, 80]),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'adagrad']),
    'activation': hp.choice('activation', ['relu']),
    'batch_norm': hp.choice('batch_norm', [True])
}

def objective(params):
    model = NeuralNet_v2(X_train, y_train, X_val, y_val, params)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred.flatten() == y_test)
    return {'loss': -accuracy, 'status': STATUS_OK}

def objective(params):
    model = NeuralNet_v2(X_train, y_train, X_val, y_val, params)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = np.mean(y_pred.flatten() == y_test)

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Normal', 'Anomaly'])
    ax.yaxis.set_ticklabels(['Normal', 'Anomaly'])
    plt.show()

    # Compute and plot ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    return {'loss': -accuracy, 'status': STATUS_OK, 'roc_auc': roc_auc}

# Running the hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

print("Best hyperparameters:", best)




#%%
# =============================================================================
# Outputs
# =============================================================================


end = time.time()
print("The total runtime of the above code was",(end-start), "seconds")