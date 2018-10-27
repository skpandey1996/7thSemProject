# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 01:17:16 2018

@author: Nayan
"""
# Import libraries
import numpy as np
import pandas as pd
import os
os.environ["PATH"] += os.pathsep + 'D:\Program Files (x86)\Graphviz2.38\bin'

# Read in data
dataset = pd.read_csv('main_assam3.csv')
#print(dataset.shape)

# Standardize columns
cols_to_std = ['districts','y05area','y05yield','y06area','y06yield',
                'y07area','y07yield','y08area','y08yield',
                'y09area','y09yield','y10area','y10yield',
                'y11area','y11yield','y12area','y12yield',
                'y13area','y13yield','y14area','y14yield']

dataset[cols_to_std] = dataset[cols_to_std].apply(lambda x: (x-x.mean()) / x.std())

# Targets are the values we want to predict
targets = np.array(dataset['y14yield'])
dist = np.array(dataset['districts'])

# axis 1 refers to the columns
dataset = dataset.drop('y14yield', axis = 1)
#dataset = dataset.drop('districts', axis = 1)

# Saving feature names for later use
feature_list = cols_to_std

# Convert to numpy array
dataset = np.array(dataset)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(dataset, targets, test_size = 0.25, random_state = 42)

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

baseline_preds = test_features[:, feature_list.index('y13yield')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
print(predictions)
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
#%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()

true_data = pd.DataFrame(data = {'districts': dist, 'actual': targets})
test_districts = test_features[:, feature_list.index('districts')]
predictions_data = pd.DataFrame(data = {'districts': test_districts , 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['districts'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['districts'], predictions_data['prediction'], 'ro', label = 'prediction') 
plt.legend()
# Graph labels
plt.xlabel('Districts')
plt.ylabel('Crop output')
plt.title('Actual and Predicted Values')