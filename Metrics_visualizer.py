import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

## Variables for angle graphs
Variable_1 = []
Variable_2 = []
Variable_3 = []
time = []

def moving_avg(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')


# Read data for the subplots
data_lists = [Variable_1, Variable_2, Variable_3]
data_labels = ['Left Arm Angle', 'Right Hand Track', 'Other']
df = pd.read_csv('Data.csv')
time = df.iloc[:,0]
for i in range(3):
    data_lists[i] = df.iloc[:,i+1]


window_size = 10

# Create the third graph with a 2x2 layout
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
colors = ['blue', 'green', 'red', 'purple']

# Each graph 
# FIRST PLOT
axes[0, 0].plot(time, moving_avg(data_lists[0], window_size), color=colors[0], label="Smoothed"+data_labels[0])
axes[0, 0].set_title(data_labels[0])

# SECOND PLOT
axes[0, 1].plot(time, moving_avg(data_lists[1], window_size), color=colors[1], label="Smoothed"+data_labels[1])
axes[0, 1].set_title(data_labels[1])

# THIRD PLOT
axes[1, 0].plot(time, moving_avg(data_lists[2], window_size), color=colors[2], label="Smoothed"+data_labels[2])
axes[1, 0].set_title(data_labels[2])

# FOURTH PLOT -> Text to show accuracy
text_kwargs = dict(ha='center', va='center', fontsize=28, color='C1')
acc = 5
text = "Global Accuracy: "
text2show = text + str(acc) + '%'
axes[1, 1].text(0.5, 0.5, text2show , **text_kwargs) #Understand how it works


# Adjust layout
plt.tight_layout()
# Show the second graph
plt.show()