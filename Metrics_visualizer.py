import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

## Variables for angle graphs
left_arm_angle_time = []
right_arm_angle_time = []
left_leg_angle_time = []
right_leg_angle_time = []
time = []

def moving_avg(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')


# Store Data for the subplots
data_lists = [left_arm_angle_time, right_arm_angle_time, left_leg_angle_time, left_leg_angle_time]
data_labels = ['Left Arm Angle', 'Right Arm Angle', 'Left Leg Angle', 'Overall Accuracy']

dfR = pd.read_csv('Data.csv')
time = dfR.iloc[:,0]
for i in range(4):
    data_lists[i] = dfR.iloc[:,i+1]

window_size = 10
# Create the third graph with a 2x2 layout
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

colors = ['blue', 'green', 'red', 'purple']
# Loop through the subplots and plot each data list
for i, ax in enumerate(axes.flatten()):
    ax.plot(time, data_lists[i], linestyle='--', color=colors[i], label=data_labels[i])
    ax.plot(time, moving_avg(data_lists[i], window_size), color=colors[i], label="Smoothed"+data_labels[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Angle')
    ax.set_title(data_labels[i])
    ax.grid(True)

# Adjust layout
plt.tight_layout()
# Show the second graph
plt.show()