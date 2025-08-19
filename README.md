# yolo_depth

1. run yolo_depth_rosbag to store Depth Anything values in new file - depth_pairs.py
  
2. run calibration file to identify scale factor / values of A and B in the following equation: current value = A * real depth value + B
   - This file will graph the values on a linear plot and show the equation on the graph from which you can identify the values 
