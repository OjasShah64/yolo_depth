# yolo_depth
This repo is a YOLO and Depth Anything Vision Pipeline trained for greenhouse environments. YOLOv8 Model is trained using Label Studio and is able to be further trained or replaced with another model. The Depth Anything model is calibrated to convert monocular depth to metric depth providing accurate depth readings in meters. This pipeline can be used with both live video and rosbag files.

Must use ROS 1 to run - can use ROS 2 but will need to update dependencies 

1. run yolo_depth_rosbag to store Depth Anything values in new file called depth_pairs.py
  
2. run calibration file to identify scale factor / values for Depth Anything of A and B in the following equation: current value = A * real depth value + B
   - This file will graph the values on a linear plot and show the equation on the graph from which you can identify the values that will be the best calibration for the model. 
