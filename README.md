# YOLO Depth: Monocular Vision Pipeline for Greenhouses

##Project Summary & Value
This project is a computer vision pipeline integrating **YOLOv8** for object detection and **Depth Anything** for monocular depth estimation, specifically calibrated for greenhouse environments. The pipeline processes both live video and ROS bag files, converting relative monocular depth maps into accurate metric depth readings (in meters) via a custom calibration script.

**Why I found this valuable:** 
This project gave me practical, hands-on experience deploying state-of-the-art AI models (YOLOv8, Depth Anything) to solve real-world agricultural problems. I learned how to handle custom datasets using Label Studio, calibrate non-metric AI outputs to physical real-world measurements using linear regression techniques, and integrate deep learning vision models seamlessly into a ROS environment.

##Features & Usage

This repo is a YOLO and Depth Anything Vision Pipeline trained for greenhouse environments. YOLOv8 Model is trained using Label Studio and is able to be further trained or replaced with another model. The Depth Anything model is calibrated to convert monocular depth to metric depth providing accurate depth readings in meters. This pipeline can be used with both live video and rosbag files.

Must use ROS 1 to run - can use ROS 2 but will need to update dependencies 

1. run yolo_depth_rosbag to store Depth Anything values in new file called depth_pairs.py
  
2. run calibration file to identify scale factor / values for Depth Anything of A and B in the following equation: current value = A * real depth value + B
   - This file will graph the values on a linear plot and show the equation on the graph from which you can identify the values that will be the best calibration for the model. 
