import math


## PARAMETERS FOR DTW-MT ##

ALPHA_T = 0.0075   # meters  
ALPHA_R = math.radians(3.75) 
BETA = 1.
GAMMA = 4.



## ENABLES ROS-based FUNCTIONS ## 
# will affect metric computation slightly.
# If you do not plan to install ROS and still be able to use tools in this directory, 
#  disable following parameter.
USE_ROS = True

