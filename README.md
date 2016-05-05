# pcml
Point Cloud and Machine Learning

## Requirements
* ROS indigo
* ROS Eigen
* Moveit!
* Kinect hardware, v1.x
* Kinect Linux driver (PrimeSense? SensorKinect?)
* openni_launch
 * $ sudo apt-get install ros-indigo-openni-launch
* openni_tracker
 * $ sudo apt-get install ros-indigo-openni-tracker
* libsvm
 * put 'svm.h', 'svm.cpp' into lib/

## Components
* msgs
 * FutureObstacleDistribution
 * FutureObstacleDistributions
* Nodes
 * future_obstacle_publisher  
     **Parameters**  
       input_stream_type (string, default: realtime)  
         One of "realtime", "cad120"  
       joints_type (string, default: upper_body)  
         One of "whole_body", "upper_body"  
       render (bool, default: false)  
       cad120_directory (string, required when input_stream_type = "cad120")  
     **Subscribed Topics**  
       Topics published by openni_tracker (/head_1, /neck_1, etc.)  
     **Published Topics**  
       future_obstacle_publisher/future_obstacle_distributions (pcml/FutureObstacleDistributions)
 * future_obstacle_visualizer  
     **Subscribed topics**  
       future_obstacle_publisher/future_obstacle_distributions (pcml/FutureObstacleDistributions)  
     **Published Topics**  
       /future_obstacle_distributions_marker_array (visualization_msgs/MarkerArray)
* Launches
 * future_obstacle_publisher.launch

## Build
* Package is organized using rosbuild  
  $ rosmake

## Run
$ roslaunch openni_launch openni.launch  
$ rosrun openni_tracker openni_tracker  
$ rosrun rviz rviz  
$ roslaunch pcml future_obstacle_publisher  
 * Make the openni_tracker track somebody
 * Modify .launch file parameters
