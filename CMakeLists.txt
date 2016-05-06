cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# compiler option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

## show all files (http://wiki.ros.org/IDEs#QtCreator)
#Add all files in subdirectories of the project in
# a dummy_target so qtcreator have access to all files
FILE(GLOB children ${CMAKE_SOURCE_DIR}/*)
FOREACH(child ${children})
  IF(IS_DIRECTORY ${child})
    file(GLOB_RECURSE dir_files "${child}/*")
    LIST(APPEND extra_files ${dir_files})
  ENDIF()
ENDFOREACH()
add_custom_target(dummy_${PROJECT_NAME} SOURCES ${extra_files})

# eigen
find_package(Eigen3 REQUIRED)

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)
#set(ROS_BUILD_TYPE Debug)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

include_directories(
    ${PROJECT_SOURCE_DIR}/lib
    ${PROJECT_SOURCE_DIR}/include
    ${Eigen_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

rosbuild_add_library(pcml
    include/${PROJECT_NAME}/data/cad120_reader.h
    include/${PROJECT_NAME}/data/skeleton_stream.h
    include/${PROJECT_NAME}/learning/train_future_motion.h
    src/data/cad120_reader.cpp
    src/data/skeleton_stream.cpp
    src/learning/train_future_motion.cpp

    # libsvm
    lib/libsvm/svm.h
    lib/libsvm/svm.cpp
)

target_link_libraries(pcml
    ${YAML_CPP_LIBRARIES}
)

rosbuild_add_executable(test_cad120 src/test_cad120.cpp)
rosbuild_add_executable(test_skeleton_stream src/test_skeleton_stream.cpp)
rosbuild_add_executable(kinect_frequency src/kinect_frequency.cpp)
rosbuild_add_executable(future_obstacle_publisher src/future_obstacle_publisher.cpp)
rosbuild_add_executable(future_obstacle_visualizer src/future_obstacle_visualizer.cpp)
add_executable(train_cad120 src/train_cad120.cpp)
add_executable(infer_cad120 src/infer_cad120.cpp)

target_link_libraries(test_cad120
    pcml
)
target_link_libraries(test_skeleton_stream
    pcml
)
target_link_libraries(kinect_frequency
)
target_link_libraries(future_obstacle_publisher
    pcml
)
target_link_libraries(future_obstacle_visualizer
    pcml
)
target_link_libraries(train_cad120
    pcml
)
target_link_libraries(infer_cad120
    pcml
)
