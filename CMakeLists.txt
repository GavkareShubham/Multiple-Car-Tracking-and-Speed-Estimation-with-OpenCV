cmake_minimum_required(VERSION 3.10)
project(CarTracking)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CarTracking main.cpp CarTracker.cpp)
target_link_libraries(CarTracking ${OpenCV_LIBS})
