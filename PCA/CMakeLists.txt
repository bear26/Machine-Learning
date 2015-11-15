project(PCA)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)

add_definitions("-std=c++11 -Wall")

find_package(OpenCV REQUIRED)
find_library(Boost REQUIRED COMPONENTS system)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
