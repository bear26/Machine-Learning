project(LinearRegression)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -Wall")

find_package(Boost REQUIRED COMPONENTS system)

add_executable(${PROJECT_NAME} ${SRC_LIST})

