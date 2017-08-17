cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0015 NEW)
#set project name
set(PRJ_NAME parser)
project(${PRJ_NAME})

#libraries to link
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED)

#### FIX THIS ##########
include_directories(/usr/local/include/eigen3)
include_directories(/home/badami/gurobi650/linux64/include)
#### /FIX THIS ##########

set(LINK_LIBS "${Boost_LIBRARIES} ${OpenCV_LIBS}")

message(STATUS "${Boost_INCLUDE_DIRS}")

#define cflags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "Use c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    message(STATUS "Use c++0x.")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#define header include directories as list
set(Parser_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
message(STATUS ${PROJECT_SOURCE_DIR})
message(STATUS ${Parser_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )
SET( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )

include_directories(lib/libforest/include)
include_directories(lib/rapidjson/include)
include_directories(lib)
include_directories(${Parser_INCLUDE_DIRS})

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 
#add_subdirectory(lib/googletest)
add_subdirectory(lib/libforest)
add_subdirectory(src)
#### FIX THIS ##########
link_directories(/home/badami/gurobi650/linux64/lib)
#### /FIX THIS ##########

add_executable (cli cli/main.cpp)
target_link_libraries(cli ${PRJ_NAME} boost_system boost_filesystem libforest ${OpenCV_LIBS} gurobi_c++ gurobi65 )

#set enable testing
enable_testing()
